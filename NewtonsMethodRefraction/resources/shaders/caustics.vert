#version 430

// Input vertex attributes
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

// Input uniform values
uniform mat4 mvp;
uniform mat4 matModel;
uniform mat4 matView;
uniform mat4 matProjection;
uniform mat4 invProjection;
uniform mat4 matVP;
uniform mat4 invView;
uniform sampler2D depthBuffer; // Depth buffer to use for refraction calculations (usually a shadow map)
uniform sampler2D normalBuffer; // Normal buffer to use for refraction calculations
uniform sampler2D texNormal; // Normal map
uniform sampler2D texNormal2; // Second normal map
uniform float time;
uniform vec3 sunPos;

// Output vertex attributes (to fragment shader)
out vec3 oldPos;
out vec3 newPos;
out float shouldDiscard;

#define SCREEN_WIDTH 2000.0
#define SCREEN_HEIGHT 1600.0

#define WATER_IOR 1.33 // Index of refraction for water

vec3 GetWorldPosFromDepth(vec2 screenSpace, float ndcDepth)
{
    vec3 ndcCoords = vec3(2.0 * screenSpace - 1.0, ndcDepth);
    vec4 viewPos = invProjection * vec4(ndcCoords, 1);
    viewPos.xyz /= viewPos.w;
    viewPos.w = 1.0;
    vec3 worldPos = (invView * viewPos).xyz;
    return worldPos;
}

struct Ray
{
    vec3 start;
    vec3 dir;
};

struct Plane
{
    vec3 point;
    vec3 normal;
};

// Returns time t along ray where hit occurs
float PlaneRayIntersection(Plane plane, Ray ray)
{
    float denom = dot(plane.normal, ray.dir);
    if (abs(denom) > 0.0001f) // epsilon
    {
        float t = dot(plane.point - ray.start, plane.normal) / denom;
        return t;
    }
    return -1;
}

// refract: get the world-space coordinates from which to fetch colors for transparency post-refraction
// Arguments
// - vec2 seed: The screen-space coordinates of the pixel to be refracted
// - Ray refractedRay: The world-space ray to trace
vec3 runRefraction(vec2 seed, Ray refractedRay, out vec3 worldPos)
{
    // Find the depth at the screen Z-buffer directly behind the water
    float depth = texture(depthBuffer, seed).r; // Get depth from depth buffer
    float ndcDepth = depth * 2.0 - 1.0; // Convert to NDC depth
    worldPos = GetWorldPosFromDepth(seed, ndcDepth); // Get world position from depth

    // Find the normal at the screen normal buffer directly behind the water
    vec3 normal = texture(normalBuffer, seed).xyz; // Get normal from normal map
    normal = normalize(normal);

    // Generate a plane using worldPos and normal
    Plane plane;
    plane.point = worldPos;
    plane.normal = normal;

    // Intersect the rafracted ray with the plane
    float t = PlaneRayIntersection(plane, refractedRay);

    if (t >= 0)
    {
        // Find the world-space intersection point
        vec3 intersectionPoint = refractedRay.start + refractedRay.dir * t;
        return intersectionPoint;
    }
    return vec3(-1, -1, -1);
}

// Check if a point is within valid screen bounds
bool isPointValid(vec2 texCoords)
{
    return (texCoords.x <= 1.0 && texCoords.x >= 0.0 && texCoords.y <= 1.0 && texCoords.y >= 0.0);
}


float distanceSquared(vec2 a, vec2 b) {
    vec2 diff = a - b;
    return dot(diff, diff);
}

void swap (inout float a, inout float b) {
    float temp = a;
    a = b;
    b = temp;
}


// Raymarch for screen-space reflection (https://jcgt.org/published/0003/04/04/)
// ------------------------------------------------------------
// traceScreenSpaceRay (single-layer DDA, adapted from McGuire&Mara 2014)
// - csOrig: camera-space ray origin (vec3)
// - csDir : camera-space ray direction (vec3) (should be normalized)
// - proj  : projection matrix (mat4) that maps camera-space -> clip
// - csZBuffer : scene depth sampler (non-linear depth [0..1])
// - csZBufferSize: depth buffer size in pixels
// - zThickness : thickness to ascribe to each depth sample (camera-space units)
// - clipInfo : unused here (kept for compatibility)
// - nearPlaneZ, farPlaneZ : near/far for depth linearization (passed through to depthBufferToCameraSpace)
// - stride, jitter, maxSteps, maxDistance : tracing params
// - out hitPixelUV : out screen-space uv (0..1) of hit
// - out cameraSpaceHitPoint : out camera-space hit point (vec3)
// Returns true if hit found.
// NOTE: This is the single-layer specialization of the paper's traceScreenSpaceRay
// ------------------------------------------------------------
bool traceScreenSpaceRay(
    in vec3 csOrig,
    in vec3 csDir,
    in mat4 proj,
    in sampler2D csZBuffer,
    in vec2 csZBufferSize,
    in float zThickness,
    in vec3 clipInfo,
    in float nearPlaneZ,
    in float farPlaneZ,
    in float stride,
    in float jitter,
    const float maxSteps,
    in float maxDistance,
    out vec2 hitPixelUV,
    out vec3 cameraSpaceHitPoint)
{
    hitPixelUV = vec2(-1.0);
    cameraSpaceHitPoint = vec3(0.0);

    // --- Clip ray against near plane (paper: clip only far endpoint to near plane) ---
    // Note: paper expects a convention where greater z is farther; adjust if your camera-space uses the opposite sign.
    float rayLength = maxDistance;
    // If the ray will cross the near plane before maxDistance, clamp so end point lies on near plane
    if ((csOrig.z + csDir.z * maxDistance) > nearPlaneZ) {
        // avoid division by zero
        if (abs(csDir.z) > 1e-6) {
            rayLength = (nearPlaneZ - csOrig.z) / csDir.z;
            // if negative or tiny, keep maxDistance
            if (rayLength < 0.0) rayLength = maxDistance;
        }
    }
    vec3 csEndPoint = csOrig + csDir * rayLength;

    // --- Project both endpoints to clip and compute perspective-correct quantities ---
    vec4 H0 = proj * vec4(csOrig, 1.0);
    vec4 H1 = proj * vec4(csEndPoint, 1.0);

    // reciprocal w (k) and Q = cs * k (homogeneous interpolation domain)
    float k0 = 1.0 / H0.w;
    float k1 = 1.0 / H1.w;
    vec3 Q0 = csOrig * k0;
    vec3 Q1 = csEndPoint * k1;

    // Convert clip -> NDC -> pixel coordinates (paper works in pixel space)
    vec2 N0 = (H0.xy / H0.w) * 0.5 + 0.5; // 0..1
    vec2 N1 = (H1.xy / H1.w) * 0.5 + 0.5; // 0..1
    vec2 P0 = N0 * csZBufferSize; // pixel space
    vec2 P1 = N1 * csZBufferSize; // pixel space

    // Ensure we don't have degenerate zero-length line in pixel-space
    if (distanceSquared(P0, P1) < 0.000001) {
        // nudge slightly so delta.x won't be zero later (paper's trick).
        P1 += vec2(0.01, 0.0);
    }

    vec2 delta = P1 - P0;

    // Permute axes so we always step along X (reduces branching in the inner loop)
    bool permute = false;
    if (abs(delta.x) < abs(delta.y)) {
        permute = true;
        // swap x/y for P0,P1,delta
        delta = delta.yx;
        P0 = P0.yx;
        P1 = P1.yx;
    }

    // Step direction and inverse delta.x in pixel units
    float stepDir = sign(delta.x);
    float invdx = stepDir / delta.x; // safe because delta.x != 0 due to nudge above

    // Compute derivatives of Q and k with respect to screen x (in pixel steps)
    vec3 dQ = (Q1 - Q0) * invdx;
    float dk = (k1 - k0) * invdx;

    // dP is the per-step delta in pixel space
    vec2 dP = vec2(stepDir, delta.y * invdx);

    // Apply stride (spacing) and jitter
    dP *= stride;
    dQ *= stride;
    dk *= stride;

    P0 += dP * jitter;
    Q0 += dQ * jitter;
    k0 += dk * jitter;

    // Used to estimate ray Z sweep per-loop iteration (paper)
    float prevZMaxEstimate = csOrig.z;

    // Loop setup
    vec2 P = P0;
    float stepCount = 0.0;
    float end = P1.x * stepDir;

    // inner loop: iterate pixels touched by the projected ray (thin-line DDA)
    for (; (P.x * stepDir) <= end && (stepCount < maxSteps); P += dP, Q0.z += dQ.z, k0 += dk, stepCount += 1.0)
    {
        // map back permuted coordinates to original pixel coords
        vec2 hitPixelF = permute ? P.yx : P;
        // Convert to integer pixel coords (texelFetch uses integer coords)
        ivec2 hitPix = ivec2(floor(hitPixelF + vec2(0.5)));

        // --- compute the z-range that the ray covers during this iteration ---
        float rayZMin = prevZMaxEstimate;
        // compute rayZMax using half-step forward (perspective-correct)
        float rayZMax = (dQ.z * 0.5 + Q0.z) / (dk * 0.5 + k0);
        prevZMaxEstimate = rayZMax;
        if (rayZMin > rayZMax) {
            // swap to ensure rayZMin <= rayZMax
            float tmp = rayZMin; rayZMin = rayZMax; rayZMax = tmp;
        }

        // --- fetch scene depth from depth buffer at this pixel ---
        // If hitPix is outside texture, texelFetch would be undefined; check bounds manually
        if (hitPix.x < 0 || hitPix.x >= int(csZBufferSize.x) || hitPix.y < 0 || hitPix.y >= int(csZBufferSize.y)) {
            // out of bounds: texelFetch would return 0; treat as miss and continue
            continue;
        }

        vec4 sceneSample = texelFetch(csZBuffer, hitPix, 0);
        float depthBuffer = sceneSample.r;

        // Convert depthBuffer (0..1) -> camera-space z using your helper (depthBufferToCameraSpace)
        // float sceneZ = depthBufferToCameraSpace(depthBuffer, nearPlaneZ, farPlaneZ);
        float ndcDepth = depthBuffer * 2.0 - 1.0;
        vec3 worldSpace = GetWorldPosFromDepth((vec2(hitPix) + 0.5) / csZBufferSize, ndcDepth);
        float sceneZ = (matView * vec4(worldSpace, 1)).z;

        // Paper uses sceneZMax as front of a voxel, then sceneZMin = sceneZMax - zThickness
        float sceneZMax = sceneZ;
        float sceneZMin = sceneZMax - zThickness;

        // --- intersection test: does ray segment in this pixel overlap the depth voxel? ---
        // If rayZMax >= sceneZMin && rayZMin <= sceneZMax => overlap (hit)
        if ((rayZMax >= sceneZMin) && (rayZMin <= sceneZMax))
        {
            // Found a hit. Compute accurate camera-space hit point via perspective-correct interpolation:
            // advance Q.xy based on the number of steps actually taken so far to produce hit point Q * (1/k)
            vec3 Q = Q0; // Note: Q0 has been incremented in the loop; it already represents current Q in homogeneous space
            float k = k0;

            // Convert Q/k back to camera space position (paper: hitPoint = Q * (1.0 / k))
            vec3 csHitPoint = Q * (1.0 / k);

            // Compute hit pixel UV (0..1)
            vec2 hitUV = (vec2(hitPix) + 0.5) / csZBufferSize; // center of texel

            hitPixelUV = hitUV;
            cameraSpaceHitPoint = csHitPoint;
            return true;
        }
    }

    // No hit found
    return false;
}

// Find the closest point on a ray to a given world position, projected to screen space
vec2 findClosestPointOnRay(vec3 resultWorldPos, Ray refractedRay)
{
    vec3 pointToRay = resultWorldPos - refractedRay.start;
    vec3 dirNorm = refractedRay.dir;
    float t = dot(pointToRay, dirNorm);
    vec3 closestPointOnRay = refractedRay.start + t * dirNorm;
    vec4 closestPointScreenSpace = matVP * vec4(closestPointOnRay, 1.0);
    vec2 ndcPoint = closestPointScreenSpace.xy / closestPointScreenSpace.w;
    vec2 screenPoint = (ndcPoint * 0.5 + 0.5);
    return screenPoint;
}

// We want to tally success/failure rates
layout(std430, binding = 0) buffer StatsBuffer
{
    uint successCount; // Count of successful refractions
    uint failureCount; // Count of failed refractions
};

void main()
{
    // Find normal from normal maps
    vec4 ndcCoords = mvp * vec4(vertexPosition, 1.0);
	vec3 postDivide = ndcCoords.xyz / ndcCoords.w;
	vec3 textureSpaceCoords = (postDivide.xyz + vec3(1)) / 2.0f;
	
    float timeScale = 0.1;
    vec2 timeOffset = vec2(time, 0) * timeScale;

    vec3 n = (texture(texNormal, vertexTexCoord + timeOffset).xyz * 2) - 1; // Normal from normal map
    n = normalize(n); // Normalize the normal vector
    n = n.xzy;
    vec3 n2 = (texture(texNormal2, vertexTexCoord - timeOffset).xyz * 2) - 1;
    n2 = normalize(n2);
    n2 = n2.xzy;

    n += n2;
    n.y *= 4;
    n = normalize(n);

    // Calculate world-space position before refraction
	oldPos = vec3(matModel * vec4(vertexPosition, 1.0));
    vec4 oldPosProj = mvp * vec4(vertexPosition, 1.0);
    oldPosProj.xyz /= oldPosProj.w;
    oldPosProj = oldPosProj * 0.5 + 0.5;
    if (oldPosProj.z > texture(depthBuffer, oldPosProj.xy).r)
    {
        // We're behind something else, discard
        shouldDiscard = -5;
        gl_Position = vec4(-1);
        return;
    }

    // Run our iterative Newton-based refraction solver
    Ray refractedRay;
    refractedRay.start = oldPos;
    vec3 toPoint = normalize(oldPos - sunPos);
    refractedRay.dir = normalize(refract(vec3(-0.5, -1, 0.5), n, 1.0 / WATER_IOR));

    shouldDiscard = 1;

    vec3 worldPosBehind;
    vec2 intersectionPoint = textureSpaceCoords.xy;
    vec3 refractionWorldPos;
    for (int i = 0; i < 2; i++)
    {
        refractionWorldPos = runRefraction(intersectionPoint, refractedRay, worldPosBehind);
        vec4 refractionNDC = matVP * vec4(refractionWorldPos, 1.0);
        refractionNDC.xyz /= refractionNDC.w; // Convert to NDC coordinates
        vec3 refractionCoords = (refractionNDC.xyz + 1.0) / 2.0; // Convert to screen space
	    intersectionPoint = refractionCoords.xy;
        if (refractionWorldPos == vec3(-1, -1, -1) || worldPosBehind.y > 0)
        {
            shouldDiscard = -5;
        }
    }

    // OR run refraction via screen-space raymarching for comparison
    /*vec3 csOrig = vec3(matView * vec4(oldPos, 1));
    vec3 csDir = vec3(matView * vec4(refractedRay.dir, 0));
    mat4 proj = matProjection;
    // sampler2D zBuffer = depthBuffer;
    vec2 csZBufferSize = vec2(2048, 2048);
    float zThickness = 0.1;
    vec3 clipInfo = vec3(1,1,0); // Unused, and who knows what the hell it does
    float nearPlaneZ = -0.01; // Should match near plane of camera
    float farPlaneZ = -1000.0; // Should match far plane of camera
    float stride = 1.0;
    float jitter = 0.0;
    const float maxSteps = 1000.0;
    float maxDistance = 100.0;
    vec2 intersectionPoint;
    vec3 cameraSpaceRefractionCoords;
    bool raymarchSuccess = traceScreenSpaceRay(csOrig, csDir, proj, depthBuffer, csZBufferSize, zThickness, clipInfo, nearPlaneZ, farPlaneZ, stride, jitter, maxSteps, maxDistance, intersectionPoint, cameraSpaceRefractionCoords);*/

    // Move the vertex to the calculated position
	float depth = texture(depthBuffer, intersectionPoint.xy).r;

	newPos = GetWorldPosFromDepth(intersectionPoint.xy, depth * 2.0 - 1.0);

    vec2 closestPoint = findClosestPointOnRay(newPos, refractedRay) * vec2(SCREEN_WIDTH, SCREEN_HEIGHT);
    vec2 curPoint = intersectionPoint.xy * vec2(SCREEN_WIDTH, SCREEN_HEIGHT);
    // If we're more than 1 pixel away from the refracted ray, mark the pixel as failed (i.e. only accept results identical to screen-space raymarching)
    if (distance(curPoint, closestPoint) > 1.0)
    {
        shouldDiscard = -5;
    }

    // Uncomment if running ray marching instead of our Newton-based solver
    /*if (!raymarchSuccess)
    {
        shouldDiscard = -5;
    }*/

    // We can tally failure rates -- disabled by default due to performance impact
    /*if (shouldDiscard < 0)
    {
        // Increment failure count
        atomicAdd(failureCount, 1);
    }
    else
    {
        // Increment success count
        atomicAdd(successCount, 1);
    }*/

	gl_Position = matVP * vec4(newPos, 1.0);
}
