#version 430

// ========================= Vertex Inputs =========================
// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;
in vec3 fragPos;

// ========================= Uniforms =========================
// Input uniform values
uniform sampler2D sceneColors;   // Opaque geometry color buffer
uniform sampler2D sceneNormals;  // Normal map of opaque geometry
uniform sampler2D sceneDepth;    // Depth buffer of opaque geometry
uniform sampler2D texNormal;     // Water normal map
uniform sampler2D texNormal2;    // Second water normal map
uniform samplerCube skybox;      // Skybox texture
uniform vec4 colDiffuse;
uniform vec3 camPos;
uniform mat4 invView;            // Inverse view matrix
uniform mat4 invProjection;      // Inverse projection matrix
uniform mat4 viewProjection;
uniform mat4 matView;
uniform mat4 matProjection;
uniform mat4 mvp;
uniform float time;

// ========================= Output =========================
out vec4 finalColor;

// ========================= Constants =========================
#define SCREEN_WIDTH 2000.0
#define SCREEN_HEIGHT 1600.0
#define WATER_IOR 1.33                  // Index of refraction for water
#define NUM_REFRACTION_ITERATIONS 4    // Number of refraction refinement steps

// ========================= Utility Functions =========================

// Fresnel function for blending refractions and reflections (Schlick approximation)
float fresnelSchlick(float cosTheta, float F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
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



// Convert screen-space and NDC depth to world position
vec3 GetWorldPosFromDepth(vec2 screenSpace, float ndcDepth)
{
    vec3 ndcCoords = vec3(2.0 * screenSpace - 1.0, ndcDepth);
    vec4 viewPos = invProjection * vec4(ndcCoords, 1.0);
    viewPos.xyz /= viewPos.w;
    viewPos.w = 1.0;
    vec3 worldPos = (invView * viewPos).xyz;
    return worldPos;
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
        float sceneDepth = sceneSample.r;

        // Convert sceneDepth (0..1) -> camera-space z using your helper (depthBufferToCameraSpace)
        // float sceneZ = depthBufferToCameraSpace(sceneDepth, nearPlaneZ, farPlaneZ);
        float ndcDepth = sceneDepth * 2.0 - 1.0;
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


// ========================= Geometry Structures =========================
struct Ray {
    vec3 start;
    vec3 dir;
};

struct Plane {
    vec3 point;
    vec3 normal;
};

// ========================= Geometry Utilities =========================
// Returns time t along ray where hit occurs with plane
float PlaneRayIntersection(Plane plane, Ray ray)
{
    float denom = dot(plane.normal, ray.dir);
    if (abs(denom) > 0.0001) {
        float t = dot(plane.point - ray.start, plane.normal) / denom;
        return t;
    }
    return -1.0;
}

// Returns a point's distance from the edge of the screen (in texture space)
float distanceFromEdge(vec2 texCoords)
{
    return texCoords.y; // Most refraction errors occur at the bottom of the screen, so we only care about how close a point is to the bottom for error fading purposes
}

// ========================= Refraction Utilities =========================
// Compute the world-space intersection of a refracted ray with geometry behind water
vec3 runRefraction(vec2 seed, Ray refractedRay, out vec3 worldPos)
{
    // Find the depth at the screen Z-buffer directly behind the water
    float depth = texture(sceneDepth, seed).r;
    float ndcDepth = depth * 2.0 - 1.0;
    worldPos = GetWorldPosFromDepth(seed, ndcDepth);
    // Find the normal at the screen normal buffer directly behind the water
    vec3 normal = texture(sceneNormals, seed).xyz;
    normal = normalize(normal);
    // Generate a plane using worldPos and normal
    Plane plane;
    plane.point = worldPos;
    plane.normal = normal;
    // Intersect the refracted ray with the plane
    float t = PlaneRayIntersection(plane, refractedRay);
    if (t >= 0.0) {
        // Find the world-space intersection point
        vec3 intersectionPoint = refractedRay.start + refractedRay.dir * t;
        return intersectionPoint;
    }
    return vec3(-1.0, -1.0, -1.0); // If no intersection, return invalid
}

// Find the closest point on a ray to a given world position, projected to screen space
vec2 findClosestPointOnRay(vec3 resultWorldPos, Ray refractedRay)
{
    vec3 pointToRay = resultWorldPos - refractedRay.start;
    vec3 dirNorm = refractedRay.dir;
    float t = dot(pointToRay, dirNorm);
    vec3 closestPointOnRay = refractedRay.start + t * dirNorm;
    vec4 closestPointScreenSpace = matProjection * matView * vec4(closestPointOnRay, 1.0);
    vec2 ndcPoint = closestPointScreenSpace.xy / closestPointScreenSpace.w;
    vec2 screenPoint = (ndcPoint * 0.5 + 0.5);
    return screenPoint;
}

// ========================= Main Shader =========================
void main()
{
    // --- Water normal calculation (animated) ---
    float timeScale = 0.1;
    vec2 timeOffset = vec2(time, 0.0) * timeScale;
    vec3 n = (texture(texNormal, fragTexCoord).xyz * 2.0) - 1.0;
    n = normalize(n);
    n = n.xzy;
    vec3 n2 = (texture(texNormal2, fragTexCoord - timeOffset).xyz * 2.0) - 1.0;
    n2 = normalize(n2);
    n2 = n2.xzy;
    n += n2;
    n.y *= 4.0;
    n = normalize(n);

    // --- View direction ---
    vec3 v = normalize(camPos - fragPos);

    // --- Screen-space coordinates ---
    vec2 sceneTexCoords = gl_FragCoord.xy / vec2(SCREEN_WIDTH, SCREEN_HEIGHT);

    // --- Early discard if behind geometry ---
    // When recording success/failure stats, this ensures that invisible fragments are not included in the tally
    // This snippet can be deleted if not recording stats
    /*if (gl_FragCoord.z > texture(sceneDepth, sceneTexCoords).r) {
        discard;
        return;
    }*/

    // --- Refraction ray setup ---
    Ray refractedRay;
    refractedRay.start = fragPos;
    refractedRay.dir = refract(-v, n, 1.0 / WATER_IOR);

    // --- Iterative refraction calculation ---
    vec3 refractionCoords;
    vec2 seed = sceneTexCoords;
    vec3 worldSpaceRefractionCoords;
    vec3 worldPosBehind;
    vec3 finalWorldPosBehind;
    for (int i = 0; i < NUM_REFRACTION_ITERATIONS; i++) {
        worldSpaceRefractionCoords = runRefraction(seed, refractedRay, finalWorldPosBehind);
        if (i == 0) worldPosBehind = finalWorldPosBehind;
        // Convert intersection point back to screen space
        vec4 ndcCoords = viewProjection * vec4(worldSpaceRefractionCoords, 1.0);
        ndcCoords.xyz /= ndcCoords.w;
        refractionCoords = (ndcCoords.xyz + 1.0) / 2.0;
        seed = refractionCoords.xy;
    }
    
    // Alternatively, we can calculate refractions with screen-space ray marching for comparison
    // Some input variables to the ray marching function are still initialized here for later use in calculating *reflections*
    vec3 csOrig = vec3(matView * vec4(fragPos, 1));
    vec3 csDir = vec3(matView * vec4(refractedRay.dir, 0));
    mat4 proj = matProjection;
    // sampler2D zBuffer = sceneDepth;
    vec2 csZBufferSize = vec2(SCREEN_WIDTH, SCREEN_HEIGHT);
    float zThickness = 0.1;
    vec3 clipInfo = vec3(1,1,0); // Unused, and who knows what the hell it does
    float nearPlaneZ = -0.01; // Should match near plane of camera
    float farPlaneZ = -1000.0; // Should match far plane of camera
    float stride = 1.0;
    float jitter = 0.0;
    const float maxSteps = 1000.0;
    float maxDistance = 100.0;
    /*vec2 refractionCoords;
    vec3 cameraSpaceRefractionCoords;
    bool raymarchSuccess = traceScreenSpaceRay(csOrig, csDir, proj, sceneDepth, csZBufferSize, zThickness, clipInfo, nearPlaneZ, farPlaneZ, stride, jitter, maxSteps, maxDistance, refractionCoords, cameraSpaceRefractionCoords);
    vec3 worldPosBehind = GetWorldPosFromDepth(sceneTexCoords, texture(sceneDepth, sceneTexCoords).r * 2.0 - 1.0);*/
    // vec3 resultWorldPos = GetWorldPosFromDepth(refractionCoords.xy, texture(sceneDepth, refractionCoords.xy).r * 2.0 - 1.0);
    // float waterDepth = distance(endPos, refractedRay.start);

    // --- Error detection and fallback ---
    vec2 fallbackRefractionCoords = sceneTexCoords + n.xz * 0.15; // GPU Gems refraction method for fallback if our method fails (https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-19-generic-refraction-simulation)
    vec3 refractionColor = vec3(1.0);
    float fade = 0.0;
    float distFromEdge = distanceFromEdge(refractionCoords.xy);
    float beginFade = 0.1;
    float resultDepth = texture(sceneDepth, refractionCoords.xy).r;
    float resultNDCDepth = resultDepth * 2.0 - 1.0;
    vec3 resultWorldPos = GetWorldPosFromDepth(refractionCoords.xy, resultNDCDepth);
    float waterDepth = distance(resultWorldPos, refractedRay.start);
    if (!isPointValid(refractionCoords.xy)) {
        fade = 1.0;
    } else if (distFromEdge < beginFade) {
        fade = (beginFade - distFromEdge) / beginFade;
    }
    vec2 closestPoint = findClosestPointOnRay(resultWorldPos, refractedRay) * vec2(SCREEN_WIDTH, SCREEN_HEIGHT);
    vec2 curPoint = refractionCoords.xy * vec2(SCREEN_WIDTH, SCREEN_HEIGHT);
    // Uncomment this if using ray marching instead of our Newton-based refraction solver
    /*if (!raymarchSuccess)
    {
        fade = 1.0;
    }*/
    // If we're more than 1 pixel away from the refracted ray, mark the pixel as failed (i.e. only accept results identical to screen-space raymarching)
    if (distance(curPoint, closestPoint) > 1.0) {
        fade = 1.0;
        waterDepth = distance(worldPosBehind, refractedRay.start);
    }

    // Can tally success/failure counts; disabled by default due to performance impact
    /*if (fade == 1.0)
    {
        // Increment failure count
        atomicAdd(failureCount, 1);
    }
    else
    {
        // Increment success count
        atomicAdd(successCount, 1);
    }*/

    // In case the GPU Gems method *also* fails, we fall back to no refractions
    if (texture(sceneDepth, fallbackRefractionCoords.xy).r < gl_FragCoord.z || !isPointValid(fallbackRefractionCoords)) {
        fallbackRefractionCoords = sceneTexCoords;
    }
    refractionCoords.xy = mix(refractionCoords.xy, fallbackRefractionCoords, fade);
    refractionColor = texture(sceneColors, refractionCoords.xy).rgb;

    // --- Water absorption (Beer-Lambert law) ---
    float absorptionCoeff = 0.05;
    vec3 waterCol = vec3(0.55, 0.71, 0.78);
    refractionColor = mix(waterCol, refractionColor, exp(-absorptionCoeff * waterDepth));

    // --- Reflection calculation ---
    vec3 reflDir = reflect(-v, n);
    vec3 reflectionColor = texture(skybox, reflDir).rgb;
    vec3 reflDirCameraSpace = vec3(matView * vec4(reflDir, 0));
    vec2 reflectionCoords;
    vec3 cameraSpaceReflectionCoords;
    bool reflectionSuccess = traceScreenSpaceRay(csOrig, reflDirCameraSpace, proj, sceneDepth, csZBufferSize, zThickness, clipInfo, nearPlaneZ, farPlaneZ, stride, jitter, maxSteps, maxDistance, reflectionCoords, cameraSpaceReflectionCoords);
    if (isPointValid(reflectionCoords) && reflectionSuccess) {
        reflectionColor = texture(sceneColors, reflectionCoords).rgb;
    }

    // --- Fresnel effect ---
    float F0 = 0.04; // Base reflectivity for water
    float cosTheta = dot(n, v);
    float fresnel = fresnelSchlick(cosTheta, F0);

    // --- Final color blend ---
    vec3 colorToRender = mix(refractionColor, reflectionColor, fresnel);

    // --- Output ---
    finalColor = vec4(colorToRender, 1.0);
}
