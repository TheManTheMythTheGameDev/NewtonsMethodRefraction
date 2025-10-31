#version 430

// === Input vertex attributes (from vertex shader) ===
in vec2 fragTexCoord;   // Texture coordinates
in vec4 fragColor;      // Vertex color (unused)
in vec3 fragNormal;     // Surface normal at fragment
in vec3 fragPos;        // World-space position of fragment

// === Uniforms ===
uniform sampler2D texture0;     // Base color texture
uniform sampler2D shadowMap;    // Shadow map for shadowing
uniform sampler2D causticsMap;  // Caustics texture for underwater lighting
uniform vec4 colDiffuse;        // Diffuse color (unused)
uniform vec3 camPos;            // Camera position in world space
uniform mat4 shadowVP;          // Shadow view-projection matrix

// === Output fragment data ===
layout(location = 0) out vec4 finalColor;   // Final color output
layout(location = 1) out vec4 finalNormal;  // Output normal (for G-buffer)

// === Poisson disk sampling for soft shadows ===
const int NUM_POINTS = 26;
const vec2 poissonPoints[NUM_POINTS] = vec2[](
    vec2(-0.012632, -0.269696),
    vec2(-0.279386, 0.245075),
    vec2(0.420386, -0.541629),
    vec2(0.077266, -0.625248),
    vec2(0.096925, 0.229906),
    vec2(-0.314555, -0.304275),
    vec2(0.296059, -0.072712),
    vec2(-0.365344, -0.614714),
    vec2(0.545623, -0.260981),
    vec2(0.537109, 0.232799),
    vec2(0.862865, -0.011350),
    vec2(0.731267, -0.525061),
    vec2(-0.190754, 0.656683),
    vec2(-0.562010, -0.008233),
    vec2(-0.581734, 0.672348),
    vec2(-0.765561, 0.431987),
    vec2(0.250228, 0.574817),
    vec2(-0.046062, 0.976276),
    vec2(-0.845756, -0.409495),
    vec2(-0.892432, -0.014966),
    vec2(-0.153369, -0.907165),
    vec2(0.845113, 0.464476),
    vec2(0.624097, 0.761748),
    vec2(0.283350, 0.948915),
    vec2(0.221667, -0.908964),
    vec2(-0.659854, -0.716601)
);

// === Utility Functions ===

// Hash function for pseudo-randomness based on a 2D point
float hash(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Returns a random angle in [0, 2*pi) based on a seed
float randomAngle(vec2 seed)
{
    return hash(seed) * 6.2831853;
}

// Rotates a 2D vector by a given angle (radians)
vec2 rotate(vec2 point, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return vec2(
        point.x * c - point.y * s,
        point.x * s + point.y * c
    );
}

// === Shadow Calculation ===

// Computes the shadow factor using Poisson disk sampling for soft shadows
float computeShadow(vec3 normal, vec2 fragTexCoord, vec3 l)
{
    // Transform fragment position to shadow map space with normal offset (for bias)
    float normalOffsetScale = 0.2;
    float normalOffsetFactor = sqrt(1.0 - pow(dot(normal, l), 2.0)); // sin^2(theta) = 1 - cos^2(theta)
    vec3 offsetPos = fragPos + normal * normalOffsetScale * normalOffsetFactor;
    vec4 shadowCoord = shadowVP * vec4(offsetPos, 1.0);
    shadowCoord.xyz /= shadowCoord.w; // Perspective divide
    shadowCoord.xyz = shadowCoord.xyz * 0.5 + 0.5; // Map to [0, 1] range

    float shadowSum = 0.0;
    float angle = randomAngle(fragTexCoord); // Randomize Poisson disk rotation per fragment
    const float radius = 5.0;

    for (int i = 0; i < NUM_POINTS; i++)
    {
        // Rotate and scale Poisson disk sample
        vec2 poisson = rotate(poissonPoints[i], angle) * radius;

        // Sample shadow map with offset
        float shadow = textureOffset(shadowMap, shadowCoord.xy, ivec2(poisson.x, poisson.y)).r;

        // Cone depth bias to reduce shadow acne
        float additionalBias = 0.0001 * length(poisson);

        // If fragment is in shadow, increment shadowSum
        if (shadowCoord.z - additionalBias > shadow)
        {
            shadowSum += 1.0;
        }
    }
    // Return average shadow factor (0 = fully lit, 1 = fully shadowed)
    return shadowSum / float(NUM_POINTS);
}

// === Caustics Calculation ===

// Returns caustic intensity factor from caustics map (for underwater lighting)
float getCausticFactor(vec3 fragPos, vec2 shadowCoord)
{
    // Only apply caustics below water surface (y < 0)
    if (fragPos.y < 0.0)
    {
        return texture(causticsMap, shadowCoord).r;
    }
    return 1.0;
}

// === Lighting Calculation ===

// Computes the final color using ambient, diffuse, and specular lighting (Phong shading model)
vec3 computeLighting(
    vec3 n, vec3 v, vec3 l,
    float ambientStrength, float diffuseStrength, float specularStrength,
    float shininess, vec3 lightColor, vec3 texColor)
{
    // Ambient
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    float diff = max(dot(n, l), 0.0);
    vec3 diffuse = diffuseStrength * diff * lightColor;

    // Specular
    vec3 reflectDir = reflect(-l, n);
    float spec = pow(max(dot(v, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;

    // Combine lighting with texture color
    return (ambient + diffuse + specular) * texColor;
}

// === Main Fragment Shader ===

void main()
{
    // === Lighting and Material Properties ===
    vec3 lightDir = normalize(vec3(0.5, 1, -0.5)); // Directional light (downward)
    vec3 lightColor = vec3(1.0);                    // White light

    float ambientStrength = 0.15;
    float diffuseStrength = 0.85;
    float specularStrength = 0.4;
    float shininess = 32.0;

    // === Surface and View Vectors ===
    vec3 n = normalize(fragNormal);         // Normal at fragment
    vec3 v = normalize(camPos - fragPos);   // View direction
    vec3 l = lightDir;                      // Light direction

    // === Shadow Calculation ===
    float shadowFactor = computeShadow(n, fragTexCoord, l);

    // Reduce diffuse and specular strength in shadowed areas
    diffuseStrength = mix(diffuseStrength, 0.0, shadowFactor);
    specularStrength = mix(specularStrength, 0.0, shadowFactor);

    // === Caustics (Underwater Light Patterns) ===
    // Compute shadow map coordinates for caustics lookup
    vec4 shadowCoord4 = shadowVP * vec4(fragPos, 1.0);
    shadowCoord4.xyz /= shadowCoord4.w;
    shadowCoord4.xyz = shadowCoord4.xyz * 0.5 + 0.5;
    vec2 shadowCoord = shadowCoord4.xy;

    float causticFactor = getCausticFactor(fragPos, shadowCoord);
    diffuseStrength *= causticFactor;
    specularStrength *= causticFactor;

    // === Texture Sampling ===
    vec3 texColor = texture(texture0, fragTexCoord).rgb;

    // === Lighting Calculation ===
    vec3 result = computeLighting(
        n, v, l,
        ambientStrength, diffuseStrength, specularStrength,
        shininess, lightColor, texColor
    );

    // === Output ===
    finalColor = vec4(result, 1.0);
    finalNormal = vec4(n, 1.0);
}
