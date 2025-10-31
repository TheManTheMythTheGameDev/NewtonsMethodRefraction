#version 430

// Input vertex attributes (from vertex shader)
in vec3 oldPos;
in vec3 newPos;
in float shouldDiscard;

// Output fragment color
layout(location = 0) out vec4 finalColor;

uniform mat4 matVP;
uniform sampler2D depthBuffer;

void main()
{
    if (shouldDiscard < 1)
    {
        discard;
    }
    // If our position differs from that stored in the shadow map, the triangle is not lying on a surface
    // This is probably because one (or more) vertex ended up on a different surface than the rest after refraction
    // The area of such triangles is erroneously large, so we should discard their pixels
    vec4 posProjected = matVP * vec4(newPos, 1.0);
    posProjected /= posProjected.w;
    posProjected = posProjected * 0.5 + 0.5;
    float depth = texture(depthBuffer, posProjected.xy).r;
    if (abs(posProjected.z - depth) > 0.0001)
    {
        discard;
    }

    // Compute the change in area (as per Wallace in https://medium.com/@evanwallace/rendering-realtime-caustics-in-webgl-2a99a29a0b2c)
    float oldArea = length(dFdx(oldPos)) * length(dFdy(oldPos));
    vec3 dx = dFdx(newPos);
    vec3 dy = dFdy(newPos);
    float newArea = length(dx) * length(dy);
    
    // Finally, take absorption into account and store result in an offscreen buffer
    float col;
    float absorptionCoeff = 0.1;
    col = oldArea / newArea * exp(-absorptionCoeff * length(newPos - oldPos));

    finalColor = vec4(col, 0.0f, 0.0f, 1.0f);
}
