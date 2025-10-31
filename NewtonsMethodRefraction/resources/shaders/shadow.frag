#version 430

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;
in vec3 fragPos;

// Output fragment normal for caustics calculations
layout(location = 0) out vec4 finalNormal;

void main()
{
    vec3 n = normalize(fragNormal);

    finalNormal = vec4(n, 1.0);
}
