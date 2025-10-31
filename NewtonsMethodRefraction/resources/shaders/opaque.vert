#version 430

// Input vertex attributes
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

// Input uniform values
uniform mat4 mvp;
uniform mat4 matNormal;
uniform mat4 matModel;

// Output vertex attributes (to fragment shader)
out vec2 fragTexCoord;
out vec4 fragColor;
out vec3 fragNormal;
out vec3 fragPos;

// NOTE: Add here your custom variables

void main()
{
    // Send vertex attributes to fragment shader
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;

    fragNormal = normalize(vec3(matNormal * vec4(vertexNormal, 0)));

    fragPos = vec3(matModel * vec4(vertexPosition, 1.0));

    // Calculate final vertex position
    gl_Position = mvp*vec4(vertexPosition, 1.0);
}
