#version 450

layout(location = 0) in vec3 inLocalPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inInstancePos;
layout(location = 3) in vec3 inInstanceColor;

layout(binding = 0) uniform UBO {
    mat4  view;
    mat4  proj;
    vec4  lightDir;
    float cubeScale;
} ubo;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec3 outNormal;

void main() {
    vec3 worldPos = inLocalPos * ubo.cubeScale + inInstancePos;
    gl_Position   = ubo.proj * ubo.view * vec4(worldPos, 1.0);
    outColor      = inInstanceColor;
    outNormal     = inNormal;
}
