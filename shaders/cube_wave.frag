#version 450

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec3 inNormal;

layout(binding = 0) uniform UBO {
    mat4  view;
    mat4  proj;
    vec4  lightDir;
    float cubeScale;
} ubo;

layout(location = 0) out vec4 outColor;

void main() {
    vec3  n     = normalize(inNormal);
    vec3  l     = normalize(ubo.lightDir.xyz);
    float diff  = max(dot(n, l), 0.0);
    float light = 0.25 + diff * 0.75;
    outColor    = vec4(inColor * light, 1.0);
}
