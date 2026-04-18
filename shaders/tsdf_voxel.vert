#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;  // unused, kept for vertex layout compat

layout(location = 0) out vec3 fragColor;

layout(set = 0, binding = 0) uniform UBO {
    mat4  model;
    mat4  view;
    mat4  proj;
    vec4  lightPos;
    vec4  viewPos;
    vec4  lightColor;
    float ambientStrength;
    float specularStrength;
    float shininess;
    float _pad;
} ubo;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor   = inColor;
}
