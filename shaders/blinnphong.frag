#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

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
    vec3 normal   = normalize(fragNormal);
    vec3 lightDir = normalize(ubo.lightPos.xyz - fragPos);
    vec3 viewDir  = normalize(ubo.viewPos.xyz  - fragPos);
    vec3 halfDir  = normalize(lightDir + viewDir);

    // Ambient
    vec3 ambient  = ubo.ambientStrength * ubo.lightColor.rgb;

    // Diffuse (Lambert)
    float diff    = max(dot(normal, lightDir), 0.0);
    vec3  diffuse = diff * ubo.lightColor.rgb;

    // Specular (Blinn-Phong)
    float spec    = pow(max(dot(normal, halfDir), 0.0), ubo.shininess);
    vec3 specular = ubo.specularStrength * spec * ubo.lightColor.rgb;

    vec3 result   = (ambient + diffuse + specular) * fragColor;
    outColor      = vec4(result, 1.0);
}
