#version 450

layout(push_constant) uniform PC {
    mat4  mvp;
    float voxelSize;
    float _pad[3];
} pc;

layout(set=0, binding=0) readonly buffer SortedPosition {
    vec4 pos[];
} positionBuffer;

layout(set=0, binding=1) readonly buffer SortedColor {
    uint col[];
} colorBuffer;

layout(location=0) out vec4 fragColor;

void main() {
    uint idx = uint(gl_VertexIndex);
    vec3 p = positionBuffer.pos[idx].xyz;
    uint c = colorBuffer.col[idx];

    gl_Position  = pc.mvp * vec4(p, 1.0);
    gl_PointSize = 3.0;

    float r = float(c & 0xFFu)          / 255.0;
    float g = float((c >> 8u) & 0xFFu)  / 255.0;
    float b = float((c >> 16u) & 0xFFu) / 255.0;
    fragColor = vec4(r, g, b, 1.0);
}
