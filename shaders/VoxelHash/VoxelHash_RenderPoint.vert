#version 450

layout(push_constant) uniform PC {
    mat4  mvp;
    float voxelSize;
    float _pad[3];
} pc;

layout(set=0, binding=0) readonly buffer HT { int raw[]; } ht;

layout(location=0) out vec4 fragColor;

#define STRIDE 12
#define EMPTY  0x7FFFFFFF

void main() {
    int base = gl_VertexIndex * STRIDE;
    int kx   = ht.raw[base + 0];

    if (kx == EMPTY) {
        gl_Position  = vec4(10.0, 10.0, 10.0, 1.0);
        gl_PointSize = 0.0;
        fragColor    = vec4(0.0);
        return;
    }

    int  ky  = ht.raw[base + 1];
    int  kz  = ht.raw[base + 2];
    uint col = uint(ht.raw[base + 5]);   // color_rgba

    vec3 center  = (vec3(float(kx), float(ky), float(kz)) + 0.5) * pc.voxelSize;
    gl_Position  = pc.mvp * vec4(center, 1.0);
    gl_PointSize = 3.0;

    float r = float(col & 0xFFu)         / 255.0;
    float g = float((col >>  8u) & 0xFFu)/ 255.0;
    float b = float((col >> 16u) & 0xFFu)/ 255.0;
    fragColor = vec4(r, g, b, 1.0);
}