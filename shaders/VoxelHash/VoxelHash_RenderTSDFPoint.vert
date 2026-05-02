#version 450

layout(push_constant) uniform PC {
    mat4  mvp;
    float voxelSize;
    float truncation;
    float projectSurface;
    float _pad;
} pc;

layout(set=0, binding=0) readonly buffer HT { int raw[]; } ht;

layout(location=0) out vec4 fragColor;

#define STRIDE 12
#define EMPTY  0x7FFFFFFF

vec3 unpackNormal(uint oct) {
    vec2 f = vec2(float(oct & 0xFFFFu), float(oct >> 16u)) / 65535.0 * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        vec2 s = sign(n.xy);
        n.xy = (1.0 - abs(n.yx)) * s;
    }
    return normalize(n);
}

void main() {
    int base = gl_VertexIndex * STRIDE;
    int kx = ht.raw[base + 0];

    if (kx == EMPTY) {
        gl_Position = vec4(10.0, 10.0, 10.0, 1.0);
        gl_PointSize = 0.0;
        fragColor = vec4(0.0);
        return;
    }

    float weight = intBitsToFloat(ht.raw[base + 4]);
    if (weight <= 0.0) {
        gl_Position = vec4(10.0, 10.0, 10.0, 1.0);
        gl_PointSize = 0.0;
        fragColor = vec4(0.0);
        return;
    }

    int ky = ht.raw[base + 1];
    int kz = ht.raw[base + 2];
    uint col = uint(ht.raw[base + 5]);
    float tsdf = intBitsToFloat(ht.raw[base + 3]);
    uint packedNormal = uint(ht.raw[base + 6]);

    vec3 center = (vec3(float(kx), float(ky), float(kz)) + 0.5) * pc.voxelSize;
    if (pc.projectSurface > 0.5) {
        vec3 n = unpackNormal(packedNormal);
        center -= n * (tsdf * pc.truncation);
    }
    gl_Position = pc.mvp * vec4(center, 1.0);
    gl_PointSize = 3.0;

    float r = float(col & 0xFFu) / 255.0;
    float g = float((col >> 8u) & 0xFFu) / 255.0;
    float b = float((col >> 16u) & 0xFFu) / 255.0;
    fragColor = vec4(r, g, b, 1.0);
}
