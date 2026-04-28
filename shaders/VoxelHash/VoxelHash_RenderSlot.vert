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

// 단위 큐브 8 꼭짓점 (중심 원점, 0.5 크기)
const vec3 kC[8] = vec3[8](
    vec3(-0.5,-0.5,-0.5), vec3( 0.5,-0.5,-0.5),
    vec3( 0.5, 0.5,-0.5), vec3(-0.5, 0.5,-0.5),
    vec3(-0.5,-0.5, 0.5), vec3( 0.5,-0.5, 0.5),
    vec3( 0.5, 0.5, 0.5), vec3(-0.5, 0.5, 0.5)
);

// 6면 × 2삼각형 × 3꼭짓점 = 36 인덱스  (-Z, +Z, -Y, +Y, -X, +X)
const int kI[36] = int[36](
    0,2,1, 0,3,2,
    4,5,6, 4,6,7,
    0,1,5, 0,5,4,
    3,7,6, 3,6,2,
    0,4,7, 0,7,3,
    1,2,6, 1,6,5
);

// 면 법선
const vec3 kN[6] = vec3[6](
    vec3( 0, 0,-1), vec3( 0, 0, 1),
    vec3( 0,-1, 0), vec3( 0, 1, 0),
    vec3(-1, 0, 0), vec3( 1, 0, 0)
);

void main() {
    int voxelIdx = gl_VertexIndex / 36;
    int triIdx   = gl_VertexIndex % 36;
    int base     = voxelIdx * STRIDE;
    int kx       = ht.raw[base + 0];

    if (kx == EMPTY) {
        gl_Position = vec4(10.0, 10.0, 10.0, 1.0);
        fragColor   = vec4(0.0);
        return;
    }

    int  ky  = ht.raw[base + 1];
    int  kz  = ht.raw[base + 2];
    uint col = uint(ht.raw[base + 5]);  // color_rgba

    vec3 centre   = (vec3(float(kx), float(ky), float(kz)) + 0.5) * pc.voxelSize;
    vec3 localPos = kC[kI[triIdx]] * pc.voxelSize * 0.92;  // 0.92 = 복셀 간 틈
    gl_Position   = pc.mvp * vec4(centre + localPos, 1.0);

    // Lambert 음영
    vec3  lightDir = normalize(vec3(1.0, 2.0, 1.5));
    float diffuse  = max(dot(kN[triIdx / 6], lightDir), 0.0);

    float r = float(col & 0xFFu)          / 255.0;
    float g = float((col >>  8u) & 0xFFu) / 255.0;
    float b = float((col >> 16u) & 0xFFu) / 255.0;
    fragColor = vec4(vec3(r, g, b) * (0.35 + 0.65 * diffuse), 1.0);
}