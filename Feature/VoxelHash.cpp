#include "VoxelHash.h"
#include <imgui.h>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cstdlib>

// ─────────────────────────────────────────────────────────────────────────────
//  GLSL 공통 매크로 (모든 compute shader 가 include)
//
//  raw int[] 레이아웃 (엔트리당 12 개의 int, stride = 48 bytes)
//    F_KX=0  F_KY=1  F_KZ=2
//    F_TSDF=3  F_W=4  F_COL=5  F_NORM=6
//    F_ACCT=7  F_ACCW=8  F_ACCC=9  F_ACCN=10
//    F_FTAG=11
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char *kVH_Common = R"GLSL(
#define STRIDE      12
#define F_KX         0
#define F_KY         1
#define F_KZ         2
#define F_TSDF       3
#define F_W          4
#define F_COL        5
#define F_NORM       6
#define F_ACCT       7
#define F_ACCW       8
#define F_ACCC       9
#define F_ACCN      10
#define F_FTAG      11
#define EMPTY       0x7FFFFFFF
#define BUCKET_SIZE 4

layout(set=0, binding=0) buffer HT  { int raw[]; } ht;
layout(set=0, binding=1) buffer Pts { vec4 pts[]; };
layout(set=0, binding=2) buffer Ctr { uint n; } ctr;

// CAS 기반 float atomic add (GLSL 에는 atomicAdd(float) 가 없으므로 직접 구현)
void casAddF(int idx, float val) {
    int expected, next;
    for (int i = 0; i < 64; i++) {
        expected = ht.raw[idx];
        next     = floatBitsToInt(intBitsToFloat(expected) + val);
        if (atomicCompSwap(ht.raw[idx], expected, next) == expected) return;
    }
}

// 공간 해시 — 소수 곱 XOR, collision-free 를 위해 numBuckets 이 충분히 크게 설정
uint hash3(ivec3 k, uint numBuckets) {
    return (uint(k.x) * 73856093u ^ uint(k.y) * 19349663u ^ uint(k.z) * 83492791u)
           % numBuckets;
}

// Oct-encoding: 단위 법선 → uint(2×uint16)
uint packNormal(vec3 n) {
    float l1 = abs(n.x) + abs(n.y) + abs(n.z);
    n /= max(l1, 1e-6);
    if (n.z < 0.0) {
        float sx = n.x >= 0.0 ? 1.0 : -1.0;
        float sy = n.y >= 0.0 ? 1.0 : -1.0;
        n.xy = (1.0 - abs(n.yx)) * vec2(sx, sy);
    }
    uint nx = uint(clamp(n.x * 0.5 + 0.5, 0.0, 1.0) * 65535.0);
    uint ny = uint(clamp(n.y * 0.5 + 0.5, 0.0, 1.0) * 65535.0);
    return (nx & 0xFFFFu) | ((ny & 0xFFFFu) << 16u);
}

// Oct-decoding: uint → 단위 법선
vec3 unpackNormal(uint packed) {
    float fx = float(packed & 0xFFFFu) / 65535.0 * 2.0 - 1.0;
    float fy = float((packed >> 16u) & 0xFFFFu) / 65535.0 * 2.0 - 1.0;
    vec3 n   = vec3(fx, fy, 1.0 - abs(fx) - abs(fy));
    if (n.z < 0.0) {
        float ox = n.x >= 0.0 ? (1.0 - abs(n.y)) : -(1.0 - abs(n.y));
        float oy = n.y >= 0.0 ? (1.0 - abs(n.x)) : -(1.0 - abs(n.x));
        n.x = ox; n.y = oy;
    }
    return normalize(n);
}

// 위치 기반 색상 (공간적 분포 확인용)
uint posToColor(ivec3 key) {
    uint h = uint(key.x)*73856093u ^ uint(key.y)*19349663u ^ uint(key.z)*83492791u;
    float t = float(h & 0xFFFFu) / 65535.0;
    float r = clamp(abs(fract(t + 0.000)*6.0 - 3.0) - 1.0, 0.0, 1.0);
    float g = clamp(abs(fract(t + 0.333)*6.0 - 3.0) - 1.0, 0.0, 1.0);
    float b = clamp(abs(fract(t + 0.667)*6.0 - 3.0) - 1.0, 0.0, 1.0);
    return (uint(r*255.0)) | (uint(g*255.0)<<8u) | (uint(b*255.0)<<16u) | 0xFF000000u;
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
//  Shader 1: Clear — 전체 해시 테이블을 빈 상태로 초기화
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char *kVH_ClearComp = R"GLSL(
#version 450
layout(local_size_x = 64) in;
layout(push_constant) uniform PC { uint total; uint _f; float _0; float _1; } pc;

layout(set=0, binding=0) buffer HT  { int raw[]; } ht;
layout(set=0, binding=1) buffer Pts { vec4 pts[]; };
layout(set=0, binding=2) buffer Ctr { uint n; } ctr;

#define STRIDE 12
#define EMPTY  0x7FFFFFFF

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    uint base = i * uint(STRIDE);
    ht.raw[base +  0] = EMPTY;  // key_x
    ht.raw[base +  1] = 0;
    ht.raw[base +  2] = 0;
    ht.raw[base +  3] = 0;      // tsdf    (0.0f)
    ht.raw[base +  4] = 0;      // weight  (0.0f)
    ht.raw[base +  5] = 0;      // color
    ht.raw[base +  6] = 0;      // normal
    ht.raw[base +  7] = 0;      // acc_tsdf
    ht.raw[base +  8] = 0;      // acc_w
    ht.raw[base +  9] = 0;      // acc_color
    ht.raw[base + 10] = 0;      // acc_norm
    ht.raw[base + 11] = -1;     // frame_tag  (-1 = 미관측)
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
//  Shader 2: Insert
//    · 각 입력 점을 해시 슬롯에 CAS 로 배치
//    · TSDF = dot(voxelCenter - point, surfaceNormal) / truncation
//    · float-CAS 로 acc_tsdf, acc_w 누산
//    · acc_color, acc_norm 은 last-write-wins
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char *kVH_InsertComp = R"GLSL(
#version 450
layout(local_size_x = 64) in;
layout(push_constant) uniform PC {
    float    sensorX, sensorY, sensorZ;
    float    voxelSize;
    uint     numPoints;
    uint     numBuckets;
    float    truncation;
    float    maxWeight;
} pc;
)GLSL"
                                              // kVH_Common を続けて埋め込む
                                              R"GLSL(
#define STRIDE      12
#define F_KX         0
#define F_KY         1
#define F_KZ         2
#define F_TSDF       3
#define F_W          4
#define F_COL        5
#define F_NORM       6
#define F_ACCT       7
#define F_ACCW       8
#define F_ACCC       9
#define F_ACCN      10
#define F_FTAG      11
#define EMPTY       0x7FFFFFFF
#define BUCKET_SIZE 4

layout(set=0, binding=0) buffer HT  { int raw[]; } ht;
layout(set=0, binding=1) readonly buffer Pts { vec4 pts[]; };
layout(set=0, binding=2) buffer Ctr { uint n; } ctr;

void casAddF(int idx, float val) {
    int expected, next;
    for (int i = 0; i < 64; i++) {
        expected = ht.raw[idx];
        next     = floatBitsToInt(intBitsToFloat(expected) + val);
        if (atomicCompSwap(ht.raw[idx], expected, next) == expected) return;
    }
}
uint hash3(ivec3 k, uint nb) {
    return (uint(k.x)*73856093u ^ uint(k.y)*19349663u ^ uint(k.z)*83492791u) % nb;
}
uint packNormal(vec3 n) {
    float l1 = abs(n.x)+abs(n.y)+abs(n.z);
    n /= max(l1,1e-6);
    if (n.z < 0.0) {
        float sx = n.x>=0.0?1.0:-1.0, sy = n.y>=0.0?1.0:-1.0;
        n.xy = (1.0 - abs(n.yx)) * vec2(sx,sy);
    }
    uint nx = uint(clamp(n.x*0.5+0.5,0.0,1.0)*65535.0);
    uint ny = uint(clamp(n.y*0.5+0.5,0.0,1.0)*65535.0);
    return (nx & 0xFFFFu) | ((ny & 0xFFFFu)<<16u);
}
uint posToColor(ivec3 k) {
    uint h = uint(k.x)*73856093u ^ uint(k.y)*19349663u ^ uint(k.z)*83492791u;
    float t = float(h & 0xFFFFu)/65535.0;
    float r = clamp(abs(fract(t+0.000)*6.0-3.0)-1.0,0.0,1.0);
    float g = clamp(abs(fract(t+0.333)*6.0-3.0)-1.0,0.0,1.0);
    float b = clamp(abs(fract(t+0.667)*6.0-3.0)-1.0,0.0,1.0);
    return uint(r*255.0)|(uint(g*255.0)<<8u)|(uint(b*255.0)<<16u)|0xFF000000u;
}

void accumulate(uint slot, float tsdf, float w, uint col, uint norm) {
    uint base = slot * uint(STRIDE);
    casAddF(int(base + F_ACCT), tsdf * w);
    casAddF(int(base + F_ACCW), w);
    atomicExchange(ht.raw[int(base + F_ACCC)], int(col));
    atomicExchange(ht.raw[int(base + F_ACCN)], int(norm));
}

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.numPoints) return;

    vec3  pt     = pts[tid].xyz;
    vec3  sensor = vec3(pc.sensorX, pc.sensorY, pc.sensorZ);
    ivec3 key    = ivec3(floor(pt / pc.voxelSize));
    uint  base   = hash3(key, pc.numBuckets) * uint(BUCKET_SIZE);

    // ── TSDF 계산 ─────────────────────────────────────────────────────────────
    // 센서 방향이 표면 법선의 최선 근사
    vec3  surfNorm  = normalize(sensor - pt);
    vec3  voxCentre = (vec3(key) + 0.5) * pc.voxelSize;
    // 복셀 중심에서 실제 표면까지의 부호 있는 거리 (법선 방향 투영)
    float tsdf      = dot(voxCentre - pt, surfNorm) / pc.truncation;
    tsdf = clamp(tsdf, -1.0, 1.0);

    // 관측 신뢰도: 센서와 표면이 수직에 가까울수록 높음
    float w = max(dot(surfNorm, normalize(sensor - voxCentre)), 0.05);

    uint packedNorm  = packNormal(surfNorm);
    uint packedColor = posToColor(key);

    // ── CAS 슬롯 탐색 + 누산 ──────────────────────────────────────────────────
    for (int s = 0; s < BUCKET_SIZE; s++) {
        uint slot = base + uint(s);
        int  b    = int(slot * uint(STRIDE));

        // 이미 이 복셀로 할당된 슬롯 → 누산만
        if (ht.raw[b+F_KX] == key.x &&
            ht.raw[b+F_KY] == key.y &&
            ht.raw[b+F_KZ] == key.z) {
            accumulate(slot, tsdf, w, packedColor, packedNorm);
            return;
        }

        // 빈 슬롯 CAS 점유
        int prev = atomicCompSwap(ht.raw[b+F_KX], EMPTY, key.x);
        if (prev == EMPTY) {
            ht.raw[b+F_KY] = key.y;
            ht.raw[b+F_KZ] = key.z;
            accumulate(slot, tsdf, w, packedColor, packedNorm);
            return;
        }
    }
    // 버킷 포화 — 이 점 드롭 (정상적인 해시 부하에서는 드묾)
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
//  Shader 3: Finalize
//    · 각 점유 슬롯에 대해 running weighted average 커밋
//    · acc_tsdf, acc_w → 가중 평균으로 tsdf, weight 갱신
//    · frame_tag 에 currentFrame 기록 → 디버그 플래시 근거
//    · 누산 필드 초기화
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char *kVH_FinalizeComp = R"GLSL(
#version 450
layout(local_size_x = 64) in;
layout(push_constant) uniform PC {
    uint  totalEntries;
    uint  currentFrame;
    float _p0, _p1;
} pc;

#define STRIDE  12
#define F_KX     0
#define F_TSDF   3
#define F_W      4
#define F_COL    5
#define F_NORM   6
#define F_ACCT   7
#define F_ACCW   8
#define F_ACCC   9
#define F_ACCN  10
#define F_FTAG  11
#define EMPTY   0x7FFFFFFF

layout(set=0, binding=0) buffer HT  { int raw[]; } ht;
layout(set=0, binding=1) buffer Pts { vec4 pts[]; };
layout(set=0, binding=2) buffer Ctr { uint n; } ctr;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.totalEntries) return;

    int b = int(i * uint(STRIDE));

    // 비어있는 슬롯 스킵
    if (ht.raw[b + F_KX] == EMPTY) return;

    float accW = intBitsToFloat(ht.raw[b + F_ACCW]);
    if (accW <= 0.0) return;  // 이번 배치에서 관측 없음

    float accT = intBitsToFloat(ht.raw[b + F_ACCT]);
    float oldW = intBitsToFloat(ht.raw[b + F_W]);
    float oldT = intBitsToFloat(ht.raw[b + F_TSDF]);

    // 가중 누진 평균 (논문 수식 ② 대응)
    float newW = min(oldW + accW, 50.0);   // maxWeight 클램핑
    float newT = (oldT * oldW + accT) / (oldW + accW);
    newT = clamp(newT, -1.0, 1.0);

    ht.raw[b + F_TSDF]  = floatBitsToInt(newT);
    ht.raw[b + F_W]     = floatBitsToInt(newW);
    ht.raw[b + F_COL]   = ht.raw[b + F_ACCC];
    ht.raw[b + F_NORM]  = ht.raw[b + F_ACCN];
    ht.raw[b + F_FTAG]  = int(pc.currentFrame); // ← 디버그 플래시 기록

    // 누산 필드 초기화 (다음 배치를 위해)
    ht.raw[b + F_ACCT] = 0;
    ht.raw[b + F_ACCW] = 0;
    ht.raw[b + F_ACCC] = 0;
    ht.raw[b + F_ACCN] = 0;
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
//  Shader 4: Count — 점유된 슬롯 수를 ctr.n 에 집계
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char *kVH_CountComp = R"GLSL(
#version 450
layout(local_size_x = 64) in;
layout(push_constant) uniform PC { uint total; uint _f; float _0; float _1; } pc;

#define STRIDE 12
#define EMPTY  0x7FFFFFFF

layout(set=0, binding=0) readonly buffer HT  { int raw[]; } ht;
layout(set=0, binding=1)          buffer Pts { vec4 pts[]; };
layout(set=0, binding=2)          buffer Ctr { uint n; } ctr;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    if (ht.raw[int(i * uint(STRIDE))] != EMPTY)
        atomicAdd(ctr.n, 1u);
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
//  Shader 5: Render (Vertex)
//    · 빈 슬롯 → NDC 밖으로 클리핑
//    · colorMode 에 따라 색상 결정
//    · frame_tag 기반 황색 플래시 오버레이 (디버그 핵심)
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char *kVH_RenderVert = R"GLSL(
#version 450
layout(push_constant) uniform PC {
    mat4     mvp;
    float    voxelSize;
    uint     colorMode;
    uint     currentFrame;
    uint     highlightFrames;
} pc;

layout(set=0, binding=0) readonly buffer HT { int raw[]; } ht;

layout(location = 0) out vec3  fragColor;
layout(location = 1) out float fragHighlight;

#define STRIDE  12
#define F_KX     0
#define F_TSDF   3
#define F_W      4
#define F_COL    5
#define F_NORM   6
#define F_FTAG  11
#define EMPTY   0x7FFFFFFF

vec3 unpackNormal(uint packed) {
    float fx = float(packed & 0xFFFFu)/65535.0*2.0-1.0;
    float fy = float((packed>>16u)&0xFFFFu)/65535.0*2.0-1.0;
    vec3 n = vec3(fx, fy, 1.0 - abs(fx) - abs(fy));
    if (n.z < 0.0) {
        float ox = n.x>=0.0?(1.0-abs(n.y)):-(1.0-abs(n.y));
        float oy = n.y>=0.0?(1.0-abs(n.x)):-(1.0-abs(n.x));
        n.x=ox; n.y=oy;
    }
    return normalize(n);
}

vec3 tsdfColor(float t) {
    if (t < 0.0) return mix(vec3(0.0,0.0,1.0), vec3(1.0,1.0,1.0), t + 1.0);
    else          return mix(vec3(1.0,1.0,1.0), vec3(1.0,0.0,0.0), t);
}

void main() {
    int b = gl_VertexIndex * STRIDE;
    int kx = ht.raw[b + F_KX];

    if (kx == EMPTY) {
        gl_Position   = vec4(10.0, 10.0, 10.0, 1.0);
        gl_PointSize  = 0.0;
        fragColor     = vec3(0.0);
        fragHighlight = 0.0;
        return;
    }

    int ky = ht.raw[b+1];
    int kz = ht.raw[b+2];
    vec3 centre = (vec3(float(kx), float(ky), float(kz)) + 0.5) * pc.voxelSize;

    // ── Integration 플래시 계산 (PointSize에도 영향) ──────────────────────────
    int   frameTag  = ht.raw[b + F_FTAG];
    float age       = float(int(pc.currentFrame) - frameTag);
    float highlightT = (frameTag >= 0)
        ? max(0.0, 1.0 - age / float(pc.highlightFrames))
        : 0.0;

    gl_Position  = pc.mvp * vec4(centre, 1.0);
    // 최근 갱신 복셀일수록 크게 표시 (2px → 8px)
    gl_PointSize = mix(2.0, 8.0, highlightT);

    // ── 기본 색상 (colorMode) ─────────────────────────────────────────────────
    vec3 base;
    if (pc.colorMode == 0u) {
        uint col = uint(ht.raw[b + F_COL]);
        base = vec3(float(col & 0xFFu), float((col>>8u)&0xFFu), float((col>>16u)&0xFFu)) / 255.0;
    } else if (pc.colorMode == 1u) {
        uint norm = uint(ht.raw[b + F_NORM]);
        vec3 n = unpackNormal(norm);
        base = n * 0.5 + 0.5;
    } else if (pc.colorMode == 2u) {
        float t = intBitsToFloat(ht.raw[b + F_TSDF]);
        base = tsdfColor(t);
    } else if (pc.colorMode == 3u) {
        float w = clamp(intBitsToFloat(ht.raw[b + F_W]) / 50.0, 0.0, 1.0);
        base = vec3(w);
    } else {
        // colorMode 4: 업데이트 최신도 히트맵 (빨강=최근, 파랑=오래됨, 회색=미관측)
        if (frameTag < 0) {
            base = vec3(0.15);
        } else {
            float normAge = clamp(age / float(max(pc.highlightFrames * 4u, 1u)), 0.0, 1.0);
            base = mix(vec3(1.0, 0.2, 0.0), vec3(0.0, 0.3, 1.0), normAge);
        }
    }

    // 황색 플래시 오버레이
    fragColor     = mix(base, vec3(1.0, 0.9, 0.1), highlightT * 0.85);
    fragHighlight = highlightT;
}
)GLSL";

static constexpr const char *kVH_RenderFrag = R"GLSL(
#version 450
layout(location = 0) in  vec3  fragColor;
layout(location = 1) in  float fragHighlight;
layout(location = 0) out vec4  outColor;
void main() {
    outColor = vec4(fragColor, 1.0);
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
//  Shader 6: 입력 포인트 클라우드 오버레이 렌더 (시안색 점)
//    · ptBuf_ (binding=1) 의 현재 배치를 시안색 점으로 표시
//    · Integration 전 원본 포인트의 위치를 시각적으로 확인
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char *kVH_PtRenderVert = R"GLSL(
#version 450
layout(push_constant) uniform PC {
    mat4  mvp;
    float voxelSize;
    uint  colorMode;
    uint  currentFrame;
    uint  highlightFrames;
} pc;

layout(set=0, binding=1) readonly buffer Pts { vec4 pts[]; };

layout(location=0) out vec3  fragColor;
layout(location=1) out float fragHighlight;

void main() {
    vec3 p = pts[gl_VertexIndex].xyz;
    gl_Position   = pc.mvp * vec4(p, 1.0);
    gl_PointSize  = 3.0;
    fragColor     = vec3(0.0, 1.0, 1.0);  // 시안: 입력 포인트
    fragHighlight = 0.0;
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
//  카메라 (BucketedHash 와 동일 패턴)
// ─────────────────────────────────────────────────────────────────────────────

Eigen::Matrix4f VoxelHashFeature::computeMVP() const
{
    float cx = std::cos(elevation_), sx = std::sin(elevation_);
    float cy = std::cos(azimuth_), sy = std::sin(azimuth_);
    Eigen::Vector3f eye(camDist_ * cx * sy, camDist_ * sx, camDist_ * cx * cy);
    Eigen::Vector3f up(0.f, 1.f, 0.f);
    Eigen::Vector3f f = (-eye).normalized();
    Eigen::Vector3f r = f.cross(up).normalized();
    Eigen::Vector3f u = r.cross(f);

    Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
    V.row(0) << r.x(), r.y(), r.z(), -r.dot(eye);
    V.row(1) << u.x(), u.y(), u.z(), -u.dot(eye);
    V.row(2) << -f.x(), -f.y(), -f.z(), f.dot(eye);

    float fovY = 60.f * static_cast<float>(M_PI) / 180.f;
    float aspect = static_cast<float>(ctx_.extent.width) / ctx_.extent.height;
    float n = 0.01f, fa = 100.f, th = std::tan(fovY * 0.5f);

    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
    P(0, 0) = 1.f / (aspect * th);
    P(1, 1) = -1.f / th;
    P(2, 2) = fa / (n - fa);
    P(2, 3) = fa * n / (n - fa);
    P(3, 2) = -1.f;

    return P * V;
}

// ─────────────────────────────────────────────────────────────────────────────
//  구면 포인트 배치 생성
// ─────────────────────────────────────────────────────────────────────────────

static float gaussRand()
{
    float u1 = (rand() + 1.f) / (static_cast<float>(RAND_MAX) + 2.f);
    float u2 = rand() / static_cast<float>(RAND_MAX);
    return std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * static_cast<float>(M_PI) * u2);
}

uint32_t VoxelHashFeature::genSphereBatch()
{
    float cx = std::cos(sensorEl_), sx = std::sin(sensorEl_);
    float cy = std::cos(sensorAz_), sy = std::sin(sensorAz_);
    Eigen::Vector3f spos(sensorDist_ * cx * sy, sensorDist_ * sx, sensorDist_ * cx * cy);

    void *mapped;
    vkMapMemory(ctx_.device, ptMem_, 0, VK_WHOLE_SIZE, 0, &mapped);
    auto *dst = static_cast<float *>(mapped);

    uint32_t written = 0;
    while (written < VH_BATCH_SIZE)
    {
        float u = (rand() / static_cast<float>(RAND_MAX)) * 2.f - 1.f;
        float t = (rand() / static_cast<float>(RAND_MAX)) * 2.f * static_cast<float>(M_PI);
        float r = std::sqrt(std::max(0.f, 1.f - u * u));
        Eigen::Vector3f n(r * std::cos(t), u, r * std::sin(t));
        Eigen::Vector3f p = n * sphereRadius_;

        Eigen::Vector3f toSensor = (spos - p).normalized();
        if (n.dot(toSensor) < 0.25f)
            continue;

        p += n * (gaussRand() * noiseStddev_);
        dst[written * 4 + 0] = p.x();
        dst[written * 4 + 1] = p.y();
        dst[written * 4 + 2] = p.z();
        dst[written * 4 + 3] = 1.f;
        written++;
    }
    vkUnmapMemory(ctx_.device, ptMem_);
    return written;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Vulkan 헬퍼
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::createBuf(VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags props,
                                 VkBuffer &buf, VkDeviceMemory &mem)
{
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx_.device, &bci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: vkCreateBuffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx_.device, buf, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(ctx_.physicalDevice, req.memoryTypeBits, props);
    if (vkAllocateMemory(ctx_.device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: vkAllocateMemory");
    vkBindBufferMemory(ctx_.device, buf, mem, 0);
}

void VoxelHashFeature::bufBarrier(VkCommandBuffer cmd, VkBuffer buf,
                                  VkAccessFlags src, VkAccessFlags dst,
                                  VkPipelineStageFlags srcS, VkPipelineStageFlags dstS)
{
    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = buf;
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 1, &b, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
//  createBuffers
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::createBuffers()
{
    // 해시 테이블: device-local (524288 × 48 bytes ≈ 24 MB)
    createBuf(sizeof(VH_Entry) * VH_TOTAL_ENTRIES,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
              htBuf_, htMem_);

    // 포인트 배치: host-visible
    createBuf(sizeof(float) * 4 * VH_BATCH_SIZE,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              ptBuf_, ptMem_);

    // 카운터: host-visible + transfer-dst
    createBuf(sizeof(uint32_t),
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              ctrBuf_, ctrMem_);
}

// ─────────────────────────────────────────────────────────────────────────────
//  createDescriptors  — binding 0: HT, 1: Pts, 2: Ctr
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::createDescriptors()
{
    VkDescriptorSetLayoutBinding b[3]{};
    for (int i = 0; i < 3; i++)
    {
        b[i].binding = static_cast<uint32_t>(i);
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 3;
    lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(ctx_.device, &lci, nullptr, &descLayout_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: descriptor layout");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(ctx_.device, &pci, nullptr, &descPool_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: descriptor pool");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &descLayout_;
    if (vkAllocateDescriptorSets(ctx_.device, &ai, &descSet_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: descriptor alloc");

    VkDescriptorBufferInfo htI{htBuf_, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo ptI{ptBuf_, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo ctrI{ctrBuf_, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[3]{};
    VkDescriptorBufferInfo *infos[3] = {&htI, &ptI, &ctrI};
    for (int i = 0; i < 3; i++)
    {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = descSet_;
        w[i].dstBinding = static_cast<uint32_t>(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = infos[i];
    }
    vkUpdateDescriptorSets(ctx_.device, 3, w, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
//  파이프라인 생성
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::createComputePipelines()
{
    // Compute 레이아웃: 최대 push constant = sizeof(VH_InsertPC) = 32 bytes
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lci.setLayoutCount = 1;
    lci.pSetLayouts = &descLayout_;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(ctx_.device, &lci, nullptr, &compLayout_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: compute layout");

    auto make = [&](const char *glsl)
    {
        VkShaderModule mod = compileGLSL(ctx_.device, glsl, shaderc_compute_shader);
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = compLayout_;
        VkPipeline p;
        if (vkCreateComputePipelines(ctx_.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p) != VK_SUCCESS)
            throw std::runtime_error("VoxelHash: compute pipeline");
        vkDestroyShaderModule(ctx_.device, mod, nullptr);
        return p;
    };

    clearPipe_ = make(kVH_ClearComp);
    insertPipe_ = make(kVH_InsertComp);
    finalizePipe_ = make(kVH_FinalizeComp);
    countPipe_ = make(kVH_CountComp);
}

void VoxelHashFeature::createRenderPipeline()
{
    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VH_RenderPC)};
    VkPipelineLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lci.setLayoutCount = 1;
    lci.pSetLayouts = &descLayout_;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(ctx_.device, &lci, nullptr, &renderLayout_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: render layout");

    auto vMod = compileGLSL(ctx_.device, kVH_RenderVert, shaderc_vertex_shader);
    auto fMod = compileGLSL(ctx_.device, kVH_RenderFrag, shaderc_fragment_shader);

    VkPipelineShaderStageCreateInfo stages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vMod, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fMod, "main"}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                              nullptr, 0, VK_PRIMITIVE_TOPOLOGY_POINT_LIST};
    VkViewport vp{0, 0, (float)ctx_.extent.width, (float)ctx_.extent.height, 0, 1};
    VkRect2D sc{{0, 0}, ctx_.extent};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                          nullptr, 0, 1, &vp, 1, &sc};
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth = 1.f;
    raster.cullMode = VK_CULL_MODE_NONE;

    // 포인트 크기 동적 설정을 위해 필요
    VkPipelineDynamicStateCreateInfo dynSt{};
    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    dynSt.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynSt.dynamicStateCount = 0; // viewport/scissor 는 고정

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                            nullptr, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              nullptr,
                                              0,
                                              VK_FALSE,
                                              {},
                                              1,
                                              &ba};
    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vps;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &ms;
    pci.pColorBlendState = &blend;
    pci.layout = renderLayout_;
    pci.renderPass = ctx_.renderPass;
    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &pci, nullptr, &renderPipe_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: render pipeline");

    vkDestroyShaderModule(ctx_.device, vMod, nullptr);

    // 입력 포인트 클라우드 파이프라인 (같은 frag, 같은 layout, vert만 교체)
    auto pvMod = compileGLSL(ctx_.device, kVH_PtRenderVert, shaderc_vertex_shader);
    stages[0].module = pvMod;
    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &pci, nullptr, &ptRenderPipe_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: ptRender pipeline");
    vkDestroyShaderModule(ctx_.device, pvMod, nullptr);

    vkDestroyShaderModule(ctx_.device, fMod, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Dispatch 헬퍼
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::dispatchClear(VkCommandBuffer cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clearPipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compLayout_,
                            0, 1, &descSet_, 0, nullptr);
    VH_FinalizePC pc{VH_TOTAL_ENTRIES, frameIndex_, 0.f, 0.f};
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &pc);
    vkCmdDispatch(cmd, (VH_TOTAL_ENTRIES + 63) / 64, 1, 1);
}

void VoxelHashFeature::dispatchInsert(VkCommandBuffer cmd)
{
    if (ptCount_ == 0)
        return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, insertPipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compLayout_,
                            0, 1, &descSet_, 0, nullptr);

    float cx = std::cos(sensorEl_), sx = std::sin(sensorEl_);
    float cy = std::cos(sensorAz_), sy = std::sin(sensorAz_);
    VH_InsertPC pc{};
    pc.sensorX = sensorDist_ * cx * sy;
    pc.sensorY = sensorDist_ * sx;
    pc.sensorZ = sensorDist_ * cx * cy;
    pc.voxelSize = voxelSize_;
    pc.numPoints = ptCount_;
    pc.numBuckets = VH_NUM_BUCKETS;
    pc.truncation = truncation_;
    pc.maxWeight = maxWeight_;
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (ptCount_ + 63) / 64, 1, 1);
}

void VoxelHashFeature::dispatchFinalize(VkCommandBuffer cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, finalizePipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compLayout_,
                            0, 1, &descSet_, 0, nullptr);
    VH_FinalizePC pc{VH_TOTAL_ENTRIES, frameIndex_, 0.f, 0.f};
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (VH_TOTAL_ENTRIES + 63) / 64, 1, 1);
}

void VoxelHashFeature::dispatchCount(VkCommandBuffer cmd)
{
    vkCmdFillBuffer(cmd, ctrBuf_, 0, sizeof(uint32_t), 0);
    bufBarrier(cmd, ctrBuf_,
               VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, countPipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compLayout_,
                            0, 1, &descSet_, 0, nullptr);
    VH_FinalizePC pc{VH_TOTAL_ENTRIES, frameIndex_, 0.f, 0.f};
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (VH_TOTAL_ENTRIES + 63) / 64, 1, 1);
    bufBarrier(cmd, ctrBuf_,
               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
}

// ─────────────────────────────────────────────────────────────────────────────
//  onInit
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::onInit(const VulkanContext &ctx)
{
    ctx_ = ctx;
    createBuffers();
    createDescriptors();
    createComputePipelines();
    createRenderPipeline();
    lastTime_ = std::chrono::steady_clock::now();
    doClear_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  onCompute
//  순서: [Clear?] → Insert → barrier → Finalize → barrier → Count
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::onCompute(VkCommandBuffer cmd)
{
    frameIndex_++;

    // 카운터 읽기 (이전 프레임 GPU 완료 후)
    if (doCountRead_)
    {
        void *m;
        vkMapMemory(ctx_.device, ctrMem_, 0, sizeof(uint32_t), 0, &m);
        occupancy_ = static_cast<int>(*static_cast<uint32_t *>(m));
        vkUnmapMemory(ctx_.device, ctrMem_);
        doCountRead_ = false;
    }

    if (doClear_)
    {
        dispatchClear(cmd);
        bufBarrier(cmd, htBuf_,
                   VK_ACCESS_SHADER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        doClear_ = false;
        occupancy_ = 0;
        totalInserted_ = 0;
    }

    if (streaming_)
    {
        ptCount_ = genSphereBatch();
        doInsert_ = true;
        sensorAz_ += sensorSpeed_;
        if (sensorAz_ > 2.f * static_cast<float>(M_PI))
            sensorAz_ -= 2.f * static_cast<float>(M_PI);
    }

    if (doInsert_ && ptCount_ > 0)
    {
        // ① Insert: 점 → 버킷 CAS 배치 + float-CAS 누산
        dispatchInsert(cmd);
        bufBarrier(cmd, htBuf_,
                   VK_ACCESS_SHADER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // ② Finalize: running-average 커밋 + frame_tag 기록
        dispatchFinalize(cmd);
        bufBarrier(cmd, htBuf_,
                   VK_ACCESS_SHADER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);

        // ③ Count
        dispatchCount(cmd);

        totalInserted_ += ptCount_;
        doInsert_ = false;
        doFinalize_ = false;
        doCountRead_ = true;

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime_).count();
        lastTime_ = now;
        if (dt > 0.f)
            ptsPerSec_ = 0.9f * ptsPerSec_ + 0.1f * (ptCount_ / dt);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  onRender
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::onRender(const RenderContext &ctx)
{
    vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, renderPipe_);
    vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            renderLayout_, 0, 1, &descSet_, 0, nullptr);

    Eigen::Matrix4f mvp = computeMVP();
    VH_RenderPC pc{};
    std::memcpy(pc.mvp, mvp.data(), 64);
    pc.voxelSize = voxelSize_;
    pc.colorMode = static_cast<uint32_t>(colorMode_);
    pc.currentFrame = frameIndex_;
    pc.highlightFrames = static_cast<uint32_t>(highlightFrames_);
    vkCmdPushConstants(ctx.commandBuffer, renderLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

    // 슬롯 수 만큼 버텍스 발행 — 빈 슬롯은 셰이더에서 NDC 밖으로 클리핑
    vkCmdDraw(ctx.commandBuffer, VH_TOTAL_ENTRIES, 1, 0, 0);

    // 입력 포인트 클라우드 오버레이 (시안색, 현재 배치만)
    if (showInputPts_ && ptCount_ > 0) {
        vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ptRenderPipe_);
        vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                renderLayout_, 0, 1, &descSet_, 0, nullptr);
        vkCmdPushConstants(ctx.commandBuffer, renderLayout_,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
        vkCmdDraw(ctx.commandBuffer, ptCount_, 1, 0, 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  onImGui
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::onImGui()
{
    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureMouse)
    {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
        {
            azimuth_ += io.MouseDelta.x * 0.005f;
            elevation_ += io.MouseDelta.y * 0.005f;
            elevation_ = std::max(-1.4f, std::min(1.4f, elevation_));
        }
        camDist_ -= io.MouseWheel * 0.3f;
        camDist_ = std::max(0.5f, std::min(20.f, camDist_));
    }

    ImGui::Begin("VoxelHash");

    // ── 테이블 통계 ───────────────────────────────────────────────────────────
    float htMB = sizeof(VH_Entry) * VH_TOTAL_ENTRIES / (1024.f * 1024.f);
    ImGui::Text("Hash table  : %u buckets × %u = %u entries  (%.1f MB)",
                VH_NUM_BUCKETS, VH_BUCKET_SIZE, VH_TOTAL_ENTRIES, htMB);
    float fillPct = 100.f * occupancy_ / static_cast<float>(VH_TOTAL_ENTRIES);
    ImGui::Text("Occupied    : %d  /  %u  (%.2f%%)", occupancy_, VH_TOTAL_ENTRIES, fillPct);
    ImGui::Text("Total pts   : %lld", static_cast<long long>(totalInserted_));
    ImGui::Text("Throughput  : %.0f pts/s", ptsPerSec_);
    ImGui::Text("Frame       : %u", frameIndex_);
    ImGui::Separator();

    // ── 카메라 ────────────────────────────────────────────────────────────────
    ImGui::Text("Camera  [drag=rotate  wheel=zoom]");
    ImGui::SliderFloat("Azimuth##vh", &azimuth_, -3.14f, 3.14f, "%.2f");
    ImGui::SliderFloat("Elevation##vh", &elevation_, -1.4f, 1.4f, "%.2f");
    ImGui::SliderFloat("Distance##vh", &camDist_, 0.5f, 20.f, "%.1f");
    ImGui::Separator();

    // ── 색상 모드 ─────────────────────────────────────────────────────────────
    ImGui::Text("Color mode");
    ImGui::RadioButton("Surface color", &colorMode_, 0); ImGui::SameLine();
    ImGui::RadioButton("Normal",        &colorMode_, 1); ImGui::SameLine();
    ImGui::RadioButton("TSDF",          &colorMode_, 2); ImGui::SameLine();
    ImGui::RadioButton("Weight",        &colorMode_, 3); ImGui::SameLine();
    ImGui::RadioButton("Recency",       &colorMode_, 4);
    if (colorMode_ == 4) {
        ImGui::TextColored(ImVec4(1.0f,0.2f,0.0f,1), "■"); ImGui::SameLine();
        ImGui::Text("= recent");  ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.0f,0.3f,1.0f,1), "■"); ImGui::SameLine();
        ImGui::Text("= old");
    }

    // ── Integration 시각화 ────────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Text("Integration visualization");
    ImGui::SliderInt("Highlight frames", &highlightFrames_, 1, 60,
                     "flash for %d frames");
    ImGui::TextColored(ImVec4(1, 0.9f, 0.1f, 1), "■"); ImGui::SameLine();
    ImGui::Text("= voxel updated this batch (size ∝ recency)");
    ImGui::Checkbox("Show input points (cyan)", &showInputPts_);
    if (showInputPts_) {
        ImGui::TextColored(ImVec4(0.0f,1.0f,1.0f,1), "●"); ImGui::SameLine();
        ImGui::Text("= current batch  (%u pts)", ptCount_);
    }
    ImGui::Separator();

    // ── Integration 파라미터 ──────────────────────────────────────────────────
    bool rebuild = false;
    rebuild |= ImGui::SliderFloat("Voxel size", &voxelSize_, 0.005f, 0.1f, "%.4f m");
    rebuild |= ImGui::SliderFloat("Truncation", &truncation_, 0.01f, 0.2f, "%.3f m");
    ImGui::SliderFloat("Max weight", &maxWeight_, 5.f, 200.f, "%.0f");
    ImGui::SliderFloat("Sphere r", &sphereRadius_, 0.1f, 3.f, "%.2f");
    ImGui::SliderFloat("Noise", &noiseStddev_, 0.f, 0.05f, "%.4f");
    ImGui::SliderFloat("Speed", &sensorSpeed_, 0.f, 0.1f, "%.3f rad/f");
    if (rebuild && streaming_)
        doClear_ = true;
    ImGui::Separator();

    if (ImGui::Button(streaming_ ? "■ Stop" : "▶ Start"))
    {
        streaming_ = !streaming_;
        if (streaming_)
        {
            doClear_ = true;
            sensorAz_ = 0.f;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset"))
    {
        doClear_ = true;
    }

    if (streaming_)
    {
        float cx = std::cos(sensorEl_), sx = std::sin(sensorEl_);
        float cy = std::cos(sensorAz_), sy = std::sin(sensorAz_);
        ImGui::Text("Sensor: (%.2f, %.2f, %.2f)",
                    sensorDist_ * cx * sy, sensorDist_ * sx, sensorDist_ * cx * cy);
    }

    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────────────────
//  onCleanup
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::onCleanup()
{
    vkDestroyPipeline(ctx_.device, clearPipe_, nullptr);
    vkDestroyPipeline(ctx_.device, insertPipe_, nullptr);
    vkDestroyPipeline(ctx_.device, finalizePipe_, nullptr);
    vkDestroyPipeline(ctx_.device, countPipe_, nullptr);
    vkDestroyPipeline(ctx_.device, renderPipe_, nullptr);
    vkDestroyPipeline(ctx_.device, ptRenderPipe_, nullptr);
    vkDestroyPipelineLayout(ctx_.device, compLayout_, nullptr);
    vkDestroyPipelineLayout(ctx_.device, renderLayout_, nullptr);
    vkDestroyDescriptorPool(ctx_.device, descPool_, nullptr);
    vkDestroyDescriptorSetLayout(ctx_.device, descLayout_, nullptr);

    vkDestroyBuffer(ctx_.device, htBuf_, nullptr);
    vkFreeMemory(ctx_.device, htMem_, nullptr);
    vkDestroyBuffer(ctx_.device, ptBuf_, nullptr);
    vkFreeMemory(ctx_.device, ptMem_, nullptr);
    vkDestroyBuffer(ctx_.device, ctrBuf_, nullptr);
    vkFreeMemory(ctx_.device, ctrMem_, nullptr);
}
