#include "VoxelHash.h"
#include <imgui.h>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <array>
#include <numeric>
#include <unordered_map>

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
//  Shader 2: GatherInsert  (논문 방식 — float-CAS 완전 제거)
//
//  설계 원칙 (vs 기존 scatter + float-CAS)
//    · CPU 에서 점을 버킷 해시 기준으로 정렬 + cellStart/cellEnd 구축
//    · 1 workgroup = 1 버킷 (32 threads 협력)
//    · Phase1: 슬롯 키 로드
//    · Phase2: thread 0 이 슬롯 할당 (같은 버킷 = 동일 workgroup → 원자 불필요)
//    · Phase3: 32 스레드 협력 gather → 스레드 별 레지스터 누산 (원자 없음)
//    · Phase4: tree reduction (shared memory)
//    · Phase5: thread t → slot t 단독 write (경쟁 없음)
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char *kVH_GatherComp = R"GLSL(
#version 450
layout(local_size_x = 32) in;

layout(push_constant) uniform PC {
    float sensorX, sensorY, sensorZ;
    float voxelSize;
    uint  _numPts;
    uint  _numBuckets;
    float truncation;
    float _maxW;
} pc;

#define STRIDE      12
#define F_KX         0
#define F_KY         1
#define F_KZ         2
#define F_ACCT       7
#define F_ACCW       8
#define F_ACCC       9
#define F_ACCN      10
#define EMPTY        0x7FFFFFFF
#define BUCKET_SIZE  4

layout(set=0, binding=0) buffer HT  { int raw[]; } ht;
layout(set=0, binding=1) readonly buffer Pts { vec4 pts[]; };
layout(set=0, binding=2) buffer Ctr { uint n; } ctr;
layout(set=0, binding=3) readonly buffer CS { uint d[]; } cellStart;
layout(set=0, binding=4) readonly buffer CE { uint d[]; } cellEnd;

shared int   sKX[BUCKET_SIZE];
shared int   sKY[BUCKET_SIZE];
shared int   sKZ[BUCKET_SIZE];
shared float sScrT[32];
shared float sScrW[32];
shared float sFT[BUCKET_SIZE];
shared float sFW[BUCKET_SIZE];
shared int   sFC[BUCKET_SIZE];
shared int   sFN[BUCKET_SIZE];

uint packNorm(vec3 n) {
    float l = abs(n.x)+abs(n.y)+abs(n.z);
    n /= max(l, 1e-6);
    if (n.z < 0.0) {
        vec2 s = vec2(n.x>=0.0?1.0:-1.0, n.y>=0.0?1.0:-1.0);
        n.xy = (1.0-abs(n.yx))*s;
    }
    return (uint(clamp(n.x*.5+.5,0.,1.)*65535.)&0xFFFFu)
          |((uint(clamp(n.y*.5+.5,0.,1.)*65535.)&0xFFFFu)<<16u);
}
uint posCol(ivec3 k) {
    uint h = uint(k.x)*73856093u^uint(k.y)*19349663u^uint(k.z)*83492791u;
    float t = float(h&0xFFFFu)/65535.;
    return uint(clamp(abs(fract(t      )*6.-3.)-1.,0.,1.)*255.)
          |(uint(clamp(abs(fract(t+.333)*6.-3.)-1.,0.,1.)*255.)<<8u)
          |(uint(clamp(abs(fract(t+.667)*6.-3.)-1.,0.,1.)*255.)<<16u)
          |0xFF000000u;
}

void main() {
    uint bucket = gl_WorkGroupID.x;
    uint tid    = gl_LocalInvocationID.x;

    // ── Phase 1: 기존 슬롯 키 로드 ───────────────────────────────────────────
    if (tid < uint(BUCKET_SIZE)) {
        int gb = int((bucket * uint(BUCKET_SIZE) + tid) * uint(STRIDE));
        sKX[tid] = ht.raw[gb + F_KX];
        sKY[tid] = ht.raw[gb + F_KY];
        sKZ[tid] = ht.raw[gb + F_KZ];
        sFT[tid] = 0.0; sFW[tid] = 0.0; sFC[tid] = 0; sFN[tid] = 0;
    }
    barrier();

    uint pStart = cellStart.d[bucket];
    uint pEnd   = cellEnd.d[bucket];
    if (pStart >= pEnd) return;  // 이 버킷에 점 없음 → early exit

    // ── Phase 2: thread 0 이 슬롯 할당 (workgroup 단독 소유 → 원자 불필요) ───
    if (tid == 0u) {
        for (uint i = pStart; i < pEnd; i++) {
            ivec3 key = ivec3(floor(pts[i].xyz / pc.voxelSize));
            bool found = false;
            for (int s = 0; s < BUCKET_SIZE; s++) {
                if (sKX[s]==key.x && sKY[s]==key.y && sKZ[s]==key.z) { found=true; break; }
            }
            if (!found) {
                for (int s = 0; s < BUCKET_SIZE; s++) {
                    if (sKX[s] == EMPTY) {
                        int gb = int((bucket*uint(BUCKET_SIZE)+uint(s))*uint(STRIDE));
                        ht.raw[gb+0] = key.x;
                        ht.raw[gb+1] = key.y;
                        ht.raw[gb+2] = key.z;
                        sKX[s]=key.x; sKY[s]=key.y; sKZ[s]=key.z;
                        break;
                    }
                }
            }
        }
    }
    barrier();

    // ── Phase 3: 32 thread 협력 gather — 스레드 별 레지스터 누산 (원자 없음) ─
    vec3  sensor = vec3(pc.sensorX, pc.sensorY, pc.sensorZ);
    float myT[4]; float myW[4]; int myC[4]; int myN[4];
    myT[0]=myT[1]=myT[2]=myT[3]=0.0;
    myW[0]=myW[1]=myW[2]=myW[3]=0.0;
    myC[0]=myC[1]=myC[2]=myC[3]=0;
    myN[0]=myN[1]=myN[2]=myN[3]=0;

    for (uint i = pStart + tid; i < pEnd; i += 32u) {
        vec3  pt  = pts[i].xyz;
        ivec3 key = ivec3(floor(pt / pc.voxelSize));
        int s = -1;
        if      (sKX[0]==key.x&&sKY[0]==key.y&&sKZ[0]==key.z) s=0;
        else if (sKX[1]==key.x&&sKY[1]==key.y&&sKZ[1]==key.z) s=1;
        else if (sKX[2]==key.x&&sKY[2]==key.y&&sKZ[2]==key.z) s=2;
        else if (sKX[3]==key.x&&sKY[3]==key.y&&sKZ[3]==key.z) s=3;
        if (s < 0) continue;

        vec3  sn   = normalize(sensor - pt);
        vec3  vc   = (vec3(key)+0.5)*pc.voxelSize;
        float tsdf = clamp(dot(vc-pt, sn)/pc.truncation, -1.0, 1.0);
        float w    = max(dot(sn, normalize(sensor-vc)), 0.05);
        myT[s] += tsdf*w; myW[s] += w;
        myC[s]  = int(posCol(key)); myN[s] = int(packNorm(sn));
    }

    // ── Phase 4: 슬롯별 tree reduction (shared memory) ───────────────────────
#define REDUCE_SLOT(S) \
    sScrT[tid]=myT[S]; sScrW[tid]=myW[S]; barrier(); \
    if(tid<16u){sScrT[tid]+=sScrT[tid+16u];sScrW[tid]+=sScrW[tid+16u];} barrier(); \
    if(tid< 8u){sScrT[tid]+=sScrT[tid+ 8u];sScrW[tid]+=sScrW[tid+ 8u];} barrier(); \
    if(tid< 4u){sScrT[tid]+=sScrT[tid+ 4u];sScrW[tid]+=sScrW[tid+ 4u];} barrier(); \
    if(tid< 2u){sScrT[tid]+=sScrT[tid+ 2u];sScrW[tid]+=sScrW[tid+ 2u];} barrier(); \
    if(tid< 1u){sScrT[0]+=sScrT[1];sScrW[0]+=sScrW[1];} barrier(); \
    if(tid==0u){sFT[S]=sScrT[0];sFW[S]=sScrW[0];}

    REDUCE_SLOT(0)
    REDUCE_SLOT(1)
    REDUCE_SLOT(2)
    REDUCE_SLOT(3)

    // 색상/법선: thread 0의 마지막 관측값 사용
    if (tid == 0u) {
        for (int s = 0; s < BUCKET_SIZE; s++) {
            if (myC[s] != 0) { sFC[s]=myC[s]; sFN[s]=myN[s]; }
        }
    }
    barrier();

    // ── Phase 5: 원자 없는 write-back (thread t → slot t, 경쟁 없음) ─────────
    if (tid < uint(BUCKET_SIZE) && sFW[tid] > 0.0 && sKX[tid] != EMPTY) {
        int gb = int((bucket*uint(BUCKET_SIZE)+tid)*uint(STRIDE));
        ht.raw[gb + F_ACCT] = floatBitsToInt(sFT[tid]);
        ht.raw[gb + F_ACCW] = floatBitsToInt(sFW[tid]);
        ht.raw[gb + F_ACCC] = sFC[tid];
        ht.raw[gb + F_ACCN] = sFN[tid];
    }
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
    mat4  mvp;
    float voxelSize;
    uint  colorMode;
    uint  currentFrame;
    uint  highlightFrames;
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

// ── 큐브 기하 테이블 ─────────────────────────────────────────────────────────
// 8 꼭짓점 (단위 큐브, 중심=원점)
const vec3 kC[8] = vec3[8](
    vec3(-0.5,-0.5,-0.5), vec3( 0.5,-0.5,-0.5),
    vec3( 0.5, 0.5,-0.5), vec3(-0.5, 0.5,-0.5),
    vec3(-0.5,-0.5, 0.5), vec3( 0.5,-0.5, 0.5),
    vec3( 0.5, 0.5, 0.5), vec3(-0.5, 0.5, 0.5)
);

// 6면 × 2삼각형 × 3꼭짓점 = 36 인덱스
// 면 순서: -Z, +Z, -Y, +Y, -X, +X  (각 6개)
const int kI[36] = int[36](
    0,2,1, 0,3,2,   // -Z
    4,5,6, 4,6,7,   // +Z
    0,1,5, 0,5,4,   // -Y
    3,7,6, 3,6,2,   // +Y
    0,4,7, 0,7,3,   // -X
    1,2,6, 1,6,5    // +X
);

// 면 법선 (Lambert 음영용)
const vec3 kN[6] = vec3[6](
    vec3( 0, 0,-1), vec3( 0, 0, 1),
    vec3( 0,-1, 0), vec3( 0, 1, 0),
    vec3(-1, 0, 0), vec3( 1, 0, 0)
);

vec3 unpackNormal(uint packed) {
    float fx = float(packed & 0xFFFFu)/65535.0*2.0-1.0;
    float fy = float((packed>>16u)&0xFFFFu)/65535.0*2.0-1.0;
    vec3 n = vec3(fx, fy, 1.0-abs(fx)-abs(fy));
    if (n.z < 0.0) {
        float ox = n.x>=0.0?(1.0-abs(n.y)):-(1.0-abs(n.y));
        float oy = n.y>=0.0?(1.0-abs(n.x)):-(1.0-abs(n.x));
        n.x=ox; n.y=oy;
    }
    return normalize(n);
}
vec3 tsdfColor(float t) {
    return t < 0.0 ? mix(vec3(0,0,1),vec3(1,1,1),t+1.0)
                   : mix(vec3(1,1,1),vec3(1,0,0),t);
}

void main() {
    int voxelIdx = gl_VertexIndex / 36;
    int triIdx   = gl_VertexIndex % 36;
    int b        = voxelIdx * STRIDE;
    int kx       = ht.raw[b + F_KX];

    if (kx == EMPTY) {
        gl_Position   = vec4(10.0, 10.0, 10.0, 1.0);
        fragColor     = vec3(0.0);
        fragHighlight = 0.0;
        return;
    }

    int ky = ht.raw[b+1];
    int kz = ht.raw[b+2];
    vec3 centre = (vec3(float(kx), float(ky), float(kz)) + 0.5) * pc.voxelSize;

    // ── Integration 플래시 ────────────────────────────────────────────────────
    int   frameTag   = ht.raw[b + F_FTAG];
    float age        = float(int(pc.currentFrame) - frameTag);
    float highlightT = (frameTag >= 0)
        ? max(0.0, 1.0 - age / float(pc.highlightFrames))
        : 0.0;

    // 기본 크기 0.92 (복셀 간 틈), 최근 업데이트 시 1.08로 팽창
    float scale   = mix(0.92, 1.08, highlightT);
    vec3 localPos = kC[kI[triIdx]] * pc.voxelSize * scale;
    gl_Position   = pc.mvp * vec4(centre + localPos, 1.0);

    // ── 기본 색상 ─────────────────────────────────────────────────────────────
    vec3 base;
    if (pc.colorMode == 0u) {
        uint col = uint(ht.raw[b + F_COL]);
        base = vec3(float(col&0xFFu), float((col>>8u)&0xFFu), float((col>>16u)&0xFFu))/255.0;
    } else if (pc.colorMode == 1u) {
        base = unpackNormal(uint(ht.raw[b + F_NORM])) * 0.5 + 0.5;
    } else if (pc.colorMode == 2u) {
        base = tsdfColor(intBitsToFloat(ht.raw[b + F_TSDF]));
    } else if (pc.colorMode == 3u) {
        float w = clamp(intBitsToFloat(ht.raw[b + F_W]) / 50.0, 0.0, 1.0);
        base = vec3(w);
    } else {
        if (frameTag < 0) {
            base = vec3(0.15);
        } else {
            float normAge = clamp(age / float(max(pc.highlightFrames * 4u, 1u)), 0.0, 1.0);
            base = mix(vec3(1.0,0.2,0.0), vec3(0.0,0.3,1.0), normAge);
        }
    }

    // ── Lambert 면 음영 ───────────────────────────────────────────────────────
    vec3  lightDir = normalize(vec3(1.0, 2.0, 1.5));
    float diffuse  = max(dot(kN[triIdx / 6], lightDir), 0.0);
    base *= 0.35 + 0.65 * diffuse;

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
    cpuPts_.clear();
    cpuPts_.reserve(VH_BATCH_SIZE);

    float cx = std::cos(sensorEl_), sx = std::sin(sensorEl_);
    float cy = std::cos(sensorAz_), sy = std::sin(sensorAz_);
    Eigen::Vector3f spos(sensorDist_ * cx * sy, sensorDist_ * sx, sensorDist_ * cx * cy);

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
        cpuPts_.push_back(VH_Point{p.x(), p.y(), p.z(), 1.f});
        written++;
    }
    return written;
}

uint32_t VoxelHashFeature::hashBucket(int32_t kx, int32_t ky, int32_t kz)
{
    uint32_t h = static_cast<uint32_t>(kx) * 73856093u ^ static_cast<uint32_t>(ky) * 19349663u ^ static_cast<uint32_t>(kz) * 83492791u;
    return h % VH_NUM_BUCKETS;
}

uint32_t VoxelHashFeature::packDebugColor(int32_t kx, int32_t ky, int32_t kz)
{
    uint32_t h = static_cast<uint32_t>(kx) * 73856093u ^ static_cast<uint32_t>(ky) * 19349663u ^ static_cast<uint32_t>(kz) * 83492791u;
    float t = static_cast<float>(h & 0xFFFFu) / 65535.0f;
    auto channel = [](float x) -> uint32_t
    {
        float v = std::clamp(std::abs(std::fmod(x, 1.0f) * 6.0f - 3.0f) - 1.0f, 0.0f, 1.0f);
        return static_cast<uint32_t>(v * 255.0f);
    };
    uint32_t r = channel(t + 0.000f);
    uint32_t g = channel(t + 0.333f);
    uint32_t b = channel(t + 0.667f);
    return r | (g << 8u) | (b << 16u) | 0xFF000000u;
}

void VoxelHashFeature::updateRecoveryStorage(const Eigen::Vector3f &sensorPos)
{
    if (cpuPts_.empty())
        return;

    const uint32_t thisFrame = frameIndex_;
    for (const VH_Point &p : cpuPts_)
    {
        Eigen::Vector3f pos(p.x, p.y, p.z);
        Eigen::Vector3f normal = (sensorPos - pos).normalized();
        int32_t kx = static_cast<int32_t>(std::floor(p.x / voxelSize_));
        int32_t ky = static_cast<int32_t>(std::floor(p.y / voxelSize_));
        int32_t kz = static_cast<int32_t>(std::floor(p.z / voxelSize_));
        recoveryRecent_.push_back(VH_RecentSample{pos, normal, packDebugColor(kx, ky, kz), thisFrame});
    }

    if (static_cast<int>(recoveryRecent_.size()) > recoveryMaxRecentPoints_)
    {
        const size_t drop = recoveryRecent_.size() - static_cast<size_t>(recoveryMaxRecentPoints_);
        recoveryRecent_.erase(recoveryRecent_.begin(), recoveryRecent_.begin() + static_cast<std::ptrdiff_t>(drop));
    }

    std::unordered_map<int64_t, size_t> mapIdx;
    std::vector<VH_RecentSample> kept;
    kept.reserve(recoveryRecent_.size());

    auto keyOf = [](int32_t x, int32_t y, int32_t z) -> int64_t
    {
        constexpr int64_t ox = 1ll << 20;
        constexpr int64_t oy = 1ll << 20;
        constexpr int64_t oz = 1ll << 20;
        int64_t ux = static_cast<int64_t>(x) + ox;
        int64_t uy = static_cast<int64_t>(y) + oy;
        int64_t uz = static_cast<int64_t>(z) + oz;
        return (ux << 42) ^ (uy << 21) ^ uz;
    };

    for (size_t i = 0; i < recoveryCompressed_.size(); ++i)
        mapIdx[keyOf(recoveryCompressed_[i].keyX, recoveryCompressed_[i].keyY, recoveryCompressed_[i].keyZ)] = i;

    for (const VH_RecentSample &s : recoveryRecent_)
    {
        if (thisFrame >= s.frame && (thisFrame - s.frame) > static_cast<uint32_t>(recoveryRecentKeepFrames_))
        {
            int32_t kx = static_cast<int32_t>(std::floor(s.pos.x() / voxelSize_));
            int32_t ky = static_cast<int32_t>(std::floor(s.pos.y() / voxelSize_));
            int32_t kz = static_cast<int32_t>(std::floor(s.pos.z() / voxelSize_));
            int64_t key = keyOf(kx, ky, kz);
            auto it = mapIdx.find(key);
            if (it == mapIdx.end())
            {
                VH_CompressedPoint cp;
                cp.centroid = s.pos;
                cp.avgNormal = s.normal;
                cp.confidence = 1.0f;
                cp.lastFrame = s.frame;
                cp.keyX = kx;
                cp.keyY = ky;
                cp.keyZ = kz;
                recoveryCompressed_.push_back(cp);
                mapIdx[key] = recoveryCompressed_.size() - 1;
            }
            else
            {
                VH_CompressedPoint &cp = recoveryCompressed_[it->second];
                const float newW = cp.confidence * recoveryCompressWeightDecay_ + 1.0f;
                cp.centroid = (cp.centroid * (newW - 1.0f) + s.pos) / newW;
                cp.avgNormal = (cp.avgNormal * (newW - 1.0f) + s.normal).normalized();
                cp.confidence = std::min(newW, 512.0f);
                cp.lastFrame = std::max(cp.lastFrame, s.frame);
            }
        }
        else
        {
            kept.push_back(s);
        }
    }

    recoveryRecent_.swap(kept);
}

std::vector<VH_RecentSample> VoxelHashFeature::snapshotRecentSamples() const
{
    return recoveryRecent_;
}

std::vector<VH_CompressedPoint> VoxelHashFeature::snapshotCompressedPoints() const
{
    return recoveryCompressed_;
}

void VoxelHashFeature::buildSpatialHashRanges()
{
    struct BucketRun
    {
        uint32_t bucket = 0;
        uint32_t begin = 0;
        uint32_t end = 0;
        uint32_t count = 0;
        uint32_t kept = 0;
    };

    auto t0 = std::chrono::steady_clock::now();

    // 1) Prepare working arrays and per-frame stats.
    const uint32_t n = static_cast<uint32_t>(cpuPts_.size());
    sortedIndex_.resize(n);
    std::iota(sortedIndex_.begin(), sortedIndex_.end(), 0u);
    cellStart_.assign(VH_NUM_BUCKETS, 0u);
    cellEnd_.assign(VH_NUM_BUCKETS, 0u);
    droppedPoints_ = 0;
    maxPtsInBucket_ = 0;
    avgPtsInActiveBucket_ = 0.f;

    // Map original point index -> hash bucket.
    auto bucketOf = [&](uint32_t idx) -> uint32_t
    {
        const VH_Point &p = cpuPts_[idx];
        int32_t kx = static_cast<int32_t>(std::floor(p.x / voxelSize_));
        int32_t ky = static_cast<int32_t>(std::floor(p.y / voxelSize_));
        int32_t kz = static_cast<int32_t>(std::floor(p.z / voxelSize_));
        return hashBucket(kx, ky, kz);
    };

    // 2) Reorder point indices so points in the same bucket are contiguous.
    std::sort(
        sortedIndex_.begin(),
        sortedIndex_.end(),
        [&](uint32_t a, uint32_t b)
        {
            return bucketOf(a) < bucketOf(b);
        });

    // Helper that extracts a contiguous [begin, end) run sharing one bucket.
    auto makeNextRun = [&](uint32_t &cursor) -> BucketRun
    {
        BucketRun run{};
        run.bucket = bucketOf(sortedIndex_[cursor]);
        run.begin = cursor;
        while (cursor < n && bucketOf(sortedIndex_[cursor]) == run.bucket)
            ++cursor;
        run.end = cursor;
        run.count = run.end - run.begin;
        run.kept = run.count;
        if (limitHeavyBuckets_ && maxPtsPerBucket_ > 0)
            run.kept = std::min<uint32_t>(run.count, static_cast<uint32_t>(maxPtsPerBucket_));
        return run;
    };

    // Generic host-visible upload helper.
    auto uploadBytes = [&](VkDeviceMemory mem, const void *src, size_t bytes)
    {
        void *mapped = nullptr;
        vkMapMemory(ctx_.device, mem, 0, VK_WHOLE_SIZE, 0, &mapped);
        if (bytes > 0)
            std::memcpy(mapped, src, bytes);
        vkUnmapMemory(ctx_.device, mem);
    };

    // 3) Build packed point array and bucket ranges [cellStart, cellEnd).
    std::vector<VH_Point> sortedPts;
    sortedPts.reserve(n);

    uint32_t cursor = 0;
    uint32_t activeBuckets = 0;
    while (cursor < n)
    {
        const BucketRun run = makeNextRun(cursor);

        droppedPoints_ += (run.count - run.kept);
        maxPtsInBucket_ = std::max(maxPtsInBucket_, run.kept);
        if (run.kept > 0)
            ++activeBuckets;

        cellStart_[run.bucket] = static_cast<uint32_t>(sortedPts.size());
        for (uint32_t i = 0; i < run.kept; ++i)
            sortedPts.push_back(cpuPts_[sortedIndex_[run.begin + i]]);
        cellEnd_[run.bucket] = static_cast<uint32_t>(sortedPts.size());
    }

    if (activeBuckets > 0)
        avgPtsInActiveBucket_ = static_cast<float>(sortedPts.size()) / static_cast<float>(activeBuckets);

    // 4) Upload packed points and bucket ranges to GPU buffers.
    ptCount_ = static_cast<uint32_t>(sortedPts.size());
    {
        void *mapped = nullptr;
        vkMapMemory(ctx_.device, ptMem_, 0, VK_WHOLE_SIZE, 0, &mapped);
        std::memset(mapped, 0, sizeof(float) * 4 * VH_BATCH_SIZE);
        if (!sortedPts.empty())
            std::memcpy(mapped, sortedPts.data(), sizeof(VH_Point) * sortedPts.size());
        vkUnmapMemory(ctx_.device, ptMem_);
    }
    uploadBytes(cellStartMem_, cellStart_.data(), sizeof(uint32_t) * cellStart_.size());
    uploadBytes(cellEndMem_, cellEnd_.data(), sizeof(uint32_t) * cellEnd_.size());

    auto t1 = std::chrono::steady_clock::now();
    sortMs_ = std::chrono::duration<float, std::milli>(t1 - t0).count();
}

void VoxelHashFeature::readPerfQueries()
{
    if (perfQueryPool_ == VK_NULL_HANDLE || !hasPerfQueries_)
        return;

    std::array<uint64_t, 6> t{};
    VkResult res = vkGetQueryPoolResults(
        ctx_.device, perfQueryPool_, 0, static_cast<uint32_t>(t.size()),
        sizeof(t), t.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (res != VK_SUCCESS)
        return;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx_.physicalDevice, &props);
    const float nsToMs = props.limits.timestampPeriod * 1e-6f;
    gatherMs_ = static_cast<float>(t[1] - t[0]) * nsToMs;
    finalizeMs_ = static_cast<float>(t[3] - t[2]) * nsToMs;
    countMs_ = static_cast<float>(t[5] - t[4]) * nsToMs;
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

    // 공간 인덱스: CPU 정렬 후 버킷별 범위 [cellStart, cellEnd) 업로드
    createBuf(sizeof(uint32_t) * VH_NUM_BUCKETS,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              cellStartBuf_, cellStartMem_);

    createBuf(sizeof(uint32_t) * VH_NUM_BUCKETS,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              cellEndBuf_, cellEndMem_);
}

// ─────────────────────────────────────────────────────────────────────────────
//  createDescriptors  — binding 0: HT, 1: Pts, 2: Ctr
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::createDescriptors()
{
    VkDescriptorSetLayoutBinding b[5]{};
    for (int i = 0; i < 5; i++)
    {
        b[i].binding = static_cast<uint32_t>(i);
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 5;
    lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(ctx_.device, &lci, nullptr, &descLayout_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: descriptor layout");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
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
    VkDescriptorBufferInfo csI{cellStartBuf_, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo ceI{cellEndBuf_, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[5]{};
    VkDescriptorBufferInfo *infos[5] = {&htI, &ptI, &ctrI, &csI, &ceI};
    for (int i = 0; i < 5; i++)
    {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = descSet_;
        w[i].dstBinding = static_cast<uint32_t>(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = infos[i];
    }
    vkUpdateDescriptorSets(ctx_.device, 5, w, 0, nullptr);
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
    gatherPipe_ = make(kVH_GatherComp);
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
                                              nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
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
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
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

void VoxelHashFeature::dispatchGather(VkCommandBuffer cmd)
{
    if (ptCount_ == 0)
        return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gatherPipe_);
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
    vkCmdDispatch(cmd, VH_NUM_BUCKETS, 1, 1);
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
    cpuPts_.reserve(VH_BATCH_SIZE);
    sortedIndex_.reserve(VH_BATCH_SIZE);
    cellStart_.assign(VH_NUM_BUCKETS, 0u);
    cellEnd_.assign(VH_NUM_BUCKETS, 0u);
    createBuffers();
    createDescriptors();
    createComputePipelines();
    createRenderPipeline();

    VkQueryPoolCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount = 6;
    if (vkCreateQueryPool(ctx_.device, &qci, nullptr, &perfQueryPool_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: perf query pool");

    lastTime_ = std::chrono::steady_clock::now();
    doClear_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  onCompute
//  순서: [Clear?] → Gather(P2G) → barrier → Finalize → barrier → Count
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
    readPerfQueries();

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
        recoveryRecent_.clear();
        recoveryCompressed_.clear();
    }

    if (streaming_)
    {
        ptCount_ = genSphereBatch();
        float cx = std::cos(sensorEl_), sx = std::sin(sensorEl_);
        float cy = std::cos(sensorAz_), sy = std::sin(sensorAz_);
        Eigen::Vector3f sensorPos(sensorDist_ * cx * sy, sensorDist_ * sx, sensorDist_ * cx * cy);
        updateRecoveryStorage(sensorPos);
        buildSpatialHashRanges();
        doInsert_ = true;
        sensorAz_ += sensorSpeed_;
        if (sensorAz_ > 2.f * static_cast<float>(M_PI))
            sensorAz_ -= 2.f * static_cast<float>(M_PI);
    }

    if (doInsert_ && ptCount_ > 0)
    {
        vkCmdResetQueryPool(cmd, perfQueryPool_, 0, 6);

        // ① Gather(P2G): 점 → 버킷 단위 협력 누산 (원자 없는 write-back)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 0);
        dispatchGather(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 1);
        bufBarrier(cmd, htBuf_,
                   VK_ACCESS_SHADER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // ② Finalize: running-average 커밋 + frame_tag 기록
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 2);
        dispatchFinalize(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 3);
        bufBarrier(cmd, htBuf_,
                   VK_ACCESS_SHADER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);

        // ③ Count
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 4);
        dispatchCount(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 5);
        hasPerfQueries_ = true;

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
    vkCmdDraw(ctx.commandBuffer, VH_TOTAL_ENTRIES * 36, 1, 0, 0);

    // 입력 포인트 클라우드 오버레이 (시안색, 현재 배치만)
    if (showInputPts_ && ptCount_ > 0)
    {
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
    ImGui::Text("P2G gather  : %.3f ms  | finalize: %.3f ms | count: %.3f ms",
                gatherMs_, finalizeMs_, countMs_);
    ImGui::Text("Hash sort   : %.3f ms  | dropped: %u", sortMs_, droppedPoints_);
    ImGui::Text("Bucket load : max %u  avg(active) %.1f", maxPtsInBucket_, avgPtsInActiveBucket_);
    ImGui::InputFloat("Baseline P2G (ms)", &baselineP2GMs_, 1.f, 5.f, "%.3f");
    if (baselineP2GMs_ > 0.0f)
    {
        float improv = (baselineP2GMs_ - gatherMs_) / baselineP2GMs_ * 100.0f;
        ImGui::Text("Gather vs baseline: %+0.1f%%", improv);
    }
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
    ImGui::RadioButton("Surface color", &colorMode_, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Normal", &colorMode_, 1);
    ImGui::SameLine();
    ImGui::RadioButton("TSDF", &colorMode_, 2);
    ImGui::SameLine();
    ImGui::RadioButton("Weight", &colorMode_, 3);
    ImGui::SameLine();
    ImGui::RadioButton("Recency", &colorMode_, 4);
    if (colorMode_ == 4)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.0f, 1), "■");
        ImGui::SameLine();
        ImGui::Text("= recent");
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.0f, 0.3f, 1.0f, 1), "■");
        ImGui::SameLine();
        ImGui::Text("= old");
    }

    // ── Integration 시각화 ────────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Text("Integration visualization");
    ImGui::SliderInt("Highlight frames", &highlightFrames_, 1, 60,
                     "flash for %d frames");
    ImGui::TextColored(ImVec4(1, 0.9f, 0.1f, 1), "■");
    ImGui::SameLine();
    ImGui::Text("= voxel updated this batch (size ∝ recency)");
    ImGui::Checkbox("Show input points (cyan)", &showInputPts_);
    if (showInputPts_)
    {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1), "●");
        ImGui::SameLine();
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
    ImGui::Checkbox("Limit heavy buckets", &limitHeavyBuckets_);
    ImGui::SliderInt("Max pts/bucket", &maxPtsPerBucket_, 32, 2048);
    ImGui::SliderInt("Recent keep frames", &recoveryRecentKeepFrames_, 1, 120);
    ImGui::SliderInt("Max recent points", &recoveryMaxRecentPoints_, 10000, 1000000);
    ImGui::SliderFloat("Compress decay", &recoveryCompressWeightDecay_, 0.90f, 1.0f, "%.3f");
    ImGui::Text("Recovered recent/compressed: %zu / %zu",
                recoveryRecent_.size(), recoveryCompressed_.size());
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
    vkDestroyPipeline(ctx_.device, gatherPipe_, nullptr);
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
    vkDestroyBuffer(ctx_.device, cellStartBuf_, nullptr);
    vkFreeMemory(ctx_.device, cellStartMem_, nullptr);
    vkDestroyBuffer(ctx_.device, cellEndBuf_, nullptr);
    vkFreeMemory(ctx_.device, cellEndMem_, nullptr);
    vkDestroyQueryPool(ctx_.device, perfQueryPool_, nullptr);
}
