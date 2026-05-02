#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
//  VoxelHashTypes.h — 모든 GPU/CPU 공유 상수, 구조체, push constant 정의
//
//  설계 원칙
//    · GPU std430 레이아웃과 CPU struct가 1:1 대응 (zero-copy 업로드)
//    · 입력 포인트: position + normal + color (VH_InputPoint, 32 bytes)
//    · 해시 슬롯:   TSDF + weight + color + normal + acc 필드 (VH_Entry, 48 bytes)
//    · 정밀도:      0.05 mm 기준, 해시 테이블 192 MB (1M buckets × 4 slots)
// ─────────────────────────────────────────────────────────────────────────────

// ── 해시 테이블 규모 ──────────────────────────────────────────────────────────
static constexpr uint32_t VH_BUCKET_SIZE      = 4;
static constexpr uint32_t VH_NUM_BUCKETS      = 1u << 20;   // 1,048,576  (192 MB HT)
static constexpr uint32_t VH_TOTAL_ENTRIES    = VH_NUM_BUCKETS * VH_BUCKET_SIZE;
static constexpr int32_t  VH_EMPTY_KEY        = 0x7FFFFFFF;

// ── 배치 크기 ────────────────────────────────────────────────────────────────
static constexpr uint32_t VH_BATCH_SIZE       = 10000;

// ── GPU sort 블록 크기 ───────────────────────────────────────────────────────
static constexpr uint32_t VH_SORT_BLOCK_SIZE  = 1024;
static constexpr uint32_t VH_SCAN_BLOCKS      = VH_NUM_BUCKETS / VH_SORT_BLOCK_SIZE;  // 1024

// ── 기본 파라미터 (0.05 mm 정밀도 기준) ─────────────────────────────────────
static constexpr float    VH_DEFAULT_VOXEL    = 0.00005f;   // 0.05 mm
static constexpr float    VH_DEFAULT_TRUNC    = 0.00030f;   // 0.30 mm (voxel × 6)
static constexpr float    VH_DEFAULT_MAXW     = 100.f;

// ── 해시 슬롯 int 필드 인덱스 (std430, 12 × int = 48 bytes) ─────────────────
static constexpr uint32_t VH_ENTRY_INTS       = 12;

// ─────────────────────────────────────────────────────────────────────────────
//  입력 포인트 구조체 (CPU → GPU 업로드 단위)
//
//  GPU std430 vec4 배열 매핑 (binding=1, vec4 d[]):
//    d[i*2+0] = { px, py, pz, col_as_float }   position + packed color
//    d[i*2+1] = { nx, ny, nz, 0.0 }            normal   + padding
// ─────────────────────────────────────────────────────────────────────────────
struct alignas(16) VH_InputPoint {
    float    px, py, pz;   // 위치        (12 bytes)
    uint32_t col;          // r8g8b8a8     (4 bytes)
    float    nx, ny, nz;   // 단위 법선    (12 bytes)
    float    _pad;         // 패딩         (4 bytes) → 총 32 bytes
};
static_assert(sizeof(VH_InputPoint) == 32, "VH_InputPoint size mismatch");

// ─────────────────────────────────────────────────────────────────────────────
//  해시 테이블 슬롯 (GPU device-local, std430 48 bytes)
//
//  int 인덱스 레이아웃:
//    0  key_x   1  key_y   2  key_z
//    3  tsdf    4  weight  5  color_rgba  6  normal_oct
//    7  acc_tsdf (float-as-int)   8  acc_w
//    9  acc_color                10  acc_norm
//   11  frame_tag
// ─────────────────────────────────────────────────────────────────────────────
struct alignas(4) VH_Entry {
    int32_t  key_x      = VH_EMPTY_KEY;
    int32_t  key_y      = 0;
    int32_t  key_z      = 0;
    float    tsdf       = 0.f;
    float    weight     = 0.f;
    uint32_t color_rgba = 0;
    uint32_t normal_oct = 0;
    int32_t  acc_tsdf   = 0;
    int32_t  acc_w      = 0;
    int32_t  acc_color  = 0;
    int32_t  acc_norm   = 0;
    int32_t  frame_tag  = -1;
};
static_assert(sizeof(VH_Entry) == VH_ENTRY_INTS * 4, "VH_Entry size mismatch");

// ─────────────────────────────────────────────────────────────────────────────
//  Push Constant 구조체 (모두 ≤ 128 bytes)
// ─────────────────────────────────────────────────────────────────────────────

// histogram / scatter / gather 공유
struct VH_SortPC {
    uint32_t numPoints;
    uint32_t numBuckets;
    float    voxelSize;
    float    _pad;
};  // 16 bytes
static_assert(sizeof(VH_SortPC) <= 128);

// gather 전용 (센서 위치 포함)
struct VH_GatherPC {
    float    sensorX, sensorY, sensorZ;
    float    voxelSize;
    uint32_t numPoints;
    uint32_t numBuckets;
    float    truncation;
    float    maxWeight;
};  // 32 bytes
static_assert(sizeof(VH_GatherPC) <= 128);

// prefix scan 전용
struct VH_ScanPC {
    uint32_t numBuckets;
    uint32_t pass;     // 0=로컬 스캔, 1=블록합 스캔, 2=전파
    uint32_t _p0, _p1;
};  // 16 bytes
static_assert(sizeof(VH_ScanPC) <= 128);

// finalize / count / clear 공유
struct VH_FinalizePC {
    uint32_t totalEntries;
    uint32_t currentFrame;
    float    _p0, _p1;
};  // 16 bytes
static_assert(sizeof(VH_FinalizePC) <= 128);

// 렌더링
struct VH_RenderPC {
    float    mvp[16];
    float    voxelSize;
    uint32_t colorMode;        // 0=color 1=normal 2=TSDF 3=weight
    uint32_t currentFrame;
    uint32_t highlightFrames;
};  // 80 bytes
static_assert(sizeof(VH_RenderPC) <= 128);
