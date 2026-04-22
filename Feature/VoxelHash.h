#pragma once

#include "IFeature.h"
#include "../Utilities.h"
#include <vulkan/vulkan.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <chrono>
#include <array>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
//  설계 원칙 (vs TSDF / BucketedHash)
//
//  TSDF(OpenVDB) 한계
//    · Dense grid → 빈 공간도 메모리 점유
//    · CPU 기반 → 매 프레임 CPU-GPU 왕복
//    · 값 + 가중치만 저장, 색상/법선 없음
//
//  BucketedHash 한계
//    · 점유 여부만 저장, 기하 정보 없음
//    · Scatter 방식 → Atomic 경합
//
//  VoxelHash 개선점
//  ① [빠른 Integration] 논문 Gather 패러다임
//     Insert Phase  : CAS 슬롯 점유 + float-CAS 배치 누산 (acc_tsdf, acc_w)
//     Finalize Phase: 슬롯 소유 스레드 단독 running-average 커밋
//                     → 슬롯 간 쓰기 충돌 구조적 제거
//  ② [고정밀 Geometry 보존]
//     슬롯당 48 bytes: TSDF + weight + oct-encoded normal + RGBA color
//     가중 누진 평균: tsdf_new = (tsdf_old·w + acc_tsdf) / (w + acc_w)
//     Oct-encoding: 법선을 2×uint16 → 32 bit 고정밀 압축
//     maxWeight 클램핑: 최신 관측 비중 유지
//  ③ [시각적 디버깅] frame_tag 필드
//     Finalize 시 현재 프레임 번호 기록
//     Render shader: 최근 N 프레임 내 갱신된 Voxel → 황색 플래시
// ─────────────────────────────────────────────────────────────────────────────

static constexpr uint32_t VH_BUCKET_SIZE = 4;
static constexpr uint32_t VH_NUM_BUCKETS = 1u << 17;
static constexpr uint32_t VH_TOTAL_ENTRIES = VH_NUM_BUCKETS * VH_BUCKET_SIZE;
static constexpr int32_t VH_EMPTY_KEY = 0x7FFFFFFF;
static constexpr uint32_t VH_BATCH_SIZE = 10000;
static constexpr uint32_t VH_ENTRY_INTS = 12; // 12 × int32 = 48 bytes

// ── GPU 엔트리 레이아웃 (std430, 48 bytes) ─────────────────────────────────────
//  Idx  Field       Type        Description
//   0   key_x       int         voxel X  (VH_EMPTY_KEY = 비어있음)
//   1   key_y       int         voxel Y
//   2   key_z       int         voxel Z
//   3   tsdf        float       확정 TSDF [-1, 1]
//   4   weight      float       누적 가중치
//   5   color_rgba  uint        확정 RGBA (r8g8b8a8)
//   6   normal_oct  uint        oct-encoded 법선 (2×uint16)
//   7   acc_tsdf    int→float   CAS 누산: Σ(tsdf_i × w_i)
//   8   acc_w       int→float   CAS 누산: Σ(w_i)
//   9   acc_color   int→uint    최신 색상 (last-write-wins)
//  10   acc_norm    int→uint    최신 법선 (last-write-wins)
//  11   frame_tag   int         마지막 Finalize 프레임 (디버그 플래시)
struct alignas(4) VH_Entry
{
    int32_t key_x = VH_EMPTY_KEY;
    int32_t key_y = 0;
    int32_t key_z = 0;
    float tsdf = 0.f;
    float weight = 0.f;
    uint32_t color_rgba = 0;
    uint32_t normal_oct = 0;
    int32_t acc_tsdf = 0;
    int32_t acc_w = 0;
    int32_t acc_color = 0;
    int32_t acc_norm = 0;
    int32_t frame_tag = -1;
};
static_assert(sizeof(VH_Entry) == VH_ENTRY_INTS * 4, "VH_Entry layout mismatch");

struct VH_InsertPC
{ // 32 bytes
    float sensorX, sensorY, sensorZ;
    float voxelSize;
    uint32_t numPoints;
    uint32_t numBuckets;
    float truncation;
    float maxWeight;
};
static_assert(sizeof(VH_InsertPC) <= 128);

struct VH_FinalizePC
{ // 16 bytes — finalize / count 공유
    uint32_t totalEntries;
    uint32_t currentFrame;
    float _pad0, _pad1;
};
static_assert(sizeof(VH_FinalizePC) <= 128);

struct VH_RenderPC
{ // 80 bytes
    float mvp[16];
    float voxelSize;
    uint32_t colorMode; // 0=color 1=normal 2=TSDF 3=weight
    uint32_t currentFrame;
    uint32_t highlightFrames; // 플래시 지속 프레임 수
};
static_assert(sizeof(VH_RenderPC) <= 128);

struct VH_Point
{
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
    float w = 1.f;
};

struct VH_RecentSample
{
    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Vector3f normal = Eigen::Vector3f::UnitY();
    uint32_t color = 0xFFFFFFFFu;
    uint32_t frame = 0u;
};

struct VH_CompressedPoint
{
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f avgNormal = Eigen::Vector3f::UnitY();
    float confidence = 0.f;
    uint32_t lastFrame = 0u;
    int32_t keyX = 0;
    int32_t keyY = 0;
    int32_t keyZ = 0;
};

class VoxelHashFeature : public IFeature
{
public:
    const char *name() const override { return "VoxelHash (TSDF+Normal+Color)"; }

    void onInit(const VulkanContext &ctx) override;
    void onCompute(VkCommandBuffer cmd) override;
    void onRender(const RenderContext &ctx) override;
    void onImGui() override;
    void onCleanup() override;

    std::vector<VH_RecentSample> snapshotRecentSamples() const;
    std::vector<VH_CompressedPoint> snapshotCompressedPoints() const;
    uint32_t snapshotFrameIndex() const { return frameIndex_; }

private:
    VulkanContext ctx_{};

    VkDescriptorSetLayout descLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;

    VkPipelineLayout compLayout_ = VK_NULL_HANDLE;
    VkPipeline clearPipe_ = VK_NULL_HANDLE;
    VkPipeline gatherPipe_ = VK_NULL_HANDLE; // replaces insertPipe_ (no float-CAS)
    VkPipeline finalizePipe_ = VK_NULL_HANDLE;
    VkPipeline countPipe_ = VK_NULL_HANDLE;

    VkPipelineLayout renderLayout_ = VK_NULL_HANDLE;
    VkPipeline renderPipe_ = VK_NULL_HANDLE;
    VkPipeline ptRenderPipe_ = VK_NULL_HANDLE; // 입력 포인트 오버레이

    VkBuffer htBuf_ = VK_NULL_HANDLE; // device-local ≈24 MB
    VkDeviceMemory htMem_ = VK_NULL_HANDLE;
    VkBuffer ptBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory ptMem_ = VK_NULL_HANDLE;
    uint32_t ptCount_ = 0;
    VkBuffer ctrBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory ctrMem_ = VK_NULL_HANDLE;
    VkBuffer cellStartBuf_ = VK_NULL_HANDLE; // sorted cell index: start
    VkDeviceMemory cellStartMem_ = VK_NULL_HANDLE;
    VkBuffer cellEndBuf_ = VK_NULL_HANDLE; // sorted cell index: end
    VkDeviceMemory cellEndMem_ = VK_NULL_HANDLE;
    VkQueryPool perfQueryPool_ = VK_NULL_HANDLE;

    bool doClear_ = true;
    bool doInsert_ = false;
    bool doFinalize_ = false;
    bool doCountRead_ = false;
    uint32_t frameIndex_ = 0;

    bool showInputPts_ = true; // 입력 포인트 클라우드 오버레이

    bool streaming_ = false;
    float sphereRadius_ = 1.0f;
    float sensorDist_ = 3.0f;
    float noiseStddev_ = 0.003f;
    float sensorAz_ = 0.0f;
    float sensorEl_ = 0.3f;
    float sensorSpeed_ = 0.015f;

    float azimuth_ = 0.5f;
    float elevation_ = 0.4f;
    float camDist_ = 4.0f;

    float voxelSize_ = 0.02f;
    float truncation_ = 0.05f;
    float maxWeight_ = 50.f;
    int colorMode_ = 0;
    int highlightFrames_ = 8;

    int occupancy_ = 0;
    int64_t totalInserted_ = 0;
    float ptsPerSec_ = 0.f;
    std::chrono::steady_clock::time_point lastTime_;
    float gatherMs_ = 0.f;
    float finalizeMs_ = 0.f;
    float countMs_ = 0.f;
    float sortMs_ = 0.f;
    uint32_t droppedPoints_ = 0;
    uint32_t maxPtsInBucket_ = 0;
    float avgPtsInActiveBucket_ = 0.f;
    int maxPtsPerBucket_ = 512;
    bool limitHeavyBuckets_ = true;
    float baselineP2GMs_ = 0.f;
    bool hasPerfQueries_ = false;
    std::vector<VH_Point> cpuPts_;
    std::vector<uint32_t> sortedIndex_;
    std::vector<uint32_t> cellStart_;
    std::vector<uint32_t> cellEnd_;
    std::vector<VH_RecentSample> recoveryRecent_;
    std::vector<VH_CompressedPoint> recoveryCompressed_;
    int recoveryRecentKeepFrames_ = 12;
    int recoveryMaxRecentPoints_ = 300000;
    float recoveryCompressWeightDecay_ = 0.995f;

    Eigen::Matrix4f computeMVP() const;
    uint32_t genSphereBatch();
    void buildSpatialHashRanges();
    static uint32_t hashBucket(int32_t kx, int32_t ky, int32_t kz);
    void readPerfQueries();
    void updateRecoveryStorage(const Eigen::Vector3f &sensorPos);
    static uint32_t packDebugColor(int32_t kx, int32_t ky, int32_t kz);

    void createBuffers();
    void createDescriptors();
    void createComputePipelines();
    void createRenderPipeline();

    void dispatchClear(VkCommandBuffer cmd);
    void dispatchGather(VkCommandBuffer cmd);
    void dispatchFinalize(VkCommandBuffer cmd);
    void dispatchCount(VkCommandBuffer cmd);

    void bufBarrier(VkCommandBuffer cmd, VkBuffer buf,
                    VkAccessFlags src, VkAccessFlags dst,
                    VkPipelineStageFlags srcStage,
                    VkPipelineStageFlags dstStage);
    void createBuf(VkDeviceSize size, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags props,
                   VkBuffer &buf, VkDeviceMemory &mem);
};
