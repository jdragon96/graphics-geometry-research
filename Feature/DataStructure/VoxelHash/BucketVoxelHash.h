#pragma once

#include "VoxelHashBase.h"
#include "VoxelHashTypes.h"

#include "../IFeature.h"
#include "../Common/Buffer.h"
#include "../Common/ComputeShader.h"
#include "../Common/RenderingShader.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

// queryTSDF 반환 타입
struct TSDFQueryResult
{
    float           tsdf;    // 정규화 거리 [-1, 1]  (실제 거리 = tsdf × truncation)
    Eigen::Vector3f normal;  // octahedral 디코딩된 단위 법선
    float           weight;  // 누적 관측 가중치
    bool            valid;   // true = 복셀 존재 & weight > 0
};

struct VHB_RenderPC
{
    float mvp[16];
    float voxelSize;
    float truncation;  // TSDF 역정규화에 사용 (실제 거리 = tsdf * truncation)
    float projectSurface;
    float _pad;
};

class BucketVoxelHash
{
public:
    void Initialize(VulkanContext &context);

    void Clear(VulkanContext &context);

    void Integrate(VulkanContext &context, const VH_InputPoint *pts, uint32_t count,
                   float sensorX = 0.f, float sensorY = 0.f, float sensorZ = 0.f);

    void RenderColor(VkCommandBuffer cmd, const float *mvp, bool useSlotPoint);

    void RenderVoxel(VkCommandBuffer cmd, const float *mvp);

    void RenderVoxelSlot(VkCommandBuffer cmd, const float *mvp);

    void RenderTSDF(VkCommandBuffer cmd, const float *mvp);

    void setVoxelSize(float v) { voxelSize_ = v; }

    void setTSDFProjectToSurface(bool v) { tsdfProjectToSurface_ = v; }

    // ── FastLIO2 지원 API ──────────────────────────────────────────────────────

    // CPU O(1): 월드 좌표 → 복셀 해시 조회 → TSDF/법선/가중치 반환
    // kHashTableBuffer의 persistent map 포인터를 직접 읽음 (vkMapMemory 오버헤드 없음)
    TSDFQueryResult queryTSDF(float wx, float wy, float wz) const;

    // GPU: box 범위 내 복셀을 EMPTY 상태로 초기화 (맵 윈도잉용)
    void deleteBox(const Eigen::AlignedBox3f& box, VulkanContext& ctx);

    float getTruncation() const { return truncation_; }

private:
    void ResetHashTable(VulkanContext &context);

    void UpdatePosition(VulkanContext &context, const VH_InputPoint *pts, uint32_t count);
    void UpdateNormal(VulkanContext &context, const VH_InputPoint *pts, uint32_t count);
    void UpdateColor(VulkanContext &context, const VH_InputPoint *pts, uint32_t count);

    void CreateBuffers(VulkanContext &context);
    void CreateClearShader(VulkanContext &context);
    void CreateHistogramShader(VulkanContext &context);
    void CreatePrefixScanShader(VulkanContext &context);
    void CreateReallocateShader(VulkanContext &context);
    void CreateUpdateTSDFShader(VulkanContext &context);
    void CreateUpdateHashTableShader(VulkanContext &context);
    void CreateCountShader(VulkanContext &context);
    void CreateColorRenderPipeline(VulkanContext &context);
    void CreatePointRenderPipeline(VulkanContext &context);
    void CreateVoxelRenderPipeline(VulkanContext &context);
    void CreateTSDFRenderPipeline(VulkanContext &context);
    void CreateDeleteBoxShader(VulkanContext &context);

    // queryTSDF 헬퍼
    static Eigen::Vector3f decodeOctNormal(uint32_t packed);
    static uint32_t        hashBucket(int32_t kx, int32_t ky, int32_t kz);

private:
    // ── 입력 버퍼 (CPU → GPU, host-visible) ──────────────────────────────────
    Buffer kHashTableBuffer;      // binding 0: HT raw int[]
    Buffer kPositionBuffer;       // binding 1: unsorted positions vec4[]
    Buffer kNormalBuffer;         // binding 2: unsorted normals vec4[]
    Buffer kColorBuffer;          // binding 3: unsorted colors uint[]
    Buffer kPatchCountBuffer;     // binding 4: occupancy counter (TRANSFER_DST)
    Buffer kHistorgramBuffer;     // binding 8: histogram / scatter offset counters
    Buffer kBlockSummationBuffer; // binding 9: prefix scan block sums

    // ── 정렬 인덱스 버퍼 (device-local) ─────────────────────────────────────
    Buffer kSortedPositionBuffer; // binding 5: scatter output (sorted positions)
    Buffer kSortedNormalBuffer;   // binding 6: scatter output (sorted normals)
    Buffer kSortedColorBuffer;    // binding 7: scatter output (sorted colors)

    // ── 버킷 범위 버퍼 ────────────────────────────────────────────────────────
    Buffer kCellStartBuffer; // binding 10: exclusive prefix sum (bucket start offset)

    // ── Render pipelines ─────────────────────────────────────────────────────
    RenderingShader kColorRenderShader_;
    RenderingShader kPointRenderShader_;
    RenderingShader kVoxelRenderShader_;
    RenderingShader kTSDFRenderShader_;

    // ── Compute Shaders ───────────────────────────────────────────────────────
    ComputeShader kClearShader;
    ComputeShader kHistogramShader;
    ComputeShader kPrefixScanShader;
    ComputeShader kReallocateShader;
    ComputeShader kUpdateTSDFShader;
    ComputeShader kUpdateHashTableShader;
    ComputeShader kCountShader;
    ComputeShader kDeleteBoxShader;

    // deleteBox push constant
    struct VH_DeleteBoxPC {
        int32_t minKx, minKy, minKz, _pad0;
        int32_t maxKx, maxKy, maxKz, _pad1;
    };
    static_assert(sizeof(VH_DeleteBoxPC) <= 128);

    float voxelSize_ = 0.02f;
    float truncation_ = VH_DEFAULT_TRUNC;
    float maxWeight_ = VH_DEFAULT_MAXW;
    bool tsdfProjectToSurface_ = false;
    uint32_t frameIndex_ = 0;
    uint32_t lastIntegratedCount_ = 0;
};
