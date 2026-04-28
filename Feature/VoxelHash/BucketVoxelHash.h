#pragma once

#include "VoxelHashBase.h"
#include "VoxelHashTypes.h"

#include "../IFeature.h"
#include "Buffer.h"
#include "ComputeShader.h"
#include "RenderingShader.h"

struct VHB_RenderPC
{
    float mvp[16];
    float voxelSize;
    float _pad[3];
};

class BucketVoxelHash
{
public:
    void Initialize(VulkanContext &context);

    void Clear(VulkanContext &context);

    void Integrate(VulkanContext &context, const VH_InputPoint *pts, uint32_t count,
                   float sensorX = 0.f, float sensorY = 0.f, float sensorZ = 0.f);

    void RenderTSDF();

    void RenderColor(VkCommandBuffer cmd, const float *mvp, bool useSlotPoint);

    void RenderVoxel(VkCommandBuffer cmd, const float *mvp);

    void RenderVoxelSlot(VkCommandBuffer cmd, const float *mvp);

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

    // ── Compute Shaders ───────────────────────────────────────────────────────
    ComputeShader kClearShader;
    ComputeShader kHistogramShader;
    ComputeShader kPrefixScanShader;
    ComputeShader kReallocateShader;
    ComputeShader kUpdateTSDFShader;
    ComputeShader kUpdateHashTableShader;
    ComputeShader kCountShader;

    float voxelSize_ = 0.02f;
    float truncation_ = VH_DEFAULT_TRUNC;
    float maxWeight_ = VH_DEFAULT_MAXW;
    uint32_t frameIndex_ = 0;
    uint32_t lastIntegratedCount_ = 0;
};
