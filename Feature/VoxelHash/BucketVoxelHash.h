#pragma once

#include "VoxelHashBase.h"
#include "VoxelHashTypes.h"

#include "../IFeature.h"
#include "Buffer.h"
#include "ComputeShader.h"

class BucketVoxelHash
{
public:
    void Initialize(VulkanContext &context);

    void Clear(VulkanContext &context);

    // GPU 전체 파이프라인: sort(histogram→prefix_scan→scatter) → gather → finalize → count
    void Integrate(VulkanContext &context, const VH_InputPoint *pts, uint32_t count,
                   float sensorX = 0.f, float sensorY = 0.f, float sensorZ = 0.f);

private:
    void ResetHashTable(VulkanContext &context);

    void UpdatePosition(VulkanContext &context, const VH_InputPoint *pts, uint32_t count);
    void UpdateNormal  (VulkanContext &context, const VH_InputPoint *pts, uint32_t count);
    void UpdateColor   (VulkanContext &context, const VH_InputPoint *pts, uint32_t count);

    void CreateBuffers       (VulkanContext &context);
    void CreateClearShader   (VulkanContext &context);
    void CreateHistogramShader(VulkanContext &context);
    void CreatePrefixScanShader(VulkanContext &context);
    void CreateScatterShader (VulkanContext &context);
    void CreateGatherShader  (VulkanContext &context);
    void CreateFinalizeShader(VulkanContext &context);
    void CreateCountShader   (VulkanContext &context);

    void BufBarrier(VkCommandBuffer cmd, VkBuffer buf,
                    VkAccessFlags src, VkAccessFlags dst,
                    VkPipelineStageFlags srcS, VkPipelineStageFlags dstS);

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
    Buffer kCellStartBuffer;      // binding 10: exclusive prefix sum (bucket start offset)

    // ── Compute Shaders ───────────────────────────────────────────────────────
    ComputeShader kClearShader;
    ComputeShader kHistogramShader;
    ComputeShader kPrefixScanShader;
    ComputeShader kScatterShader;
    ComputeShader kGatherShader;
    ComputeShader kFinalizeShader;
    ComputeShader kCountShader;

    float    voxelSize_  = 0.02f;
    float    truncation_ = VH_DEFAULT_TRUNC;
    float    maxWeight_  = VH_DEFAULT_MAXW;
    uint32_t frameIndex_ = 0;
};
