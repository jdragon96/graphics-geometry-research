#include "BucketVoxelHash.h"
#include "../../Utilities.h"

void BucketVoxelHash::Initialize(VulkanContext &context)
{
    CreateBuffers(context);
    CreateClearShader(context);
    CreateHistogramShader(context);
    CreatePrefixScanShader(context);
    CreateReallocateShader(context);
    CreateUpdateTSDFShader(context);
    CreateUpdateHashTableShader(context);
    CreateCountShader(context);
    CreateColorRenderPipeline(context);
    CreatePointRenderPipeline(context);
    CreateVoxelRenderPipeline(context);
    Clear(context);
}

void BucketVoxelHash::ResetHashTable(VulkanContext &context)
{
    struct ClearPC
    {
        uint32_t numberOfEntries, p0, p1, p2;
    };

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = context.commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(context.device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    kClearShader.Bind(cmd);
    kClearShader.Push(cmd, ClearPC{VH_TOTAL_ENTRIES, 0, 0, 0});
    kClearShader.Dispatch(cmd, (VH_TOTAL_ENTRIES + 63u) / 64u);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(context.graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(context.graphicsQueue);
    vkFreeCommandBuffers(context.device, context.commandPool, 1, &cmd);
}

void BucketVoxelHash::Clear(VulkanContext &context)
{
    ResetHashTable(context);
}

void BucketVoxelHash::CreateColorRenderPipeline(VulkanContext &context)
{
    RenderingShader::PipelineOptions options{};
    options.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    options.cullMode = VK_CULL_MODE_NONE;
    kColorRenderShader_.Initialize(
        context.device,
        context.renderPass,
        context.extent,
        context.basePath + "/shaders/VoxelHash_RenderPoint.vert.spv",
        context.basePath + "/shaders/VoxelHash_RenderPoint.frag.spv",
        {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}},
        options,
        sizeof(VHB_RenderPC));
    kColorRenderShader_.BindBuffer(0, kHashTableBuffer.GetBuffer());
}

void BucketVoxelHash::RenderVoxel(VkCommandBuffer cmd, const float *mvp)
{
    VHB_RenderPC pc{};
    for (int i = 0; i < 16; ++i)
    {
        pc.mvp[i] = mvp[i];
    }
    pc.voxelSize = voxelSize_;
    kVoxelRenderShader_.Bind(cmd);
    kVoxelRenderShader_.Push(cmd, VK_SHADER_STAGE_VERTEX_BIT, pc);

    // 복셀당 36 vertices (큐브 6면 × 2삼각형 × 3꼭짓점)
    // 빈 슬롯은 셰이더에서 NDC 밖으로 클리핑
    kVoxelRenderShader_.Draw(cmd, VH_TOTAL_ENTRIES * 36);
}

void BucketVoxelHash::RenderColor(VkCommandBuffer cmd, const float *mvp, bool useSlotPoint)
{
    VHB_RenderPC pc{};
    for (int i = 0; i < 16; ++i)
        pc.mvp[i] = mvp[i];
    pc.voxelSize = voxelSize_;
    if (useSlotPoint)
    {
        kPointRenderShader_.Bind(cmd);
        kPointRenderShader_.Push(cmd, VK_SHADER_STAGE_VERTEX_BIT, pc);
        kPointRenderShader_.Draw(cmd, lastIntegratedCount_);
        return;
    }

    kColorRenderShader_.Bind(cmd);
    kColorRenderShader_.Push(cmd, VK_SHADER_STAGE_VERTEX_BIT, pc);
    // 슬롯 수만큼 버텍스 발행 — 빈 슬롯은 셰이더에서 클리핑
    kColorRenderShader_.Draw(cmd, VH_TOTAL_ENTRIES);
}

void BucketVoxelHash::RenderVoxelSlot(VkCommandBuffer cmd, const float *mvp)
{
    VHB_RenderPC pc{};
    for (int i = 0; i < 16; ++i)
        pc.mvp[i] = mvp[i];
    pc.voxelSize = voxelSize_;
    kPointRenderShader_.Bind(cmd);
    kPointRenderShader_.Push(cmd, VK_SHADER_STAGE_VERTEX_BIT, pc);
    kPointRenderShader_.Draw(cmd, lastIntegratedCount_);
}

void BucketVoxelHash::CreatePointRenderPipeline(VulkanContext &context)
{
    RenderingShader::PipelineOptions options{};
    options.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    options.cullMode = VK_CULL_MODE_NONE;
    kPointRenderShader_.Initialize(
        context.device,
        context.renderPass,
        context.extent,
        context.basePath + "/shaders/VoxelHash_RenderPointBySample.vert.spv",
        context.basePath + "/shaders/VoxelHash_RenderPoint.frag.spv",
        {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
        },
        options,
        sizeof(VHB_RenderPC));
    kPointRenderShader_.BindBuffer(0, kSortedPositionBuffer.GetBuffer());
    kPointRenderShader_.BindBuffer(1, kSortedColorBuffer.GetBuffer());
}

void BucketVoxelHash::CreateVoxelRenderPipeline(VulkanContext &context)
{
    RenderingShader::PipelineOptions options{};
    options.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    options.cullMode = VK_CULL_MODE_BACK_BIT;
    options.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    options.depthTestEnable = VK_TRUE;
    options.depthWriteEnable = VK_TRUE;
    options.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    kVoxelRenderShader_.Initialize(
        context.device,
        context.renderPass,
        context.extent,
        context.basePath + "/shaders/VoxelHash_RenderVoxel.vert.spv",
        context.basePath + "/shaders/VoxelHash_RenderVoxel.frag.spv",
        {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT}},
        options,
        sizeof(VHB_RenderPC));
    kVoxelRenderShader_.BindBuffer(0, kHashTableBuffer.GetBuffer());
}

void BucketVoxelHash::RenderTSDF() {}

void BucketVoxelHash::Integrate(VulkanContext &context,
                                const VH_InputPoint *pts, uint32_t count,
                                float sensorX, float sensorY, float sensorZ)
{
    if (count == 0)
    {
        lastIntegratedCount_ = 0;
        return;
    }
    count = std::min(count, VH_BATCH_SIZE);
    lastIntegratedCount_ = count;

    // ── 1. CPU 업로드: unsorted 포인트 → host-visible 버퍼 ──────────────────
    UpdatePosition(context, pts, count);
    UpdateNormal(context, pts, count);
    UpdateColor(context, pts, count);

    // ── 2. One-shot command buffer ────────────────────────────────────────────
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = context.commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(context.device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    const VH_SortPC sortPC{count, VH_NUM_BUCKETS, voxelSize_, 0.f};

    VkAccessFlags transferWriteFlag = VK_ACCESS_TRANSFER_WRITE_BIT;
    VkAccessFlags shaderReadFlag = VK_ACCESS_SHADER_READ_BIT;
    VkAccessFlags shaderWriteFlag = VK_ACCESS_SHADER_WRITE_BIT;
    VkAccessFlags hostReadFlag = VK_ACCESS_HOST_READ_BIT;
    VkPipelineStageFlags transferStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags computeStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags hostStage = VK_PIPELINE_STAGE_HOST_BIT;

    {
        // Clear Histogram
        vkCmdFillBuffer(cmd, kHistorgramBuffer.GetBuffer(), 0, VK_WHOLE_SIZE, 0);
        kHistorgramBuffer.Barrier(cmd, transferWriteFlag, shaderReadFlag | shaderWriteFlag, transferStage, computeStage);
    }
    {
        // Update Histogram
        kHistogramShader.Bind(cmd);
        kHistogramShader.Push(cmd, sortPC);
        kHistogramShader.Dispatch(cmd, (count + 63u) / 64u);
        kHistorgramBuffer.Barrier(cmd, shaderWriteFlag, shaderWriteFlag | shaderReadFlag, computeStage, computeStage);
    }

    // ── Sort Pass 2: Prefix Scan (3 passes) ───────────────────────────────────
    {
        kPrefixScanShader.Bind(cmd);
        kPrefixScanShader.Push(cmd, VH_ScanPC{VH_NUM_BUCKETS, 0u, 0u, 0u});
        kPrefixScanShader.Dispatch(cmd, VH_SCAN_BLOCKS);
        kCellStartBuffer.Barrier(cmd, shaderWriteFlag, shaderReadFlag | shaderWriteFlag, computeStage, computeStage);
        kBlockSummationBuffer.Barrier(cmd, shaderWriteFlag, shaderReadFlag | shaderWriteFlag, computeStage, computeStage);
    }
    {
        kPrefixScanShader.Push(cmd, VH_ScanPC{VH_NUM_BUCKETS, 1u, 0u, 0u});
        kPrefixScanShader.Dispatch(cmd, 1u); // pass 1: 단일 workgroup(1024 threads)으로 1024개 blockSums 처리
        kBlockSummationBuffer.Barrier(cmd, shaderWriteFlag, shaderReadFlag | shaderWriteFlag, computeStage, computeStage);
    }
    {
        kPrefixScanShader.Push(cmd, VH_ScanPC{VH_NUM_BUCKETS, 2u, 0u, 0u});
        kPrefixScanShader.Dispatch(cmd, VH_SCAN_BLOCKS);
        kCellStartBuffer.Barrier(cmd, shaderWriteFlag, shaderReadFlag | shaderWriteFlag, computeStage, computeStage);
    }

    // ── Sort Pass 3: Scatter (histogram 재초기화 후 offset 카운터로 재사용) ──
    {
        vkCmdFillBuffer(cmd, kHistorgramBuffer.GetBuffer(), 0, VK_WHOLE_SIZE, 0);
        kHistorgramBuffer.Barrier(cmd, transferWriteFlag, shaderReadFlag | shaderWriteFlag, transferStage, computeStage);
    }
    {
        kReallocateShader.Bind(cmd);
        kReallocateShader.Push(cmd, sortPC);
        kReallocateShader.Dispatch(cmd, (count + 63u) / 64u);
        kSortedPositionBuffer.Barrier(cmd, shaderWriteFlag, shaderReadFlag, computeStage, computeStage);
        kHistorgramBuffer.Barrier(cmd, shaderWriteFlag, shaderReadFlag, computeStage, computeStage);
    }

    // ── Integrate ①: Gather — P2G TSDF 누산 ──────────────────────────────────
    {
        kUpdateTSDFShader.Bind(cmd);
        kUpdateTSDFShader.Push(cmd, VH_GatherPC{
                                        sensorX, sensorY, sensorZ,
                                        voxelSize_, count, VH_NUM_BUCKETS,
                                        truncation_, maxWeight_});
        kUpdateTSDFShader.Dispatch(cmd, VH_NUM_BUCKETS);
        kHashTableBuffer.Barrier(cmd, shaderWriteFlag, shaderReadFlag | shaderWriteFlag, computeStage, computeStage);
    }

    // ── Integrate ②: Finalize — acc → tsdf/weight 커밋 ───────────────────────
    {
        ++frameIndex_;
        kUpdateHashTableShader.Bind(cmd);
        kUpdateHashTableShader.Push(cmd, VH_FinalizePC{VH_TOTAL_ENTRIES, frameIndex_, 0.f, 0.f});
        kUpdateHashTableShader.Dispatch(cmd, (VH_TOTAL_ENTRIES + 63u) / 64u);
        kHashTableBuffer.Barrier(cmd, shaderWriteFlag, shaderReadFlag | shaderWriteFlag, computeStage, computeStage);
    }

    // ── Integrate ③: Count ────────────────────────────────────────────────────
    {
        vkCmdFillBuffer(cmd, kPatchCountBuffer.GetBuffer(), 0, sizeof(uint32_t), 0);
        kPatchCountBuffer.Barrier(cmd, transferWriteFlag, shaderReadFlag | shaderWriteFlag, transferStage, computeStage);
    }
    {
        kCountShader.Bind(cmd);
        kCountShader.Push(cmd, VH_FinalizePC{VH_TOTAL_ENTRIES, frameIndex_, 0.f, 0.f});
        kCountShader.Dispatch(cmd, (VH_TOTAL_ENTRIES + 63u) / 64u);
        kPatchCountBuffer.Barrier(cmd, shaderWriteFlag, hostReadFlag, computeStage, hostStage);
    }

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(context.graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(context.graphicsQueue);
    vkFreeCommandBuffers(context.device, context.commandPool, 1, &cmd);
}

void BucketVoxelHash::UpdatePosition(VulkanContext &, const VH_InputPoint *pts, uint32_t count)
{
    auto m = kPositionBuffer.Access<float>(sizeof(float) * 4 * count);
    for (uint32_t i = 0; i < count; i++)
    {
        m.get()[i * 4 + 0] = pts[i].px;
        m.get()[i * 4 + 1] = pts[i].py;
        m.get()[i * 4 + 2] = pts[i].pz;
        m.get()[i * 4 + 3] = 1.f;
    }
}

void BucketVoxelHash::UpdateNormal(VulkanContext &, const VH_InputPoint *pts, uint32_t count)
{
    auto m = kNormalBuffer.Access<float>(sizeof(float) * 4 * count);
    for (uint32_t i = 0; i < count; i++)
    {
        m.get()[i * 4 + 0] = pts[i].nx;
        m.get()[i * 4 + 1] = pts[i].ny;
        m.get()[i * 4 + 2] = pts[i].nz;
        m.get()[i * 4 + 3] = 0.f;
    }
}

void BucketVoxelHash::UpdateColor(VulkanContext &, const VH_InputPoint *pts, uint32_t count)
{
    auto m = kColorBuffer.Access<uint32_t>(sizeof(uint32_t) * count);
    for (uint32_t i = 0; i < count; i++)
        m.get()[i] = pts[i].col;
}

void BucketVoxelHash::CreateClearShader(VulkanContext &context)
{
    kClearShader.Initialize(
        context.device,
        context.basePath + "/shaders/VoxelHash_Clear.comp.spv",
        {{0}},
        16);
    kClearShader.BindBuffer(0, kHashTableBuffer.GetBuffer());
}

void BucketVoxelHash::CreateHistogramShader(VulkanContext &context)
{
    kHistogramShader.Initialize(
        context.device,
        context.basePath + "/shaders/VoxelHash_UpdateHistogram.comp.spv",
        {{1}, {8}},
        sizeof(VH_SortPC));
    kHistogramShader.BindBuffer(1, kPositionBuffer.GetBuffer());
    kHistogramShader.BindBuffer(8, kHistorgramBuffer.GetBuffer());
}

void BucketVoxelHash::CreatePrefixScanShader(VulkanContext &context)
{
    kPrefixScanShader.Initialize(
        context.device,
        context.basePath + "/shaders/VoxelHash_UpdateBucket.comp.spv",
        {{8}, {9}, {10}},
        sizeof(VH_ScanPC));
    kPrefixScanShader.BindBuffer(8, kHistorgramBuffer.GetBuffer());
    kPrefixScanShader.BindBuffer(9, kBlockSummationBuffer.GetBuffer());
    kPrefixScanShader.BindBuffer(10, kCellStartBuffer.GetBuffer());
}

void BucketVoxelHash::CreateReallocateShader(VulkanContext &context)
{
    kReallocateShader.Initialize(
        context.device,
        context.basePath + "/shaders/VoxelHash_ReallocatePoint.comp.spv",
        {{1}, {2}, {3}, {5}, {6}, {7}, {8}, {10}},
        sizeof(VH_SortPC));
    kReallocateShader.BindBuffer(1, kPositionBuffer.GetBuffer());
    kReallocateShader.BindBuffer(2, kNormalBuffer.GetBuffer());
    kReallocateShader.BindBuffer(3, kColorBuffer.GetBuffer());
    kReallocateShader.BindBuffer(5, kSortedPositionBuffer.GetBuffer());
    kReallocateShader.BindBuffer(6, kSortedNormalBuffer.GetBuffer());
    kReallocateShader.BindBuffer(7, kSortedColorBuffer.GetBuffer());
    kReallocateShader.BindBuffer(8, kHistorgramBuffer.GetBuffer());
    kReallocateShader.BindBuffer(10, kCellStartBuffer.GetBuffer());
}

void BucketVoxelHash::CreateUpdateTSDFShader(VulkanContext &context)
{
    kUpdateTSDFShader.Initialize(
        context.device,
        context.basePath + "/shaders/VoxelHash_UpdateTSDF.comp.spv",
        {{0}, {5}, {8}, {10}},
        sizeof(VH_GatherPC));
    kUpdateTSDFShader.BindBuffer(0, kHashTableBuffer.GetBuffer());
    kUpdateTSDFShader.BindBuffer(5, kSortedPositionBuffer.GetBuffer());
    kUpdateTSDFShader.BindBuffer(8, kHistorgramBuffer.GetBuffer());
    kUpdateTSDFShader.BindBuffer(10, kCellStartBuffer.GetBuffer());
}

void BucketVoxelHash::CreateUpdateHashTableShader(VulkanContext &context)
{
    kUpdateHashTableShader.Initialize(
        context.device,
        context.basePath + "/shaders/VoxelHash_UpdateHashTable.comp.spv",
        {{0}},
        sizeof(VH_FinalizePC));
    kUpdateHashTableShader.BindBuffer(0, kHashTableBuffer.GetBuffer());
}

void BucketVoxelHash::CreateCountShader(VulkanContext &context)
{

    kCountShader.Initialize(
        context.device,
        context.basePath + "/shaders/VoxelHash_CheckOccupiedVoxel.comp.spv",
        {{0}, {1}},
        sizeof(VH_FinalizePC));
    kCountShader.BindBuffer(0, kHashTableBuffer.GetBuffer());
    kCountShader.BindBuffer(1, kPatchCountBuffer.GetBuffer());
}

void BucketVoxelHash::CreateBuffers(VulkanContext &context)
{
    constexpr auto HOST = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr auto DEVICE = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    constexpr auto XFER = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    constexpr auto SSBO = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    constexpr VkDeviceSize VEC4_BATCH = sizeof(float) * 4 * VH_BATCH_SIZE;
    constexpr VkDeviceSize COL_BATCH = sizeof(uint32_t) * VH_BATCH_SIZE;
    constexpr VkDeviceSize BUCKETS = sizeof(uint32_t) * VH_NUM_BUCKETS;
    constexpr VkDeviceSize SCAN_BLOCKS = sizeof(uint32_t) * VH_SCAN_BLOCKS;

    // 입력 버퍼 (host-visible)
    kHashTableBuffer.Initialize(context.physicalDevice, context.device,
                                sizeof(VH_Entry) * VH_TOTAL_ENTRIES, SSBO, HOST, 0);
    kPositionBuffer.Initialize(context.physicalDevice, context.device,
                               VEC4_BATCH, SSBO, HOST, 1);
    kNormalBuffer.Initialize(context.physicalDevice, context.device,
                             VEC4_BATCH, SSBO, HOST, 2);
    kColorBuffer.Initialize(context.physicalDevice, context.device,
                            COL_BATCH, SSBO, HOST, 3);
    kPatchCountBuffer.Initialize(context.physicalDevice, context.device,
                                 sizeof(uint32_t), SSBO | XFER, HOST, 4);
    kHistorgramBuffer.Initialize(context.physicalDevice, context.device,
                                 BUCKETS, SSBO | XFER, HOST, 8);
    kBlockSummationBuffer.Initialize(context.physicalDevice, context.device,
                                     SCAN_BLOCKS, SSBO | XFER, HOST, 9);

    // GPU sort 출력 버퍼 (device-local)
    kSortedPositionBuffer.Initialize(context.physicalDevice, context.device,
                                     VEC4_BATCH, SSBO, DEVICE, 5);
    kSortedNormalBuffer.Initialize(context.physicalDevice, context.device,
                                   VEC4_BATCH, SSBO, DEVICE, 6);
    kSortedColorBuffer.Initialize(context.physicalDevice, context.device,
                                  COL_BATCH, SSBO, DEVICE, 7);

    // 버킷 시작 오프셋 (prefix scan 출력)
    kCellStartBuffer.Initialize(context.physicalDevice, context.device,
                                BUCKETS, SSBO | XFER, HOST, 10);
}
