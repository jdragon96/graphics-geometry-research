#include "VoxelHash.h"

// ─────────────────────────────────────────────────────────────────────────────
//  bufBarrier — 버퍼 메모리 배리어 헬퍼
// ─────────────────────────────────────────────────────────────────────────────

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
//  dispatchClear — 해시 테이블 전체 초기화
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::dispatchClear(VkCommandBuffer cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clearPipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compLayout_,
                            0, 1, &descSet_, 0, nullptr);
    VH_FinalizePC pc{VH_TOTAL_ENTRIES, frameIndex_, 0.f, 0.f};
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &pc);
    vkCmdDispatch(cmd, (VH_TOTAL_ENTRIES + 63u) / 64u, 1, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  dispatchGather — sortedPtBuf_ 기반 P2G 누산
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::dispatchGather(VkCommandBuffer cmd)
{
    if (ptCount_ == 0)
        return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gatherPipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compLayout_, 0, 1, &descSet_, 0, nullptr);
    VH_GatherPC pc{};
    pc.sensorX = sensorX_;
    pc.sensorY = sensorY_;
    pc.sensorZ = sensorZ_;
    pc.voxelSize = voxelSize_;
    pc.numPoints = ptCount_;
    pc.numBuckets = VH_NUM_BUCKETS;
    pc.truncation = truncation_;
    pc.maxWeight = maxWeight_;
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, VH_NUM_BUCKETS, 1, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  dispatchFinalize — running-average 커밋 + frame_tag 기록
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::dispatchFinalize(VkCommandBuffer cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, finalizePipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compLayout_, 0, 1, &descSet_, 0, nullptr);
    VH_FinalizePC pc{VH_TOTAL_ENTRIES, frameIndex_, 0.f, 0.f};
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &pc);
    vkCmdDispatch(cmd, (VH_TOTAL_ENTRIES + 63u) / 64u, 1, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  dispatchCount — 점유된 슬롯 수 집계 (다음 프레임에서 CPU가 읽음)
// ─────────────────────────────────────────────────────────────────────────────

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
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &pc);
    vkCmdDispatch(cmd, (VH_TOTAL_ENTRIES + 63u) / 64u, 1, 1);
    bufBarrier(cmd, ctrBuf_,
               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
}
