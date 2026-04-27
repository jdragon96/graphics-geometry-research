#include "VoxelHash.h"

// ─────────────────────────────────────────────────────────────────────────────
//  dispatchGPUSort — CPU buildSpatialHashRanges를 완전히 대체하는 GPU 3-패스 정렬
//
//  Pass 1 — vh_histogram.comp
//    포인트마다 bucket 계산 → atomicAdd(histBuf_[b], 1)
//
//  Pass 2 — vh_prefix_scan.comp (3회 dispatch)
//    pass=0  각 블록(1024 스레드) 로컬 exclusive scan + blockSums 저장
//    pass=1  blockSums exclusive scan (단일 워크그룹)
//    pass=2  각 블록에 blockSums 오프셋 전파 → cellStart 확정
//
//  Pass 3 — vh_scatter.comp
//    atomicAdd(histBuf_[b], 1)을 scatter offset으로 사용
//    sortedPtBuf_[cellStart[b] + offset] = ptBuf_[i]
//
//  결과: sortedPtBuf_ → 버킷 순으로 정렬된 포인트
//        cellStartBuf_[b] → 버킷 b의 시작 오프셋
//        (cellEnd[b] = cellStart[b+1] 로 gather shader 내부에서 계산)
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::dispatchGPUSort(VkCommandBuffer cmd)
{
    if (ptCount_ == 0)
        return;

    auto barrier = [&](VkBuffer buf, VkAccessFlags src, VkAccessFlags dst,
                       VkPipelineStageFlags srcS, VkPipelineStageFlags dstS)
    {
        bufBarrier(cmd, buf, src, dst, srcS, dstS);
    };

    constexpr VkPipelineStageFlags CS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    constexpr VkAccessFlags RW = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    constexpr VkAccessFlags W = VK_ACCESS_SHADER_WRITE_BIT;
    constexpr VkAccessFlags TR = VK_ACCESS_TRANSFER_WRITE_BIT;

    const VH_SortPC sortPC{ptCount_, VH_NUM_BUCKETS, voxelSize_, 0.f};

    // ── Pass 1: 히스토그램 ─────────────────────────────────────────────────────
    // histBuf_ 초기화 (0으로)
    vkCmdFillBuffer(cmd, histBuf_, 0, sizeof(uint32_t) * VH_NUM_BUCKETS, 0);
    barrier(histBuf_, TR, RW, VK_PIPELINE_STAGE_TRANSFER_BIT, CS);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, histogramPipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compLayout_, 0, 1, &descSet_, 0, nullptr);
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sortPC), &sortPC);
    vkCmdDispatch(cmd, (ptCount_ + 63u) / 64u, 1, 1);
    barrier(histBuf_, W, RW, CS, CS);

    // ── Pass 2a: 로컬 prefix scan (1024 블록 × 1024 스레드) ──────────────────
    VH_ScanPC scanPC{VH_NUM_BUCKETS, 0, 0, 0};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prefixScanPipe_);
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(scanPC), &scanPC);
    vkCmdDispatch(cmd, VH_SCAN_BLOCKS, 1, 1);
    barrier(cellStartBuf_, W, RW, CS, CS);
    barrier(blockSumsBuf_, W, RW, CS, CS);

    // ── Pass 2b: blockSums exclusive scan (단일 워크그룹) ────────────────────
    scanPC.pass = 1;
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(scanPC), &scanPC);
    vkCmdDispatch(cmd, 1, 1, 1);
    barrier(blockSumsBuf_, W, RW, CS, CS);

    // ── Pass 2c: 전역 오프셋 전파 ─────────────────────────────────────────────
    scanPC.pass = 2;
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(scanPC), &scanPC);
    vkCmdDispatch(cmd, VH_SCAN_BLOCKS, 1, 1);
    barrier(cellStartBuf_, W, RW, CS, CS);

    // ── Pass 3: scatter — histBuf_ 재초기화 후 scatter offset으로 재사용 ──────
    vkCmdFillBuffer(cmd, histBuf_, 0, sizeof(uint32_t) * VH_NUM_BUCKETS, 0);
    barrier(histBuf_, TR, RW, VK_PIPELINE_STAGE_TRANSFER_BIT, CS);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scatterPipe_);
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sortPC), &sortPC);
    vkCmdDispatch(cmd, (ptCount_ + 63u) / 64u, 1, 1);
    barrier(sortedPosBuf_, W, RW, CS, CS);
    barrier(sortedNrmBuf_, W, RW, CS, CS);
    barrier(sortedColBuf_, W, RW, CS, CS);
}
