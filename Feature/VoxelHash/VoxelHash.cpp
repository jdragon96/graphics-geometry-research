#include "VoxelHash.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
//  computeMVP — 구형 카메라 (azimuth / elevation / distance)
// ─────────────────────────────────────────────────────────────────────────────

Eigen::Matrix4f VoxelHashFeature::computeMVP() const
{
    float cx = std::cos(elevation_), sx = std::sin(elevation_);
    float cy = std::cos(azimuth_), sy = std::sin(azimuth_);
    Eigen::Vector3f eye(camDist_ * cx * sy, camDist_ * sx, camDist_ * cx * cy);
    Eigen::Vector3f up(0.f, 1.f, 0.f);
    Eigen::Vector3f f = Eigen::Vector3f(-eye).normalized();
    Eigen::Vector3f r = Eigen::Vector3f(f.cross(up)).normalized();
    Eigen::Vector3f u = Eigen::Vector3f(r.cross(f));

    Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
    V.row(0) << r.x(), r.y(), r.z(), -r.dot(eye);
    V.row(1) << u.x(), u.y(), u.z(), -u.dot(eye);
    V.row(2) << -f.x(), -f.y(), -f.z(), f.dot(eye);

    float fovY = 60.f * static_cast<float>(M_PI) / 180.f;
    float aspect = static_cast<float>(ctx_.extent.width) / ctx_.extent.height;
    float n = 0.0001f, fa = 1.f, th = std::tan(fovY * 0.5f);

    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
    P(0, 0) = 1.f / (aspect * th);
    P(1, 1) = -1.f / th;
    P(2, 2) = fa / (n - fa);
    P(2, 3) = fa * n / (n - fa);
    P(3, 2) = -1.f;

    return P * V;
}

// ─────────────────────────────────────────────────────────────────────────────
//  submitPoints — 외부 포인트 주입 API
//
//  구조광 스캐너 또는 다른 소스에서 position+normal+color 포인트를
//  직접 넣을 수 있는 공개 인터페이스.
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::submitPoints(const VH_InputPoint *pts, uint32_t count,
                                    float sx, float sy, float sz)
{
    if (count == 0)
        return;
    count = std::min(count, VH_BATCH_SIZE);

    // SoA 분리 업로드: VH_InputPoint → posBuf_ / nrmBuf_ / colBuf_
    void *mapped;

    vkMapMemory(ctx_.device, posMem_, 0, sizeof(float) * 4 * count, 0, &mapped);
    auto *posD = static_cast<float *>(mapped);
    for (uint32_t i = 0; i < count; i++)
    {
        posD[i * 4 + 0] = pts[i].px;
        posD[i * 4 + 1] = pts[i].py;
        posD[i * 4 + 2] = pts[i].pz;
        posD[i * 4 + 3] = 1.f;
    }
    vkUnmapMemory(ctx_.device, posMem_);

    vkMapMemory(ctx_.device, nrmMem_, 0, sizeof(float) * 4 * count, 0, &mapped);
    auto *nrmD = static_cast<float *>(mapped);
    for (uint32_t i = 0; i < count; i++)
    {
        nrmD[i * 4 + 0] = pts[i].nx;
        nrmD[i * 4 + 1] = pts[i].ny;
        nrmD[i * 4 + 2] = pts[i].nz;
        nrmD[i * 4 + 3] = 0.f;
    }
    vkUnmapMemory(ctx_.device, nrmMem_);

    vkMapMemory(ctx_.device, colMem_, 0, sizeof(uint32_t) * count, 0, &mapped);
    auto *colD = static_cast<uint32_t *>(mapped);
    for (uint32_t i = 0; i < count; i++)
        colD[i] = pts[i].col;
    vkUnmapMemory(ctx_.device, colMem_);

    ptCount_ = count;
    sensorX_ = sx;
    sensorY_ = sy;
    sensorZ_ = sz;
    doInsert_ = true;
}

void VoxelHashFeature::reset()
{
    doClear_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  onInit
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::onInit(const VulkanContext &ctx)
{
    ctx_ = ctx;

    // GPU가 충분한 workgroup 수를 지원하는지 확인 (gather = VH_NUM_BUCKETS workgroups)
    VkPhysicalDeviceProperties devProps{};
    vkGetPhysicalDeviceProperties(ctx_.physicalDevice, &devProps);
    if (devProps.limits.maxComputeWorkGroupCount[0] < VH_NUM_BUCKETS)
        throw std::runtime_error("GPU maxComputeWorkGroupCount too small for VH_NUM_BUCKETS");

    createBuffers();
    createDescriptors();
    createComputePipelines();
    createRenderPipeline();

    // 성능 쿼리 풀 생성 (gather / finalize / count / sort 각 2개 timestamp)
    VkQueryPoolCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount = 8;
    if (vkCreateQueryPool(ctx_.device, &qci, nullptr, &perfQueryPool_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: query pool");

    lastTime_ = std::chrono::steady_clock::now();
    doClear_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  onCompute — 프레임당 컴퓨트 커맨드 기록
//
//  순서: [Clear?] → GPUSort → Gather → Finalize → Count
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::onCompute(VkCommandBuffer cmd)
{
    frameIndex_++;

    // 이전 프레임 카운터 읽기
    if (doCountRead_)
    {
        void *m;
        vkMapMemory(ctx_.device, ctrMem_, 0, sizeof(uint32_t), 0, &m);
        occupancy_ = static_cast<int>(*static_cast<uint32_t *>(m));
        vkUnmapMemory(ctx_.device, ctrMem_);
        doCountRead_ = false;
    }
    readPerfQueries();

    // 해시 테이블 초기화
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

    // 합성 데이터 스트리밍 — genSphereBatch 내부에서 submitPoints 호출
    if (streaming_)
    {
        genSphereBatch(); // → submitPoints(batch, count, sx, sy, sz) 경유
        sensorAz_ += sensorSpeed_;
        if (sensorAz_ > 2.f * static_cast<float>(M_PI))
            sensorAz_ -= 2.f * static_cast<float>(M_PI);
    }

    if (doInsert_ && ptCount_ > 0)
    {
        vkCmdResetQueryPool(cmd, perfQueryPool_, 0, 8);

        // ① GPU Sort (histogram → prefix scan → scatter)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 0);
        dispatchGPUSort(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 1);

        // ② Gather (P2G 누산, 실제 법선 사용)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 2);
        dispatchGather(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 3);
        bufBarrier(cmd, htBuf_,
                   VK_ACCESS_SHADER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        // ③ Finalize (running-average 커밋)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 4);
        dispatchFinalize(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 5);
        bufBarrier(cmd, htBuf_,
                   VK_ACCESS_SHADER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);

        // ④ Count
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 6);
        dispatchCount(cmd);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, perfQueryPool_, 7);
        hasPerfQuery_ = true;

        totalInserted_ += ptCount_;
        doInsert_ = false;
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

void VoxelHashFeature::onRender(const RenderContext &rctx)
{
    vkCmdBindPipeline(rctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, renderPipe_);
    vkCmdBindDescriptorSets(rctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            renderLayout_, 0, 1, &descSet_, 0, nullptr);

    Eigen::Matrix4f mvp = computeMVP();
    VH_RenderPC pc{};
    std::memcpy(pc.mvp, mvp.data(), 64);
    pc.voxelSize = voxelSize_;
    pc.colorMode = static_cast<uint32_t>(colorMode_);
    pc.currentFrame = frameIndex_;
    pc.highlightFrames = static_cast<uint32_t>(highlightFrames_);
    vkCmdPushConstants(rctx.commandBuffer, renderLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(rctx.commandBuffer, VH_TOTAL_ENTRIES * 36, 1, 0, 0);

    // 입력 포인트 오버레이
    if (showInputPts_ && ptCount_ > 0)
    {
        vkCmdBindPipeline(rctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ptRenderPipe_);
        vkCmdBindDescriptorSets(rctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                renderLayout_, 0, 1, &descSet_, 0, nullptr);
        vkCmdPushConstants(rctx.commandBuffer, renderLayout_,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
        vkCmdDraw(rctx.commandBuffer, ptCount_, 1, 0, 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  onCleanup
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::onCleanup()
{
    // 파이프라인
    vkDestroyPipeline(ctx_.device, clearPipe_, nullptr);
    vkDestroyPipeline(ctx_.device, histogramPipe_, nullptr);
    vkDestroyPipeline(ctx_.device, prefixScanPipe_, nullptr);
    vkDestroyPipeline(ctx_.device, scatterPipe_, nullptr);
    vkDestroyPipeline(ctx_.device, gatherPipe_, nullptr);
    vkDestroyPipeline(ctx_.device, finalizePipe_, nullptr);
    vkDestroyPipeline(ctx_.device, countPipe_, nullptr);
    vkDestroyPipeline(ctx_.device, renderPipe_, nullptr);
    vkDestroyPipeline(ctx_.device, ptRenderPipe_, nullptr);
    vkDestroyPipelineLayout(ctx_.device, compLayout_, nullptr);
    vkDestroyPipelineLayout(ctx_.device, renderLayout_, nullptr);

    // 디스크립터
    vkDestroyDescriptorPool(ctx_.device, descPool_, nullptr);
    vkDestroyDescriptorSetLayout(ctx_.device, descLayout_, nullptr);

    // 버퍼 (binding 순서 일치)
    auto destroyBuf = [&](VkBuffer buf, VkDeviceMemory mem)
    {
        vkDestroyBuffer(ctx_.device, buf, nullptr);
        vkFreeMemory(ctx_.device, mem, nullptr);
    };
    destroyBuf(htBuf_, htMem_);
    destroyBuf(posBuf_, posMem_);
    destroyBuf(nrmBuf_, nrmMem_);
    destroyBuf(colBuf_, colMem_);
    destroyBuf(ctrBuf_, ctrMem_);
    destroyBuf(sortedPosBuf_, sortedPosMem_);
    destroyBuf(sortedNrmBuf_, sortedNrmMem_);
    destroyBuf(sortedColBuf_, sortedColMem_);
    destroyBuf(histBuf_, histMem_);
    destroyBuf(blockSumsBuf_, blockSumsMem_);
    destroyBuf(cellStartBuf_, cellStartMem_);

    // 쿼리 풀
    vkDestroyQueryPool(ctx_.device, perfQueryPool_, nullptr);
}
