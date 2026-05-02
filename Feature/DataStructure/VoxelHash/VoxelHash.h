#pragma once

#include "../IFeature.h"
#include "VoxelHashTypes.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>

// ─────────────────────────────────────────────────────────────────────────────
//  VoxelHashFeature
//
//  GPU Gather TSDF 복셀 해시 (0.05 mm 정밀도 기준 설계)
//
//  외부 연결 지점:
//    submitPoints(pts, count, sensorX, sensorY, sensorZ)
//      → 구조광 스캐너 / 기타 외부 소스에서 포인트를 주입
//      → VH_InputPoint 배열 (position + normal + color)을 직접 전달
//
//  내부 파이프라인 (onCompute 호출 시):
//    [선택] Clear → GPUSort(histogram→prefixscan→scatter) → Gather → Finalize → Count
// ─────────────────────────────────────────────────────────────────────────────

class VoxelHashFeature : public IFeature
{
public:
    const char *name() const override { return "VoxelHash (0.05mm Gather)"; }

    // ── IFeature lifecycle ────────────────────────────────────────────────────
    void onInit(const VulkanContext &ctx) override;
    void onCompute(VkCommandBuffer cmd) override;
    void onRender(const RenderContext &ctx) override;
    void onImGui() override;
    void onCleanup() override;

    // ── 외부 포인트 주입 API ──────────────────────────────────────────────────
    void submitPoints(const VH_InputPoint *pts, uint32_t count,
                      float sensorX, float sensorY, float sensorZ);
    void reset();
    int getOccupancy() const { return occupancy_; }

private:
    // ── Vulkan 핸들 ───────────────────────────────────────────────────────────
    VulkanContext ctx_{};

    // ── 디스크립터 ────────────────────────────────────────────────────────────
    VkDescriptorSetLayout descLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;

    // ── 컴퓨트 파이프라인 ─────────────────────────────────────────────────────
    VkPipelineLayout compLayout_ = VK_NULL_HANDLE;
    VkPipeline clearPipe_ = VK_NULL_HANDLE;
    VkPipeline histogramPipe_ = VK_NULL_HANDLE;  // GPU sort pass 1
    VkPipeline prefixScanPipe_ = VK_NULL_HANDLE; // GPU sort pass 2
    VkPipeline scatterPipe_ = VK_NULL_HANDLE;    // GPU sort pass 3
    VkPipeline gatherPipe_ = VK_NULL_HANDLE;     // P2G 누산
    VkPipeline finalizePipe_ = VK_NULL_HANDLE;   // running-average 커밋
    VkPipeline countPipe_ = VK_NULL_HANDLE;      // 점유율 집계

    // ── 렌더 파이프라인 ───────────────────────────────────────────────────────
    VkPipelineLayout renderLayout_ = VK_NULL_HANDLE;
    VkPipeline renderPipe_ = VK_NULL_HANDLE;   // 복셀 큐브
    VkPipeline ptRenderPipe_ = VK_NULL_HANDLE; // 입력 포인트 오버레이

    // ── GPU 버퍼 (binding 번호와 1:1 대응) ──────────────────────────────────
    //  binding  이름             내용                    접근 셰이더
    //  ──────   ──────────────   ─────────────────────   ──────────────────────
    //   0       htBuf_           해시 테이블 슬롯         gather/finalize/count/render
    //   1       posBuf_          입력 위치  vec4[]        histogram/scatter
    //   2       nrmBuf_          입력 법선  vec4[]        scatter
    //   3       colBuf_          입력 색상  uint[]        scatter
    //   4       ctrBuf_          점유 카운터               count
    //   5       sortedPosBuf_    정렬된 위치 vec4[]        scatter/gather/ptrender
    //   6       sortedNrmBuf_    정렬된 법선 vec4[]        scatter/gather
    //   7       sortedColBuf_    정렬된 색상 uint[]        scatter/gather
    //   8       histBuf_         히스토그램/scatter offset histogram/scatter
    //   9       blockSumsBuf_    prefix scan 블록합        prefix_scan
    //  10       cellStartBuf_    버킷 시작 오프셋          prefix_scan/scatter/gather

    VkBuffer htBuf_ = VK_NULL_HANDLE; // binding  0
    VkDeviceMemory htMem_ = VK_NULL_HANDLE;
    VkBuffer posBuf_ = VK_NULL_HANDLE; // binding  1  (host-coherent)
    VkDeviceMemory posMem_ = VK_NULL_HANDLE;
    VkBuffer nrmBuf_ = VK_NULL_HANDLE; // binding  2  (host-coherent)
    VkDeviceMemory nrmMem_ = VK_NULL_HANDLE;
    VkBuffer colBuf_ = VK_NULL_HANDLE; // binding  3  (host-coherent)
    VkDeviceMemory colMem_ = VK_NULL_HANDLE;
    VkBuffer ctrBuf_ = VK_NULL_HANDLE; // binding  4
    VkDeviceMemory ctrMem_ = VK_NULL_HANDLE;
    VkBuffer sortedPosBuf_ = VK_NULL_HANDLE; // binding  5  (device-local)
    VkDeviceMemory sortedPosMem_ = VK_NULL_HANDLE;
    VkBuffer sortedNrmBuf_ = VK_NULL_HANDLE; // binding  6  (device-local)
    VkDeviceMemory sortedNrmMem_ = VK_NULL_HANDLE;
    VkBuffer sortedColBuf_ = VK_NULL_HANDLE; // binding  7  (device-local)
    VkDeviceMemory sortedColMem_ = VK_NULL_HANDLE;
    VkBuffer histBuf_ = VK_NULL_HANDLE; // binding  8
    VkDeviceMemory histMem_ = VK_NULL_HANDLE;
    VkBuffer blockSumsBuf_ = VK_NULL_HANDLE; // binding  9
    VkDeviceMemory blockSumsMem_ = VK_NULL_HANDLE;
    VkBuffer cellStartBuf_ = VK_NULL_HANDLE; // binding 10
    VkDeviceMemory cellStartMem_ = VK_NULL_HANDLE;

    // ── 제어 플래그 ───────────────────────────────────────────────────────────
    bool doClear_ = true;
    bool doInsert_ = false;
    bool doCountRead_ = false;
    uint32_t frameIndex_ = 0;

    // ── 센서 파라미터 (현재 배치) ─────────────────────────────────────────────
    float sensorX_ = 0.f, sensorY_ = 0.f, sensorZ_ = 3.f;
    uint32_t ptCount_ = 0;

    // ── UI 파라미터 ───────────────────────────────────────────────────────────
    bool showInputPts_ = true;
    float azimuth_ = 0.5f;
    float elevation_ = 0.4f;
    float camDist_ = 0.2f; // 0.05mm 스케일에 맞게 가까이
    float voxelSize_ = VH_DEFAULT_VOXEL;
    float truncation_ = VH_DEFAULT_TRUNC;
    float maxWeight_ = VH_DEFAULT_MAXW;
    int colorMode_ = 0;
    int highlightFrames_ = 8;

    // ── 테스트 합성 데이터 (구면 생성기) ──────────────────────────────────────
    bool streaming_ = false;
    float sphereRadius_ = 0.01f;    // 10 mm 반구
    float sensorDist_ = 0.03f;      // 30 mm
    float noiseStddev_ = 0.000005f; // 5 μm 노이즈
    float sensorAz_ = 0.f;
    float sensorEl_ = 0.3f;
    float sensorSpeed_ = 0.015f;

    // ── 성능 계측 ─────────────────────────────────────────────────────────────
    VkQueryPool perfQueryPool_ = VK_NULL_HANDLE;
    bool hasPerfQuery_ = false;
    float sortMs_ = 0.f;
    float gatherMs_ = 0.f;
    float finalizeMs_ = 0.f;
    float countMs_ = 0.f;
    float baselineMs_ = 0.f;

    // ── 통계 ──────────────────────────────────────────────────────────────────
    int occupancy_ = 0;
    int64_t totalInserted_ = 0;
    float ptsPerSec_ = 0.f;
    std::chrono::steady_clock::time_point lastTime_;

    // ── 내부 헬퍼 선언 ────────────────────────────────────────────────────────

    // VoxelHashBuffers.cpp
    void createBuffers();
    void createBuf(VkDeviceSize size, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags props,
                   VkBuffer &buf, VkDeviceMemory &mem);

    // VoxelHashPipelines.cpp
    void createDescriptors();
    void createComputePipelines();
    void createRenderPipeline();
    VkPipeline makeComputePipeline(const char *glsl);
    VkPipeline makeComputePipelineFromSpv(const std::string &spvPath);

    // VoxelHashDispatch.cpp
    void dispatchClear(VkCommandBuffer cmd);
    void dispatchGather(VkCommandBuffer cmd);
    void dispatchFinalize(VkCommandBuffer cmd);
    void dispatchCount(VkCommandBuffer cmd);
    void bufBarrier(VkCommandBuffer cmd, VkBuffer buf,
                    VkAccessFlags src, VkAccessFlags dst,
                    VkPipelineStageFlags srcStage,
                    VkPipelineStageFlags dstStage);

    // VoxelHashSort.cpp
    void dispatchGPUSort(VkCommandBuffer cmd);

    // VoxelHashSynth.cpp
    uint32_t genSphereBatch();

    // VoxelHash.cpp (카메라)
    Eigen::Matrix4f computeMVP() const;

    // VoxelHashImGui.cpp
    void readPerfQueries();
};
