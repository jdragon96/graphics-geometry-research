#pragma once
#include "IFeature.h"
#include <vector>
#include <string>

class ComputeTest : public IFeature {
public:
    const char* name() const override { return "Compute Wave Cubes"; }
    void onInit   (const VulkanContext& ctx) override;
    void onCompute(VkCommandBuffer cmd) override;
    void onRender (const RenderContext& ctx) override;
    void onImGui  () override;
    void onKey    (int key, int action, int mods) override;
    void onCleanup() override;

private:
    VulkanContext ctx_{};

    // 시뮬레이션 파라미터
    int   gridSize_   = 40;
    float spacing_    = 0.35f;
    float amplitude_  = 0.5f;
    float frequency_  = 1.2f;
    float speed_      = 1.0f;
    float cubeScale_  = 0.28f;
    bool  paused_     = false;
    float pausedTime_ = 0.0f;

    // 카메라
    float camDist_    = 9.0f;
    float camAngle_   = 0.6f;
    float rotY_       = 0.0f;
    bool  autoRotate_ = true;

    // ── 컴퓨트 파이프라인 ──────────────────────────────────────────────────────
    VkPipeline            computePipeline_       = VK_NULL_HANDLE;
    VkPipelineLayout      computePipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeDescLayout_     = VK_NULL_HANDLE;
    VkDescriptorPool      computeDescPool_       = VK_NULL_HANDLE;
    VkDescriptorSet       computeDescSet_        = VK_NULL_HANDLE;

    // ── SSBO (컴퓨트 출력 + 인스턴스 버텍스 버퍼 겸용) ────────────────────────
    VkBuffer       ssboBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory ssboMemory_ = VK_NULL_HANDLE;

    // ── 그래픽스 파이프라인 ───────────────────────────────────────────────────
    VkPipeline            graphicsPipeline_       = VK_NULL_HANDLE;
    VkPipelineLayout      graphicsPipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout graphicsDescLayout_     = VK_NULL_HANDLE;
    VkDescriptorPool      graphicsDescPool_       = VK_NULL_HANDLE;
    VkDescriptorSet       graphicsDescSet_        = VK_NULL_HANDLE;

    // ── UBO ───────────────────────────────────────────────────────────────────
    VkBuffer       uboBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory uboMemory_ = VK_NULL_HANDLE;
    void*          uboMapped_ = nullptr;

    // ── 큐브 지오메트리 ───────────────────────────────────────────────────────
    VkBuffer       cubeVertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory cubeVertexMemory_ = VK_NULL_HANDLE;
    VkBuffer       cubeIndexBuffer_  = VK_NULL_HANDLE;
    VkDeviceMemory cubeIndexMemory_  = VK_NULL_HANDLE;
    uint32_t       cubeIndexCount_   = 0;

    // ── 헬퍼 ──────────────────────────────────────────────────────────────────
    void createSSBO();
    void createComputePipeline();
    void createUBO();
    void createGraphicsPipeline();
    void createCubeGeometry();
    void updateUBO();

    void           createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags props,
                                VkBuffer& buf, VkDeviceMemory& mem);
    uint32_t       findMemoryType(uint32_t filter, VkMemoryPropertyFlags props);
    VkShaderModule loadShader(const std::string& path);
    static std::vector<char> readFile(const std::string& path);
};
