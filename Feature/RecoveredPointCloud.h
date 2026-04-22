#pragma once

#include "IFeature.h"
#include "VoxelHash.h"
#include "../Utilities.h"
#include <vulkan/vulkan.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

struct RPC_RenderPoint {
    float x = 0.f;
    float y = 0.f;
    float z = 0.f;
    float aux = 0.f; // recent: age, compressed: confidence
    uint32_t color = 0xFFFFFFFFu;
    uint32_t mode = 0u; // 0 recent, 1 compressed
    uint32_t frame = 0u;
    uint32_t _pad = 0u;
};

class RecoveredPointCloudFeature : public IFeature {
public:
    explicit RecoveredPointCloudFeature(const VoxelHashFeature *source) : source_(source) {}
    const char *name() const override { return "Recovered Point Cloud"; }

    void onInit(const VulkanContext &ctx) override;
    void onCompute(VkCommandBuffer cmd) override;
    void onRender(const RenderContext &ctx) override;
    void onImGui() override;
    void onCleanup() override;

private:
    VulkanContext ctx_{};
    const VoxelHashFeature *source_ = nullptr;

    VkDescriptorSetLayout descLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;
    VkPipelineLayout renderLayout_ = VK_NULL_HANDLE;
    VkPipeline renderPipe_ = VK_NULL_HANDLE;

    VkBuffer ptBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory ptMem_ = VK_NULL_HANDLE;

    std::vector<RPC_RenderPoint> drawPts_;
    uint32_t drawCount_ = 0;
    uint32_t latestSourceFrame_ = 0;

    int mode_ = 2; // 0 recent, 1 compressed, 2 hybrid
    int colorMode_ = 0; // 0 sourceColor, 1 normalHint, 2 ageConfidence
    float pointSize_ = 3.0f;
    float confidenceThreshold_ = 0.0f;
    int recentAgeClampFrames_ = 60;
    int maxDrawPoints_ = 600000;
    bool freezeSnapshot_ = false;

    float azimuth_ = 0.5f;
    float elevation_ = 0.4f;
    float camDist_ = 4.0f;
    float rebuildMs_ = 0.f;
    float compressionRatioPct_ = 0.f;

    Eigen::Matrix4f computeMVP() const;
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer &buf, VkDeviceMemory &mem);
    void createDescriptors();
    void createPipeline();
    void rebuildSnapshot();
    static uint32_t packColor(const Eigen::Vector3f &rgb);
};

