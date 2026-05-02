#pragma once
#include "IFeature.h"
#include "../SceneObject.h"
#include <vector>
#include <string>

class BlinnPhongFeature : public IFeature {
public:
    const char* name() const override { return "BlinnPhong Sphere"; }
    void onInit(const VulkanContext& ctx) override;
    void onRender(const RenderContext& ctx) override;
    void onImGui() override;
    void onKey(int key, int action, int mods) override;
    void onCleanup() override;

private:
    VulkanContext ctx_{};

    // Vulkan objects
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool_      = VK_NULL_HANDLE;
    VkDescriptorSet       descriptorSet_       = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout_      = VK_NULL_HANDLE;
    VkPipeline            pipeline_            = VK_NULL_HANDLE;

    VkBuffer       vertexBuffer_  = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_  = VK_NULL_HANDLE;
    VkBuffer       indexBuffer_   = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory_   = VK_NULL_HANDLE;
    VkBuffer       uniformBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory uniformMemory_ = VK_NULL_HANDLE;
    void*          uniformMapped_ = nullptr;

    std::vector<Vertex>   vertices_;
    std::vector<uint32_t> indices_;

    // Light & material
    float lightPos_[3]    = { 2.0f, 2.0f,  2.0f };
    float lightColor_[3]  = { 1.0f, 1.0f,  1.0f };
    float objectColor_[3] = { 0.5f, 0.35f, 0.8f };
    float ambient_        = 0.15f;
    float specular_       = 0.6f;
    float shininess_      = 64.0f;

    // Camera
    float camDist_  = 3.0f;
    float fovDeg_   = 45.0f;

    // Rotation
    float rotY_      = 0.0f;
    bool  autoRotate_ = true;

    // ── helpers ──────────────────────────────────────────────────────────────
    void generateSphere(int stacks, int sectors);
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createUniformBuffer();
    void createDescriptorSet();
    void createVertexBuffer();
    void createIndexBuffer();
    void createPipeline();
    void updateUniformBuffer();

    void           createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags props,
                                VkBuffer& buf, VkDeviceMemory& mem);
    uint32_t       findMemoryType(uint32_t filter, VkMemoryPropertyFlags props);
    VkShaderModule loadShader(const std::string& path);
    static std::vector<char> readFile(const std::string& path);
};
