#pragma once
#include "IFeature.h"
#include "../SceneObject.h"
#include <vector>
#include <string>

class TriangleFeature : public IFeature {
public:
    const char* name() const override { return "Triangle"; }
    void onInit(const VulkanContext& ctx) override;
    void onRender(const RenderContext& ctx) override;
    void onImGui() override;
    void onKey(int key, int action, int mods) override;
    void onCleanup() override;

private:
    VulkanContext ctx_{};

    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline       pipeline_       = VK_NULL_HANDLE;
    VkBuffer         vertexBuffer_   = VK_NULL_HANDLE;
    VkDeviceMemory   vertexMemory_   = VK_NULL_HANDLE;

    std::vector<Vertex> vertices_;
    float               scale_    = 1.0f;
    float               color_[3] = {1.0f, 1.0f, 1.0f};

    void createPipeline();
    void createVertexBuffer();

    uint32_t       findMemoryType(uint32_t filter, VkMemoryPropertyFlags props);
    VkShaderModule loadShader(const std::string& path);

    static std::vector<char> readFile(const std::string& path);
};
