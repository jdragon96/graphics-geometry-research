#pragma once

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#include <string>
#include <vector>

class RenderingShader
{
public:
    struct Binding
    {
        uint32_t slot;
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT;
    };

    struct PipelineOptions
    {
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL;
        VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
        VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkBool32 depthTestEnable = VK_FALSE;
        VkBool32 depthWriteEnable = VK_FALSE;
        VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkBool32 blendEnable = VK_FALSE;
        std::vector<VkVertexInputBindingDescription> vertexBindings;
        std::vector<VkVertexInputAttributeDescription> vertexAttributes;
    };

    ~RenderingShader() { Clear(); }

    void Initialize(
        VkDevice device,
        VkRenderPass renderPass,
        VkExtent2D extent,
        const std::string &vertSpvPath,
        const std::string &fragSpvPath,
        const std::vector<Binding> &bindings,
        const PipelineOptions &options,
        uint32_t pushConstantBytes = 0);

    void InitializeGLSL(
        VkDevice device,
        VkRenderPass renderPass,
        VkExtent2D extent,
        const std::string &vertSrc,
        const std::string &fragSrc,
        const std::vector<Binding> &bindings,
        const PipelineOptions &options,
        uint32_t pushConstantBytes = 0);

    void BindBuffer(uint32_t slot, VkBuffer buffer,
                    VkDeviceSize offset = 0,
                    VkDeviceSize size = VK_WHOLE_SIZE,
                    VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    void BindImage(uint32_t slot, VkImageView imageView, VkSampler sampler,
                   VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VkDescriptorType type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    void Bind(VkCommandBuffer cmd) const;

    template <typename T>
    void Push(VkCommandBuffer cmd, VkShaderStageFlags stages, const T &data) const
    {
        vkCmdPushConstants(cmd, layout_, stages, 0, sizeof(T), &data);
    }

    void Draw(VkCommandBuffer cmd, uint32_t vertexCount,
              uint32_t instanceCount = 1,
              uint32_t firstVertex = 0,
              uint32_t firstInstance = 0) const;

    void Clear();

private:
    void initDescAndLayout(VkDevice device,
                           const std::vector<Binding> &bindings,
                           uint32_t pushConstantBytes);
    void buildPipeline(VkDevice device,
                       VkRenderPass renderPass,
                       VkExtent2D extent,
                       VkShaderModule vertMod,
                       VkShaderModule fragMod,
                       const PipelineOptions &options);

    VkDevice storedDevice_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;
    uint32_t pcBytes_ = 0;
};
