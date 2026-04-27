#pragma once

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#include <string>
#include <vector>

class ComputeShader
{
public:
    struct Binding
    {
        uint32_t slot;
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        VkShaderStageFlags stages = VK_SHADER_STAGE_COMPUTE_BIT;
    };

    ~ComputeShader() { Clear(); }

    // .spv 파일에서 로드
    void Initialize(
        VkDevice device,
        const std::string &spvPath,
        const std::vector<Binding> &bindings,
        uint32_t pushConstantBytes = 0);

    // 런타임 GLSL 소스에서 컴파일
    void InitializeGLSL(
        VkDevice device,
        const std::string &glslSrc,
        shaderc_shader_kind kind,
        const std::vector<Binding> &bindings,
        uint32_t pushConstantBytes = 0);

    void BindBuffer(uint32_t slot, VkBuffer buffer,
                    VkDeviceSize offset = 0,
                    VkDeviceSize size = VK_WHOLE_SIZE);

    void Bind(VkCommandBuffer cmd) const;

    template <typename T>
    void Push(VkCommandBuffer cmd, const T &data) const
    {
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(T), &data);
    }

    void Dispatch(VkCommandBuffer cmd,
                  uint32_t x, uint32_t y = 1, uint32_t z = 1) const;

    void Clear();

private:
    void initDescAndLayout(VkDevice device,
                           const std::vector<Binding> &bindings,
                           uint32_t pushConstantBytes);
    void buildPipeline(VkDevice device, VkShaderModule mod);

    VkDevice storedDevice_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet_ = VK_NULL_HANDLE;
    uint32_t pcBytes_ = 0;
};
