#pragma once

/*
 * ============================================================
 * 2주차 1번 과제 - 삼각형 회전시키기
 * - Uniform Buffer
 * - Eigen::Matirx
 * ============================================================
 * Uniform Buffer를 활용하여 삼각형을 회전시킨다.
 *
 * ============================================================
 * TIL
 * 1. Uniform Buffer를 통해 상수값을 Shader로 전송한다.
 * 2. Rotate 행렬을 이해한다.
 * 3.
 * ============================================================
 * (1) TODO: Vertex Shader에 Uniform Buffer 정의
 * (2) TODO: Uniform Buffer 메모리 할당하기
 * (3) Render 시점에 상수값 전달하기
 * (4) ImGUI에 회전각도 변경을 연결하고 시각화 하기
 * (5)
 *
 */

#include "../Feature/IFeature.h"
#include "../SceneObject.h"
#include "../Utilities.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <imgui.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#include <cstring>

struct alignas(16) RotateTriangleUBO
{
    Eigen::Matrix4f modelMatrix;
};

class RotateTriangleSolv : public IFeature
{
public:
    const char *name() const override { return "3. Rotate Triangle (Solution)"; }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    void onInit(const VulkanContext &ctx) override
    {
        ctx_ = ctx;
        vertices_ = {
            {{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
            {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
            {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        };
        indices_ = {0, 1, 2};
        createDescriptorSetLayout();
        createPipeline();
        createVertexBuffer();
        createIndexBuffer();
        createUBO();
        createDescriptorSet();
    }

    void onRender(const RenderContext &ctx) override
    {
        auto deg2rad = [](float deg)
        {
            return (deg / 180.f) * 3.141592f;
        };

        float fRadAngle = deg2rad(fAngleOfAxisZ);
        Eigen::Matrix4f rot = Eigen::Matrix4f::Identity();
        rot.block<3, 3>(0, 0) = Eigen::AngleAxisf(fRadAngle, Eigen::Vector3f::UnitZ()).toRotationMatrix();
        ubo_.modelMatrix = rot;
        memcpy(uboAccessPointer, &ubo_, sizeof(ubo_));

        VkDeviceSize offset = 0;
        vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
        vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 1, &vertexBuffer_, &offset);
        vkCmdBindIndexBuffer(ctx.commandBuffer, indexBuffer_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(ctx.commandBuffer, static_cast<uint32_t>(indices_.size()), 1, 0, 0, 0);
    }

    void onCleanup() override
    {
        vkDestroyBuffer(ctx_.device, uboBuffer_, nullptr);
        vkFreeMemory(ctx_.device, uboMemory_, nullptr);
        vkDestroyBuffer(ctx_.device, indexBuffer_, nullptr);
        vkFreeMemory(ctx_.device, indexMemory_, nullptr);
        vkDestroyBuffer(ctx_.device, vertexBuffer_, nullptr);
        vkFreeMemory(ctx_.device, vertexMemory_, nullptr);
        vkDestroyDescriptorPool(ctx_.device, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(ctx_.device, descriptorSetLayout_, nullptr);
        vkDestroyPipeline(ctx_.device, pipeline_, nullptr);
        vkDestroyPipelineLayout(ctx_.device, pipelineLayout_, nullptr);
    }

    void onImGui() override
    {
        ImGui::Begin("Rotate Triangle");

        ImGui::SliderFloat("Angle of Axis Z(Degree)", &fAngleOfAxisZ, -180.f, 180.f);

        ImGui::End();
    }

private:
    VulkanContext ctx_{};

    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;
    std::vector<Vertex> vertices_;

    VkBuffer indexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory_ = VK_NULL_HANDLE;
    std::vector<uint32_t> indices_;

    // ── TODO: ─────────────────────────────────────────────────
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;

    VkBuffer uboBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory uboMemory_ = VK_NULL_HANDLE;
    RotateTriangleUBO ubo_;
    void *uboAccessPointer = nullptr;
    float fAngleOfAxisZ = 0.f;

    // ── Embedded GLSL shaders ─────────────────────────────────────────────────

    static constexpr const char *kVertSrc = R"GLSL(
#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;

layout(set = 0, binding = 0) uniform UBO {
    mat4  model;
} ubo;

void main() {
    vec4 pos = vec4(inPosition, 1.0);
    gl_Position = ubo.model * pos;
    fragColor   = inColor;
}
)GLSL";

    // (C) 정답: outColor 출력 변수 선언 및 fragColor 출력
    static constexpr const char *kFragSrc = R"GLSL(
#version 450
layout(location = 0) in  vec3 fragColor;
layout(location = 0) out vec4 outColor;   // ← 출력 변수 선언

void main() {
    outColor = vec4(fragColor, 1.0);       // ← fragColor를 rgba로 출력
}
)GLSL";

    // ── Pipeline & Buffer creation ────────────────────────────────────────────

    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings = &binding;
        if (vkCreateDescriptorSetLayout(ctx_.device, &ci, nullptr, &descriptorSetLayout_) != VK_SUCCESS)
            throw std::runtime_error("failed to create descriptor set layout");
    }

    void createDescriptorSet()
    {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.maxSets = 1;
        poolCI.poolSizeCount = 1;
        poolCI.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(ctx_.device, &poolCI, nullptr, &descriptorPool_) != VK_SUCCESS)
            throw std::runtime_error("failed to create descriptor pool");

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool_;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout_;
        if (vkAllocateDescriptorSets(ctx_.device, &allocInfo, &descriptorSet_) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate descriptor set");

        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = uboBuffer_;
        bufInfo.offset = 0;
        bufInfo.range = sizeof(RotateTriangleUBO);

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSet_;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &bufInfo;
        vkUpdateDescriptorSets(ctx_.device, 1, &write, 0, nullptr);
    }

    void createPipeline()
    {
        auto vertMod = compileGLSL(ctx_.device, kVertSrc, shaderc_vertex_shader);
        auto fragMod = compileGLSL(ctx_.device, kFragSrc, shaderc_fragment_shader);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertMod;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragMod;
        stages[1].pName = "main";

        auto bindDesc = Vertex::getBindingDescription();
        auto attrDesc = Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bindDesc;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
        vi.pVertexAttributeDescriptions = attrDesc.data();

        // (A) 정답: 세 정점으로 삼각형 하나를 만드는 TRIANGLE_LIST 사용
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport{0.f, 0.f, (float)ctx_.extent.width, (float)ctx_.extent.height, 0.f, 1.f};
        VkRect2D scissor{{0, 0}, ctx_.extent};
        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.pViewports = &viewport;
        vp.scissorCount = 1;
        vp.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.lineWidth = 1.0f;
        raster.cullMode = VK_CULL_MODE_BACK_BIT;
        raster.frontFace = VK_FRONT_FACE_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState blendAtt{};
        blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blendAtt.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo blend{};
        blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend.attachmentCount = 1;
        blend.pAttachments = &blendAtt;

        VkPipelineLayoutCreateInfo layoutCI{};
        layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCI.setLayoutCount = 1;
        layoutCI.pSetLayouts = &descriptorSetLayout_;
        if (vkCreatePipelineLayout(ctx_.device, &layoutCI, nullptr, &pipelineLayout_) != VK_SUCCESS)
            throw std::runtime_error("failed to create pipeline layout");

        VkGraphicsPipelineCreateInfo pCI{};
        pCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pCI.stageCount = 2;
        pCI.pStages = stages;
        pCI.pVertexInputState = &vi;
        pCI.pInputAssemblyState = &ia;
        pCI.pViewportState = &vp;
        pCI.pRasterizationState = &raster;
        pCI.pMultisampleState = &ms;
        pCI.pColorBlendState = &blend;
        pCI.layout = pipelineLayout_;
        pCI.renderPass = ctx_.renderPass;
        pCI.subpass = 0;

        if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &pCI, nullptr, &pipeline_) != VK_SUCCESS)
            throw std::runtime_error("failed to create graphics pipeline");

        vkDestroyShaderModule(ctx_.device, vertMod, nullptr);
        vkDestroyShaderModule(ctx_.device, fragMod, nullptr);
    }

    void createVertexBuffer()
    {
        // (B) 정답: VkBuffer 생성 → 메모리 할당 → 바인딩 → 데이터 업로드

        VkDeviceSize size = sizeof(Vertex) * vertices_.size();

        // Step 1: Buffer 생성
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(ctx_.device, &bci, nullptr, &vertexBuffer_) != VK_SUCCESS)
            throw std::runtime_error("failed to create vertex buffer");

        // Step 2: 메모리 요구사항 조회
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(ctx_.device, vertexBuffer_, &req);

        // Step 3 & 4: 메모리 타입 선택 후 할당
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(
            ctx_.physicalDevice,
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(ctx_.device, &ai, nullptr, &vertexMemory_) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate vertex memory");

        // Step 5: 바인딩
        vkBindBufferMemory(ctx_.device, vertexBuffer_, vertexMemory_, 0);

        // Step 6: 데이터 업로드
        void *data;
        vkMapMemory(ctx_.device, vertexMemory_, 0, size, 0, &data);
        memcpy(data, vertices_.data(), static_cast<size_t>(size));
        vkUnmapMemory(ctx_.device, vertexMemory_);
    }

    void createIndexBuffer()
    {
        VkDeviceSize size = sizeof(Vertex) * indices_.size();
        VkBufferCreateInfo kBufferCreateInfo{};
        kBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        kBufferCreateInfo.size = size;
        kBufferCreateInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        if (vkCreateBuffer(ctx_.device, &kBufferCreateInfo, nullptr, &indexBuffer_))
        {
            throw std::runtime_error("failed to create buffer");
        }

        VkMemoryRequirements kMemRequires;
        vkGetBufferMemoryRequirements(ctx_.device, indexBuffer_, &kMemRequires);

        VkMemoryAllocateInfo kAllocInfo;
        kAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        kAllocInfo.allocationSize = size;
        kAllocInfo.memoryTypeIndex = findMemoryType(
            ctx_.physicalDevice,
            kMemRequires.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(ctx_.device, &kAllocInfo, nullptr, &indexMemory_))
        {
            throw std::runtime_error("failed to create buffer");
        }
        vkBindBufferMemory(ctx_.device, indexBuffer_, indexMemory_, 0);

        void *data;
        vkMapMemory(ctx_.device, indexMemory_, 0, size, 0, &data);
        memcpy(data, indices_.data(), size);
        vkUnmapMemory(ctx_.device, indexMemory_);
    }

    void createUBO()
    {
        VkDeviceSize size = sizeof(RotateTriangleUBO);

        VkBufferCreateInfo kBufferCreateInfo{};
        kBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        kBufferCreateInfo.size = size;
        kBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (vkCreateBuffer(ctx_.device, &kBufferCreateInfo, nullptr, &uboBuffer_))
        {
            throw std::runtime_error("failed to create buffer");
        }

        VkMemoryRequirements kMemRequires;
        vkGetBufferMemoryRequirements(ctx_.device, uboBuffer_, &kMemRequires);

        VkMemoryAllocateInfo kAllocInfo;
        kAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        kAllocInfo.allocationSize = size;
        kAllocInfo.memoryTypeIndex = findMemoryType(
            ctx_.physicalDevice,
            kMemRequires.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(ctx_.device, &kAllocInfo, nullptr, &uboMemory_))
        {
            throw std::runtime_error("failed to create buffer");
        }
        vkBindBufferMemory(ctx_.device, uboBuffer_, uboMemory_, 0);
        vkMapMemory(ctx_.device, uboMemory_, 0, size, 0, &uboAccessPointer);
    }
};
