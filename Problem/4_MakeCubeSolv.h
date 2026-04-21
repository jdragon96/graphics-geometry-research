#pragma once

/*
 * ============================================================
 * TIL
 * 1. Cube를 그려본다.(Vulkan 3차원 공간 구조에 대해 이해한다.)
 *  - https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/
 * 2. Projection 행렬에 대해 이해한다.
 * 3. View 행렬에 대해 이해한다.
 * 4. Euler Rotation에 대해 이해한다.
 * 5. MVP 행렬에 대해 이해한다.
 */

#include "../Feature/IFeature.h"
#include "../SceneObject.h"
#include "../Utilities.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <imgui.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cmath>

struct alignas(16) MakeCubeUBO
{
    Eigen::Matrix4f mvpMatrix;
};

class MakeCubeSolv : public IFeature
{
public:
    const char *name() const override { return "4. Make Cube (Solution)"; }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    void onInit(const VulkanContext &ctx) override
    {
        ctx_ = ctx;
        vertices_ = {
            {{-0.5f, 0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, -1.0f, 0.0f}},
            {{-0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}},
            {{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, -1.0f, 0.0f}},
            {{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, -1.0f, 0.0f}},

            {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 1.0f}, {0.0f, -1.0f, 0.0f}},
            {{-0.5f, -0.5f, 0.5f}, {0.0f, 1.0f, 1.0f}, {0.0f, -1.0f, 0.0f}},
            {{0.5f, -0.5f, 0.5f}, {1.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}},
            {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 1.0f}, {0.0f, -1.0f, 0.0f}},
        };
        for (auto &vertex : vertices_)
        {
            Eigen::Vector3f direction = Eigen::Vector3f(
                                            vertex.position.x(),
                                            vertex.position.y(),
                                            vertex.position.z()) -
                                        Eigen::Vector3f(0, 0, 0);
            vertex.normal = direction.normalized();
        }
        indices_ = {
            0,
            2,
            1,
            0,
            3,
            2,
            5,
            0,
            1,
            5,
            4,
            0,
            6,
            1,
            2,
            6,
            5,
            1,
            7,
            2,
            3,
            7,
            6,
            2,
            4,
            3,
            0,
            4,
            7,
            3,
            5,
            7,
            4,
            5,
            6,
            7,
        };
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

        Eigen::Matrix3f rot =
            Eigen::AngleAxisf(deg2rad(fAngleOfAxisZ), Eigen::Vector3f::UnitZ()).toRotationMatrix() *
            Eigen::AngleAxisf(deg2rad(fAngleOfAxisY), Eigen::Vector3f::UnitY()).toRotationMatrix() *
            Eigen::AngleAxisf(deg2rad(fAngleOfAxisX), Eigen::Vector3f::UnitX()).toRotationMatrix();
        Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
        model.block<3, 3>(0, 0) = rot;
        ubo_.mvpMatrix = computeProj() * computeView() * model;
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
        ImGuiIO &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                camAzimuth_ += io.MouseDelta.x * 0.005f;
                camElevation_ += io.MouseDelta.y * 0.005f;
                camElevation_ = std::max(-1.4f, std::min(1.4f, camElevation_));
            }
            camDist_ -= io.MouseWheel * 0.3f;
            camDist_ = std::max(0.5f, std::min(20.f, camDist_));
        }

        ImGui::Begin("Make Cube");

        ImGui::SeparatorText("Model Rotation");
        ImGui::SliderFloat("Rot X (deg)", &fAngleOfAxisX, -180.f, 180.f);
        ImGui::SliderFloat("Rot Y (deg)", &fAngleOfAxisY, -180.f, 180.f);
        ImGui::SliderFloat("Rot Z (deg)", &fAngleOfAxisZ, -180.f, 180.f);

        ImGui::SeparatorText("Camera  [drag=rotate  wheel=zoom]");
        ImGui::SliderFloat("Azimuth", &camAzimuth_, -3.14f, 3.14f, "%.2f rad");
        ImGui::SliderFloat("Elevation", &camElevation_, -1.4f, 1.4f, "%.2f rad");
        ImGui::SliderFloat("Distance", &camDist_, 0.5f, 20.f, "%.1f");

        ImGui::SeparatorText("Projection");
        ImGui::SliderFloat("FOV Y (deg)", &fovY_, 10.f, 120.f, "%.0f");
        ImGui::SliderFloat("Near", &nearZ_, 0.001f, 1.f, "%.3f");
        ImGui::SliderFloat("Far", &farZ_, 10.f, 500.f, "%.0f");

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
    MakeCubeUBO ubo_;
    void *uboAccessPointer = nullptr;

    // ── Model rotation ────────────────────────────────────────────────────────
    float fAngleOfAxisX = 0.f;
    float fAngleOfAxisY = 0.f;
    float fAngleOfAxisZ = 0.f;

    // ── View (orbit camera) ───────────────────────────────────────────────────
    float camAzimuth_ = 0.5f;
    float camElevation_ = 0.3f;
    float camDist_ = 3.0f;

    // ── Projection ────────────────────────────────────────────────────────────
    float fovY_ = 60.f;
    float nearZ_ = 0.01f;
    float farZ_ = 100.f;

    // ── MVP helpers ───────────────────────────────────────────────────────────

    Eigen::Matrix4f computeView() const
    {
        float cx = std::cos(camElevation_), sx = std::sin(camElevation_);
        float cy = std::cos(camAzimuth_), sy = std::sin(camAzimuth_);
        Eigen::Vector3f eye(camDist_ * cx * sy, camDist_ * sx, camDist_ * cx * cy);
        Eigen::Vector3f up(0.f, 1.f, 0.f);
        Eigen::Vector3f f = (-eye).normalized();
        Eigen::Vector3f r = f.cross(up).normalized();
        Eigen::Vector3f u = r.cross(f);

        Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
        V.row(0) << r.x(), r.y(), r.z(), -r.dot(eye);
        V.row(1) << u.x(), u.y(), u.z(), -u.dot(eye);
        V.row(2) << -f.x(), -f.y(), -f.z(), f.dot(eye);
        return V;
    }

    Eigen::Matrix4f computeProj() const
    {
        float fovRad = fovY_ * static_cast<float>(M_PI) / 180.f;
        float aspect = static_cast<float>(ctx_.extent.width) / ctx_.extent.height;
        float th = std::tan(fovRad * 0.5f);

        Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
        P(0, 0) = 1.f / (aspect * th);
        P(1, 1) = -1.f / th; // Vulkan Y-flip
        P(2, 2) = farZ_ / (nearZ_ - farZ_);
        P(2, 3) = farZ_ * nearZ_ / (nearZ_ - farZ_);
        P(3, 2) = -1.f;
        return P;
    }

    // ── Embedded GLSL shaders ─────────────────────────────────────────────────

    static constexpr const char *kVertSrc = R"GLSL(
#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;

layout(set = 0, binding = 0) uniform UBO {
    mat4 mvp;
} ubo;

void main() {
    gl_Position = ubo.mvp * vec4(inPosition, 1.0);
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
        bufInfo.range = sizeof(MakeCubeUBO);

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
        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS;

        pCI.pRasterizationState = &raster;
        pCI.pMultisampleState = &ms;
        pCI.pColorBlendState = &blend;
        pCI.pDepthStencilState = &ds;
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
        VkDeviceSize size = sizeof(MakeCubeUBO);

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
