#pragma once

/*
 * ============================================================
 *  [Problem 1] Triangle 그리기
 * ============================================================
 *
 * 목표: RGB 삼각형을 화면에 렌더링하세요.
 *
 * TODO 목록
 *  (A) createPipeline()  → Input Assembly의 topology 값 설정
 *  (B) createVertexBuffer() → VkBuffer / VkDeviceMemory 직접 생성
 *  (C) kFragSrc           → Fragment Shader 출력 변수 선언 및 출력
 *
 * 힌트
 *  - Utilities.h 의 compileGLSL(), findMemoryType() 을 사용하세요.
 *  - VkBufferCreateInfo, VkMemoryAllocateInfo 구조체를 참고하세요.
 *  - vkMapMemory / memcpy / vkUnmapMemory 순서로 데이터를 업로드합니다.
 * ============================================================
 */

#include "../Feature/IFeature.h"
#include "../SceneObject.h"
#include "../Utilities.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <stdexcept>
#include <cstring>

class MakeTriangle : public IFeature
{
public:
    const char *name() const override { return "1. Make Triangle"; }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    void onInit(const VulkanContext &ctx) override
    {
        ctx_ = ctx;

        // Geometry 데이터는 제공됩니다. 수정하지 마세요.
        vertices_ = {
            {{0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
            {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
            {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}},
        };

        createPipeline();
        createVertexBuffer();
    }

    void onRender(const RenderContext &ctx) override
    {
        vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 1, &vertexBuffer_, &offset);
        vkCmdDraw(ctx.commandBuffer, static_cast<uint32_t>(vertices_.size()), 1, 0, 0);
    }

    void onCleanup() override
    {
        vkDestroyBuffer(ctx_.device, vertexBuffer_, nullptr);
        vkFreeMemory(ctx_.device, vertexMemory_, nullptr);
        vkDestroyPipeline(ctx_.device, pipeline_, nullptr);
        vkDestroyPipelineLayout(ctx_.device, pipelineLayout_, nullptr);
    }

private:
    VulkanContext ctx_{};

    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory_ = VK_NULL_HANDLE;

    std::vector<Vertex> vertices_;

    // ── Embedded GLSL shaders ─────────────────────────────────────────────────

    // Vertex Shader: 완성되어 있습니다.
    static constexpr const char *kVertSrc = R"GLSL(
#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = vec4(inPosition, 1.0);
    fragColor   = inColor;
}
)GLSL";

    // TODO (C): 아래 Fragment Shader 에는 출력 변수가 빠져 있습니다.
    //   1. outColor 출력 변수를 선언하세요.
    //      예) layout(location = 0) out vec4 outColor;
    //   2. main() 안에서 outColor 에 색상을 출력하세요.
    //      예) outColor = vec4(fragColor, 1.0);
    static constexpr const char *kFragSrc = R"GLSL(
#version 450
layout(location = 0) in vec3 fragColor;

// TODO: 출력 변수를 여기에 선언하세요.

void main() {
    // TODO: 선언한 출력 변수에 fragColor를 출력하세요. (alpha = 1.0)
}
)GLSL";

    // ── Pipeline & Buffer creation ────────────────────────────────────────────

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

        // TODO (A): Triangle을 그리기 위한 topology 값을 설정하세요.
        //   힌트: VK_PRIMITIVE_TOPOLOGY_??? 중 세 정점으로 삼각형을 만드는 값은?
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; // TODO: Triangle을 그리는 올바른 값으로 교체하세요

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
        // TODO (B): Vertex Buffer를 직접 만들고 데이터를 업로드하세요.
        //
        // Step 1. VkBufferCreateInfo 를 채우고 vkCreateBuffer() 를 호출하세요.
        //         - size        : sizeof(Vertex) * vertices_.size()
        //         - usage       : VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
        //         - sharingMode : VK_SHARING_MODE_EXCLUSIVE
        //         결과 핸들: vertexBuffer_
        //
        // Step 2. vkGetBufferMemoryRequirements() 로 메모리 요구사항을 조회하세요.
        //
        // Step 3. findMemoryType(ctx_.physicalDevice, ...) 을 사용해
        //         HOST_VISIBLE | HOST_COHERENT 메모리 타입 인덱스를 구하세요.
        //
        // Step 4. VkMemoryAllocateInfo 를 채우고 vkAllocateMemory() 를 호출하세요.
        //         결과 핸들: vertexMemory_
        //
        // Step 5. vkBindBufferMemory() 로 버퍼와 메모리를 바인딩하세요.
        //
        // Step 6. vkMapMemory() → memcpy(vertices_.data()) → vkUnmapMemory()
        //         순서로 정점 데이터를 GPU 메모리에 업로드하세요.
    }
};
