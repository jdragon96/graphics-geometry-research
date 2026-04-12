#include "TriangleFeature.h"
#include <imgui.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <stdexcept>
#include <cstring>

// ── Helpers ───────────────────────────────────────────────────────────────────

std::vector<char> TriangleFeature::readFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("cannot open shader: " + path);
    size_t size = (size_t)file.tellg();
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), size);
    return buf;
}

VkShaderModule TriangleFeature::loadShader(const std::string& path) {
    auto code = readFile(path);
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(ctx_.device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("failed to create shader module");
    return mod;
}

uint32_t TriangleFeature::findMemoryType(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(ctx_.physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("no suitable memory type");
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

void TriangleFeature::onInit(const VulkanContext& ctx) {
    ctx_ = ctx;

    vertices_ = {
        { { 0.0f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },
        { { 0.5f,  0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f} },
        { {-0.5f,  0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f} },
    };

    createPipeline();
    createVertexBuffer();
}

void TriangleFeature::createPipeline() {
    auto vertMod = loadShader(ctx_.basePath + "/shaders/triangle.vert.spv");
    auto fragMod = loadShader(ctx_.basePath + "/shaders/triangle.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName  = "main";

    auto bindDesc = Vertex::getBindingDescription();
    auto attrDesc = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &bindDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
    vertexInput.pVertexAttributeDescriptions    = attrDesc.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{ 0, 0, (float)ctx_.extent.width, (float)ctx_.extent.height, 0.f, 1.f };
    VkRect2D   scissor { {0,0}, ctx_.extent };

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.pViewports = &viewport;
    vp.scissorCount  = 1; vp.pScissors  = &scissor;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth   = 1.0f;
    raster.cullMode    = VK_CULL_MODE_BACK_BIT;
    raster.frontFace   = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAtt.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments    = &blendAtt;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(float);

    VkPipelineLayoutCreateInfo layoutCI{};
    layoutCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges    = &pushRange;
    if (vkCreatePipelineLayout(ctx_.device, &layoutCI, nullptr, &pipelineLayout_) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout");

    VkGraphicsPipelineCreateInfo pCI{};
    pCI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pCI.stageCount          = 2;
    pCI.pStages             = stages;
    pCI.pVertexInputState   = &vertexInput;
    pCI.pInputAssemblyState = &ia;
    pCI.pViewportState      = &vp;
    pCI.pRasterizationState = &raster;
    pCI.pMultisampleState   = &ms;
    pCI.pColorBlendState    = &blend;
    pCI.layout              = pipelineLayout_;
    pCI.renderPass          = ctx_.renderPass;
    pCI.subpass             = 0;

    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &pCI, nullptr, &pipeline_) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline");

    vkDestroyShaderModule(ctx_.device, vertMod, nullptr);
    vkDestroyShaderModule(ctx_.device, fragMod, nullptr);
}

void TriangleFeature::createVertexBuffer() {
    VkDeviceSize size = sizeof(Vertex) * vertices_.size();

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(ctx_.device, &bci, nullptr, &vertexBuffer_) != VK_SUCCESS)
        throw std::runtime_error("failed to create vertex buffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx_.device, vertexBuffer_, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(
        req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(ctx_.device, &ai, nullptr, &vertexMemory_) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate vertex memory");

    vkBindBufferMemory(ctx_.device, vertexBuffer_, vertexMemory_, 0);

    void* data;
    vkMapMemory(ctx_.device, vertexMemory_, 0, size, 0, &data);
    memcpy(data, vertices_.data(), size);
    vkUnmapMemory(ctx_.device, vertexMemory_);
}

// ── Per-frame ─────────────────────────────────────────────────────────────────

void TriangleFeature::onRender(const RenderContext& ctx) {
    vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdPushConstants(ctx.commandBuffer, pipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float), &scale_);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 1, &vertexBuffer_, &offset);
    vkCmdDraw(ctx.commandBuffer, static_cast<uint32_t>(vertices_.size()), 1, 0, 0);
}

void TriangleFeature::onImGui() {
    ImGui::Begin("Triangle");
    ImGui::SliderFloat("Scale", &scale_, 0.1f, 2.0f);
    ImGui::ColorEdit3("Vertex Tint", color_);
    ImGui::Separator();
    ImGui::Text("Vertices : %zu", vertices_.size());
    ImGui::Text("Press R  : reset scale");
    ImGui::End();
}

void TriangleFeature::onKey(int key, int action, int mods) {
    if (key == GLFW_KEY_R && action == GLFW_PRESS)
        scale_ = 1.0f;
}

void TriangleFeature::onCleanup() {
    vkDestroyBuffer(ctx_.device, vertexBuffer_, nullptr);
    vkFreeMemory(ctx_.device, vertexMemory_, nullptr);
    vkDestroyPipeline(ctx_.device, pipeline_, nullptr);
    vkDestroyPipelineLayout(ctx_.device, pipelineLayout_, nullptr);
}
