#include "RecoveredPointCloud.h"
#include <imgui.h>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace {
struct RPC_PC {
    float mvp[16];
    float pointSize;
    uint32_t colorMode;
    uint32_t currentFrame;
    uint32_t _pad;
};

static constexpr const char *kRPC_Vert = R"GLSL(
#version 450
layout(push_constant) uniform PC {
    mat4 mvp;
    float pointSize;
    uint colorMode;
    uint currentFrame;
    uint _pad;
} pc;

layout(set=0, binding=0) readonly buffer Pts { uint raw[]; } pts;
layout(location = 0) out vec3 fragColor;

vec3 unpackRgb(uint c) {
    return vec3(float(c & 0xFFu), float((c >> 8u) & 0xFFu), float((c >> 16u) & 0xFFu)) / 255.0;
}

void main() {
    int base = gl_VertexIndex * 8;
    vec3 pos = vec3(uintBitsToFloat(pts.raw[base + 0]),
                    uintBitsToFloat(pts.raw[base + 1]),
                    uintBitsToFloat(pts.raw[base + 2]));
    float aux = uintBitsToFloat(pts.raw[base + 3]);
    uint color = pts.raw[base + 4];
    uint mode = pts.raw[base + 5];
    uint frame = pts.raw[base + 6];

    gl_Position = pc.mvp * vec4(pos, 1.0);
    gl_PointSize = pc.pointSize;

    if (pc.colorMode == 0u) {
        fragColor = unpackRgb(color);
    } else if (pc.colorMode == 1u) {
        fragColor = (mode == 0u) ? vec3(0.1, 1.0, 1.0) : vec3(1.0, 0.6, 0.1);
    } else {
        if (mode == 0u) {
            float age = clamp((float(pc.currentFrame) - float(frame)) / max(aux, 1.0), 0.0, 1.0);
            fragColor = mix(vec3(1.0, 0.2, 0.1), vec3(0.1, 0.2, 1.0), age);
        } else {
            float conf = clamp(aux / 32.0, 0.0, 1.0);
            fragColor = mix(vec3(0.1, 0.1, 0.1), vec3(1.0, 1.0, 0.3), conf);
        }
    }
}
)GLSL";

static constexpr const char *kRPC_Frag = R"GLSL(
#version 450
layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(fragColor, 1.0); }
)GLSL";
} // namespace

Eigen::Matrix4f RecoveredPointCloudFeature::computeMVP() const
{
    float cx = std::cos(elevation_), sx = std::sin(elevation_);
    float cy = std::cos(azimuth_), sy = std::sin(azimuth_);
    Eigen::Vector3f eye(camDist_ * cx * sy, camDist_ * sx, camDist_ * cx * cy);
    Eigen::Vector3f up(0.f, 1.f, 0.f);
    Eigen::Vector3f f = (-eye).normalized();
    Eigen::Vector3f r = f.cross(up).normalized();
    Eigen::Vector3f u = r.cross(f);

    Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
    V.row(0) << r.x(), r.y(), r.z(), -r.dot(eye);
    V.row(1) << u.x(), u.y(), u.z(), -u.dot(eye);
    V.row(2) << -f.x(), -f.y(), -f.z(), f.dot(eye);

    float fovY = 60.f * static_cast<float>(M_PI) / 180.f;
    float aspect = static_cast<float>(ctx_.extent.width) / std::max(1u, ctx_.extent.height);
    float n = 0.01f, fa = 100.f, th = std::tan(fovY * 0.5f);

    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
    P(0, 0) = 1.f / (aspect * th);
    P(1, 1) = -1.f / th;
    P(2, 2) = fa / (n - fa);
    P(2, 3) = fa * n / (n - fa);
    P(3, 2) = -1.f;

    return P * V;
}

void RecoveredPointCloudFeature::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer &buf, VkDeviceMemory &mem)
{
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx_.device, &bci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("RecoveredPointCloud: vkCreateBuffer");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(ctx_.device, buf, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(ctx_.physicalDevice, req.memoryTypeBits, props);
    if (vkAllocateMemory(ctx_.device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("RecoveredPointCloud: vkAllocateMemory");
    vkBindBufferMemory(ctx_.device, buf, mem, 0);
}

void RecoveredPointCloudFeature::createDescriptors()
{
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 1;
    lci.pBindings = &b;
    if (vkCreateDescriptorSetLayout(ctx_.device, &lci, nullptr, &descLayout_) != VK_SUCCESS)
        throw std::runtime_error("RecoveredPointCloud: descriptor layout");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(ctx_.device, &pci, nullptr, &descPool_) != VK_SUCCESS)
        throw std::runtime_error("RecoveredPointCloud: descriptor pool");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &descLayout_;
    if (vkAllocateDescriptorSets(ctx_.device, &ai, &descSet_) != VK_SUCCESS)
        throw std::runtime_error("RecoveredPointCloud: descriptor alloc");

    VkDescriptorBufferInfo bi{ptBuf_, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = descSet_;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(ctx_.device, 1, &w, 0, nullptr);
}

void RecoveredPointCloudFeature::createPipeline()
{
    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(RPC_PC)};
    VkPipelineLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lci.setLayoutCount = 1;
    lci.pSetLayouts = &descLayout_;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(ctx_.device, &lci, nullptr, &renderLayout_) != VK_SUCCESS)
        throw std::runtime_error("RecoveredPointCloud: pipeline layout");

    VkShaderModule vMod = compileGLSL(ctx_.device, kRPC_Vert, shaderc_vertex_shader);
    VkShaderModule fMod = compileGLSL(ctx_.device, kRPC_Frag, shaderc_fragment_shader);

    VkPipelineShaderStageCreateInfo stages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vMod, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fMod, "main"}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, VK_PRIMITIVE_TOPOLOGY_POINT_LIST};
    VkViewport vp{0.f, 0.f, static_cast<float>(ctx_.extent.width), static_cast<float>(ctx_.extent.height), 0.f, 1.f};
    VkRect2D sc{{0, 0}, ctx_.extent};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &vp, 1, &sc};
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, {}, 1, &ba};

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vps;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pColorBlendState = &blend;
    pci.layout = renderLayout_;
    pci.renderPass = ctx_.renderPass;
    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &pci, nullptr, &renderPipe_) != VK_SUCCESS)
        throw std::runtime_error("RecoveredPointCloud: graphics pipeline");

    vkDestroyShaderModule(ctx_.device, vMod, nullptr);
    vkDestroyShaderModule(ctx_.device, fMod, nullptr);
}

uint32_t RecoveredPointCloudFeature::packColor(const Eigen::Vector3f &rgb)
{
    uint32_t r = static_cast<uint32_t>(std::clamp(rgb.x(), 0.f, 1.f) * 255.f);
    uint32_t g = static_cast<uint32_t>(std::clamp(rgb.y(), 0.f, 1.f) * 255.f);
    uint32_t b = static_cast<uint32_t>(std::clamp(rgb.z(), 0.f, 1.f) * 255.f);
    return r | (g << 8u) | (b << 16u) | 0xFF000000u;
}

void RecoveredPointCloudFeature::rebuildSnapshot()
{
    if (source_ == nullptr)
        return;

    auto t0 = std::chrono::steady_clock::now();
    const std::vector<VH_RecentSample> recent = source_->snapshotRecentSamples();
    const std::vector<VH_CompressedPoint> compressed = source_->snapshotCompressedPoints();
    latestSourceFrame_ = source_->snapshotFrameIndex();

    drawPts_.clear();
    drawPts_.reserve(std::min<int>(maxDrawPoints_, static_cast<int>(recent.size() + compressed.size())));

    if (mode_ == 0 || mode_ == 2)
    {
        for (const auto &s : recent)
        {
            if (static_cast<int>(drawPts_.size()) >= maxDrawPoints_)
                break;
            float ageClamp = static_cast<float>(std::max(1, recentAgeClampFrames_));
            drawPts_.push_back(RPC_RenderPoint{s.pos.x(), s.pos.y(), s.pos.z(), ageClamp, s.color, 0u, s.frame, 0u});
        }
    }

    if (mode_ == 1 || mode_ == 2)
    {
        for (const auto &c : compressed)
        {
            if (static_cast<int>(drawPts_.size()) >= maxDrawPoints_)
                break;
            if (c.confidence < confidenceThreshold_)
                continue;
            uint32_t cColor = packColor(c.avgNormal * 0.5f + Eigen::Vector3f::Constant(0.5f));
            drawPts_.push_back(RPC_RenderPoint{
                c.centroid.x(), c.centroid.y(), c.centroid.z(),
                c.confidence, cColor, 1u, c.lastFrame, 0u});
        }
    }

    drawCount_ = static_cast<uint32_t>(drawPts_.size());
    compressionRatioPct_ = recent.empty() ? 0.f : (100.0f * static_cast<float>(compressed.size()) / static_cast<float>(recent.size()));

    void *mapped = nullptr;
    vkMapMemory(ctx_.device, ptMem_, 0, VK_WHOLE_SIZE, 0, &mapped);
    std::memset(mapped, 0, sizeof(RPC_RenderPoint) * static_cast<size_t>(maxDrawPoints_));
    if (!drawPts_.empty())
        std::memcpy(mapped, drawPts_.data(), sizeof(RPC_RenderPoint) * drawPts_.size());
    vkUnmapMemory(ctx_.device, ptMem_);

    auto t1 = std::chrono::steady_clock::now();
    rebuildMs_ = std::chrono::duration<float, std::milli>(t1 - t0).count();
}

void RecoveredPointCloudFeature::onInit(const VulkanContext &ctx)
{
    ctx_ = ctx;
    drawPts_.reserve(static_cast<size_t>(maxDrawPoints_));
    createBuffer(sizeof(RPC_RenderPoint) * static_cast<size_t>(maxDrawPoints_),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 ptBuf_, ptMem_);
    createDescriptors();
    createPipeline();
}

void RecoveredPointCloudFeature::onCompute(VkCommandBuffer)
{
    if (!freezeSnapshot_)
        rebuildSnapshot();
}

void RecoveredPointCloudFeature::onRender(const RenderContext &ctx)
{
    if (drawCount_ == 0)
        return;

    vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, renderPipe_);
    vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, renderLayout_, 0, 1, &descSet_, 0, nullptr);

    RPC_PC pc{};
    Eigen::Matrix4f mvp = computeMVP();
    std::memcpy(pc.mvp, mvp.data(), sizeof(float) * 16);
    pc.pointSize = pointSize_;
    pc.colorMode = static_cast<uint32_t>(colorMode_);
    pc.currentFrame = latestSourceFrame_;
    vkCmdPushConstants(ctx.commandBuffer, renderLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
    vkCmdDraw(ctx.commandBuffer, drawCount_, 1, 0, 0);
}

void RecoveredPointCloudFeature::onImGui()
{
    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureMouse)
    {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
        {
            azimuth_ += io.MouseDelta.x * 0.005f;
            elevation_ += io.MouseDelta.y * 0.005f;
            elevation_ = std::clamp(elevation_, -1.4f, 1.4f);
        }
        camDist_ = std::clamp(camDist_ - io.MouseWheel * 0.3f, 0.5f, 20.0f);
    }

    ImGui::Begin("RecoveredPointCloud");
    ImGui::Text("Source frame: %u", latestSourceFrame_);
    ImGui::Text("Draw points : %u", drawCount_);
    ImGui::Text("Rebuild time: %.3f ms", rebuildMs_);
    ImGui::Text("Compression : %.2f%% (compressed/recent)", compressionRatioPct_);
    ImGui::Separator();

    ImGui::RadioButton("Recent only", &mode_, 0); ImGui::SameLine();
    ImGui::RadioButton("Compressed only", &mode_, 1); ImGui::SameLine();
    ImGui::RadioButton("Hybrid", &mode_, 2);
    ImGui::SliderFloat("Point size", &pointSize_, 1.0f, 8.0f, "%.1f");
    ImGui::SliderFloat("Confidence threshold", &confidenceThreshold_, 0.0f, 64.0f, "%.1f");
    ImGui::SliderInt("Recent age clamp", &recentAgeClampFrames_, 1, 240);
    ImGui::SliderInt("Max draw points", &maxDrawPoints_, 50000, 1000000);

    ImGui::RadioButton("Color: source", &colorMode_, 0); ImGui::SameLine();
    ImGui::RadioButton("Color: type", &colorMode_, 1); ImGui::SameLine();
    ImGui::RadioButton("Color: age/conf", &colorMode_, 2);

    ImGui::Checkbox("Freeze snapshot", &freezeSnapshot_);
    if (ImGui::Button("Refresh now"))
        rebuildSnapshot();
    ImGui::End();
}

void RecoveredPointCloudFeature::onCleanup()
{
    vkDestroyPipeline(ctx_.device, renderPipe_, nullptr);
    vkDestroyPipelineLayout(ctx_.device, renderLayout_, nullptr);
    vkDestroyDescriptorPool(ctx_.device, descPool_, nullptr);
    vkDestroyDescriptorSetLayout(ctx_.device, descLayout_, nullptr);
    vkDestroyBuffer(ctx_.device, ptBuf_, nullptr);
    vkFreeMemory(ctx_.device, ptMem_, nullptr);
}

