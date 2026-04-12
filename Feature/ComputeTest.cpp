#include "ComputeTest.h"
#include <imgui.h>
#include <GLFW/glfw3.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <array>

// ── 내부 데이터 구조 ──────────────────────────────────────────────────────────

struct alignas(16) WaveUBO {
    Eigen::Matrix4f view;
    Eigen::Matrix4f proj;
    Eigen::Vector4f lightDir;
    float           cubeScale;
    float           _pad[3];
};

struct CubeVertex {
    float pos[3];
    float normal[3];
};

struct ComputePushConstants {
    float time;
    float amplitude;
    float frequency;
    float speed;
    int   gridSize;
    float spacing;
};

// ── 수학 헬퍼 ─────────────────────────────────────────────────────────────────

static Eigen::Matrix4f lookAt(Eigen::Vector3f eye, Eigen::Vector3f center, Eigen::Vector3f up) {
    Eigen::Vector3f f = (center - eye).normalized();
    Eigen::Vector3f s = f.cross(up).normalized();
    Eigen::Vector3f u = s.cross(f);
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0,0)=s.x(); m(0,1)=s.y(); m(0,2)=s.z(); m(0,3)=-s.dot(eye);
    m(1,0)=u.x(); m(1,1)=u.y(); m(1,2)=u.z(); m(1,3)=-u.dot(eye);
    m(2,0)=-f.x();m(2,1)=-f.y();m(2,2)=-f.z();m(2,3)= f.dot(eye);
    m(3,0)=0;     m(3,1)=0;     m(3,2)=0;     m(3,3)=1;
    return m;
}

static Eigen::Matrix4f perspective(float fovYRad, float aspect, float nearZ, float farZ) {
    float f = 1.0f / std::tan(fovYRad * 0.5f);
    Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
    m(0,0) =  f / aspect;
    m(1,1) = -f;
    m(2,2) =  farZ / (nearZ - farZ);
    m(2,3) =  farZ * nearZ / (nearZ - farZ);
    m(3,2) = -1.0f;
    return m;
}

// ── 파일 / 셰이더 헬퍼 ───────────────────────────────────────────────────────

std::vector<char> ComputeTest::readFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("cannot open: " + path);
    size_t sz = (size_t)file.tellg();
    std::vector<char> buf(sz);
    file.seekg(0);
    file.read(buf.data(), sz);
    return buf;
}

VkShaderModule ComputeTest::loadShader(const std::string& path) {
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

uint32_t ComputeTest::findMemoryType(uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(ctx_.physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("no suitable memory type");
}

void ComputeTest::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                               VkMemoryPropertyFlags props,
                               VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx_.device, &bci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("failed to create buffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx_.device, buf, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    if (vkAllocateMemory(ctx_.device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate memory");
    vkBindBufferMemory(ctx_.device, buf, mem, 0);
}

// ── 큐브 지오메트리 ───────────────────────────────────────────────────────────

void ComputeTest::createCubeGeometry() {
    constexpr float h = 0.5f;
    const std::vector<CubeVertex> verts = {
        // Front  (z=+h, n=0,0,1)
        {{-h, h, h},{0,0,1}}, {{ h, h, h},{0,0,1}}, {{ h,-h, h},{0,0,1}}, {{-h,-h, h},{0,0,1}},
        // Back   (z=-h, n=0,0,-1)
        {{ h, h,-h},{0,0,-1}}, {{-h, h,-h},{0,0,-1}}, {{-h,-h,-h},{0,0,-1}}, {{ h,-h,-h},{0,0,-1}},
        // Left   (x=-h, n=-1,0,0)
        {{-h, h,-h},{-1,0,0}}, {{-h, h, h},{-1,0,0}}, {{-h,-h, h},{-1,0,0}}, {{-h,-h,-h},{-1,0,0}},
        // Right  (x=+h, n=1,0,0)
        {{ h, h, h},{1,0,0}},  {{ h, h,-h},{1,0,0}},  {{ h,-h,-h},{1,0,0}},  {{ h,-h, h},{1,0,0}},
        // Top    (y=+h, n=0,1,0)
        {{-h, h,-h},{0,1,0}},  {{ h, h,-h},{0,1,0}},  {{ h, h, h},{0,1,0}},  {{-h, h, h},{0,1,0}},
        // Bottom (y=-h, n=0,-1,0)
        {{-h,-h, h},{0,-1,0}}, {{ h,-h, h},{0,-1,0}}, {{ h,-h,-h},{0,-1,0}}, {{-h,-h,-h},{0,-1,0}},
    };

    std::vector<uint32_t> idxs;
    for (uint32_t f = 0; f < 6; ++f) {
        uint32_t b = f * 4;
        idxs.insert(idxs.end(), {b, b+1, b+2, b, b+2, b+3});
    }
    cubeIndexCount_ = (uint32_t)idxs.size();

    VkDeviceSize vs = sizeof(CubeVertex) * verts.size();
    createBuffer(vs, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 cubeVertexBuffer_, cubeVertexMemory_);
    void* d;
    vkMapMemory(ctx_.device, cubeVertexMemory_, 0, vs, 0, &d);
    memcpy(d, verts.data(), vs);
    vkUnmapMemory(ctx_.device, cubeVertexMemory_);

    VkDeviceSize is = sizeof(uint32_t) * idxs.size();
    createBuffer(is, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 cubeIndexBuffer_, cubeIndexMemory_);
    vkMapMemory(ctx_.device, cubeIndexMemory_, 0, is, 0, &d);
    memcpy(d, idxs.data(), is);
    vkUnmapMemory(ctx_.device, cubeIndexMemory_);
}

// ── SSBO ─────────────────────────────────────────────────────────────────────

void ComputeTest::createSSBO() {
    // Particle = { vec4 position, vec4 color } = 32 bytes per particle
    VkDeviceSize size = (VkDeviceSize)(gridSize_ * gridSize_) * 32;
    createBuffer(size,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 ssboBuffer_, ssboMemory_);
}

// ── 컴퓨트 파이프라인 ─────────────────────────────────────────────────────────

void ComputeTest::createComputePipeline() {
    // Descriptor set layout: binding 0 = SSBO
    VkDescriptorSetLayoutBinding binding{};
    binding.binding         = 0;
    binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = 1;
    dslCI.pBindings    = &binding;
    if (vkCreateDescriptorSetLayout(ctx_.device, &dslCI, nullptr, &computeDescLayout_) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute desc layout");

    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets       = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &poolSize;
    if (vkCreateDescriptorPool(ctx_.device, &poolCI, nullptr, &computeDescPool_) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute desc pool");

    VkDescriptorSetAllocateInfo dsAI{};
    dsAI.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAI.descriptorPool     = computeDescPool_;
    dsAI.descriptorSetCount = 1;
    dsAI.pSetLayouts        = &computeDescLayout_;
    if (vkAllocateDescriptorSets(ctx_.device, &dsAI, &computeDescSet_) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate compute desc set");

    VkDescriptorBufferInfo bufInfo{ ssboBuffer_, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = computeDescSet_;
    write.dstBinding      = 0;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo     = &bufInfo;
    vkUpdateDescriptorSets(ctx_.device, 1, &write, 0, nullptr);

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(ComputePushConstants);

    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount         = 1;
    plCI.pSetLayouts            = &computeDescLayout_;
    plCI.pushConstantRangeCount = 1;
    plCI.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(ctx_.device, &plCI, nullptr, &computePipelineLayout_) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute pipeline layout");

    auto compMod = loadShader(ctx_.basePath + "/shaders/compute_wave.comp.spv");
    VkPipelineShaderStageCreateInfo stageCI{};
    stageCI.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageCI.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCI.module = compMod;
    stageCI.pName  = "main";

    VkComputePipelineCreateInfo cpCI{};
    cpCI.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpCI.stage  = stageCI;
    cpCI.layout = computePipelineLayout_;
    if (vkCreateComputePipelines(ctx_.device, VK_NULL_HANDLE, 1, &cpCI, nullptr, &computePipeline_) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute pipeline");

    vkDestroyShaderModule(ctx_.device, compMod, nullptr);
}

// ── UBO ───────────────────────────────────────────────────────────────────────

void ComputeTest::createUBO() {
    createBuffer(sizeof(WaveUBO),
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 uboBuffer_, uboMemory_);
    vkMapMemory(ctx_.device, uboMemory_, 0, sizeof(WaveUBO), 0, &uboMapped_);
}

void ComputeTest::updateUBO() {
    if (autoRotate_)
        rotY_ = (float)glfwGetTime() * 0.3f;

    float cx = camDist_ * std::cos(camAngle_) * std::sin(rotY_);
    float cy = camDist_ * std::sin(camAngle_);
    float cz = camDist_ * std::cos(camAngle_) * std::cos(rotY_);
    Eigen::Vector3f camPos(cx, cy, cz);

    WaveUBO ubo{};
    ubo.view = lookAt(camPos, Eigen::Vector3f::Zero(), Eigen::Vector3f::UnitY());
    float aspect = (float)ctx_.extent.width / (float)ctx_.extent.height;
    ubo.proj = perspective(45.0f * (float)M_PI / 180.0f, aspect, 0.1f, 100.0f);
    ubo.lightDir  = Eigen::Vector4f(0.5f, 1.0f, 0.3f, 0.0f).normalized();
    ubo.cubeScale = cubeScale_;
    memcpy(uboMapped_, &ubo, sizeof(ubo));
}

// ── 그래픽스 파이프라인 ───────────────────────────────────────────────────────

void ComputeTest::createGraphicsPipeline() {
    // Descriptor set layout: binding 0 = UBO (vert + frag)
    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding         = 0;
    uboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = 1;
    dslCI.pBindings    = &uboBinding;
    if (vkCreateDescriptorSetLayout(ctx_.device, &dslCI, nullptr, &graphicsDescLayout_) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics desc layout");

    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets       = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &poolSize;
    if (vkCreateDescriptorPool(ctx_.device, &poolCI, nullptr, &graphicsDescPool_) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics desc pool");

    VkDescriptorSetAllocateInfo dsAI{};
    dsAI.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAI.descriptorPool     = graphicsDescPool_;
    dsAI.descriptorSetCount = 1;
    dsAI.pSetLayouts        = &graphicsDescLayout_;
    if (vkAllocateDescriptorSets(ctx_.device, &dsAI, &graphicsDescSet_) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate graphics desc set");

    VkDescriptorBufferInfo bufInfo{ uboBuffer_, 0, sizeof(WaveUBO) };
    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = graphicsDescSet_;
    write.dstBinding      = 0;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo     = &bufInfo;
    vkUpdateDescriptorSets(ctx_.device, 1, &write, 0, nullptr);

    auto vertMod = loadShader(ctx_.basePath + "/shaders/cube_wave.vert.spv");
    auto fragMod = loadShader(ctx_.basePath + "/shaders/cube_wave.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_VERTEX_BIT,   vertMod, "main", nullptr };
    stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                  VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "main", nullptr };

    // 버텍스 입력: binding 0 = 큐브 버텍스(per-vertex), binding 1 = SSBO(per-instance)
    VkVertexInputBindingDescription bindings[2]{};
    bindings[0] = { 0, sizeof(CubeVertex), VK_VERTEX_INPUT_RATE_VERTEX };
    bindings[1] = { 1, 32,                 VK_VERTEX_INPUT_RATE_INSTANCE }; // sizeof(Particle)

    VkVertexInputAttributeDescription attrs[4]{};
    attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,  0  }; // local pos
    attrs[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT,  12 }; // normal
    attrs[2] = { 2, 1, VK_FORMAT_R32G32B32_SFLOAT,  0  }; // instance pos xyz
    attrs[3] = { 3, 1, VK_FORMAT_R32G32B32_SFLOAT,  16 }; // instance color rgb

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 2;
    vi.pVertexBindingDescriptions      = bindings;
    vi.vertexAttributeDescriptionCount = 4;
    vi.pVertexAttributeDescriptions    = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{ 0, 0, (float)ctx_.extent.width, (float)ctx_.extent.height, 0.f, 1.f };
    VkRect2D   sc{ {0,0}, ctx_.extent };
    VkPipelineViewportStateCreateInfo vps{};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1; vps.pViewports = &vp;
    vps.scissorCount  = 1; vps.pScissors  = &sc;

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
    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments    = &blendAtt;

    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts    = &graphicsDescLayout_;
    if (vkCreatePipelineLayout(ctx_.device, &plCI, nullptr, &graphicsPipelineLayout_) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics pipeline layout");

    VkGraphicsPipelineCreateInfo gCI{};
    gCI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gCI.stageCount          = 2;
    gCI.pStages             = stages;
    gCI.pVertexInputState   = &vi;
    gCI.pInputAssemblyState = &ia;
    gCI.pViewportState      = &vps;
    gCI.pRasterizationState = &raster;
    gCI.pMultisampleState   = &ms;
    gCI.pColorBlendState    = &blend;
    gCI.layout              = graphicsPipelineLayout_;
    gCI.renderPass          = ctx_.renderPass;
    gCI.subpass             = 0;
    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &gCI, nullptr, &graphicsPipeline_) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics pipeline");

    vkDestroyShaderModule(ctx_.device, vertMod, nullptr);
    vkDestroyShaderModule(ctx_.device, fragMod, nullptr);
}

// ── 라이프사이클 ──────────────────────────────────────────────────────────────

void ComputeTest::onInit(const VulkanContext& ctx) {
    ctx_ = ctx;
    createCubeGeometry();
    createSSBO();
    createComputePipeline();
    createUBO();
    createGraphicsPipeline();
}

void ComputeTest::onCompute(VkCommandBuffer cmd) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            computePipelineLayout_, 0, 1, &computeDescSet_, 0, nullptr);

    float t = paused_ ? pausedTime_ : (float)glfwGetTime();
    ComputePushConstants pc{ t, amplitude_, frequency_, speed_, gridSize_, spacing_ };
    vkCmdPushConstants(cmd, computePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t groups = ((uint32_t)(gridSize_ * gridSize_) + 63u) / 64u;
    vkCmdDispatch(cmd, groups, 1, 1);

    // 배리어: 컴퓨트 쓰기 완료 → 버텍스 읽기 시작
    VkBufferMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask       = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer              = ssboBuffer_;
    barrier.offset              = 0;
    barrier.size                = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void ComputeTest::onRender(const RenderContext& ctx) {
    updateUBO();

    vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);
    vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            graphicsPipelineLayout_, 0, 1, &graphicsDescSet_, 0, nullptr);

    VkBuffer     vbufs[2]   = { cubeVertexBuffer_, ssboBuffer_ };
    VkDeviceSize offsets[2] = { 0, 0 };
    vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 2, vbufs, offsets);
    vkCmdBindIndexBuffer(ctx.commandBuffer, cubeIndexBuffer_, 0, VK_INDEX_TYPE_UINT32);

    uint32_t instanceCount = (uint32_t)(gridSize_ * gridSize_);
    vkCmdDrawIndexed(ctx.commandBuffer, cubeIndexCount_, instanceCount, 0, 0, 0);
}

void ComputeTest::onImGui() {
    ImGui::Begin("Compute Wave Cubes");
    ImGui::Text("Grid: %d x %d = %d cubes  |  Draw call: 1", gridSize_, gridSize_, gridSize_*gridSize_);

    ImGui::SeparatorText("Wave");
    ImGui::SliderFloat("Amplitude",  &amplitude_, 0.0f, 2.0f);
    ImGui::SliderFloat("Frequency",  &frequency_, 0.1f, 5.0f);
    ImGui::SliderFloat("Speed",      &speed_,     0.0f, 3.0f);

    ImGui::SeparatorText("Geometry");
    ImGui::SliderFloat("Cube Scale", &cubeScale_, 0.05f, 0.5f);

    ImGui::SeparatorText("Camera");
    ImGui::SliderFloat("Distance",   &camDist_,   2.0f, 25.0f);
    ImGui::SliderFloat("Elevation",  &camAngle_,  0.1f, 1.4f);
    ImGui::Checkbox("Auto Rotate",   &autoRotate_);
    if (!autoRotate_)
        ImGui::SliderFloat("Rotate Y", &rotY_, -3.14f, 3.14f);

    ImGui::SeparatorText("Controls");
    if (ImGui::Button(paused_ ? "Resume  [Space]" : "Pause  [Space]")) {
        paused_ = !paused_;
        if (paused_) pausedTime_ = (float)glfwGetTime();
    }
    ImGui::Text("R : reset camera");
    ImGui::End();
}

void ComputeTest::onKey(int key, int action, int mods) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_SPACE) {
        paused_ = !paused_;
        if (paused_) pausedTime_ = (float)glfwGetTime();
    }
    if (key == GLFW_KEY_R) {
        rotY_       = 0.0f;
        autoRotate_ = true;
    }
}

void ComputeTest::onCleanup() {
    vkDestroyPipeline(ctx_.device,            computePipeline_,       nullptr);
    vkDestroyPipelineLayout(ctx_.device,      computePipelineLayout_,  nullptr);
    vkDestroyDescriptorPool(ctx_.device,      computeDescPool_,        nullptr);
    vkDestroyDescriptorSetLayout(ctx_.device, computeDescLayout_,      nullptr);

    vkDestroyBuffer(ctx_.device,              ssboBuffer_,             nullptr);
    vkFreeMemory(ctx_.device,                 ssboMemory_,             nullptr);

    vkUnmapMemory(ctx_.device,                uboMemory_);
    vkDestroyBuffer(ctx_.device,              uboBuffer_,              nullptr);
    vkFreeMemory(ctx_.device,                 uboMemory_,              nullptr);

    vkDestroyPipeline(ctx_.device,            graphicsPipeline_,       nullptr);
    vkDestroyPipelineLayout(ctx_.device,      graphicsPipelineLayout_,  nullptr);
    vkDestroyDescriptorPool(ctx_.device,      graphicsDescPool_,        nullptr);
    vkDestroyDescriptorSetLayout(ctx_.device, graphicsDescLayout_,      nullptr);

    vkDestroyBuffer(ctx_.device,              cubeVertexBuffer_,       nullptr);
    vkFreeMemory(ctx_.device,                 cubeVertexMemory_,       nullptr);
    vkDestroyBuffer(ctx_.device,              cubeIndexBuffer_,        nullptr);
    vkFreeMemory(ctx_.device,                 cubeIndexMemory_,        nullptr);
}
