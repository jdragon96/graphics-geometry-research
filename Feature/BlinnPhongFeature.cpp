#include "BlinnPhongFeature.h"
#include <imgui.h>
#include <GLFW/glfw3.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cmath>

// ── UBO (std140 layout) ───────────────────────────────────────────────────────
struct alignas(16) UBO
{
    Eigen::Matrix4f model;
    Eigen::Matrix4f view;
    Eigen::Matrix4f proj;
    Eigen::Vector4f lightPos;
    Eigen::Vector4f viewPos;
    Eigen::Vector4f lightColor;
    float ambientStrength;
    float specularStrength;
    float shininess;
    float _pad = 0;
};

// ── Math helpers ──────────────────────────────────────────────────────────────

static Eigen::Matrix4f lookAt(Eigen::Vector3f eye, Eigen::Vector3f center, Eigen::Vector3f up)
{
    Eigen::Vector3f f = (center - eye).normalized();
    Eigen::Vector3f s = f.cross(up).normalized();
    Eigen::Vector3f u = s.cross(f);
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0, 0) = s.x();
    m(0, 1) = s.y();
    m(0, 2) = s.z();
    m(0, 3) = -s.dot(eye);
    m(1, 0) = u.x();
    m(1, 1) = u.y();
    m(1, 2) = u.z();
    m(1, 3) = -u.dot(eye);
    m(2, 0) = -f.x();
    m(2, 1) = -f.y();
    m(2, 2) = -f.z();
    m(2, 3) = f.dot(eye);
    m(3, 0) = 0;
    m(3, 1) = 0;
    m(3, 2) = 0;
    m(3, 3) = 1;
    return m;
}

// Vulkan perspective: right-handed, depth [0,1], Y-flipped
static Eigen::Matrix4f perspective(float fovYRad, float aspect, float nearZ, float farZ)
{
    float f = 1.0f / std::tan(fovYRad * 0.5f);
    Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
    m(0, 0) = f / aspect;
    m(1, 1) = -f; // flip Y for Vulkan
    m(2, 2) = farZ / (nearZ - farZ);
    m(2, 3) = farZ * nearZ / (nearZ - farZ);
    m(3, 2) = -1.0f;
    return m;
}

// ── File / Shader helpers ─────────────────────────────────────────────────────

std::vector<char> BlinnPhongFeature::readFile(const std::string &path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("cannot open shader: " + path);
    size_t size = (size_t)file.tellg();
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), size);
    return buf;
}

VkShaderModule BlinnPhongFeature::loadShader(const std::string &path)
{
    auto code = readFile(path);
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(ctx_.device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("failed to create shader module");
    return mod;
}

uint32_t BlinnPhongFeature::findMemoryType(uint32_t filter, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(ctx_.physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("no suitable memory type");
}

void BlinnPhongFeature::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                     VkMemoryPropertyFlags props,
                                     VkBuffer &buf, VkDeviceMemory &mem)
{
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx_.device, &bci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("failed to create buffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx_.device, buf, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    if (vkAllocateMemory(ctx_.device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate memory");

    vkBindBufferMemory(ctx_.device, buf, mem, 0);
}

// ── Sphere generation ─────────────────────────────────────────────────────────

void BlinnPhongFeature::generateSphere(int stacks, int sectors)
{
    vertices_.clear();
    indices_.clear();

    for (int i = 0; i <= stacks; i++)
    {
        float phi = static_cast<float>(M_PI) / 2.0f - i * static_cast<float>(M_PI) / stacks;
        float y = std::sin(phi);
        float r = std::cos(phi);

        for (int j = 0; j <= sectors; j++)
        {
            float theta = j * 2.0f * static_cast<float>(M_PI) / sectors;
            float x = r * std::cos(theta);
            float z = r * std::sin(theta);

            Vertex v;
            v.position = {x, y, z};
            v.normal = {x, y, z}; // unit sphere: normal == position
            v.color = {objectColor_[0], objectColor_[1], objectColor_[2]};
            vertices_.push_back(v);
        }
    }

    for (int i = 0; i < stacks; i++)
    {
        for (int j = 0; j < sectors; j++)
        {
            uint32_t p1 = i * (sectors + 1) + j;
            uint32_t p2 = p1 + (sectors + 1);
            indices_.push_back(p1);
            indices_.push_back(p2);
            indices_.push_back(p1 + 1);
            indices_.push_back(p1 + 1);
            indices_.push_back(p2);
            indices_.push_back(p2 + 1);
        }
    }
}

// ── Descriptor set layout ─────────────────────────────────────────────────────

void BlinnPhongFeature::createDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 1;
    ci.pBindings = &binding;

    if (vkCreateDescriptorSetLayout(ctx_.device, &ci, nullptr, &descriptorSetLayout_) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor set layout");
}

// ── Descriptor pool & set ─────────────────────────────────────────────────────

void BlinnPhongFeature::createDescriptorPool()
{
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = 1;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(ctx_.device, &ci, nullptr, &descriptorPool_) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor pool");
}

void BlinnPhongFeature::createUniformBuffer()
{
    VkDeviceSize size = sizeof(UBO);
    createBuffer(size,
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 uniformBuffer_, uniformMemory_);
    vkMapMemory(ctx_.device, uniformMemory_, 0, size, 0, &uniformMapped_);
}

void BlinnPhongFeature::createDescriptorSet()
{
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descriptorPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &descriptorSetLayout_;
    if (vkAllocateDescriptorSets(ctx_.device, &ai, &descriptorSet_) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate descriptor set");

    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = uniformBuffer_;
    bufInfo.offset = 0;
    bufInfo.range = sizeof(UBO);

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet_;
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &bufInfo;
    vkUpdateDescriptorSets(ctx_.device, 1, &write, 0, nullptr);
}

// ── Vertex / Index buffers ────────────────────────────────────────────────────

void BlinnPhongFeature::createVertexBuffer()
{
    VkDeviceSize size = sizeof(Vertex) * vertices_.size();
    createBuffer(size,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vertexBuffer_, vertexMemory_);
    void *data;
    vkMapMemory(ctx_.device, vertexMemory_, 0, size, 0, &data);
    memcpy(data, vertices_.data(), size);
    vkUnmapMemory(ctx_.device, vertexMemory_);
}

void BlinnPhongFeature::createIndexBuffer()
{
    VkDeviceSize size = sizeof(uint32_t) * indices_.size();
    createBuffer(size,
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 indexBuffer_, indexMemory_);
    void *data;
    vkMapMemory(ctx_.device, indexMemory_, 0, size, 0, &data);
    memcpy(data, indices_.data(), size);
    vkUnmapMemory(ctx_.device, indexMemory_);
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

void BlinnPhongFeature::createPipeline()
{
    auto vertMod = loadShader(ctx_.basePath + "/shaders/blinnphong.vert.spv");
    auto fragMod = loadShader(ctx_.basePath + "/shaders/blinnphong.frag.spv");

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

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
    vertexInput.pVertexAttributeDescriptions = attrDesc.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{0, 0, (float)ctx_.extent.width, (float)ctx_.extent.height, 0.f, 1.f};
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
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

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
    pCI.pVertexInputState = &vertexInput;
    pCI.pInputAssemblyState = &ia;
    pCI.pViewportState = &vp;
    pCI.pRasterizationState = &raster;
    pCI.pMultisampleState = &ms;
    pCI.pColorBlendState = &blend;
    pCI.layout = pipelineLayout_;
    pCI.renderPass = ctx_.renderPass;
    pCI.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &pCI, nullptr, &pipeline_) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline");

    vkDestroyShaderModule(ctx_.device, vertMod, nullptr);
    vkDestroyShaderModule(ctx_.device, fragMod, nullptr);
}

// ── UBO update (every frame) ──────────────────────────────────────────────────

void BlinnPhongFeature::updateUniformBuffer()
{
    if (autoRotate_)
        rotY_ = static_cast<float>(glfwGetTime()) * 0.8f;

    Eigen::Vector3f camPos(0.0f, 0.0f, camDist_);

    UBO ubo{};
    // Model: rotate around Y
    Eigen::AngleAxisf rot(rotY_, Eigen::Vector3f::UnitY());
    ubo.model = Eigen::Matrix4f::Identity();
    ubo.model.block<3, 3>(0, 0) = rot.toRotationMatrix();

    ubo.view = lookAt(camPos, Eigen::Vector3f::Zero(), Eigen::Vector3f::UnitY());

    float aspect = (float)ctx_.extent.width / (float)ctx_.extent.height;
    ubo.proj = perspective(fovDeg_ * static_cast<float>(M_PI) / 180.0f, aspect, 0.1f, 100.0f);

    ubo.lightPos = Eigen::Vector4f(lightPos_[0], lightPos_[1], lightPos_[2], 1.0f);
    ubo.viewPos = Eigen::Vector4f(camPos.x(), camPos.y(), camPos.z(), 1.0f);
    ubo.lightColor = Eigen::Vector4f(lightColor_[0], lightColor_[1], lightColor_[2], 1.0f);

    ubo.ambientStrength = ambient_;
    ubo.specularStrength = specular_;
    ubo.shininess = shininess_;

    memcpy(uniformMapped_, &ubo, sizeof(ubo));
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

void BlinnPhongFeature::onInit(const VulkanContext &ctx)
{
    ctx_ = ctx;
    generateSphere(36, 36);
    createDescriptorSetLayout();
    createDescriptorPool();
    createUniformBuffer();
    createDescriptorSet();
    createVertexBuffer();
    createIndexBuffer();
    createPipeline();
}

void BlinnPhongFeature::onRender(const RenderContext &ctx)
{
    updateUniformBuffer();

    vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 1, &vertexBuffer_, &offset);
    vkCmdBindIndexBuffer(ctx.commandBuffer, indexBuffer_, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(ctx.commandBuffer, static_cast<uint32_t>(indices_.size()), 1, 0, 0, 0);
}

void BlinnPhongFeature::onImGui()
{
    ImGui::Begin("BlinnPhong Sphere");

    ImGui::SeparatorText("Light");
    ImGui::DragFloat3("Position##L", lightPos_, 0.05f, -10.f, 10.f);
    ImGui::ColorEdit3("Color##L", lightColor_);

    ImGui::SeparatorText("Material");
    ImGui::ColorEdit3("Object Color", objectColor_);
    ImGui::SliderFloat("Ambient", &ambient_, 0.0f, 1.0f);
    ImGui::SliderFloat("Specular", &specular_, 0.0f, 1.0f);
    ImGui::SliderFloat("Shininess", &shininess_, 1.0f, 256.0f);

    ImGui::SeparatorText("Camera");
    ImGui::SliderFloat("Distance", &camDist_, 1.0f, 10.0f);
    ImGui::SliderFloat("FOV", &fovDeg_, 20.0f, 120.0f);

    ImGui::SeparatorText("Rotation");
    ImGui::Checkbox("Auto Rotate", &autoRotate_);
    if (!autoRotate_)
        ImGui::SliderFloat("Rotate Y", &rotY_, -3.14f, 3.14f);

    ImGui::Separator();
    ImGui::Text("Vertices: %zu  Indices: %zu", vertices_.size(), indices_.size());
    ImGui::Text("Press R: reset rotation");
    ImGui::End();
}

void BlinnPhongFeature::onKey(int key, int action, int mods)
{
    if (key == GLFW_KEY_R && action == GLFW_PRESS)
    {
        rotY_ = 0.0f;
        autoRotate_ = true;
    }
}

void BlinnPhongFeature::onCleanup()
{
    vkUnmapMemory(ctx_.device, uniformMemory_);
    vkDestroyBuffer(ctx_.device, uniformBuffer_, nullptr);
    vkFreeMemory(ctx_.device, uniformMemory_, nullptr);
    vkDestroyBuffer(ctx_.device, vertexBuffer_, nullptr);
    vkFreeMemory(ctx_.device, vertexMemory_, nullptr);
    vkDestroyBuffer(ctx_.device, indexBuffer_, nullptr);
    vkFreeMemory(ctx_.device, indexMemory_, nullptr);
    vkDestroyPipeline(ctx_.device, pipeline_, nullptr);
    vkDestroyPipelineLayout(ctx_.device, pipelineLayout_, nullptr);
    vkDestroyDescriptorPool(ctx_.device, descriptorPool_, nullptr);
    vkDestroyDescriptorSetLayout(ctx_.device, descriptorSetLayout_, nullptr);
}
