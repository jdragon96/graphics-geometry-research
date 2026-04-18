#include "TSDF.h"
#include <imgui.h>
#include <GLFW/glfw3.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>

// ── UBO (std140, matches blinnphong shaders) ──────────────────────────────────
struct alignas(16) TSDFUBO
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

// ── Camera math ───────────────────────────────────────────────────────────────
static Eigen::Matrix4f tsdfLookAt(Eigen::Vector3f eye, Eigen::Vector3f c, Eigen::Vector3f up)
{
    Eigen::Vector3f f = (c - eye).normalized();
    Eigen::Vector3f s = f.cross(up).normalized();
    Eigen::Vector3f u = s.cross(f);
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m(0, 0) = s.x();  m(0, 1) = s.y();  m(0, 2) = s.z();  m(0, 3) = -s.dot(eye);
    m(1, 0) = u.x();  m(1, 1) = u.y();  m(1, 2) = u.z();  m(1, 3) = -u.dot(eye);
    m(2, 0) = -f.x(); m(2, 1) = -f.y(); m(2, 2) = -f.z(); m(2, 3) = f.dot(eye);
    return m;
}

static Eigen::Matrix4f tsdfPerspective(float fovYRad, float aspect, float nearZ, float farZ)
{
    float f = 1.0f / std::tan(fovYRad * 0.5f);
    Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
    m(0, 0) = f / aspect;
    m(1, 1) = -f;
    m(2, 2) = farZ / (nearZ - farZ);
    m(2, 3) = farZ * nearZ / (nearZ - farZ);
    m(3, 2) = -1.0f;
    return m;
}

// ── File / shader helpers ─────────────────────────────────────────────────────
std::vector<char> TSDFFeature::readFile(const std::string &path)
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

VkShaderModule TSDFFeature::loadShader(const std::string &path)
{
    auto code = readFile(path);
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode    = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(ctx_.device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("failed to create shader module");
    return mod;
}

uint32_t TSDFFeature::findMemoryType(uint32_t filter, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(ctx_.physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("no suitable memory type");
}

void TSDFFeature::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags props,
                                VkBuffer &buf, VkDeviceMemory &mem)
{
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

// ── Scene SDF (smooth-union of two spheres) ───────────────────────────────────
float TSDFFeature::sceneSDF(float x, float y, float z) const
{
    float r = 0.5f, d = 0.35f;
    float s1 = std::sqrt((x - d) * (x - d) + y * y + z * z) - r;
    float s2 = std::sqrt((x + d) * (x + d) + y * y + z * z) - r;
    float k  = 0.25f;
    float h  = std::max(k - std::abs(s1 - s2), 0.0f) / k;
    return std::min(s1, s2) - h * h * k * 0.25f;
}

Eigen::Vector3f TSDFFeature::sdfNormal(float x, float y, float z) const
{
    const float e = 0.002f;
    return Eigen::Vector3f(
               sceneSDF(x + e, y, z) - sceneSDF(x - e, y, z),
               sceneSDF(x, y + e, z) - sceneSDF(x, y - e, z),
               sceneSDF(x, y, z + e) - sceneSDF(x, y, z - e))
        .normalized();
}

// TSDF value → RGB: outside(blue) → surface(green) → inside(red)
Eigen::Vector3f TSDFFeature::tsdfColor(float t) const
{
    t = std::clamp(t, -1.0f, 1.0f);
    if (t >= 0.0f)
        return Eigen::Vector3f(0.0f, 1.0f - 0.7f * t, 0.3f + 0.7f * t);  // green→blue
    else {
        float s = -t;
        return Eigen::Vector3f(s, 1.0f - 0.7f * s, 0.0f);                 // green→red
    }
}

// ── OpenVDB init / reset ──────────────────────────────────────────────────────
void TSDFFeature::initVDB()
{
    openvdb::initialize();
    tsdfGrid_   = openvdb::FloatGrid::create(1.0f);  // background: outside
    weightGrid_ = openvdb::FloatGrid::create(0.0f);  // background: unobserved
    tsdfGrid_->setName("tsdf");
    weightGrid_->setName("weight");
    auto xform = openvdb::math::Transform::createLinearTransform(double(voxelSize_));
    tsdfGrid_->setTransform(xform);
    weightGrid_->setTransform(xform->copy());
}

void TSDFFeature::resetTSDF()
{
    tsdfGrid_->clear();
    weightGrid_->clear();
    integratedFrames_ = 0;
    voxelDirty_       = false;

    vkDeviceWaitIdle(ctx_.device);
    auto destroy = [&](VkBuffer &b, VkDeviceMemory &m) {
        if (b != VK_NULL_HANDLE) { vkDestroyBuffer(ctx_.device, b, nullptr); b = VK_NULL_HANDLE; }
        if (m != VK_NULL_HANDLE) { vkFreeMemory(ctx_.device, m, nullptr);    m = VK_NULL_HANDLE; }
    };
    destroy(voxelBuffer_,   voxelMemory_);
    destroy(inputPtBuffer_, inputPtMemory_);
    destroy(accPtBuffer_,   accPtMemory_);
    voxelCount_   = 0;
    inputPtCount_ = 0;
    accPtCount_   = 0;
    accPts_.clear();
}

// ── Surface point cloud (precomputed once) ────────────────────────────────────
void TSDFFeature::buildSurfaceCloud()
{
    surfacePts_.clear();
    surfaceNormals_.clear();

    // Scan grid, keep voxels within half a voxel of the zero-level set
    const float threshold = voxelSize_ * 0.5f;
    for (int iz = -gridHalfDim_; iz < gridHalfDim_; iz++) {
        for (int iy = -gridHalfDim_; iy < gridHalfDim_; iy++) {
            for (int ix = -gridHalfDim_; ix < gridHalfDim_; ix++) {
                float wx = ix * voxelSize_;
                float wy = iy * voxelSize_;
                float wz = iz * voxelSize_;
                if (std::abs(sceneSDF(wx, wy, wz)) < threshold) {
                    surfacePts_.push_back({wx, wy, wz});
                    surfaceNormals_.push_back(sdfNormal(wx, wy, wz));
                }
            }
        }
    }
}

// ── Per-frame visible point cloud ─────────────────────────────────────────────
std::vector<Eigen::Vector3f>
TSDFFeature::filterVisiblePoints(Eigen::Vector3f camPos) const
{
    std::vector<Eigen::Vector3f> visible;
    visible.reserve(surfacePts_.size() / 2);
    for (size_t i = 0; i < surfacePts_.size(); i++) {
        Eigen::Vector3f toCamera = (camPos - surfacePts_[i]).normalized();
        if (surfaceNormals_[i].dot(toCamera) > 0.0f)
            visible.push_back(surfacePts_[i]);
    }
    return visible;
}

// ── Core TSDF integration: ray-based weighted update ─────────────────────────
//
// For each input point p (measured surface hit from camera C):
//   d = normalize(p - C),  D = |p - C|
//   March along the ray from D-trunc to D+trunc (step = voxelSize)
//   sdf_along_ray = D - t   (>0 = in front of surface, <0 = behind)
//   tsdf_new = clamp(sdf / trunc, -1, 1)
//   Weighted average update: tsdf = (W*tsdf + tsdf_new) / (W + 1)
//
void TSDFFeature::integratePointCloud(const std::vector<Eigen::Vector3f> &cloud,
                                       Eigen::Vector3f camPos)
{
    auto tsdfAcc   = tsdfGrid_->getAccessor();
    auto weightAcc = weightGrid_->getAccessor();

    for (const auto &noisyPt : cloud) {  // noise already applied by caller

        Eigen::Vector3f dir = (noisyPt - camPos);
        float D = dir.norm();
        if (D < 1e-5f) continue;
        dir /= D;

        float t = D - truncation_;
        float t_end = D + truncation_;

        while (t <= t_end) {
            Eigen::Vector3f v = camPos + t * dir;

            // World → voxel coordinate
            int ix = (int)std::round(v.x() / voxelSize_);
            int iy = (int)std::round(v.y() / voxelSize_);
            int iz = (int)std::round(v.z() / voxelSize_);

            if (std::abs(ix) < gridHalfDim_ &&
                std::abs(iy) < gridHalfDim_ &&
                std::abs(iz) < gridHalfDim_)
            {
                float sdf_ray = D - t;
                float tsdf_new = std::clamp(sdf_ray / truncation_, -1.0f, 1.0f);

                openvdb::Coord c(ix, iy, iz);
                float w_old    = weightAcc.getValue(c);
                float tsdf_old = (w_old > 0.0f) ? tsdfAcc.getValue(c) : tsdf_new;

                float tsdf_fused = (w_old * tsdf_old + tsdf_new) / (w_old + 1.0f);
                float w_fused    = std::min(w_old + 1.0f, 100.0f);

                tsdfAcc.setValue(c, tsdf_fused);
                weightAcc.setValue(c, w_fused);
            }
            t += voxelSize_;
        }
    }
}

void TSDFFeature::integrateFrame(int frameIndex)
{
    float theta = frameIndex * (2.0f * float(M_PI) / totalFrames_);
    float elev  = 0.3f;
    float R     = 3.0f;
    Eigen::Vector3f camPos(
        std::sin(theta) * R * std::cos(elev),
        std::sin(elev)  * R,
        std::cos(theta) * R * std::cos(elev));

    // Filter to camera-visible surface points
    auto cleanPts = filterVisiblePoints(camPos);

    // Apply noise + random dropout to simulate a real depth sensor
    static std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> gaussian(0.0f, sensorNoise_);
    std::uniform_real_distribution<float> dropout(0.0f, 1.0f);

    std::vector<Eigen::Vector3f> noisyPts;
    noisyPts.reserve(cleanPts.size());
    for (const auto &p : cleanPts) {
        if (dropout(rng) < 0.15f) continue;  // ~15% random dropout breaks regularity
        noisyPts.push_back(p + Eigen::Vector3f(
            gaussian(rng), gaussian(rng), gaussian(rng)));
    }

    // Integrate into TSDF
    integratePointCloud(noisyPts, camPos);

    // Current frame input cloud (yellow-white) — show noisy points
    {
        std::vector<Vertex> vtx;
        vtx.reserve(noisyPts.size());
        for (const auto &p : noisyPts) {
            Vertex v;
            v.position = p;
            v.color    = Eigen::Vector3f(1.0f, 1.0f, 0.5f);
            v.normal   = Eigen::Vector3f(0, 1, 0);
            vtx.push_back(v);
        }
        uploadPointBuffer(inputPtBuffer_, inputPtMemory_, inputPtCount_, vtx);
    }

    // Accumulate raw geometry — noisy points, never discarded until Reset
    for (const auto &p : noisyPts)
        accPts_.push_back(p);

    {
        std::vector<Vertex> vtx;
        vtx.reserve(accPts_.size());
        for (const auto &p : accPts_) {
            Vertex v;
            v.position = p;
            v.color    = Eigen::Vector3f(0.85f, 0.85f, 0.85f);  // light grey
            v.normal   = Eigen::Vector3f(0, 1, 0);
            vtx.push_back(v);
        }
        uploadPointBuffer(accPtBuffer_, accPtMemory_, accPtCount_, vtx);
    }

    integratedFrames_++;
    voxelDirty_ = true;
}

// ── Voxel visualization buffer ────────────────────────────────────────────────
void TSDFFeature::updateVoxelBuffer()
{
    voxelDirty_ = false;

    std::vector<Vertex> pts;
    pts.reserve(8192);

    auto weightAcc = weightGrid_->getConstAccessor();
    auto tsdfAcc   = tsdfGrid_->getConstAccessor();

    for (auto iter = weightGrid_->cbeginValueOn(); iter; ++iter) {
        float w = iter.getValue();
        if (w <= 0.0f) continue;

        openvdb::Coord c = iter.getCoord();
        float t = tsdfAcc.getValue(c);

        Eigen::Vector3f col;
        if (voxelColorMode_ == 0)
            col = tsdfColor(t);
        else {
            float wt = std::min(w / 20.0f, 1.0f);
            col = Eigen::Vector3f(1.0f - wt, 0.8f + 0.2f * wt, wt);
        }

        Vertex v;
        v.position = Eigen::Vector3f(c.x() * voxelSize_, c.y() * voxelSize_, c.z() * voxelSize_);
        v.color    = col;
        v.normal   = Eigen::Vector3f(0, 1, 0);
        pts.push_back(v);
    }

    uploadPointBuffer(voxelBuffer_, voxelMemory_, voxelCount_, pts);
}

void TSDFFeature::uploadPointBuffer(VkBuffer &buf, VkDeviceMemory &mem, uint32_t &count,
                                     const std::vector<Vertex> &pts)
{
    vkDeviceWaitIdle(ctx_.device);
    if (buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(ctx_.device, buf, nullptr); buf = VK_NULL_HANDLE;
    }
    if (mem != VK_NULL_HANDLE) {
        vkFreeMemory(ctx_.device, mem, nullptr);    mem = VK_NULL_HANDLE;
    }
    count = 0;
    if (pts.empty()) return;

    VkDeviceSize sz = sizeof(Vertex) * pts.size();
    createBuffer(sz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 buf, mem);
    void *data;
    vkMapMemory(ctx_.device, mem, 0, sz, 0, &data);
    memcpy(data, pts.data(), sz);
    vkUnmapMemory(ctx_.device, mem);
    count = static_cast<uint32_t>(pts.size());
}

// ── Vulkan setup ──────────────────────────────────────────────────────────────
void TSDFFeature::createDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding b{};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 1;
    ci.pBindings    = &b;
    if (vkCreateDescriptorSetLayout(ctx_.device, &ci, nullptr, &descLayout_) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor set layout");
}

void TSDFFeature::createDescriptorPool()
{
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
    VkDescriptorPoolCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets       = 1;
    ci.poolSizeCount = 1;
    ci.pPoolSizes    = &ps;
    if (vkCreateDescriptorPool(ctx_.device, &ci, nullptr, &descPool_) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor pool");
}

void TSDFFeature::createUBO()
{
    createBuffer(sizeof(TSDFUBO),
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 uboBuffer_, uboMemory_);
    vkMapMemory(ctx_.device, uboMemory_, 0, sizeof(TSDFUBO), 0, &uboMapped_);
}

void TSDFFeature::createDescriptorSet()
{
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = descPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &descLayout_;
    if (vkAllocateDescriptorSets(ctx_.device, &ai, &descSet_) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate descriptor set");
    VkDescriptorBufferInfo bi{uboBuffer_, 0, sizeof(TSDFUBO)};
    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet          = descSet_;
    w.dstBinding      = 0;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w.descriptorCount = 1;
    w.pBufferInfo     = &bi;
    vkUpdateDescriptorSets(ctx_.device, 1, &w, 0, nullptr);
}

void TSDFFeature::createPipeline()
{
    // tsdf_voxel shaders: unlit, just position + color, POINT_LIST topology
    auto vertMod = loadShader(ctx_.basePath + "/shaders/tsdf_voxel.vert.spv");
    auto fragMod = loadShader(ctx_.basePath + "/shaders/tsdf_voxel.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod; stages[1].pName = "main";

    auto bindDesc = Vertex::getBindingDescription();
    auto attrDesc = Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &bindDesc;
    vi.vertexAttributeDescriptionCount = (uint32_t)attrDesc.size();
    vi.pVertexAttributeDescriptions    = attrDesc.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkViewport viewport{0, 0, (float)ctx_.extent.width, (float)ctx_.extent.height, 0.f, 1.f};
    VkRect2D   scissor{{0, 0}, ctx_.extent};
    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.pViewports = &viewport;
    vp.scissorCount  = 1; vp.pScissors  = &scissor;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth   = 1.0f;
    raster.cullMode    = VK_CULL_MODE_NONE;
    raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

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

    VkPipelineLayoutCreateInfo layoutCI{};
    layoutCI.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.setLayoutCount = 1;
    layoutCI.pSetLayouts    = &descLayout_;
    if (vkCreatePipelineLayout(ctx_.device, &layoutCI, nullptr, &pipelineLayout_) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout");

    VkGraphicsPipelineCreateInfo pCI{};
    pCI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pCI.stageCount          = 2;
    pCI.pStages             = stages;
    pCI.pVertexInputState   = &vi;
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

void TSDFFeature::updateUBO()
{
    if (autoRotate_)
        rotY_ = float(glfwGetTime()) * 0.5f;

    float cx = std::sin(rotY_) * std::cos(camAngleV_) * camDist_;
    float cy = std::sin(camAngleV_) * camDist_;
    float cz = std::cos(rotY_) * std::cos(camAngleV_) * camDist_;

    TSDFUBO ubo{};
    ubo.model      = Eigen::Matrix4f::Identity();
    ubo.view       = tsdfLookAt({cx, cy, cz}, Eigen::Vector3f::Zero(), Eigen::Vector3f::UnitY());
    float aspect   = float(ctx_.extent.width) / float(ctx_.extent.height);
    ubo.proj       = tsdfPerspective(fovDeg_ * float(M_PI) / 180.0f, aspect, 0.1f, 100.0f);
    ubo.lightPos   = Eigen::Vector4f(lightPos_[0],   lightPos_[1],   lightPos_[2],   1.0f);
    ubo.viewPos    = Eigen::Vector4f(cx, cy, cz, 1.0f);
    ubo.lightColor = Eigen::Vector4f(lightColor_[0], lightColor_[1], lightColor_[2], 1.0f);
    ubo.ambientStrength  = ambient_;
    ubo.specularStrength = specular_;
    ubo.shininess        = shininess_;
    memcpy(uboMapped_, &ubo, sizeof(ubo));
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────
void TSDFFeature::onInit(const VulkanContext &ctx)
{
    ctx_ = ctx;
    initVDB();
    buildSurfaceCloud();  // precompute surface points once
    createDescriptorSetLayout();
    createDescriptorPool();
    createUBO();
    createDescriptorSet();
    createPipeline();
}

void TSDFFeature::onRender(const RenderContext &ctx)
{
    float now = float(glfwGetTime());
    float dt  = (lastTime_ > 0.0f) ? (now - lastTime_) : 0.0f;
    lastTime_ = now;

    if (autoIntegrate_ && integratedFrames_ < totalFrames_) {
        integrationTimer_ += dt;
        if (integrationTimer_ >= integrationRate_) {
            integrationTimer_ = 0.0f;
            integrateFrame(integratedFrames_);
        }
    }

    if (voxelDirty_)
        updateVoxelBuffer();

    updateUBO();

    // Bind pipeline + descriptor once, draw both buffers
    vkCmdBindPipeline(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout_, 0, 1, &descSet_, 0, nullptr);

    VkDeviceSize offset = 0;
    if (showTSDFVoxels_ && voxelCount_ > 0) {
        vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 1, &voxelBuffer_, &offset);
        vkCmdDraw(ctx.commandBuffer, voxelCount_, 1, 0, 0);
    }
    if (showInputCloud_ && inputPtCount_ > 0) {
        vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 1, &inputPtBuffer_, &offset);
        vkCmdDraw(ctx.commandBuffer, inputPtCount_, 1, 0, 0);
    }
    if (showAccCloud_ && accPtCount_ > 0) {
        vkCmdBindVertexBuffers(ctx.commandBuffer, 0, 1, &accPtBuffer_, &offset);
        vkCmdDraw(ctx.commandBuffer, accPtCount_, 1, 0, 0);
    }
}

void TSDFFeature::onImGui()
{
    ImGui::Begin("TSDF Visualization");

    ImGui::SeparatorText("State");
    ImGui::Text("Surface samples: %zu", surfacePts_.size());
    ImGui::Text("Frames: %d / %d", integratedFrames_, totalFrames_);
    ImGui::Text("Active voxels: %u  |  Input pts: %u  |  Acc pts: %u",
                voxelCount_, inputPtCount_, accPtCount_);

    ImGui::SeparatorText("Integration");
    bool canStep = integratedFrames_ < totalFrames_;
    if (ImGui::Button("Step (+1 frame)") && canStep) {
        integrateFrame(integratedFrames_);
        updateVoxelBuffer();
    }
    ImGui::SameLine();
    if (ImGui::Button("Integrate All") && canStep) {
        while (integratedFrames_ < totalFrames_)
            integrateFrame(integratedFrames_);
        updateVoxelBuffer();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) resetTSDF();

    ImGui::Checkbox("Auto Integrate", &autoIntegrate_);
    ImGui::SliderFloat("Rate (s)", &integrationRate_, 0.05f, 2.0f);
    ImGui::SliderFloat("Sensor Noise (m)", &sensorNoise_, 0.0f, 0.02f);

    ImGui::SeparatorText("TSDF Params");
    if (ImGui::SliderFloat("Truncation (m)", &truncation_, 0.04f, 0.4f)) resetTSDF();
    ImGui::SliderInt("Total Frames", &totalFrames_, 4, 72);

    ImGui::SeparatorText("Visualization");
    ImGui::Checkbox("TSDF Voxels",       &showTSDFVoxels_);
    ImGui::SameLine();
    ImGui::Checkbox("Current Frame",     &showInputCloud_);
    ImGui::SameLine();
    ImGui::Checkbox("Accumulated Cloud", &showAccCloud_);
    ImGui::TextDisabled("Acc cloud = all observed raw input points (light grey)");
    const char *colorModes[] = {"TSDF value (blue/green/red)", "Weight (yellow→cyan)"};
    ImGui::Combo("Voxel Color", &voxelColorMode_, colorModes, 2);
    if (ImGui::IsItemEdited()) updateVoxelBuffer();

    ImGui::TextDisabled("TSDF: blue=outside  green=surface  red=inside");

    ImGui::SeparatorText("Camera");
    ImGui::SliderFloat("Distance",  &camDist_,    1.0f, 10.0f);
    ImGui::SliderFloat("Elevation", &camAngleV_, -1.0f,  1.0f);
    ImGui::Checkbox  ("Auto Rotate", &autoRotate_);
    if (!autoRotate_)
        ImGui::SliderFloat("Rotate Y", &rotY_, -3.14f, 3.14f);

    ImGui::Separator();
    ImGui::Text("[Space] Step  [A] Toggle Auto  [R] Reset");
    ImGui::End();
}

void TSDFFeature::onKey(int key, int action, int mods)
{
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_SPACE && integratedFrames_ < totalFrames_) {
        integrateFrame(integratedFrames_);
        updateVoxelBuffer();
    }
    if (key == GLFW_KEY_R) resetTSDF();
    if (key == GLFW_KEY_A) autoIntegrate_ = !autoIntegrate_;
}

void TSDFFeature::onCleanup()
{
    vkDeviceWaitIdle(ctx_.device);
    vkUnmapMemory(ctx_.device, uboMemory_);
    auto destroy = [&](VkBuffer b, VkDeviceMemory m) {
        if (b) vkDestroyBuffer(ctx_.device, b, nullptr);
        if (m) vkFreeMemory(ctx_.device, m, nullptr);
    };
    destroy(uboBuffer_,     uboMemory_);
    destroy(voxelBuffer_,   voxelMemory_);
    destroy(inputPtBuffer_, inputPtMemory_);
    destroy(accPtBuffer_,   accPtMemory_);
    vkDestroyPipeline(ctx_.device, pipeline_, nullptr);
    vkDestroyPipelineLayout(ctx_.device, pipelineLayout_, nullptr);
    vkDestroyDescriptorPool(ctx_.device, descPool_, nullptr);
    vkDestroyDescriptorSetLayout(ctx_.device, descLayout_, nullptr);
}
