#include "BucketedHash.h"
#include <imgui.h>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cstdlib>

// ─────────────────────────────────────────────────────────────────────────────
//  Embedded GLSL
//  Descriptor layout (all shaders share the same VkDescriptorSet):
//    binding 0 → hash table  Entry[]
//    binding 1 → point batch vec4[]
//    binding 2 → counter     uint
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char* kClearComp = R"GLSL(
#version 450
layout(local_size_x = 64) in;
layout(push_constant) uniform PC { uint total; float _f; uint _b; uint _c; } pc;
struct Entry { int kx, ky, kz, val; };
layout(set=0, binding=0) buffer HT { Entry e[]; } ht;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    ht.e[i] = Entry(0x7FFFFFFF, 0, 0, -1);
}
)GLSL";

static constexpr const char* kInsertComp = R"GLSL(
#version 450
layout(local_size_x = 64) in;
layout(push_constant) uniform PC {
    uint  numPoints;
    float voxelSize;
    uint  numBuckets;
    uint  _pad;
} pc;
struct Entry { int kx, ky, kz, val; };
layout(set=0, binding=0) buffer HT { Entry e[]; } ht;
layout(set=0, binding=1) readonly buffer Pts { vec4 pts[]; };

const int BUCKET_SIZE = 4;
const int EMPTY = 0x7FFFFFFF;

// Spatial hash — three coprime primes, good distribution on integer grids
uint hash3(ivec3 k) {
    return (uint(k.x) * 73856093u ^ uint(k.y) * 19349663u ^ uint(k.z) * 83492791u)
           % pc.numBuckets;
}

void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= pc.numPoints) return;

    ivec3 key = ivec3(floor(pts[tid].xyz / pc.voxelSize));
    uint  base = hash3(key) * uint(BUCKET_SIZE);

    // Linear probe within the bucket (cache-friendly: 4 × 16B = 64B = 1 cache line)
    for (int s = 0; s < BUCKET_SIZE; s++) {
        uint slot = base + uint(s);

        // Already voxelised → nothing to do
        if (ht.e[slot].kx == key.x &&
            ht.e[slot].ky == key.y &&
            ht.e[slot].kz == key.z) return;

        // Race-free slot claim via CAS on kx (sentinel = EMPTY)
        int prev = atomicCompSwap(ht.e[slot].kx, EMPTY, key.x);
        if (prev == EMPTY) {
            ht.e[slot].ky  = key.y;
            ht.e[slot].kz  = key.z;
            atomicExchange(ht.e[slot].val, int(tid));
            return;
        }
    }
    // Bucket full — this point is dropped (rare at normal load factors)
}
)GLSL";

static constexpr const char* kCountComp = R"GLSL(
#version 450
layout(local_size_x = 64) in;
layout(push_constant) uniform PC { uint total; float _f; uint _b; uint _c; } pc;
struct Entry { int kx, ky, kz, val; };
layout(set=0, binding=0) readonly buffer HT { Entry e[]; } ht;
layout(set=0, binding=2)          buffer Ctr { uint n; }  ctr;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    if (ht.e[i].val >= 0) atomicAdd(ctr.n, 1u);
}
)GLSL";

// ── Render: point-cloud view of occupied voxels ───────────────────────────────
// No vertex buffer — gl_VertexIndex indexes directly into the SSBO.
// Empty slots are clipped by placing them far outside NDC.
static constexpr const char* kRenderVert = R"GLSL(
#version 450

layout(push_constant) uniform PC {
    mat4  mvp;
    float voxelSize;
    float _pad[3];
} pc;

// Raw-int view so we can read without declaring the struct (reinterpretation safe in std430)
layout(set=0, binding=0) readonly buffer HT { int raw[]; } ht;

layout(location = 0) out vec3 fragColor;

const int EMPTY = 0x7FFFFFFF;

void main() {
    int base = gl_VertexIndex * 4;
    int kx   = ht.raw[base + 0];
    int ky   = ht.raw[base + 1];
    int kz   = ht.raw[base + 2];
    int val  = ht.raw[base + 3];

    if (kx == EMPTY || val < 0) {
        gl_Position  = vec4(10.0, 10.0, 10.0, 1.0); // outside clip → discarded
        gl_PointSize = 0.0;
        fragColor    = vec3(0.0);
        return;
    }

    // Voxel centre in world space
    vec3 centre = (vec3(float(kx), float(ky), float(kz)) + 0.5) * pc.voxelSize;
    gl_Position  = pc.mvp * vec4(centre, 1.0);
    gl_PointSize = 3.0;

    // Colour by spatial hash for easy visual inspection
    uint h  = uint(kx) * 73856093u ^ uint(ky) * 19349663u ^ uint(kz) * 83492791u;
    float t = float(h & 0xFFFFu) / 65535.0;
    fragColor = vec3(fract(t), fract(t * 2.0 + 0.33), fract(t * 4.0 + 0.67));
}
)GLSL";

static constexpr const char* kRenderFrag = R"GLSL(
#version 450
layout(location = 0) in  vec3 fragColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(fragColor, 1.0); }
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
//  Camera math  (Eigen, column-major → matches GLSL mat4 layout directly)
// ─────────────────────────────────────────────────────────────────────────────

Eigen::Matrix4f BucketedHash::computeMVP() const
{
    // ── View: standard look-at, camera orbits origin ──────────────────────────
    float cx = std::cos(elevation_), sx = std::sin(elevation_);
    float cy = std::cos(azimuth_),   sy = std::sin(azimuth_);
    Eigen::Vector3f eye(camDist_ * cx * sy, camDist_ * sx, camDist_ * cx * cy);
    Eigen::Vector3f up(0.f, 1.f, 0.f);

    Eigen::Vector3f f = (-eye).normalized();           // forward = toward origin
    Eigen::Vector3f r = f.cross(up).normalized();
    Eigen::Vector3f u = r.cross(f);

    Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
    V.row(0) << r.x(), r.y(), r.z(), -r.dot(eye);
    V.row(1) << u.x(), u.y(), u.z(), -u.dot(eye);
    V.row(2) <<-f.x(),-f.y(),-f.z(),  f.dot(eye);

    // ── Perspective (Vulkan: Y-down NDC, depth [0,1]) ─────────────────────────
    float fovY   = 60.f * static_cast<float>(M_PI) / 180.f;
    float aspect = static_cast<float>(ctx_.extent.width) / ctx_.extent.height;
    float n = 0.01f, fa = 100.f;
    float th = std::tan(fovY * 0.5f);

    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
    P(0,0) =  1.f / (aspect * th);
    P(1,1) = -1.f / th;             // flip Y for Vulkan
    P(2,2) =  fa  / (n - fa);
    P(2,3) =  (fa * n) / (n - fa);
    P(3,2) = -1.f;

    return P * V;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Sphere surface point generation (CPU)
//  Simulates a depth sensor at (sensorAz_, sensorEl_) looking at the origin.
//  Returns the number of points actually written into ptBuf_.
// ─────────────────────────────────────────────────────────────────────────────

static float gaussRand()
{
    // Box-Muller (simple, good enough for noise simulation)
    float u1 = (rand() + 1.f) / (static_cast<float>(RAND_MAX) + 2.f);
    float u2 =  rand()        /  static_cast<float>(RAND_MAX);
    return std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * static_cast<float>(M_PI) * u2);
}

uint32_t BucketedHash::genSphereBatch()
{
    // Sensor position on the orbit sphere
    float cx = std::cos(sensorEl_), sx = std::sin(sensorEl_);
    float cy = std::cos(sensorAz_), sy = std::sin(sensorAz_);
    Eigen::Vector3f sensorPos(sensorDist_ * cx * sy, sensorDist_ * sx, sensorDist_ * cx * cy);

    // Map ptBuf_ for writing
    void* mapped;
    vkMapMemory(ctx_.device, ptMem_, 0, VK_WHOLE_SIZE, 0, &mapped);
    auto* dst = static_cast<float*>(mapped);

    uint32_t written = 0;
    while (written < BH_BATCH_SIZE) {
        // Uniform random point on sphere surface (Marsaglia method)
        float u = (rand() / static_cast<float>(RAND_MAX)) * 2.f - 1.f;
        float t = (rand() / static_cast<float>(RAND_MAX)) * 2.f * static_cast<float>(M_PI);
        float r = std::sqrt(std::max(0.f, 1.f - u * u));
        Eigen::Vector3f n(r * std::cos(t), u, r * std::sin(t)); // unit normal
        Eigen::Vector3f p = n * sphereRadius_;

        // Visibility: front-facing AND within sensor FOV (cos > 0.25 → ±75°)
        Eigen::Vector3f toSensor = (sensorPos - p).normalized();
        if (n.dot(toSensor) < 0.25f) continue;

        // Radial noise (simulates depth sensor measurement error)
        p += n * (gaussRand() * noiseStddev_);

        dst[written * 4 + 0] = p.x();
        dst[written * 4 + 1] = p.y();
        dst[written * 4 + 2] = p.z();
        dst[written * 4 + 3] = 1.f;
        written++;
    }

    vkUnmapMemory(ctx_.device, ptMem_);
    return written;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Vulkan helpers
// ─────────────────────────────────────────────────────────────────────────────

void BucketedHash::createBuf(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags props,
                              VkBuffer& buf, VkDeviceMemory& mem)
{
    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx_.device, &bci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("BucketedHash: vkCreateBuffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx_.device, buf, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(ctx_.physicalDevice, req.memoryTypeBits, props);
    if (vkAllocateMemory(ctx_.device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("BucketedHash: vkAllocateMemory");

    vkBindBufferMemory(ctx_.device, buf, mem, 0);
}

void BucketedHash::bufBarrier(VkCommandBuffer cmd, VkBuffer buf,
                               VkAccessFlags src, VkAccessFlags dst,
                               VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
    VkBufferMemoryBarrier b{};
    b.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask       = src;
    b.dstAccessMask       = dst;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer              = buf;
    b.offset              = 0;
    b.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 1, &b, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
//  createBuffers
// ─────────────────────────────────────────────────────────────────────────────

void BucketedHash::createBuffers()
{
    // Hash table: device-local for throughput
    createBuf(sizeof(BH_Entry) * BH_TOTAL_ENTRIES,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
              htBuf_, htMem_);

    // Point batch: host-visible, fixed BH_BATCH_SIZE capacity
    createBuf(sizeof(float) * 4 * BH_BATCH_SIZE,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              ptBuf_, ptMem_);

    // Occupancy counter: host-visible for readback, transfer-dst for vkCmdFillBuffer
    createBuf(sizeof(uint32_t),
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              ctrBuf_, ctrMem_);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Descriptors
// ─────────────────────────────────────────────────────────────────────────────

void BucketedHash::createDescriptors()
{
    VkDescriptorSetLayoutBinding b[3]{};
    for (int i = 0; i < 3; i++) {
        b[i].binding        = static_cast<uint32_t>(i);
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount= 1;
        b[i].stageFlags     = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 3;
    lci.pBindings    = b;
    if (vkCreateDescriptorSetLayout(ctx_.device, &lci, nullptr, &descLayout_) != VK_SUCCESS)
        throw std::runtime_error("BucketedHash: descriptor layout");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo pci{};
    pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets       = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &ps;
    if (vkCreateDescriptorPool(ctx_.device, &pci, nullptr, &descPool_) != VK_SUCCESS)
        throw std::runtime_error("BucketedHash: descriptor pool");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = descPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &descLayout_;
    if (vkAllocateDescriptorSets(ctx_.device, &ai, &descSet_) != VK_SUCCESS)
        throw std::runtime_error("BucketedHash: descriptor alloc");

    VkDescriptorBufferInfo htInfo {htBuf_,  0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo ptInfo {ptBuf_,  0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo ctrInfo{ctrBuf_, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet   w[3]{};
    VkDescriptorBufferInfo* infos[3] = {&htInfo, &ptInfo, &ctrInfo};
    for (int i = 0; i < 3; i++) {
        w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet          = descSet_;
        w[i].dstBinding      = static_cast<uint32_t>(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo     = infos[i];
    }
    vkUpdateDescriptorSets(ctx_.device, 3, w, 0, nullptr);
}

void BucketedHash::updatePtDescriptor()
{
    VkDescriptorBufferInfo info{ptBuf_, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet   w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet          = descSet_;
    w.dstBinding      = 1;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo     = &info;
    vkUpdateDescriptorSets(ctx_.device, 1, &w, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pipelines
// ─────────────────────────────────────────────────────────────────────────────

void BucketedHash::createComputePipelines()
{
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BH_ComputePC)};
    VkPipelineLayoutCreateInfo lci{};
    lci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lci.setLayoutCount         = 1;
    lci.pSetLayouts            = &descLayout_;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx_.device, &lci, nullptr, &compLayout_) != VK_SUCCESS)
        throw std::runtime_error("BucketedHash: compute layout");

    auto make = [&](const char* glsl) {
        VkShaderModule mod = compileGLSL(ctx_.device, glsl, shaderc_compute_shader);
        VkComputePipelineCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName  = "main";
        ci.layout       = compLayout_;
        VkPipeline p;
        if (vkCreateComputePipelines(ctx_.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p) != VK_SUCCESS)
            throw std::runtime_error("BucketedHash: compute pipeline");
        vkDestroyShaderModule(ctx_.device, mod, nullptr);
        return p;
    };

    clearPipe_  = make(kClearComp);
    insertPipe_ = make(kInsertComp);
    countPipe_  = make(kCountComp);
}

void BucketedHash::createRenderPipeline()
{
    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(BH_RenderPC)};
    VkPipelineLayoutCreateInfo lci{};
    lci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lci.setLayoutCount         = 1;
    lci.pSetLayouts            = &descLayout_;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(ctx_.device, &lci, nullptr, &renderLayout_) != VK_SUCCESS)
        throw std::runtime_error("BucketedHash: render layout");

    auto vMod = compileGLSL(ctx_.device, kRenderVert, shaderc_vertex_shader);
    auto fMod = compileGLSL(ctx_.device, kRenderFrag, shaderc_fragment_shader);

    VkPipelineShaderStageCreateInfo stages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,  vMod,"main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,fMod,"main"}
    };
    VkPipelineVertexInputStateCreateInfo   vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                              nullptr, 0, VK_PRIMITIVE_TOPOLOGY_POINT_LIST};
    VkViewport vp{0,0,(float)ctx_.extent.width,(float)ctx_.extent.height,0,1};
    VkRect2D   sc{{0,0},ctx_.extent};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                          nullptr,0,1,&vp,1,&sc};
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth   = 1.f;
    raster.cullMode    = VK_CULL_MODE_NONE;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                            nullptr,0,VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              nullptr,0,VK_FALSE,{},1,&ba};
    VkGraphicsPipelineCreateInfo pci{};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState      = &vps;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState   = &ms;
    pci.pColorBlendState    = &blend;
    pci.layout              = renderLayout_;
    pci.renderPass          = ctx_.renderPass;
    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &pci, nullptr, &renderPipe_) != VK_SUCCESS)
        throw std::runtime_error("BucketedHash: render pipeline");

    vkDestroyShaderModule(ctx_.device, vMod, nullptr);
    vkDestroyShaderModule(ctx_.device, fMod, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Dispatch helpers
// ─────────────────────────────────────────────────────────────────────────────

void BucketedHash::dispatchClear(VkCommandBuffer cmd)
{
    vkCmdBindPipeline      (cmd, VK_PIPELINE_BIND_POINT_COMPUTE, clearPipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compLayout_,
                            0, 1, &descSet_, 0, nullptr);
    BH_ComputePC pc{BH_TOTAL_ENTRIES, 0.f, 0, 0};
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (BH_TOTAL_ENTRIES + 63) / 64, 1, 1);
}

void BucketedHash::dispatchInsert(VkCommandBuffer cmd)
{
    if (ptCount_ == 0) return;
    vkCmdBindPipeline      (cmd, VK_PIPELINE_BIND_POINT_COMPUTE, insertPipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compLayout_,
                            0, 1, &descSet_, 0, nullptr);
    BH_ComputePC pc{ptCount_, voxelSize_, BH_NUM_BUCKETS, 0};
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (ptCount_ + 63) / 64, 1, 1);
}

void BucketedHash::dispatchCount(VkCommandBuffer cmd)
{
    // Reset counter via transfer, then compute
    vkCmdFillBuffer(cmd, ctrBuf_, 0, sizeof(uint32_t), 0);
    bufBarrier(cmd, ctrBuf_,
               VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdBindPipeline      (cmd, VK_PIPELINE_BIND_POINT_COMPUTE, countPipe_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compLayout_,
                            0, 1, &descSet_, 0, nullptr);
    BH_ComputePC pc{BH_TOTAL_ENTRIES, 0.f, 0, 0};
    vkCmdPushConstants(cmd, compLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (BH_TOTAL_ENTRIES + 63) / 64, 1, 1);
    // Make result visible to host after fence
    bufBarrier(cmd, ctrBuf_,
               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
}

// ─────────────────────────────────────────────────────────────────────────────
//  onInit
// ─────────────────────────────────────────────────────────────────────────────

void BucketedHash::onInit(const VulkanContext& ctx)
{
    ctx_ = ctx;
    createBuffers();
    createDescriptors();
    createComputePipelines();
    createRenderPipeline();
    lastTime_ = std::chrono::steady_clock::now();
    doClear_  = true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  onCompute  — called each frame before the render pass
// ─────────────────────────────────────────────────────────────────────────────

void BucketedHash::onCompute(VkCommandBuffer cmd)
{
    // ── Counter readback from PREVIOUS frame (fence ensures GPU is done) ──────
    if (doCountRead_) {
        void* m;
        vkMapMemory(ctx_.device, ctrMem_, 0, sizeof(uint32_t), 0, &m);
        occupancy_ = static_cast<int>(*static_cast<uint32_t*>(m));
        vkUnmapMemory(ctx_.device, ctrMem_);
        doCountRead_ = false;
    }

    // ── Hash table clear ──────────────────────────────────────────────────────
    if (doClear_) {
        dispatchClear(cmd);
        bufBarrier(cmd, htBuf_,
                   VK_ACCESS_SHADER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        doClear_    = false;
        occupancy_  = 0;
        totalInserted_ = 0;
    }

    // ── Streaming: generate one sphere batch per frame ────────────────────────
    if (streaming_) {
        ptCount_ = genSphereBatch();
        doInsert_ = true;

        // Advance sensor orbit
        sensorAz_ += sensorSpeed_;
        if (sensorAz_ > 2.f * static_cast<float>(M_PI)) sensorAz_ -= 2.f * static_cast<float>(M_PI);
    }

    // ── Insert ────────────────────────────────────────────────────────────────
    if (doInsert_ && ptCount_ > 0) {
        dispatchInsert(cmd);
        bufBarrier(cmd, htBuf_,
                   VK_ACCESS_SHADER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
        dispatchCount(cmd);
        totalInserted_ += ptCount_;
        doInsert_    = false;
        doCountRead_ = true;

        // Throughput estimate
        auto now  = std::chrono::steady_clock::now();
        float dt  = std::chrono::duration<float>(now - lastTime_).count();
        lastTime_ = now;
        if (dt > 0.f) ptsPerSec_ = 0.9f * ptsPerSec_ + 0.1f * (ptCount_ / dt);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  onRender
// ─────────────────────────────────────────────────────────────────────────────

void BucketedHash::onRender(const RenderContext& ctx)
{
    vkCmdBindPipeline      (ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, renderPipe_);
    vkCmdBindDescriptorSets(ctx.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            renderLayout_, 0, 1, &descSet_, 0, nullptr);

    Eigen::Matrix4f mvp = computeMVP();
    BH_RenderPC pc{};
    std::memcpy(pc.mvp, mvp.data(), 64);
    pc.voxelSize = voxelSize_;
    vkCmdPushConstants(ctx.commandBuffer, renderLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

    // One vertex per hash-table slot; empty slots self-clip in the shader
    vkCmdDraw(ctx.commandBuffer, BH_TOTAL_ENTRIES, 1, 0, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  onImGui
// ─────────────────────────────────────────────────────────────────────────────

void BucketedHash::onImGui()
{
    // ── Mouse-driven orbit camera (outside any ImGui window) ─────────────────
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            azimuth_   += io.MouseDelta.x * 0.005f;
            elevation_ += io.MouseDelta.y * 0.005f;
            elevation_  = std::max(-1.4f, std::min(1.4f, elevation_));
        }
        camDist_ -= io.MouseWheel * 0.3f;
        camDist_  = std::max(0.5f, std::min(20.f, camDist_));
    }

    ImGui::Begin("Bucketed Hash");

    // ── Table info ────────────────────────────────────────────────────────────
    float htMB = sizeof(BH_Entry) * BH_TOTAL_ENTRIES / (1024.f * 1024.f);
    ImGui::Text("Buckets     : %u  ×  %u entries  =  %.1f MB",
                BH_NUM_BUCKETS, BH_BUCKET_SIZE, htMB);
    float fillPct = BH_TOTAL_ENTRIES > 0
                    ? 100.f * occupancy_ / static_cast<float>(BH_TOTAL_ENTRIES) : 0.f;
    ImGui::Text("Occupied    : %d  /  %u  (%.2f%%)", occupancy_, BH_TOTAL_ENTRIES, fillPct);
    ImGui::Text("Total pts   : %lld", (long long)totalInserted_);
    ImGui::Text("Throughput  : %.0f  pts/s", ptsPerSec_);
    ImGui::Separator();

    // ── Camera ────────────────────────────────────────────────────────────────
    ImGui::Text("Camera  [drag to rotate, scroll to zoom]");
    ImGui::SliderFloat("Azimuth",   &azimuth_,   -3.14f, 3.14f, "%.2f rad");
    ImGui::SliderFloat("Elevation", &elevation_, -1.4f,  1.4f,  "%.2f rad");
    ImGui::SliderFloat("Distance",  &camDist_,   0.5f,  20.f,  "%.1f m");
    ImGui::Separator();

    // ── Sphere integration ────────────────────────────────────────────────────
    ImGui::Text("Sphere integration");
    bool rebuild = false;
    rebuild |= ImGui::SliderFloat("Voxel size",    &voxelSize_,   0.005f, 0.1f, "%.4f m");
    ImGui::SliderFloat("Sphere radius",  &sphereRadius_, 0.1f,  3.f,  "%.2f m");
    ImGui::SliderFloat("Sensor dist",    &sensorDist_,   1.f,  10.f,  "%.1f m");
    ImGui::SliderFloat("Noise stddev",   &noiseStddev_,  0.f,  0.05f, "%.4f m");
    ImGui::SliderFloat("Sensor speed",   &sensorSpeed_,  0.f,  0.1f,  "%.3f rad/f");

    if (rebuild && streaming_) { doClear_ = true; }

    ImGui::Separator();
    if (ImGui::Button(streaming_ ? "■ Stop" : "▶ Start integration")) {
        streaming_ = !streaming_;
        if (streaming_) { doClear_ = true; sensorAz_ = 0.f; }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset table")) { doClear_ = true; }

    // ── Sensor orbit indicator ────────────────────────────────────────────────
    if (streaming_) {
        float cx = std::cos(sensorEl_), sx = std::sin(sensorEl_);
        float cy = std::cos(sensorAz_), sy = std::sin(sensorAz_);
        ImGui::Text("Sensor pos  : (%.2f, %.2f, %.2f)",
                    sensorDist_*cx*sy, sensorDist_*sx, sensorDist_*cx*cy);
    }

    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────────────────
//  onCleanup
// ─────────────────────────────────────────────────────────────────────────────

void BucketedHash::onCleanup()
{
    vkDestroyPipeline      (ctx_.device, clearPipe_,   nullptr);
    vkDestroyPipeline      (ctx_.device, insertPipe_,  nullptr);
    vkDestroyPipeline      (ctx_.device, countPipe_,   nullptr);
    vkDestroyPipeline      (ctx_.device, renderPipe_,  nullptr);
    vkDestroyPipelineLayout(ctx_.device, compLayout_,  nullptr);
    vkDestroyPipelineLayout(ctx_.device, renderLayout_,nullptr);
    vkDestroyDescriptorPool(ctx_.device, descPool_,    nullptr);
    vkDestroyDescriptorSetLayout(ctx_.device, descLayout_, nullptr);

    vkDestroyBuffer(ctx_.device, htBuf_,  nullptr); vkFreeMemory(ctx_.device, htMem_,  nullptr);
    vkDestroyBuffer(ctx_.device, ptBuf_,  nullptr); vkFreeMemory(ctx_.device, ptMem_,  nullptr);
    vkDestroyBuffer(ctx_.device, ctrBuf_, nullptr); vkFreeMemory(ctx_.device, ctrMem_, nullptr);
}
