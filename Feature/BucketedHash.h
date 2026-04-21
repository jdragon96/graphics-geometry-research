#pragma once

#include "IFeature.h"
#include "../Utilities.h"
#include <vulkan/vulkan.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <chrono>

// ── Compile-time constants (CPU & GPU must agree) ─────────────────────────────
static constexpr uint32_t BH_BUCKET_SIZE   = 4;
static constexpr uint32_t BH_NUM_BUCKETS   = 1u << 17;   // 131 072
static constexpr uint32_t BH_TOTAL_ENTRIES = BH_NUM_BUCKETS * BH_BUCKET_SIZE;
static constexpr int32_t  BH_EMPTY_KEY     = 0x7FFFFFFF;
static constexpr uint32_t BH_BATCH_SIZE    = 10000;       // points per frame (streaming)

// ── GPU-side entry (std430, 16 bytes) ─────────────────────────────────────────
struct alignas(16) BH_Entry {
    int32_t key_x = BH_EMPTY_KEY;
    int32_t key_y = 0;
    int32_t key_z = 0;
    int32_t value = -1;
};

// ── Push constants ────────────────────────────────────────────────────────────
struct BH_ComputePC {           // ≤ 16 bytes, shared by all compute shaders
    uint32_t p0;                // clear/count → totalEntries | insert → numPoints
    float    p1;                // insert → voxelSize
    uint32_t p2;                // insert → numBuckets
    uint32_t _pad = 0;
};

struct BH_RenderPC {            // 80 bytes (within 128-byte Vulkan minimum)
    float    mvp[16];           // column-major Eigen mat4 → GLSL mat4
    float    voxelSize;
    float    _pad[3];
};

// ── Descriptor bindings ───────────────────────────────────────────────────────
//  0 → hash table  Entry[]  (device-local, always bound)
//  1 → point batch vec4[]   (host-visible, replaced each streaming frame)
//  2 → counter     uint     (host-visible, always bound)

class BucketedHash : public IFeature {
public:
    const char* name() const override { return "Bucketed Hash"; }

    void onInit   (const VulkanContext& ctx) override;
    void onCompute(VkCommandBuffer cmd)      override;
    void onRender (const RenderContext& ctx) override;
    void onImGui  ()                         override;
    void onCleanup()                         override;

private:
    VulkanContext ctx_{};

    // ── Vulkan resources ──────────────────────────────────────────────────────
    VkDescriptorSetLayout descLayout_  = VK_NULL_HANDLE;
    VkDescriptorPool      descPool_    = VK_NULL_HANDLE;
    VkDescriptorSet       descSet_     = VK_NULL_HANDLE;

    VkPipelineLayout compLayout_   = VK_NULL_HANDLE;
    VkPipeline       clearPipe_    = VK_NULL_HANDLE;
    VkPipeline       insertPipe_   = VK_NULL_HANDLE;
    VkPipeline       countPipe_    = VK_NULL_HANDLE;

    VkPipelineLayout renderLayout_ = VK_NULL_HANDLE;
    VkPipeline       renderPipe_   = VK_NULL_HANDLE;

    // hash table: device-local
    VkBuffer       htBuf_  = VK_NULL_HANDLE;
    VkDeviceMemory htMem_  = VK_NULL_HANDLE;
    // point batch: host-visible, fixed BH_BATCH_SIZE capacity
    VkBuffer       ptBuf_  = VK_NULL_HANDLE;
    VkDeviceMemory ptMem_  = VK_NULL_HANDLE;
    uint32_t       ptCount_= 0;
    // occupancy counter: host-visible
    VkBuffer       ctrBuf_ = VK_NULL_HANDLE;
    VkDeviceMemory ctrMem_ = VK_NULL_HANDLE;

    // ── Frame flags ───────────────────────────────────────────────────────────
    bool doClear_     = true;
    bool doInsert_    = false;
    bool doCountRead_ = false;

    // ── Orbit camera ──────────────────────────────────────────────────────────
    float azimuth_   = 0.5f;   // horizontal angle (radians)
    float elevation_ = 0.4f;   // vertical angle   (radians, clamped ±1.4)
    float camDist_   = 4.0f;   // distance from origin

    // ── Sphere integration ────────────────────────────────────────────────────
    bool  streaming_     = false;
    float sphereRadius_  = 1.0f;
    float sensorDist_    = 3.0f;
    float noiseStddev_   = 0.005f; // meters
    float sensorAz_      = 0.0f;   // sensor orbit angle (advances each frame)
    float sensorEl_      = 0.2f;
    float sensorSpeed_   = 0.02f;  // radians per frame

    // ── Stats ─────────────────────────────────────────────────────────────────
    float   voxelSize_      = 0.02f;
    int     occupancy_      = 0;
    int64_t totalInserted_  = 0;    // cumulative points inserted
    float   ptsPerSec_      = 0.f;
    std::chrono::steady_clock::time_point lastTime_;

    // ── Helpers ───────────────────────────────────────────────────────────────
    Eigen::Matrix4f computeMVP() const;
    uint32_t genSphereBatch();          // fills ptBuf_, returns count

    void createBuffers();
    void createDescriptors();
    void createComputePipelines();
    void createRenderPipeline();
    void updatePtDescriptor();

    void dispatchClear (VkCommandBuffer cmd);
    void dispatchInsert(VkCommandBuffer cmd);
    void dispatchCount (VkCommandBuffer cmd);

    void bufBarrier(VkCommandBuffer cmd, VkBuffer buf,
                    VkAccessFlags src, VkAccessFlags dst,
                    VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage);
    void createBuf(VkDeviceSize size, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags props,
                   VkBuffer& buf, VkDeviceMemory& mem);
};
