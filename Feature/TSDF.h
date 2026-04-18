#pragma once
#include "IFeature.h"
#include "../SceneObject.h"
#include <openvdb/openvdb.h>
#include <vector>
#include <string>

class TSDFFeature : public IFeature
{
public:
    const char *name() const override { return "TSDF Visualization"; }
    void onInit   (const VulkanContext &ctx) override;
    void onRender (const RenderContext  &ctx) override;
    void onImGui  () override;
    void onKey    (int key, int action, int mods) override;
    void onCleanup() override;

private:
    VulkanContext ctx_{};

    // ── OpenVDB TSDF ───────────────────────────────────────────────────────────
    openvdb::FloatGrid::Ptr tsdfGrid_;
    openvdb::FloatGrid::Ptr weightGrid_;
    int   integratedFrames_ = 0;
    int   totalFrames_      = 36;
    float truncation_       = 0.15f;
    float voxelSize_        = 0.04f;
    int   gridHalfDim_      = 40;    // voxel coords in [-40, 40)
    float sensorNoise_      = 0.005f; // simulated depth noise (m)
    bool  autoIntegrate_    = false;
    float integrationTimer_ = 0.0f;
    float integrationRate_  = 0.4f;
    bool  voxelDirty_       = false;
    float lastTime_         = 0.0f;

    // ── Precomputed surface point cloud ────────────────────────────────────────
    std::vector<Eigen::Vector3f> surfacePts_;
    std::vector<Eigen::Vector3f> surfaceNormals_;

    // ── Accumulated raw input geometry (all frames) ────────────────────────────
    std::vector<Eigen::Vector3f> accPts_;   // grows frame-by-frame, never cleared mid-run
    VkBuffer       accPtBuffer_  = VK_NULL_HANDLE;
    VkDeviceMemory accPtMemory_  = VK_NULL_HANDLE;
    uint32_t       accPtCount_   = 0;
    bool           showAccCloud_ = true;

    // ── Camera / view ──────────────────────────────────────────────────────────
    float camDist_    = 4.0f;
    float camAngleV_  = 0.4f;
    float rotY_       = 0.0f;
    bool  autoRotate_ = true;
    float fovDeg_     = 45.0f;

    // ── Light / material ───────────────────────────────────────────────────────
    float lightPos_[3]   = {3.0f, 3.0f, 3.0f};
    float lightColor_[3] = {1.0f, 1.0f, 1.0f};
    float ambient_   = 0.3f;
    float specular_  = 0.0f;
    float shininess_ = 1.0f;

    // ── Visualization toggles ──────────────────────────────────────────────────
    bool showTSDFVoxels_  = true;
    bool showInputCloud_  = true;
    int  voxelColorMode_  = 0;  // 0=TSDF value, 1=weight

    // ── Input point cloud buffer (current frame's depth cloud) ─────────────────
    VkBuffer       inputPtBuffer_  = VK_NULL_HANDLE;
    VkDeviceMemory inputPtMemory_  = VK_NULL_HANDLE;
    uint32_t       inputPtCount_   = 0;

    // ── TSDF voxel visualization buffer ────────────────────────────────────────
    VkBuffer       voxelBuffer_    = VK_NULL_HANDLE;
    VkDeviceMemory voxelMemory_    = VK_NULL_HANDLE;
    uint32_t       voxelCount_     = 0;

    // ── Vulkan pipeline (point list, tsdf_voxel shaders) ──────────────────────
    VkPipeline            pipeline_       = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout_     = VK_NULL_HANDLE;
    VkDescriptorPool      descPool_       = VK_NULL_HANDLE;
    VkDescriptorSet       descSet_        = VK_NULL_HANDLE;
    VkBuffer              uboBuffer_      = VK_NULL_HANDLE;
    VkDeviceMemory        uboMemory_      = VK_NULL_HANDLE;
    void                 *uboMapped_      = nullptr;

    // ── Core TSDF operations ───────────────────────────────────────────────────
    float           sceneSDF        (float x, float y, float z) const;
    Eigen::Vector3f sdfNormal       (float x, float y, float z) const;
    Eigen::Vector3f tsdfColor       (float tsdfVal) const;

    void initVDB              ();
    void resetTSDF            ();
    void buildSurfaceCloud    ();

    // Returns the visible subset of surfacePts_ from this camera
    std::vector<Eigen::Vector3f> filterVisiblePoints(Eigen::Vector3f camPos) const;

    void integratePointCloud  (const std::vector<Eigen::Vector3f> &cloud,
                                Eigen::Vector3f camPos);
    void integrateFrame       (int frameIndex);

    void updateVoxelBuffer    ();
    void uploadPointBuffer    (VkBuffer &buf, VkDeviceMemory &mem, uint32_t &count,
                                const std::vector<Vertex> &pts);

    // ── Vulkan setup ───────────────────────────────────────────────────────────
    void createDescriptorSetLayout();
    void createDescriptorPool     ();
    void createUBO                ();
    void createDescriptorSet      ();
    void createPipeline           ();
    void updateUBO                ();

    void           createBuffer   (VkDeviceSize size, VkBufferUsageFlags usage,
                                   VkMemoryPropertyFlags props,
                                   VkBuffer &buf, VkDeviceMemory &mem);
    uint32_t       findMemoryType (uint32_t filter, VkMemoryPropertyFlags props);
    VkShaderModule loadShader     (const std::string &path);
    static std::vector<char> readFile(const std::string &path);
};
