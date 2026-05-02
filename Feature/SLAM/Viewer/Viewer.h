#pragma once

#include "../../DataStructure/VoxelHash/BucketVoxelHash.h"
#include "../../DataStructure/VoxelHash/VoxelHashTypes.h"
#include "../../IFeature.h"

#include <Eigen/Core>
#include <string>
#include <vector>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
//  KITTILoader — KITTI odometry 데이터 로더 (순수 C++, Vulkan 없음)
//
//  경로 규칙:
//    velodyne : {base}/kitti/00/velodyne/{frame:06d}.bin
//    poses    : {base}/kitti_pose/00.txt
//    calib    : {base}/kitti_cal/00/calib.txt
// ─────────────────────────────────────────────────────────────────────────────
class KITTILoader
{
public:
    KITTILoader() = default;

    void initialize(const std::string &datasetBase);

    uint32_t frameCount() const { return static_cast<uint32_t>(poses_.size()); }

    // 프레임 i의 센서(velodyne) 위치 — 첫 프레임 기준 로컬 좌표
    Eigen::Vector3f sensorPos(uint32_t frameIdx) const;

    // velodyne → 시퀀스 로컬 (첫 프레임 원점): T_base_inv_ * poses_[i] * Tr_
    Eigen::Matrix4f veloToLocal(uint32_t frameIdx) const;

    // .bin 로드 → 좌표 변환 → 서브샘플(≤VH_BATCH_SIZE) → VH_InputPoint 반환
    std::vector<VH_InputPoint> loadFrame(uint32_t frameIdx) const;

private:
    void parseCalibration(const std::string &path);
    void parsePoses(const std::string &path);

    std::string binPath(uint32_t idx) const;

    // reflectance [0,1] → r8g8b8a8  (hot colormap)
    static uint32_t toColor(float reflectance);

    std::string     velodyneDir_;
    Eigen::Matrix4f Tr_;                      // T_cam_velo  (3×4 → 4×4)
    std::vector<Eigen::Matrix4f> poses_;      // T_world_cam per frame (3×4 → 4×4)
    Eigen::Matrix4f T_base_inv_;             // (poses_[0] × Tr_)^{-1}
};

// ─────────────────────────────────────────────────────────────────────────────
//  Viewer — BucketVoxelHash 래핑 (GPU 관리)
// ─────────────────────────────────────────────────────────────────────────────
class Viewer
{
public:
    void Initialize(const VulkanContext &ctx, const std::string &datasetBase);

    void IntegrateFrame(uint32_t frameIdx);
    void ClearMap();

    void RenderColor(VkCommandBuffer cmd, const float *mvp, bool useSlotPoint = false);
    void RenderVoxel(VkCommandBuffer cmd, const float *mvp);

    void     setVoxelSize(float v)            { vhb_.setVoxelSize(v); }
    uint32_t frameCount()              const  { return loader_.frameCount(); }
    Eigen::Vector3f sensorPos(uint32_t i) const { return loader_.sensorPos(i); }

private:
    VulkanContext   ctx_{};
    BucketVoxelHash vhb_;
    KITTILoader     loader_;
};
