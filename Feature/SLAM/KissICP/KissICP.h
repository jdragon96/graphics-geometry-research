#pragma once

// KISS-ICP — IEEE RA-L 2023 (Vizzo et al.), LiDAR-only odometry.
// Reference: https://github.com/PRBonn/kiss-icp

#include <Eigen/Core>
#include <memory>
#include <vector>

struct KissICPConfig
{
    float r_max          = 100.f;
    float delta_t_sweep  = 0.1f;
    float tau0           = 2.f;
    float delta_min      = 0.1f;
    int   N_max          = 20;
    float alpha          = 0.5f;
    float beta           = 1.5f;
    float icp_gamma      = 1e-4f;
    int   max_icp_iters  = 100;
    bool  use_deskew     = true; // KITTI pre-motion-compensated scans: set false (paper IV)
};

struct KissICPVoxelMap;

struct KissICPResult
{
    Eigen::Matrix4f T_world_velo = Eigen::Matrix4f::Identity();
    float           tau            = 2.f;
    float           sigma          = 0.67f;
    int             icp_iterations = 0;
    int             num_correspondences = 0;
};

class KissICP
{
public:
    explicit KissICP(KissICPConfig cfg = {});
    ~KissICP();

    KissICP(const KissICP &)            = delete;
    KissICP &operator=(const KissICP &) = delete;

    // Velodyne-frame points. rel_times: optional [0,1] along sweep; if null, linear by index.
    KissICPResult registerFrame(const std::vector<Eigen::Vector3f> &points_velo,
                                const std::vector<float>          *rel_times = nullptr);

    void reset();

    float voxelMapSize() const { return voxel_map_; }

    // Last frame's merge-resolution cloud (deskewed, voxel α), velodyne frame — for visualization.
    const std::vector<Eigen::Vector3f> &lastMergeDeskewed() const { return last_merge_; }

    KissICPConfig       &config() { return cfg_; }
    const KissICPConfig &config() const { return cfg_; }

private:
    KissICPConfig cfg_;
    float         voxel_map_{};
    float         voxel_merge_{};
    float         voxel_icp_{};

    std::vector<Eigen::Matrix4f> T_history_;
    std::vector<float>           delta_samples_;
    float                        sigma_{};

    std::unique_ptr<KissICPVoxelMap> map_;

    std::vector<Eigen::Vector3f> last_merge_;
};
