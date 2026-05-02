#pragma once

// KISS-ICP odometry + BucketVoxelHash visualization (KITTI sequence 00).
// GT from KITTILoader is evaluation-only (ImGui), not fed to the estimator.

#include "KissICP.h"
#include "../Viewer/Viewer.h"
#include "../../DataStructure/VoxelHash/BucketVoxelHash.h"
#include "../../DataStructure/VoxelHash/VoxelHashTypes.h"
#include "../../IFeature.h"

#include <Eigen/Core>
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace kitti_velo
{
// Velodyne .bin in file order (time along sweep). Stride subsamples to maxPts; rel_times[i] in [0,1]
// is the original-scan fractional index k/(N-1) for deskew (KissICP::registerFrame).
struct LoadedVeloBin
{
    std::vector<Eigen::Vector3f> points;
    std::vector<float>           rel_times;
};

inline LoadedVeloBin loadBin(const std::string &binPath, uint32_t maxPts = VH_BATCH_SIZE)
{
    LoadedVeloBin out;
    FILE          *fp = fopen(binPath.c_str(), "rb");
    if (!fp) return out;

    fseek(fp, 0, SEEK_END);
    uint32_t N = static_cast<uint32_t>(ftell(fp) / (4 * sizeof(float)));
    fseek(fp, 0, SEEK_SET);

    std::vector<float> raw(static_cast<size_t>(N) * 4u);
    fread(raw.data(), sizeof(float), static_cast<size_t>(N) * 4u, fp);
    fclose(fp);

    if (N == 0u) return out;

    const uint32_t count = std::min(N, maxPts);
    out.points.reserve(count);
    out.rel_times.reserve(count);

    const float denom = (N > 1u) ? static_cast<float>(N - 1u) : 1.f;

    for (uint32_t j = 0; j < count; ++j)
    {
        const uint32_t k =
            (count > 1u) ? (j * (N - 1u)) / (count - 1u) : 0u; // 0 .. N-1, monotone in j
        out.points.emplace_back(raw[static_cast<size_t>(k) * 4u + 0u],
                                  raw[static_cast<size_t>(k) * 4u + 1u],
                                  raw[static_cast<size_t>(k) * 4u + 2u]);
        out.rel_times.push_back((N > 1u) ? static_cast<float>(k) / denom : 0.f);
    }
    return out;
}

inline std::string binPath(const std::string &velodyneDir, uint32_t idx)
{
    std::ostringstream oss;
    oss << velodyneDir << "/" << std::setfill('0') << std::setw(6) << idx << ".bin";
    return oss.str();
}
} // namespace kitti_velo

class KissICPFeature : public IFeature
{
public:
    const char *name() const override { return "KISS_ICP"; }

    void onInit(const VulkanContext &ctx) override
    {
        ctx_         = ctx;
        const std::string base =
            "/Users/sjy/Desktop/Project/Vision-3D/dataset";
        velodyneDir_ = base + "/kitti/00/velodyne";
        gtLoader_.initialize(base);

        vhb_.Initialize(ctx_);
        vhb_.setVoxelSize(voxelSize_);

        kiss_.reset();
        kiss_.config().use_deskew = use_deskew_;

        totalFrames_   = gtLoader_.frameCount();
        lastFrameTime_ = std::chrono::steady_clock::now();
        lastSeqFrame_  = kNoPrev;
        loadFrame(0);
    }

    void onCompute(VkCommandBuffer) override
    {
        if (isPlaying_ && totalFrames_ > 0)
        {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - lastFrameTime_).count();
            if (dt >= 1.f / playSpeed_)
            {
                lastFrameTime_ = now;
                uint32_t next = currentFrame_ + 1;
                if (next < totalFrames_) loadFrame(next);
                else
                    isPlaying_ = false;
            }
        }
    }

    void onRender(const RenderContext &ctx) override
    {
        auto mvp = computeMVP();
        if (renderMode_ == 0)
            vhb_.RenderColor(ctx.commandBuffer, mvp.data(), false);
        else
            vhb_.RenderVoxel(ctx.commandBuffer, mvp.data());
    }

    void onImGui() override
    {
        ImGuiIO &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                azimuth_ += io.MouseDelta.x * 0.005f;
                elevation_ += io.MouseDelta.y * 0.005f;
                elevation_ = std::max(-1.4f, std::min(1.4f, elevation_));
            }
            camDist_ -= io.MouseWheel * 2.f;
            camDist_ = std::max(1.f, std::min(500.f, camDist_));
        }

        ImGui::Begin("KISS-ICP");
        if (ImGui::Button(isPlaying_ ? "Pause" : "Play "))
            isPlaying_ = !isPlaying_;
        ImGui::SameLine();
        ImGui::SliderFloat("FPS##kiss", &playSpeed_, 1.f, 20.f, "%.0f");

        int frame = (int)currentFrame_;
        if (ImGui::SliderInt("##kissf", &frame, 0,
                             totalFrames_ > 0 ? (int)totalFrames_ - 1 : 0))
        {
            isPlaying_ = false;
            loadFrame((uint32_t)frame);
        }
        ImGui::Text("Frame %u / %u", currentFrame_,
                    totalFrames_ > 0 ? totalFrames_ - 1 : 0u);

        if (ImGui::Checkbox("Deskew (CV)", &use_deskew_))
            kiss_.config().use_deskew = use_deskew_;

        ImGui::Text("tau=%.2f  sigma=%.3f  icp_iter=%d  corr=%d", last_tau_, last_sigma_,
                    last_icp_iters_, last_corr_);

        Eigen::Vector3f est = last_T_.block<3, 1>(0, 3);
        ImGui::Text("Pos est: %.2f %.2f %.2f", est.x(), est.y(), est.z());
        if (currentFrame_ < totalFrames_)
        {
            Eigen::Vector3f gt = gtLoader_.sensorPos(currentFrame_);
            ImGui::Text("Pos GT : %.2f %.2f %.2f", gt.x(), gt.y(), gt.z());
            ImGui::Text("Pos err: %.3f m", (est - gt).norm());
        }

        if (ImGui::RadioButton("Point##kiss", renderMode_ == 0)) renderMode_ = 0;
        ImGui::SameLine();
        if (ImGui::RadioButton("Voxel##kiss", renderMode_ == 1)) renderMode_ = 1;

        if (ImGui::SliderFloat("Voxel##kissv", &voxelSize_, 0.1f, 2.f, "%.2f"))
            vhb_.setVoxelSize(voxelSize_);

        if (ImGui::Button("Reset")) resetOdometry();

        ImGui::End();
    }

    void onKey(int key, int action, int /*mods*/) override
    {
        if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
        switch (key)
        {
        case GLFW_KEY_SPACE:
            isPlaying_ = !isPlaying_;
            break;
        case GLFW_KEY_RIGHT:
            if (currentFrame_ + 1 < totalFrames_)
            {
                isPlaying_ = false;
                loadFrame(currentFrame_ + 1);
            }
            break;
        case GLFW_KEY_LEFT:
            if (currentFrame_ > 0)
            {
                isPlaying_ = false;
                loadFrame(currentFrame_ - 1);
            }
            break;
        }
    }

    void onCleanup() override {}

private:
    static KissICPConfig makeKissConfig() { return KissICPConfig{}; }

    void resetOdometry()
    {
        vhb_.Clear(ctx_);
        kiss_.reset();
        kiss_.config().use_deskew = use_deskew_;
        lastSeqFrame_ = kNoPrev;
        loadFrame(0);
    }

    void loadFrame(uint32_t idx)
    {
        if (lastSeqFrame_ != kNoPrev && idx != lastSeqFrame_ && idx != lastSeqFrame_ + 1u
            && idx + 1u != lastSeqFrame_)
        {
            vhb_.Clear(ctx_);
            kiss_.reset();
            kiss_.config().use_deskew = use_deskew_;
            lastSeqFrame_ = kNoPrev;
        }

        currentFrame_ = idx;
        std::string path = kitti_velo::binPath(velodyneDir_, idx);
        auto        frame = kitti_velo::loadBin(path, VH_BATCH_SIZE);
        if (frame.points.empty()) return;

        KissICPResult r =
            kiss_.registerFrame(frame.points,
                                 frame.rel_times.empty() ? nullptr : &frame.rel_times);
        last_T_         = r.T_world_velo;
        last_tau_       = r.tau;
        last_sigma_     = r.sigma;
        last_icp_iters_ = r.icp_iterations;
        last_corr_      = r.num_correspondences;

        Eigen::Vector3f sensorW = last_T_.block<3, 1>(0, 3);

        std::vector<VH_InputPoint> vh;
        vh.reserve(kiss_.lastMergeDeskewed().size());
        for (const Eigen::Vector3f &pl : kiss_.lastMergeDeskewed())
        {
            Eigen::Vector3f pw = last_T_.block<3, 3>(0, 0) * pl + last_T_.block<3, 1>(0, 3);
            VH_InputPoint   vp{};
            vp.px = pw.x();
            vp.py = pw.y();
            vp.pz = pw.z();
            vp.nx = vp.ny = vp.nz = 0.f;
            vp.col                 = 0xFF808080;
            vh.push_back(vp);
        }
        uint32_t N = static_cast<uint32_t>(std::min<size_t>(vh.size(), VH_BATCH_SIZE));
        if (N > 0) vhb_.Integrate(ctx_, vh.data(), N, sensorW.x(), sensorW.y(), sensorW.z());

        lastSeqFrame_ = idx;
    }

    Eigen::Matrix4f computeMVP() const
    {
        float cx = std::cos(elevation_), sx = std::sin(elevation_);
        float cy = std::cos(azimuth_), sy = std::sin(azimuth_);
        Eigen::Vector3f eye(camDist_ * cx * sy, camDist_ * sx, camDist_ * cx * cy);
        Eigen::Vector3f up(0.f, 1.f, 0.f);
        Eigen::Vector3f f  = (-eye).normalized();
        Eigen::Vector3f rr = f.cross(up).normalized();
        Eigen::Vector3f u  = rr.cross(f);

        Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
        V.row(0) << rr.x(), rr.y(), rr.z(), -rr.dot(eye);
        V.row(1) << u.x(), u.y(), u.z(), -u.dot(eye);
        V.row(2) << -f.x(), -f.y(), -f.z(), f.dot(eye);

        float fovY   = 60.f * 3.14159f / 180.f;
        float aspect = (float)ctx_.extent.width / (float)ctx_.extent.height;
        float n = 0.1f, fa = 1000.f, th = std::tan(fovY * 0.5f);

        Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
        P(0, 0) = 1.f / (aspect * th);
        P(1, 1) = -1.f / th;
        P(2, 2) = fa / (n - fa);
        P(2, 3) = fa * n / (n - fa);
        P(3, 2) = -1.f;
        return P * V;
    }

    VulkanContext       ctx_{};
    BucketVoxelHash     vhb_;
    KissICP kiss_{makeKissConfig()};
    KITTILoader         gtLoader_;
    std::string         velodyneDir_;
    static constexpr uint32_t kNoPrev = 0xFFFFFFFFu;
    uint32_t            lastSeqFrame_ = kNoPrev;

    uint32_t currentFrame_ = 0;
    uint32_t totalFrames_  = 0;
    bool     isPlaying_    = false;
    float    playSpeed_    = 5.f;
    std::chrono::steady_clock::time_point lastFrameTime_;

    Eigen::Matrix4f last_T_ = Eigen::Matrix4f::Identity();
    float           last_tau_{2.f};
    float           last_sigma_{0.67f};
    int             last_icp_iters_{};
    int             last_corr_{};

    int   renderMode_ = 0;
    float voxelSize_  = 0.5f;
    bool  use_deskew_ = true;

    float azimuth_   = 0.f;
    float elevation_ = 0.3f;
    float camDist_   = 100.f;
};
