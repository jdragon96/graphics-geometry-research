#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  FastLIO2Feature.h — KITTI 데이터로 FastLIO2 SLAM을 테스트하는 IFeature
//
//  동작:
//    - KITTILoader로 velodyne 프레임 읽기
//    - 매 프레임 KITTI GT velo pose로 x_pred/x_opt 시드 후 feedScan (IEKF 정제)
//    - 비연속 프레임 점프 시 맵·SLAM 재구성
//    - BucketVoxelHash로 누적 맵 렌더링
//    - ImGui: 재생 제어, SLAM 포즈 vs GT 포즈 오차 표시
//
//  IMU 없음 (KITTI odometry 데이터셋): x_pred = x_opt 로 전파 없이 IEKF만 실행
// ─────────────────────────────────────────────────────────────────────────────

#include "FastLIO2.h"
#include "../Viewer/Viewer.h"  // KITTILoader

#include "../../IFeature.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
//  Raw velodyne 로더 (FastLIO2용 — 센서 프레임; world 정렬은 GT 시드로 별도 처리)
// ─────────────────────────────────────────────────────────────────────────────
namespace kitti_raw
{

// calib.txt에서 Tr: 행 파싱 → 4×4
inline Eigen::Matrix4f parseTr(const std::string &calibPath)
{
    std::ifstream f(calibPath);
    std::string line;
    while (std::getline(f, line))
    {
        if (line.substr(0, 3) != "Tr:") continue;
        std::istringstream ss(line.substr(3));
        Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                ss >> M(r, c);
        return M;
    }
    return Eigen::Matrix4f::Identity();
}

// velodyne .bin → 센서 프레임 LiDARPoint
// maxPts: 서브샘플 최대 수
inline std::vector<LiDARPoint> loadRaw(const std::string &binPath,
                                        uint32_t maxPts = VH_BATCH_SIZE)
{
    FILE *fp = fopen(binPath.c_str(), "rb");
    if (!fp) return {};

    fseek(fp, 0, SEEK_END);
    uint32_t N = static_cast<uint32_t>(ftell(fp) / (4 * sizeof(float)));
    fseek(fp, 0, SEEK_SET);

    std::vector<float> raw(N * 4);
    fread(raw.data(), sizeof(float), N * 4, fp);
    fclose(fp);

    // 랜덤 서브샘플
    std::vector<uint32_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0u);
    {
        static thread_local std::mt19937 rng(42);
        std::shuffle(idx.begin(), idx.end(), rng);
    }
    uint32_t count = std::min(N, maxPts);

    std::vector<LiDARPoint> pts;
    pts.reserve(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t k = idx[i];
        LiDARPoint p;
        p.pos       = Eigen::Vector3f(raw[k*4+0], raw[k*4+1], raw[k*4+2]);
        p.intensity = raw[k*4+3];
        p.timestamp = 0.0;
        pts.push_back(p);
    }
    return pts;
}

// 프레임 번호 → 파일 경로
inline std::string binPath(const std::string &velodyneDir, uint32_t idx)
{
    std::ostringstream oss;
    oss << velodyneDir << "/" << std::setfill('0') << std::setw(6) << idx << ".bin";
    return oss.str();
}

} // namespace kitti_raw

// ─────────────────────────────────────────────────────────────────────────────
//  FastLIO2Feature
// ─────────────────────────────────────────────────────────────────────────────
class FastLIO2Feature : public IFeature
{
public:
    const char *name() const override { return "FastLIO2_SLAM"; }

    void onInit(const VulkanContext &ctx) override
    {
        ctx_ = ctx;

        // BucketVoxelHash 초기화
        vhb_.Initialize(ctx_);
        vhb_.setVoxelSize(voxelSize_);

        // KITTI 데이터 로드
        const std::string base = "/Users/sjy/Desktop/Project/Vision-3D/dataset";
        velodyneDir_ = base + "/kitti/00/velodyne";
        gtLoader_.initialize(base);   // GT 포즈 + veloToLocal

        slam_ = FastLIO2(makeSlamConfig());
        slam_.setMap(vhb_, ctx_);

        totalFrames_ = gtLoader_.frameCount();
        lastFrameTime_ = std::chrono::steady_clock::now();
        lastSlamSeqFrame_ = kNoPrevSlamFrame;

        // 첫 프레임으로 초기 맵 구성
        loadAndProcess(0);
    }

    void onCompute(VkCommandBuffer) override
    {
        // 자동 재생
        if (isPlaying_ && totalFrames_ > 0)
        {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - lastFrameTime_).count();
            if (dt >= 1.f / playSpeed_)
            {
                lastFrameTime_ = now;
                uint32_t next = currentFrame_ + 1;
                if (next < totalFrames_)
                {
                    loadAndProcess(next);
                }
                else
                {
                    isPlaying_ = false;  // 시퀀스 끝
                }
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
        // 마우스 카메라
        ImGuiIO &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                azimuth_   += io.MouseDelta.x * 0.005f;
                elevation_ += io.MouseDelta.y * 0.005f;
                elevation_  = std::max(-1.4f, std::min(1.4f, elevation_));
            }
            camDist_ -= io.MouseWheel * 2.f;
            camDist_  = std::max(1.f, std::min(500.f, camDist_));
        }

        ImGui::Begin("FastLIO2 SLAM");

        // ── 재생 제어 ─────────────────────────────────────────────────────────
        if (ImGui::Button(isPlaying_ ? "Pause" : "Play "))
            isPlaying_ = !isPlaying_;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::SliderFloat("FPS##slam", &playSpeed_, 1.f, 20.f, "%.0f");

        ImGui::SetNextItemWidth(-1);
        int frame = (int)currentFrame_;
        if (ImGui::SliderInt("##slamframe", &frame, 0,
                              totalFrames_ > 0 ? (int)totalFrames_ - 1 : 0))
        {
            isPlaying_ = false;
            loadAndProcess((uint32_t)frame);
        }
        ImGui::Text("Frame %u / %u", currentFrame_,
                    totalFrames_ > 0 ? totalFrames_ - 1 : 0u);
        ImGui::Separator();

        // ── SLAM 상태 ─────────────────────────────────────────────────────────
        ImGui::Text("SLAM Status: %s",
                    lastValid_ ? "Converged" : "Initializing / Failed");
        ImGui::Text("IEKF: %s | m=%d",
                    FastLIO2::iekfFailString(slam_.lastIEKFFail()),
                    slam_.lastIEKFResidualCount());

        if (slam_.isInitialized())
        {
            const auto &s = slam_.getState();
            ImGui::Text("Pos  SLAM: %.2f  %.2f  %.2f", s.p.x(), s.p.y(), s.p.z());

            // GT 포즈와 비교
            if (currentFrame_ < totalFrames_)
            {
                Eigen::Vector3f gtPos = gtLoader_.sensorPos(currentFrame_);
                float err = (s.p - gtPos).norm();
                ImGui::Text("Pos  GT  : %.2f  %.2f  %.2f", gtPos.x(), gtPos.y(), gtPos.z());
                ImGui::Text("Pos Error: %.3f m", err);
            }
        }
        ImGui::Separator();

        // ── 렌더 설정 ─────────────────────────────────────────────────────────
        if (ImGui::RadioButton("Point", renderMode_ == 0)) renderMode_ = 0;
        ImGui::SameLine();
        if (ImGui::RadioButton("Voxel", renderMode_ == 1)) renderMode_ = 1;

        if (ImGui::SliderFloat("Voxel Size##slam", &voxelSize_, 0.1f, 2.f, "%.2f"))
        {
            vhb_.setVoxelSize(voxelSize_);
            resetSLAM();
        }
        ImGui::Separator();

        if (ImGui::Button("Reset SLAM")) resetSLAM();

        ImGui::TextDisabled("Mouse drag: orbit | Scroll: zoom");
        ImGui::TextDisabled("Space: play/pause | Arrow: step frame");
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
                loadAndProcess(currentFrame_ + 1);
            }
            break;
        case GLFW_KEY_LEFT:
            if (currentFrame_ > 0)
            {
                isPlaying_ = false;
                loadAndProcess(currentFrame_ - 1);
            }
            break;
        }
    }

    void onCleanup() override {}

private:
    // ── 핵심: 프레임 로드 + FastLIO2 처리 ──────────────────────────────────────
    void loadAndProcess(uint32_t frameIdx)
    {
        currentFrame_ = frameIdx;

        if (slam_.isInitialized() && lastSlamSeqFrame_ != kNoPrevSlamFrame
            && frameIdx != lastSlamSeqFrame_ + 1u)
            rebuildSlamOnly();

        // 센서 프레임 raw 포인트 로드
        std::string path = kitti_raw::binPath(velodyneDir_, frameIdx);
        auto rawPts = kitti_raw::loadRaw(path, VH_BATCH_SIZE);
        if (rawPts.empty()) return;

        Eigen::Matrix4f Tvl = gtLoader_.veloToLocal(frameIdx);
        Eigen::Matrix3f R = Tvl.block<3, 3>(0, 0);
        Eigen::Vector3f t = Tvl.block<3, 1>(0, 3);

        // 첫 스캔 전: x_opt_를 GT에 맞춰 맵 좌표계를 Viewer/KITTI 로컬과 일치
        if (!slam_.isInitialized())
            slam_.setWorldPoseFromVelo(R, t);
        else
            slam_.setPredictFromVelo(R, t);

        double scanTime = static_cast<double>(frameIdx) * 0.1;  // 10Hz 가정
        auto result = slam_.feedScan(rawPts, scanTime);
        lastValid_ = result.valid;

        lastSlamSeqFrame_ = frameIdx;
    }

    void resetSLAM()
    {
        vhb_.Clear(ctx_);
        slam_ = FastLIO2(makeSlamConfig());
        slam_.setMap(vhb_, ctx_);
        lastSlamSeqFrame_ = kNoPrevSlamFrame;
        currentFrame_     = 0;
        lastValid_        = false;
        loadAndProcess(0);
    }

    void rebuildSlamOnly()
    {
        vhb_.Clear(ctx_);
        slam_ = FastLIO2(makeSlamConfig());
        slam_.setMap(vhb_, ctx_);
        lastSlamSeqFrame_ = kNoPrevSlamFrame;
    }

    static FastLIO2::Config makeSlamConfig()
    {
        FastLIO2::Config cfg;
        // KITTI 데모: 상태 R,p = velodyne world pose (LiDAR 바디 = nominal IMU 바디)
        cfg.R_LI_init = Eigen::Matrix3f::Identity();
        cfg.p_LI_init = Eigen::Vector3f::Zero();
        cfg.meas_noise         = 0.05f;
        cfg.converge_eps       = 0.01f;
        cfg.max_iterations     = 20;
        cfg.map_move_threshold = 20.f;
        cfg.map_size_L         = 200.f;
        return cfg;
    }

    Eigen::Matrix4f computeMVP() const
    {
        float cx = std::cos(elevation_), sx = std::sin(elevation_);
        float cy = std::cos(azimuth_),   sy = std::sin(azimuth_);
        Eigen::Vector3f eye(camDist_*cx*sy, camDist_*sx, camDist_*cx*cy);
        Eigen::Vector3f up(0.f, 1.f, 0.f);
        Eigen::Vector3f f  = (-eye).normalized();
        Eigen::Vector3f rr = f.cross(up).normalized();
        Eigen::Vector3f u  = rr.cross(f);

        Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
        V.row(0) << rr.x(), rr.y(), rr.z(), -rr.dot(eye);
        V.row(1) << u.x(),  u.y(),  u.z(),  -u.dot(eye);
        V.row(2) << -f.x(), -f.y(), -f.z(),  f.dot(eye);

        float fovY   = 60.f * 3.14159f / 180.f;
        float aspect = (float)ctx_.extent.width / ctx_.extent.height;
        float n = 0.1f, fa = 1000.f, th = std::tan(fovY * 0.5f);

        Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
        P(0,0) =  1.f / (aspect * th);
        P(1,1) = -1.f / th;
        P(2,2) =  fa / (n - fa);
        P(2,3) =  fa * n / (n - fa);
        P(3,2) = -1.f;

        return P * V;
    }

    // ── 멤버 변수 ──────────────────────────────────────────────────────────────
    VulkanContext   ctx_{};
    BucketVoxelHash vhb_;
    FastLIO2        slam_{FastLIO2::Config{}};
    KITTILoader     gtLoader_;   // GT 포즈 비교용

    std::string velodyneDir_;

    static constexpr uint32_t kNoPrevSlamFrame = 0xFFFFFFFFu;
    uint32_t lastSlamSeqFrame_ = kNoPrevSlamFrame;

    uint32_t currentFrame_ = 0;
    uint32_t totalFrames_  = 0;
    bool     lastValid_    = false;
    bool     isPlaying_    = false;
    float    playSpeed_    = 5.f;
    std::chrono::steady_clock::time_point lastFrameTime_;

    int   renderMode_  = 0;
    float voxelSize_   = 0.5f;    // KITTI 야외 스케일에 맞게 50cm

    float azimuth_   = 0.f;
    float elevation_ = 0.3f;
    float camDist_   = 100.f;
};
