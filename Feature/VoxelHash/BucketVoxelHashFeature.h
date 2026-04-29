#pragma once

#include "BucketVoxelHash.h"
#include "../IFeature.h"
#include "../Acquisition/SimpleLiDAR.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <GLFW/glfw3.h>
#include <cmath>
#include <imgui.h>
#include <vector>

// IFeature 래퍼: BucketVoxelHash를 Application 프레임 루프에 연결
class BucketVoxelHashFeature : public IFeature
{
public:
    const char *name() const override { return "BucketVoxelHash"; }

    void onInit(const VulkanContext &ctx) override
    {
        ctx_ = ctx;
        vhb_.Initialize(ctx_);
        lidar_.addDefaultScene();
        doIntegrate();
    }

    // ── 매 프레임: 키 상태 반영 → 이동 → 실시간 integrate ─────────────────────
    void onCompute(VkCommandBuffer) override
    {
        bool moved = false;
        if (keyW_) { sensorZ_ += moveSpeed_; moved = true; }
        if (keyS_) { sensorZ_ -= moveSpeed_; moved = true; }
        if (keyA_) { sensorX_ -= moveSpeed_; moved = true; }
        if (keyD_) { sensorX_ += moveSpeed_; moved = true; }
        if (keyQ_) { sensorY_ -= moveSpeed_; moved = true; }
        if (keyE_) { sensorY_ += moveSpeed_; moved = true; }

        if (!realtimeIntegrate_) return;

        if (liveMode_)
            vhb_.Clear(ctx_);   // 매 프레임 초기화 → 현재 스캔만 표시

        if (!liveMode_ && !moved) return;   // accumulate 모드에선 이동 시에만 갱신

        doIntegrate();
    }

    void onRender(const RenderContext &ctx) override
    {
        Eigen::Matrix4f mvp = computeMVP();
        if (renderMode_ == 0)
            vhb_.RenderColor(ctx.commandBuffer, mvp.data(), useSlotPoint_);
        else
        {
            vhb_.RenderVoxel(ctx.commandBuffer, mvp.data());
            if (showVoxelSlotPoint_)
                vhb_.RenderVoxelSlot(ctx.commandBuffer, mvp.data());
        }
    }

    void onImGui() override
    {
        // ── 마우스 카메라 조작 ────────────────────────────────────────────────
        ImGuiIO &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                azimuth_   += io.MouseDelta.x * 0.005f;
                elevation_ += io.MouseDelta.y * 0.005f;
                elevation_  = std::max(-1.4f, std::min(1.4f, elevation_));
            }
            camDist_ -= io.MouseWheel * 0.3f;
            camDist_  = std::max(0.5f, std::min(50.f, camDist_));
        }

        ImGui::Begin("BucketVoxelHash");

        // ── 렌더 모드 ──────────────────────────────────────────────────────────
        ImGui::Text("Render Mode");
        if (ImGui::RadioButton("Point Cloud", renderMode_ == 0)) renderMode_ = 0;
        ImGui::SameLine();
        if (ImGui::RadioButton("Voxel Cube",  renderMode_ == 1)) renderMode_ = 1;
        ImGui::Checkbox("Use Slot Point", &useSlotPoint_);
        if (renderMode_ == 1)
            ImGui::Checkbox("Show Voxel Slot Points", &showVoxelSlotPoint_);
        ImGui::Separator();

        // ── 센서 위치 + 이동 ──────────────────────────────────────────────────
        ImGui::Text("Sensor  X %.2f  Y %.2f  Z %.2f", sensorX_, sensorY_, sensorZ_);
        ImGui::SliderFloat("Move Speed", &moveSpeed_, 0.01f, 1.f, "%.2f");
        ImGui::TextDisabled("WASD: XZ   Q/E: Y");
        if (ImGui::Button("Reset Sensor"))
            sensorX_ = sensorY_ = sensorZ_ = 0.f;
        ImGui::Separator();

        // ── LiDAR 파라미터 ────────────────────────────────────────────────────
        ImGui::Text("LiDAR Config");
        SimpleLiDAR::Config cfg = lidar_.getConfig();
        bool cfgChanged = false;
        int rings = (int)cfg.numRings;
        int ppr   = (int)cfg.pointsPerRing;
        cfgChanged |= ImGui::SliderInt  ("Rings",          &rings,          1,   64);
        cfgChanged |= ImGui::SliderInt  ("Points/Ring",    &ppr,           45,  720);
        cfgChanged |= ImGui::SliderFloat("Elev Min (°)",  &cfg.elevMinDeg, -45.f,  0.f, "%.1f");
        cfgChanged |= ImGui::SliderFloat("Elev Max (°)",  &cfg.elevMaxDeg,   0.f, 45.f, "%.1f");
        cfgChanged |= ImGui::SliderFloat("Max Range (m)", &cfg.maxRange,    1.f, 50.f,  "%.1f");
        if (cfgChanged)
        {
            cfg.numRings      = (uint32_t)rings;
            cfg.pointsPerRing = (uint32_t)ppr;
            lidar_.setConfig(cfg);
        }
        ImGui::Separator();

        // ── 실시간 제어 ───────────────────────────────────────────────────────
        ImGui::Checkbox("Realtime Integrate", &realtimeIntegrate_);
        if (realtimeIntegrate_)
        {
            ImGui::SameLine();
            ImGui::Checkbox("Live Mode", &liveMode_);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Live: 매 프레임 초기화 후 스캔\nAccumulate: 이동 시에만 누적");
        }
        ImGui::SliderFloat("Voxel Size", &voxelSize_, 0.005f, 0.5f, "%.3f");
        if (ImGui::Button("Re-integrate"))
        {
            vhb_.Clear(ctx_);
            doIntegrate();
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear Map"))
            vhb_.Clear(ctx_);

        ImGui::End();
    }

    // ── 키 입력: press/release 으로 플래그 토글 ───────────────────────────────
    void onKey(int key, int action, int /*mods*/) override
    {
        bool down = (action == GLFW_PRESS || action == GLFW_REPEAT);
        switch (key)
        {
        case GLFW_KEY_W: keyW_ = down; break;
        case GLFW_KEY_S: keyS_ = down; break;
        case GLFW_KEY_A: keyA_ = down; break;
        case GLFW_KEY_D: keyD_ = down; break;
        case GLFW_KEY_Q: keyQ_ = down; break;
        case GLFW_KEY_E: keyE_ = down; break;
        }
    }

    void onCleanup() override {}

private:
    VulkanContext   ctx_{};
    BucketVoxelHash vhb_;
    SimpleLiDAR     lidar_;

    // ── 렌더 옵션 ─────────────────────────────────────────────────────────────
    int   renderMode_         = 0;
    bool  useSlotPoint_       = false;
    bool  showVoxelSlotPoint_ = false;
    float voxelSize_          = 0.05f;

    // ── 카메라 ────────────────────────────────────────────────────────────────
    float azimuth_   = 0.5f;
    float elevation_ = 0.4f;
    float camDist_   = 10.f;

    // ── 센서 위치 + 이동 ──────────────────────────────────────────────────────
    float sensorX_ = 0.f, sensorY_ = 0.f, sensorZ_ = 0.f;
    float moveSpeed_ = 0.1f;
    bool  keyW_ = false, keyS_ = false;
    bool  keyA_ = false, keyD_ = false;
    bool  keyQ_ = false, keyE_ = false;

    // ── 실시간 모드 ───────────────────────────────────────────────────────────
    bool realtimeIntegrate_ = true;
    bool liveMode_          = false;  // true = 매프레임 clear, false = 누적

    // ─────────────────────────────────────────────────────────────────────────

    void doIntegrate()
    {
        vhb_.setVoxelSize(voxelSize_);

        auto pts = lidar_.scan(sensorX_, sensorY_, sensorZ_);
        if (pts.empty()) return;

        uint32_t N = std::min((uint32_t)pts.size(), VH_BATCH_SIZE);
        vhb_.Integrate(ctx_, pts.data(), N, sensorX_, sensorY_, sensorZ_);
    }

    Eigen::Matrix4f computeMVP() const
    {
        float cx = std::cos(elevation_), sx = std::sin(elevation_);
        float cy = std::cos(azimuth_),   sy = std::sin(azimuth_);
        Eigen::Vector3f eye(camDist_ * cx * sy, camDist_ * sx, camDist_ * cx * cy);
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
        float n = 0.01f, fa = 100.f, th = std::tan(fovY * 0.5f);

        Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
        P(0, 0) =  1.f / (aspect * th);
        P(1, 1) = -1.f / th;
        P(2, 2) =  fa / (n - fa);
        P(2, 3) =  fa * n / (n - fa);
        P(3, 2) = -1.f;

        return P * V;
    }
};
