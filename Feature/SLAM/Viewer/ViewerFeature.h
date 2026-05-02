#pragma once

#include "Viewer.h"
#include "../../IFeature.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <chrono>
#include <cmath>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
//  KITTIViewerFeature — KITTI sequence 00 재생 + VoxelHash 시각화
// ─────────────────────────────────────────────────────────────────────────────
class KITTIViewerFeature : public IFeature
{
public:
    const char *name() const override { return "KITTIViewer"; }

    void onInit(const VulkanContext &ctx) override
    {
        const std::string kDatasetBase =
            "/Users/sjy/Desktop/Project/Vision-3D/dataset";

        viewer_.Initialize(ctx, kDatasetBase);
        viewer_.setVoxelSize(voxelSize_);
        lastFrameTime_ = std::chrono::steady_clock::now();

        // 첫 프레임 미리 integrate
        viewer_.IntegrateFrame(0);
    }

    // ── 매 프레임: 타이머 → 프레임 advance → integrate ────────────────────────
    void onCompute(VkCommandBuffer) override
    {
        if (viewer_.frameCount() == 0) return;

        bool advance = false;

        if (isPlaying_)
        {
            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - lastFrameTime_).count();
            if (dt >= 1.f / playSpeed_)
            {
                currentFrame_ = (currentFrame_ + 1) % viewer_.frameCount();
                lastFrameTime_ = now;
                advance = true;
            }
        }

        if (!needsUpdate_ && !advance) return;
        needsUpdate_ = false;

        if (!accumulate_) viewer_.ClearMap();
        viewer_.IntegrateFrame(currentFrame_);
    }

    void onRender(const RenderContext &ctx) override
    {
        if (viewer_.frameCount() == 0) return;

        Eigen::Matrix4f mvp = computeMVP();
        if (renderMode_ == 0)
            viewer_.RenderColor(ctx.commandBuffer, mvp.data());
        else
            viewer_.RenderVoxel(ctx.commandBuffer, mvp.data());
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
            camDist_ -= io.MouseWheel * 1.f;
            camDist_  = std::max(1.f, std::min(200.f, camDist_));
        }

        ImGui::Begin("KITTIViewer");

        const uint32_t total = viewer_.frameCount();

        // ── 재생 제어 ──────────────────────────────────────────────────────────
        if (ImGui::Button(isPlaying_ ? "Pause" : "Play "))
            isPlaying_ = !isPlaying_;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        ImGui::SliderFloat("FPS", &playSpeed_, 1.f, 30.f, "%.0f");

        ImGui::SetNextItemWidth(-1);
        int frame = static_cast<int>(currentFrame_);
        if (ImGui::SliderInt("##frame", &frame, 0,
                             total > 0 ? (int)total - 1 : 0))
        {
            currentFrame_ = static_cast<uint32_t>(frame);
            needsUpdate_  = true;
        }
        ImGui::Text("Frame %u / %u", currentFrame_, total > 0 ? total - 1 : 0);
        ImGui::Separator();

        // ── 맵 설정 ───────────────────────────────────────────────────────────
        if (ImGui::Checkbox("Accumulate (Map)", &accumulate_) && !accumulate_)
        {
            viewer_.ClearMap();
            needsUpdate_ = true;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("OFF: 프레임마다 Clear (단일 스캔)\nON: 누적 맵 구축");

        if (ImGui::SliderFloat("Voxel Size", &voxelSize_, 0.05f, 2.f, "%.2f"))
        {
            viewer_.setVoxelSize(voxelSize_);
            viewer_.ClearMap();
            needsUpdate_ = true;
        }
        ImGui::Separator();

        // ── 렌더 모드 ─────────────────────────────────────────────────────────
        ImGui::Text("Render");
        if (ImGui::RadioButton("Point Cloud", renderMode_ == 0)) renderMode_ = 0;
        ImGui::SameLine();
        if (ImGui::RadioButton("Voxel Cube",  renderMode_ == 1)) renderMode_ = 1;
        ImGui::Separator();

        if (ImGui::Button("Clear Map"))
        {
            viewer_.ClearMap();
            needsUpdate_ = true;
        }

        ImGui::TextDisabled("Mouse drag: orbit | Scroll: zoom");
        ImGui::TextDisabled("Arrow keys: step frame | Space: play/pause");

        ImGui::End();
    }

    void onKey(int key, int action, int /*mods*/) override
    {
        if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

        const uint32_t total = viewer_.frameCount();
        if (total == 0) return;

        switch (key)
        {
        case GLFW_KEY_RIGHT:
            currentFrame_ = std::min(currentFrame_ + 1, total - 1);
            needsUpdate_  = true;
            break;
        case GLFW_KEY_LEFT:
            if (currentFrame_ > 0) { --currentFrame_; needsUpdate_ = true; }
            break;
        case GLFW_KEY_SPACE:
            isPlaying_ = !isPlaying_;
            break;
        }
    }

    void onCleanup() override {}

private:
    Viewer viewer_;

    // 재생 상태
    uint32_t currentFrame_ = 0;
    bool     isPlaying_    = false;
    float    playSpeed_    = 10.f;
    bool     accumulate_   = false;
    bool     needsUpdate_  = false;
    std::chrono::steady_clock::time_point lastFrameTime_;

    // 렌더
    int   renderMode_ = 0;
    float voxelSize_  = 0.2f;

    // 카메라
    float azimuth_   = 0.f;
    float elevation_ = 0.3f;
    float camDist_   = 50.f;

    // ─────────────────────────────────────────────────────────────────────────

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
        float aspect = 800.f / 600.f;  // 앱 초기 해상도
        float n = 0.1f, fa = 1000.f, th = std::tan(fovY * 0.5f);

        Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
        P(0, 0) =  1.f / (aspect * th);
        P(1, 1) = -1.f / th;
        P(2, 2) =  fa / (n - fa);
        P(2, 3) =  fa * n / (n - fa);
        P(3, 2) = -1.f;

        return P * V;
    }
};
