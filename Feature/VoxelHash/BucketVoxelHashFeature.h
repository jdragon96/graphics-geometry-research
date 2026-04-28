#pragma once

#include "BucketVoxelHash.h"
#include "../IFeature.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdlib>
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
        doIntegrate();
    }

    void onRender(const RenderContext &ctx) override
    {
        Eigen::Matrix4f mvp = computeMVP();
        if (renderMode_ == 0)
        {
            vhb_.RenderColor(ctx.commandBuffer, mvp.data(), useSlotPoint_);
        }
        else
        {
            vhb_.RenderVoxel(ctx.commandBuffer, mvp.data());
            if (showVoxelSlotPoint_)
                vhb_.RenderVoxelSlot(ctx.commandBuffer, mvp.data());
        }
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
            camDist_ -= io.MouseWheel * 0.3f;
            camDist_ = std::max(0.5f, std::min(20.f, camDist_));
        }

        ImGui::Begin("BucketVoxelHash");

        ImGui::Text("Render Mode");
        if (ImGui::RadioButton("Point Cloud", renderMode_ == 0)) renderMode_ = 0;
        ImGui::SameLine();
        if (ImGui::RadioButton("Voxel Cube",  renderMode_ == 1)) renderMode_ = 1;
        ImGui::Checkbox("Use Slot Point", &useSlotPoint_);
        if (renderMode_ == 1)
            ImGui::Checkbox("Show Voxel Slot Points", &showVoxelSlotPoint_);
        ImGui::Separator();

        ImGui::SliderFloat("Sphere R", &sphereRadius_, 0.1f, 3.f, "%.2f");
        ImGui::SliderFloat("Voxel",    &voxelSize_,    0.005f, 0.1f, "%.4f");
        ImGui::SliderFloat("Points",   &numPoints_,    100.f, 10000.f, "%.0f");
        if (ImGui::Button("Re-integrate"))
        {
            vhb_.Clear(ctx_);
            doIntegrate();
        }
        ImGui::End();
    }

    void onCompute(VkCommandBuffer) override {}
    void onCleanup() override {}

private:
    VulkanContext ctx_{};
    BucketVoxelHash vhb_;

    int   renderMode_   = 0;   // 0 = point cloud, 1 = voxel cube
    bool  useSlotPoint_ = false;
    bool  showVoxelSlotPoint_ = false;
    float azimuth_ = 0.5f;
    float elevation_ = 0.4f;
    float camDist_ = 4.0f;
    float sphereRadius_ = 1.0f;
    float voxelSize_ = 0.02f;
    float numPoints_ = 5000.f;

    void doIntegrate()
    {
        const uint32_t N = static_cast<uint32_t>(numPoints_);
        std::vector<VH_InputPoint> pts(N);

        srand(42);
        for (uint32_t i = 0; i < N; ++i)
        {
            // 구면 균등 분포 (Marsaglia)
            float u = (rand() / (float)RAND_MAX) * 2.f - 1.f;
            float t = (rand() / (float)RAND_MAX) * 2.f * 3.14159f;
            float r = std::sqrt(std::max(0.f, 1.f - u * u));
            float nx = r * std::cos(t), ny = u, nz = r * std::sin(t);
            float px = nx * sphereRadius_, py = ny * sphereRadius_, pz = nz * sphereRadius_;

            pts[i].px = px;
            pts[i].py = py;
            pts[i].pz = pz;
            pts[i].nx = nx;
            pts[i].ny = ny;
            pts[i].nz = nz;
            // 위치 기반 색상
            uint32_t h = uint32_t(int(px * 100)) ^ uint32_t(int(py * 100) * 19349663u) ^ uint32_t(int(pz * 100) * 83492791u);
            pts[i].col = 0xFF000000u | (h & 0xFFFFFFu);
            pts[i]._pad = 0.f;
        }

        // 센서 = (0, 0, 5) 방향에서 관측
        vhb_.Integrate(ctx_, pts.data(), N, 0.f, 0.f, 5.f);
    }

    Eigen::Matrix4f computeMVP() const
    {
        float cx = std::cos(elevation_), sx = std::sin(elevation_);
        float cy = std::cos(azimuth_), sy = std::sin(azimuth_);
        Eigen::Vector3f eye(camDist_ * cx * sy, camDist_ * sx, camDist_ * cx * cy);
        Eigen::Vector3f up(0.f, 1.f, 0.f);
        Eigen::Vector3f f = (-eye).normalized();
        Eigen::Vector3f rr = f.cross(up).normalized();
        Eigen::Vector3f u = rr.cross(f);

        Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
        V.row(0) << rr.x(), rr.y(), rr.z(), -rr.dot(eye);
        V.row(1) << u.x(), u.y(), u.z(), -u.dot(eye);
        V.row(2) << -f.x(), -f.y(), -f.z(), f.dot(eye);

        float fovY = 60.f * 3.14159f / 180.f;
        float aspect = (float)ctx_.extent.width / ctx_.extent.height;
        float n = 0.01f, fa = 100.f, th = std::tan(fovY * 0.5f);

        Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
        P(0, 0) = 1.f / (aspect * th);
        P(1, 1) = -1.f / th;
        P(2, 2) = fa / (n - fa);
        P(2, 3) = fa * n / (n - fa);
        P(3, 2) = -1.f;

        return P * V;
    }
};
