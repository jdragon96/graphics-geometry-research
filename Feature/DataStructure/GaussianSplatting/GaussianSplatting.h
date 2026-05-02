#pragma once

#include "../IFeature.h"
#include "../Common/Buffer.h"
#include "../Common/RenderingShader.h"
#include "../Acquisition/SimpleLiDAR.h"
#include "../VoxelHash/VoxelHashTypes.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
static constexpr uint32_t GS_MAX_SPLATS = 50000;

// ── GPU splat struct (48 bytes, alignas(16)) ───────────────────────────────────
struct alignas(16) GS_Gaussian {
    float    px, py, pz;   // center position
    float    scaleX;       // billboard half-width
    float    nx, ny, nz;   // surface normal (Lambertian shading)
    float    scaleY;       // billboard half-height
    float    opacity;
    uint32_t col;          // r8g8b8a8
    float    _pad[2];
};
static_assert(sizeof(GS_Gaussian) == 48, "GS_Gaussian size mismatch");

// ── Push constant (96 bytes ≤ 128) ────────────────────────────────────────────
struct GS_RenderPC {
    float mvp[16];
    float camRight[3];
    float splatScale;
    float camUp[3];
    float _pad;
};
static_assert(sizeof(GS_RenderPC) <= 128, "GS_RenderPC too large");

// ─────────────────────────────────────────────────────────────────────────────
//  GaussianSplatting — GPU 버퍼 + 렌더 파이프라인 관리
// ─────────────────────────────────────────────────────────────────────────────
class GaussianSplatting
{
public:
    void Initialize(const VulkanContext &ctx);

    // 새 스캔을 기존 splat 뒤에 누적 append (SLAM 맵 구축)
    // 버퍼가 가득 차면 더 이상 추가하지 않음 → Clear() 호출로 초기화
    void Integrate(const VH_InputPoint *pts, uint32_t count);

    // 렌더 패스 내에서 호출. splatCount_*6 vertices draw.
    void Render(VkCommandBuffer cmd, const GS_RenderPC &pc);

    // 누적 카운터 초기화 (버퍼 메모리는 유지, 덮어쓰기로 재사용됨)
    void Clear() { splatCount_ = 0; }

    uint32_t getSplatCount()    const { return splatCount_; }
    bool     isBufferFull()     const { return splatCount_ >= GS_MAX_SPLATS; }

private:
    Buffer          splatBuffer_;
    RenderingShader shader_;
    uint32_t        splatCount_ = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  GaussianSplattingFeature — IFeature 래퍼
// ─────────────────────────────────────────────────────────────────────────────
class GaussianSplattingFeature : public IFeature
{
public:
    const char *name() const override { return "GaussianSplatting"; }

    void onInit(const VulkanContext &ctx) override
    {
        ctx_ = ctx;
        gs_.Initialize(ctx_);
        lidar_.addDefaultScene();
        doIntegrate();
    }

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
        if (liveMode_) gs_.Clear();
        if (!liveMode_ && !moved) return;
        doIntegrate();
    }

    void onRender(const RenderContext &ctx) override
    {
        Eigen::Matrix4f mvp = computeMVP();

        float cx = std::cos(elevation_), sx = std::sin(elevation_);
        float cy = std::cos(azimuth_),   sy = std::sin(azimuth_);
        Eigen::Vector3f eye(camDist_ * cx * sy, camDist_ * sx, camDist_ * cx * cy);
        Eigen::Vector3f f     = (-eye).normalized();
        Eigen::Vector3f right = f.cross(Eigen::Vector3f(0.f, 1.f, 0.f)).normalized();
        Eigen::Vector3f camUp = right.cross(f);

        GS_RenderPC pc{};
        const float *m = mvp.data();
        for (int i = 0; i < 16; ++i) pc.mvp[i] = m[i];
        pc.camRight[0] = right.x(); pc.camRight[1] = right.y(); pc.camRight[2] = right.z();
        pc.splatScale  = splatScale_;
        pc.camUp[0]    = camUp.x();  pc.camUp[1]    = camUp.y();  pc.camUp[2]    = camUp.z();
        pc._pad        = 0.f;

        gs_.Render(ctx.commandBuffer, pc);
    }

    void onImGui() override
    {
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

        ImGui::Begin("GaussianSplatting");

        float fillRatio = (float)gs_.getSplatCount() / GS_MAX_SPLATS;
        ImGui::ProgressBar(fillRatio, ImVec2(-1, 0),
            gs_.isBufferFull() ? "Buffer Full!" : "Map Capacity");
        ImGui::Text("Splats: %u / %u", gs_.getSplatCount(), GS_MAX_SPLATS);
        if (gs_.isBufferFull())
            ImGui::TextColored({1,0.4f,0,1}, "Buffer full — Clear Map to reset");
        ImGui::SliderFloat("Splat Scale", &splatScale_, 0.001f, 1.f, "%.3f");
        ImGui::Separator();

        ImGui::Text("Sensor  X %.2f  Y %.2f  Z %.2f", sensorX_, sensorY_, sensorZ_);
        ImGui::SliderFloat("Move Speed", &moveSpeed_, 0.01f, 1.f, "%.2f");
        ImGui::TextDisabled("WASD: XZ   Q/E: Y");
        if (ImGui::Button("Reset Sensor"))
            sensorX_ = sensorY_ = sensorZ_ = 0.f;
        ImGui::Separator();

        ImGui::Text("LiDAR Config");
        SimpleLiDAR::Config cfg = lidar_.getConfig();
        bool cfgChanged = false;
        int rings = (int)cfg.numRings, ppr = (int)cfg.pointsPerRing;
        cfgChanged |= ImGui::SliderInt  ("GS Rings",       &rings,          1,   64);
        cfgChanged |= ImGui::SliderInt  ("GS Points/Ring", &ppr,           45,  720);
        cfgChanged |= ImGui::SliderFloat("GS Elev Min",   &cfg.elevMinDeg, -45.f,  0.f, "%.1f");
        cfgChanged |= ImGui::SliderFloat("GS Elev Max",   &cfg.elevMaxDeg,   0.f, 45.f, "%.1f");
        cfgChanged |= ImGui::SliderFloat("GS Max Range",  &cfg.maxRange,    1.f, 50.f,  "%.1f");
        if (cfgChanged)
        {
            cfg.numRings      = (uint32_t)rings;
            cfg.pointsPerRing = (uint32_t)ppr;
            lidar_.setConfig(cfg);
        }
        ImGui::Separator();

        ImGui::Checkbox("Realtime", &realtimeIntegrate_);
        if (realtimeIntegrate_) { ImGui::SameLine(); ImGui::Checkbox("Live Mode", &liveMode_); }
        if (ImGui::Button("Re-integrate")) { gs_.Clear(); doIntegrate(); }
        ImGui::SameLine();
        if (ImGui::Button("Clear")) gs_.Clear();

        ImGui::End();
    }

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
    void doIntegrate()
    {
        auto pts = lidar_.scan(sensorX_, sensorY_, sensorZ_);
        if (pts.empty()) return;
        uint32_t N = std::min((uint32_t)pts.size(), GS_MAX_SPLATS);
        gs_.Integrate(pts.data(), N);
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

    VulkanContext     ctx_{};
    GaussianSplatting gs_;
    SimpleLiDAR       lidar_;

    float azimuth_   = 0.5f;
    float elevation_ = 0.4f;
    float camDist_   = 10.f;

    float sensorX_ = 0.f, sensorY_ = 0.f, sensorZ_ = 0.f;
    float moveSpeed_  = 0.1f;
    float splatScale_ = 0.05f;

    bool keyW_ = false, keyS_ = false;
    bool keyA_ = false, keyD_ = false;
    bool keyQ_ = false, keyE_ = false;

    bool realtimeIntegrate_ = true;
    bool liveMode_          = false;
};
