#include "VoxelHash.h"
#include <imgui.h>
#include <array>

// ─────────────────────────────────────────────────────────────────────────────
//  readPerfQueries — GPU timestamp 쿼리 결과 읽기
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::readPerfQueries()
{
    if (perfQueryPool_ == VK_NULL_HANDLE || !hasPerfQuery_) return;

    std::array<uint64_t, 8> t{};
    VkResult res = vkGetQueryPoolResults(
        ctx_.device, perfQueryPool_, 0, static_cast<uint32_t>(t.size()),
        sizeof(t), t.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (res != VK_SUCCESS) return;

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx_.physicalDevice, &props);
    const float nsToMs = props.limits.timestampPeriod * 1e-6f;

    sortMs_     = static_cast<float>(t[1] - t[0]) * nsToMs;
    gatherMs_   = static_cast<float>(t[3] - t[2]) * nsToMs;
    finalizeMs_ = static_cast<float>(t[5] - t[4]) * nsToMs;
    countMs_    = static_cast<float>(t[7] - t[6]) * nsToMs;
}

// ─────────────────────────────────────────────────────────────────────────────
//  onImGui
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::onImGui()
{
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            azimuth_   += io.MouseDelta.x * 0.005f;
            elevation_ += io.MouseDelta.y * 0.005f;
            elevation_  = std::max(-1.4f, std::min(1.4f, elevation_));
        }
        camDist_ -= io.MouseWheel * 0.002f;
        camDist_  = std::max(0.005f, std::min(0.5f, camDist_));
    }

    ImGui::Begin("VoxelHash");

    // ── 해시 테이블 통계 ──────────────────────────────────────────────────────
    float htMB    = sizeof(VH_Entry) * VH_TOTAL_ENTRIES / (1024.f * 1024.f);
    float fillPct = 100.f * occupancy_ / static_cast<float>(VH_TOTAL_ENTRIES);
    ImGui::Text("Hash table : %u buckets × %u = %u entries  (%.0f MB)",
                VH_NUM_BUCKETS, VH_BUCKET_SIZE, VH_TOTAL_ENTRIES, htMB);
    ImGui::Text("Occupied   : %d / %u  (%.2f%%)", occupancy_, VH_TOTAL_ENTRIES, fillPct);
    ImGui::Text("Total pts  : %lld  |  %.0f pts/s", static_cast<long long>(totalInserted_), ptsPerSec_);
    ImGui::Text("Frame      : %u", frameIndex_);
    ImGui::Separator();

    // ── GPU 성능 타이밍 ───────────────────────────────────────────────────────
    ImGui::Text("GPU Timings (ms)");
    ImGui::Text("  Sort   : %.3f  |  Gather : %.3f", sortMs_, gatherMs_);
    ImGui::Text("  Finalize: %.3f |  Count  : %.3f", finalizeMs_, countMs_);
    ImGui::InputFloat("Baseline sort (ms)", &baselineMs_, 0.1f, 1.f, "%.3f");
    if (baselineMs_ > 0.f)
        ImGui::Text("  Sort improvement: %+.1f%%",
                    (baselineMs_ - sortMs_) / baselineMs_ * 100.f);
    ImGui::Separator();

    // ── 카메라 ────────────────────────────────────────────────────────────────
    ImGui::Text("Camera  [drag=rotate  wheel=zoom]");
    ImGui::SliderFloat("Azimuth##vh",   &azimuth_,   -3.14f, 3.14f, "%.2f");
    ImGui::SliderFloat("Elevation##vh", &elevation_, -1.4f,  1.4f,  "%.2f");
    ImGui::SliderFloat("Distance##vh",  &camDist_,    0.005f, 0.5f, "%.4f m");
    ImGui::Separator();

    // ── 색상 모드 ─────────────────────────────────────────────────────────────
    ImGui::Text("Color mode");
    ImGui::RadioButton("Surface color", &colorMode_, 0); ImGui::SameLine();
    ImGui::RadioButton("Normal",        &colorMode_, 1); ImGui::SameLine();
    ImGui::RadioButton("TSDF",          &colorMode_, 2); ImGui::SameLine();
    ImGui::RadioButton("Weight",        &colorMode_, 3); ImGui::SameLine();
    ImGui::RadioButton("Recency",       &colorMode_, 4);
    ImGui::Separator();

    // ── 디버그 플래시 ─────────────────────────────────────────────────────────
    ImGui::Text("Integration visualization");
    ImGui::SliderInt("Highlight frames", &highlightFrames_, 1, 60, "flash %d frames");
    ImGui::TextColored(ImVec4(1,0.9f,0.1f,1), "■"); ImGui::SameLine();
    ImGui::Text("= recently updated voxel");
    ImGui::Checkbox("Show input points (cyan)", &showInputPts_);
    if (showInputPts_) {
        ImGui::TextColored(ImVec4(0,1,1,1), "●"); ImGui::SameLine();
        ImGui::Text("= current batch (%u pts)", ptCount_);
    }
    ImGui::Separator();

    // ── TSDF 파라미터 ────────────────────────────────────────────────────────
    bool rebuild = false;
    rebuild |= ImGui::SliderFloat("Voxel size (m)", &voxelSize_,   0.00005f, 0.002f, "%.5f");
    rebuild |= ImGui::SliderFloat("Truncation (m)", &truncation_,  0.0001f,  0.002f, "%.4f");
    ImGui::SliderFloat("Max weight", &maxWeight_, 5.f, 200.f, "%.0f");
    ImGui::Separator();

    // ── 합성 데이터 파라미터 ──────────────────────────────────────────────────
    ImGui::Text("Synthetic sphere generator");
    ImGui::SliderFloat("Sphere r (m)",  &sphereRadius_, 0.001f, 0.05f, "%.4f");
    ImGui::SliderFloat("Noise (m)",     &noiseStddev_,  0.f,    0.00005f, "%.6f");
    ImGui::SliderFloat("Speed (rad/f)", &sensorSpeed_,  0.f,    0.1f,    "%.3f");
    if (rebuild && streaming_) doClear_ = true;
    ImGui::Separator();

    // ── 제어 ─────────────────────────────────────────────────────────────────
    if (ImGui::Button(streaming_ ? "■ Stop" : "▶ Start")) {
        streaming_ = !streaming_;
        if (streaming_) { doClear_ = true; sensorAz_ = 0.f; }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) doClear_ = true;

    if (streaming_) {
        float cx = std::cos(sensorEl_), sx = std::sin(sensorEl_);
        float cy = std::cos(sensorAz_), sy = std::sin(sensorAz_);
        ImGui::Text("Sensor: (%.4f, %.4f, %.4f) m",
                    sensorDist_*cx*sy, sensorDist_*sx, sensorDist_*cx*cy);
    }

    ImGui::End();
}
