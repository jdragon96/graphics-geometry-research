#include "Viewer.h"

#include <Eigen/Dense> // inverse()에 필요한 LU 분해

#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <iomanip>

// ─────────────────────────────────────────────────────────────────────────────
//  helpers
// ─────────────────────────────────────────────────────────────────────────────

// 12-float 행 우선 → Eigen 4×4 (마지막 행 = [0 0 0 1])
static Eigen::Matrix4f parse3x4(const float *v)
{
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            M(r, c) = v[r * 4 + c];
    return M;
}

// ─────────────────────────────────────────────────────────────────────────────
//  KITTILoader
// ─────────────────────────────────────────────────────────────────────────────

void KITTILoader::initialize(const std::string &base)
{
    velodyneDir_ = base + "/kitti/00/velodyne";
    parseCalibration(base + "/kitti_cal/00/calib.txt");
    parsePoses(base + "/kitti_gt_pose/00.txt");

    // 첫 프레임 기준 정규화 행렬
    Eigen::Matrix4f T0 = poses_[0] * Tr_;
    T_base_inv_ = T0.inverse();
}

void KITTILoader::parseCalibration(const std::string &path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("calib.txt not found: " + path);

    std::string line;
    while (std::getline(f, line))
    {
        if (line.substr(0, 3) != "Tr:")
            continue;
        std::istringstream ss(line.substr(3));
        float v[12];
        for (int i = 0; i < 12; ++i)
            ss >> v[i];
        Tr_ = parse3x4(v);
        return;
    }
    throw std::runtime_error("Tr: not found in " + path);
}

void KITTILoader::parsePoses(const std::string &path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("poses file not found: " + path);

    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty())
            continue;
        std::istringstream ss(line);
        float v[12];
        for (int i = 0; i < 12; ++i)
            ss >> v[i];
        poses_.push_back(parse3x4(v));
    }
}

std::string KITTILoader::binPath(uint32_t idx) const
{
    std::ostringstream oss;
    oss << velodyneDir_ << "/" << std::setfill('0') << std::setw(6) << idx << ".bin";
    return oss.str();
}

uint32_t KITTILoader::toColor(float r)
{
    r = std::clamp(r, 0.f, 1.f);
    // hot colormap: 0→흑, 0.5→적, 0.75→황, 1→백
    uint8_t red = static_cast<uint8_t>(std::min(r * 2.f, 1.f) * 255.f);
    uint8_t green = static_cast<uint8_t>(std::max(r * 2.f - 1.f, 0.f) * 255.f);
    uint8_t blue = static_cast<uint8_t>(std::max(r * 4.f - 3.f, 0.f) * 255.f);
    return 0xFF000000u | (uint32_t(blue) << 16) | (uint32_t(green) << 8) | red;
}

Eigen::Vector3f KITTILoader::sensorPos(uint32_t idx) const
{
    // velodyne 원점을 로컬 좌표계로 변환
    return veloToLocal(idx).block<3, 1>(0, 3);
}

Eigen::Matrix4f KITTILoader::veloToLocal(uint32_t frameIdx) const
{
    return T_base_inv_ * poses_[frameIdx] * Tr_;
}

std::vector<VH_InputPoint> KITTILoader::loadFrame(uint32_t idx) const
{
    // ── 1. .bin 읽기 ─────────────────────────────────────────────────────────
    std::string path = binPath(idx);
    FILE *fp = fopen(path.c_str(), "rb");
    if (!fp)
        throw std::runtime_error("Cannot open: " + path);

    fseek(fp, 0, SEEK_END);
    long bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint32_t N = static_cast<uint32_t>(bytes / (4 * sizeof(float)));
    std::vector<float> raw(N * 4);
    fread(raw.data(), sizeof(float), N * 4, fp);
    fclose(fp);

    // ── 2. 변환 행렬 (velodyne → 로컬 좌표계) ─────────────────────────────
    Eigen::Matrix4f T = T_base_inv_ * poses_[idx] * Tr_;

    // ── 3. 랜덤 서브샘플 (VH_BATCH_SIZE 이하로) ──────────────────────────────
    std::vector<uint32_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0u);
    {
        static thread_local std::mt19937 rng(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    uint32_t count = std::min(N, VH_BATCH_SIZE);

    // ── 4. 변환 + 색상 매핑 ───────────────────────────────────────────────────
    std::vector<VH_InputPoint> result;
    result.reserve(count);

    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t k = indices[i];
        float x = raw[k * 4 + 0];
        float y = raw[k * 4 + 1];
        float z = raw[k * 4 + 2];
        float r = raw[k * 4 + 3];

        Eigen::Vector4f pw = T * Eigen::Vector4f(x, y, z, 1.f);

        VH_InputPoint pt{};
        pt.px = pw.x();
        pt.py = pw.y();
        pt.pz = pw.z();
        pt.nx = 0.f;
        pt.ny = 1.f;
        pt.nz = 0.f;
        pt.col = toColor(r);
        result.push_back(pt);
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Viewer
// ─────────────────────────────────────────────────────────────────────────────

void Viewer::Initialize(const VulkanContext &ctx, const std::string &datasetBase)
{
    ctx_ = ctx;
    loader_.initialize(datasetBase);
    vhb_.Initialize(ctx_);
}

void Viewer::IntegrateFrame(uint32_t frameIdx)
{
    auto pts = loader_.loadFrame(frameIdx);
    if (pts.empty())
        return;

    auto sp = loader_.sensorPos(frameIdx);
    uint32_t N = static_cast<uint32_t>(pts.size());
    vhb_.Integrate(ctx_, pts.data(), N, sp.x(), sp.y(), sp.z());
}

void Viewer::ClearMap()
{
    vhb_.Clear(ctx_);
}

void Viewer::RenderColor(VkCommandBuffer cmd, const float *mvp, bool useSlotPoint)
{
    vhb_.RenderColor(cmd, mvp, useSlotPoint);
}

void Viewer::RenderVoxel(VkCommandBuffer cmd, const float *mvp)
{
    vhb_.RenderVoxel(cmd, mvp);
}
