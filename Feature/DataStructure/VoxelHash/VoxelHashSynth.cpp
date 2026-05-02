#include "VoxelHash.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  genSphereBatch — 합성 구면 포인트 배치 생성 후 submitPoints()로 업로드
//
//  CPU에서 VH_InputPoint 배열을 생성하고 submitPoints()에 넘깁니다.
//  GPU 버퍼에 직접 접근하지 않으며, 외부 스캐너 경로와 동일한 입력 경로를 사용합니다.
// ─────────────────────────────────────────────────────────────────────────────

static float gaussRand()
{
    float u1 = (rand() + 1.f) / (static_cast<float>(RAND_MAX) + 2.f);
    float u2 = rand() / static_cast<float>(RAND_MAX);
    return std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * static_cast<float>(M_PI) * u2);
}

static uint32_t posToColor(float x, float y, float z)
{
    uint32_t h = static_cast<uint32_t>(x * 73856093.f) ^ static_cast<uint32_t>(y * 19349663.f) ^ static_cast<uint32_t>(z * 83492791.f);
    float t = static_cast<float>(h & 0xFFFFu) / 65535.f;
    auto ch = [](float base) -> uint32_t
    {
        float v = std::clamp(std::abs(std::fmod(base, 1.f) * 6.f - 3.f) - 1.f, 0.f, 1.f);
        return static_cast<uint32_t>(v * 255.f);
    };
    return ch(t) | (ch(t + .333f) << 8u) | (ch(t + .667f) << 16u) | 0xFF000000u;
}

uint32_t VoxelHashFeature::genSphereBatch()
{
    float cx = std::cos(sensorEl_), sx = std::sin(sensorEl_);
    float cy = std::cos(sensorAz_), sy = std::sin(sensorAz_);
    float spx = sensorDist_ * cx * sy;
    float spy = sensorDist_ * sx;
    float spz = sensorDist_ * cx * cy;

    std::vector<VH_InputPoint> batch;
    batch.reserve(VH_BATCH_SIZE);

    while (batch.size() < VH_BATCH_SIZE)
    {
        float u = (rand() / static_cast<float>(RAND_MAX)) * 2.f - 1.f;
        float t = (rand() / static_cast<float>(RAND_MAX)) * 2.f * static_cast<float>(M_PI);
        float r = std::sqrt(std::max(0.f, 1.f - u * u));
        float nx = r * std::cos(t);
        float ny = u;
        float nz = r * std::sin(t);

        // 센서 반대편 면 제외
        float toSx = spx - nx * sphereRadius_;
        float toSy = spy - ny * sphereRadius_;
        float toSz = spz - nz * sphereRadius_;
        float len = std::sqrt(toSx * toSx + toSy * toSy + toSz * toSz);
        if (len > 1e-6f && (nx * toSx + ny * toSy + nz * toSz) / len < 0.25f)
            continue;

        float noise = gaussRand() * noiseStddev_;
        float px = nx * sphereRadius_ + nx * noise;
        float py = ny * sphereRadius_ + ny * noise;
        float pz = nz * sphereRadius_ + nz * noise;

        batch.push_back({px, py, pz, posToColor(px, py, pz), nx, ny, nz, 0.f});
    }

    // 외부 스캐너와 동일한 경로로 업로드
    submitPoints(batch.data(), static_cast<uint32_t>(batch.size()), spx, spy, spz);
    return static_cast<uint32_t>(batch.size());
}
