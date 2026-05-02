#pragma once

#include "../DataStructure/VoxelHash/VoxelHashTypes.h"
#include <vector>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
//  SimpleLiDAR — CPU 레이캐스팅 기반 LiDAR 시뮬레이터
//
//  사용 방법:
//    SimpleLiDAR lidar;
//    lidar.setConfig({.numRings=32, .pointsPerRing=360});
//    lidar.addSphere(0, 0, 5, 1.5f, 0xFF0000FF);
//    lidar.addPlane(0, 1, 0, 1.5f, 0xFF808080);   // y = -1.5
//    auto pts = lidar.scan(0.f, 0.f, 0.f);
//    vhb.Integrate(ctx, pts.data(), pts.size(), 0,0,0);
// ─────────────────────────────────────────────────────────────────────────────
class SimpleLiDAR
{
public:
    struct Config
    {
        uint32_t numRings      = 32;    // 수직 채널 수 (Velodyne-32 기본값)
        uint32_t pointsPerRing = 360;   // 채널당 수평 샘플 수
        float    elevMinDeg    = -15.f; // 최소 앙각 (도)
        float    elevMaxDeg    =  15.f; // 최대 앙각 (도)
        float    minRange      = 0.05f; // 최소 감지 거리 (m)
        float    maxRange      = 30.f;  // 최대 감지 거리 (m)
        float    noiseStddev   = 0.005f; // 위치 노이즈 표준편차 (m)
    };

    // ── 씬 구성 ───────────────────────────────────────────────────────────────
    void clearScene();

    // 구: 중심 (cx,cy,cz), 반지름 r, RGBA 색상
    void addSphere(float cx, float cy, float cz, float r, uint32_t rgba);

    // 박스: 중심 (cx,cy,cz), 반-엣지(half-extent) hx/hy/hz, RGBA 색상
    void addBox(float cx, float cy, float cz, float hx, float hy, float hz, uint32_t rgba);

    // 무한 평면: 법선 (nx,ny,nz), 오프셋 d  →  nx*x + ny*y + nz*z + d = 0
    void addPlane(float nx, float ny, float nz, float d, uint32_t rgba);

    // 테스트용 기본 씬 (지면 + 구 + 박스)
    void addDefaultScene();

    // ── 스캔 ──────────────────────────────────────────────────────────────────
    // 센서 위치 (sx,sy,sz) 에서 한 프레임 스캔. 히트된 포인트만 반환.
    std::vector<VH_InputPoint> scan(float sx = 0.f, float sy = 0.f, float sz = 0.f) const;

    void        setConfig(const Config &cfg) { cfg_ = cfg; }
    const Config &getConfig() const           { return cfg_; }

private:
    struct Hit   { float t, nx, ny, nz; uint32_t col; };
    struct Sphere { float cx, cy, cz, r; uint32_t col; };
    struct Box    { float cx, cy, cz, hx, hy, hz; uint32_t col; };
    struct Plane  { float nx, ny, nz, d; uint32_t col; };

    bool intersectSphere(const Sphere &s,
                         float ox, float oy, float oz,
                         float dx, float dy, float dz, Hit &h) const;

    bool intersectBox(const Box &b,
                      float ox, float oy, float oz,
                      float dx, float dy, float dz, Hit &h) const;

    bool intersectPlane(const Plane &p,
                        float ox, float oy, float oz,
                        float dx, float dy, float dz, Hit &h) const;

    Config             cfg_;
    std::vector<Sphere> spheres_;
    std::vector<Box>    boxes_;
    std::vector<Plane>  planes_;
};
