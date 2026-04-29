#include "SimpleLiDAR.h"

#include <cmath>
#include <algorithm>
#include <limits>

static constexpr float kPi = 3.14159265358979f;

// ── 씬 구성 ───────────────────────────────────────────────────────────────────

void SimpleLiDAR::clearScene()
{
    spheres_.clear();
    boxes_.clear();
    planes_.clear();
}

void SimpleLiDAR::addSphere(float cx, float cy, float cz, float r, uint32_t rgba)
{
    spheres_.push_back({cx, cy, cz, r, rgba});
}

void SimpleLiDAR::addBox(float cx, float cy, float cz, float hx, float hy, float hz, uint32_t rgba)
{
    boxes_.push_back({cx, cy, cz, hx, hy, hz, rgba});
}

void SimpleLiDAR::addPlane(float nx, float ny, float nz, float d, uint32_t rgba)
{
    // 법선 정규화
    float len = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (len < 1e-8f) return;
    planes_.push_back({nx/len, ny/len, nz/len, d, rgba});
}

void SimpleLiDAR::addDefaultScene()
{
    // 지면: y = -1.5  →  0*x + 1*y + 0*z + 1.5 = 0
    addPlane(0.f, 1.f, 0.f, 1.5f, 0xFF808080u);

    // 정면 구
    addSphere(0.f, 0.f, 5.f, 1.f, 0xFF0000FFu);  // 파랑

    // 왼쪽 박스
    addBox(-3.f, 0.f, 4.f, 0.6f, 0.6f, 0.6f, 0xFF00FF00u);  // 초록

    // 오른쪽 박스
    addBox(3.f, 0.f, 4.f, 0.6f, 0.6f, 0.6f, 0xFFFF0000u);   // 빨강
}

// ── 교차 검사 ─────────────────────────────────────────────────────────────────

bool SimpleLiDAR::intersectSphere(const Sphere &s,
                                   float ox, float oy, float oz,
                                   float dx, float dy, float dz, Hit &h) const
{
    // oc = O - C
    float ocx = ox - s.cx, ocy = oy - s.cy, ocz = oz - s.cz;
    // half-b 공식: hb = oc·D, c = |oc|² - r²
    float hb   = ocx*dx + ocy*dy + ocz*dz;
    float c    = ocx*ocx + ocy*ocy + ocz*ocz - s.r*s.r;
    float disc = hb*hb - c;
    if (disc < 0.f) return false;

    float sq = std::sqrt(disc);
    float t  = -hb - sq;
    if (t < 0.f) t = -hb + sq;
    if (t < 0.f) return false;

    float hx = ox + dx*t - s.cx;
    float hy = oy + dy*t - s.cy;
    float hz = oz + dz*t - s.cz;
    float inv = 1.f / s.r;
    h = {t, hx*inv, hy*inv, hz*inv, s.col};
    return true;
}

bool SimpleLiDAR::intersectBox(const Box &b,
                                float ox, float oy, float oz,
                                float dx, float dy, float dz, Hit &h) const
{
    const float lo[3] = {b.cx - b.hx, b.cy - b.hy, b.cz - b.hz};
    const float hi[3] = {b.cx + b.hx, b.cy + b.hy, b.cz + b.hz};
    const float o[3]  = {ox, oy, oz};
    const float d[3]  = {dx, dy, dz};

    float tmin = 0.f, tmax = std::numeric_limits<float>::max();
    int   entryAxis = -1;
    int   entrySign =  1; // +1 = lo면 진입, -1 = hi면 진입

    for (int i = 0; i < 3; ++i)
    {
        if (std::fabs(d[i]) < 1e-8f)
        {
            if (o[i] < lo[i] || o[i] > hi[i]) return false;
            continue;
        }
        float t0 = (lo[i] - o[i]) / d[i];
        float t1 = (hi[i] - o[i]) / d[i];
        int sign = 1;
        if (t0 > t1) { std::swap(t0, t1); sign = -1; }
        if (t0 > tmin) { tmin = t0; entryAxis = i; entrySign = sign; }
        tmax = std::min(tmax, t1);
    }

    if (tmin > tmax || tmax < 0.f) return false;
    float t = (tmin >= 0.f) ? tmin : tmax;
    if (t < 0.f) return false;

    // lo면 진입(sign=1) → 법선 = -e_i (외향)
    // hi면 진입(sign=-1) → 법선 = +e_i (외향)
    float nx = 0.f, ny = 0.f, nz = 0.f;
    float *n[3] = {&nx, &ny, &nz};
    if (entryAxis >= 0) *n[entryAxis] = -(float)entrySign;
    h = {t, nx, ny, nz, b.col};
    return true;
}

bool SimpleLiDAR::intersectPlane(const Plane &p,
                                  float ox, float oy, float oz,
                                  float dx, float dy, float dz, Hit &h) const
{
    // 평면: N·P + d = 0
    float denom = p.nx*dx + p.ny*dy + p.nz*dz;
    if (std::fabs(denom) < 1e-8f) return false;

    float t = -(p.nx*ox + p.ny*oy + p.nz*oz + p.d) / denom;
    if (t < 0.f) return false;

    // 법선은 항상 레이 방향을 향해 뒤집음 (앞면만 감지)
    float nx = p.nx, ny = p.ny, nz = p.nz;
    if (denom > 0.f) { nx = -nx; ny = -ny; nz = -nz; }
    h = {t, nx, ny, nz, p.col};
    return true;
}

// ── 스캔 ──────────────────────────────────────────────────────────────────────

std::vector<VH_InputPoint> SimpleLiDAR::scan(float sx, float sy, float sz) const
{
    std::vector<VH_InputPoint> result;
    result.reserve(cfg_.numRings * cfg_.pointsPerRing);

    const float toRad = kPi / 180.f;

    for (uint32_t r = 0; r < cfg_.numRings; ++r)
    {
        float elevDeg = (cfg_.numRings > 1)
            ? cfg_.elevMinDeg + (cfg_.elevMaxDeg - cfg_.elevMinDeg) * r / (cfg_.numRings - 1)
            : 0.f;
        float elev = elevDeg * toRad;
        float cosE = std::cos(elev);
        float sinE = std::sin(elev);

        for (uint32_t a = 0; a < cfg_.pointsPerRing; ++a)
        {
            float az   = (a * 2.f * kPi) / cfg_.pointsPerRing;
            float dx   = cosE * std::cos(az);
            float dy   = sinE;
            float dz   = cosE * std::sin(az);

            // 가장 가까운 히트 탐색
            Hit best{cfg_.maxRange, 0.f, 0.f, 0.f, 0u};
            bool hitAny = false;

            for (const auto &s : spheres_)
            {
                Hit hit;
                if (intersectSphere(s, sx, sy, sz, dx, dy, dz, hit)
                    && hit.t >= cfg_.minRange && hit.t < best.t)
                { best = hit; hitAny = true; }
            }
            for (const auto &b : boxes_)
            {
                Hit hit;
                if (intersectBox(b, sx, sy, sz, dx, dy, dz, hit)
                    && hit.t >= cfg_.minRange && hit.t < best.t)
                { best = hit; hitAny = true; }
            }
            for (const auto &p : planes_)
            {
                Hit hit;
                if (intersectPlane(p, sx, sy, sz, dx, dy, dz, hit)
                    && hit.t >= cfg_.minRange && hit.t < best.t)
                { best = hit; hitAny = true; }
            }

            if (!hitAny) continue;

            VH_InputPoint pt{};
            pt.px  = sx + dx * best.t;
            pt.py  = sy + dy * best.t;
            pt.pz  = sz + dz * best.t;
            pt.nx  = best.nx;
            pt.ny  = best.ny;
            pt.nz  = best.nz;
            pt.col = best.col;
            result.push_back(pt);
        }
    }

    return result;
}
