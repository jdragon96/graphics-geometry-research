#include "KissICP.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <unordered_map>

struct KissVoxelKey
{
    int ix, iy, iz;
    bool operator==(const KissVoxelKey &o) const { return ix == o.ix && iy == o.iy && iz == o.iz; }
};

struct KissVoxelKeyHash
{
    size_t operator()(const KissVoxelKey &k) const noexcept
    {
        const uint64_t ux = static_cast<uint32_t>(k.ix);
        const uint64_t uy = static_cast<uint32_t>(k.iy);
        const uint64_t uz = static_cast<uint32_t>(k.iz);
        return static_cast<size_t>((ux * 1315423911u) ^ (uy * 2654435761u) ^ (uz * 2246822519u));
    }
};

inline KissVoxelKey kissVoxelKey(const Eigen::Vector3f &p, float voxel)
{
    return {static_cast<int>(std::floor(p.x() / voxel)),
            static_cast<int>(std::floor(p.y() / voxel)),
            static_cast<int>(std::floor(p.z() / voxel))};
}

namespace {

inline Eigen::Matrix3f hat(const Eigen::Vector3f &v)
{
    Eigen::Matrix3f M;
    M << 0.f, -v.z(), v.y(), v.z(), 0.f, -v.x(), -v.y(), v.x(), 0.f;
    return M;
}

inline Eigen::Matrix3f SO3Exp(const Eigen::Vector3f &phi)
{
    float theta = phi.norm();
    if (theta < 1e-7f) return Eigen::Matrix3f::Identity() + hat(phi);
    Eigen::Matrix3f K = hat(phi / theta);
    return Eigen::Matrix3f::Identity() + std::sin(theta) * K
         + (1.f - std::cos(theta)) * (K * K);
}

inline Eigen::Vector3f SO3Log(const Eigen::Matrix3f &R)
{
    float cosT = std::clamp((R.trace() - 1.f) * 0.5f, -1.f, 1.f);
    float theta = std::acos(cosT);
    if (std::abs(theta) < 1e-7f) return Eigen::Vector3f::Zero();
    Eigen::Matrix3f skew = (R - R.transpose()) * (theta / (2.f * std::sin(theta)));
    return Eigen::Vector3f(skew(2, 1), skew(0, 2), skew(1, 0));
}

inline Eigen::Matrix4f mult44(const Eigen::Matrix4f &A, const Eigen::Matrix4f &B)
{
    Eigen::Matrix4f C = Eigen::Matrix4f::Identity();
    C.block<3, 3>(0, 0) = A.block<3, 3>(0, 0) * B.block<3, 3>(0, 0);
    C.block<3, 1>(0, 3) = A.block<3, 3>(0, 0) * B.block<3, 1>(0, 3) + A.block<3, 1>(0, 3);
    return C;
}

inline Eigen::Vector3f transform3(const Eigen::Matrix4f &T, const Eigen::Vector3f &p)
{
    return T.block<3, 3>(0, 0) * p + T.block<3, 1>(0, 3);
}

// se(3) increment: xi = [t; omega]  (translation, axis-angle vector)
inline Eigen::Matrix4f deltaSE3(const Eigen::Matrix<float, 6, 1> &xi)
{
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3, 3>(0, 0) = SO3Exp(xi.tail<3>());
    T.block<3, 1>(0, 3) = xi.head<3>();
    return T;
}

inline float deltaFromCorrection(const Eigen::Matrix4f &dT, float r_max)
{
    Eigen::Matrix3f dR = dT.block<3, 3>(0, 0);
    Eigen::Vector3f dt = dT.block<3, 1>(0, 3);
    float cosT         = std::clamp((dR.trace() - 1.f) * 0.5f, -1.f, 1.f);
    float theta        = std::acos(cosT);
    float delta_rot    = 2.f * r_max * std::sin(0.5f * theta);
    float delta_trans  = dt.norm();
    return delta_rot + delta_trans;
}

} // namespace

struct KissICPVoxelMap
{
    float                                              voxel_{};
    int                                                n_max_{};
    std::unordered_map<KissVoxelKey, std::vector<Eigen::Vector3f>, KissVoxelKeyHash> cells_;

    void clear() { cells_.clear(); }

    bool empty() const { return cells_.empty(); }

    void addPoint(const Eigen::Vector3f &p_world)
    {
        KissVoxelKey k = kissVoxelKey(p_world, voxel_);
        auto        &vec = cells_[k];
        if (static_cast<int>(vec.size()) >= n_max_) return;
        if (vec.empty()) vec.push_back(p_world);
    }

    void getClosest(const Eigen::Vector3f &q, Eigen::Vector3f &out_pt, float &out_d2) const
    {
        KissVoxelKey k0 = kissVoxelKey(q, voxel_);
        out_d2          = std::numeric_limits<float>::infinity();
        bool found      = false;
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz)
                {
                    KissVoxelKey k{k0.ix + dx, k0.iy + dy, k0.iz + dz};
                    auto         it = cells_.find(k);
                    if (it == cells_.end()) continue;
                    for (const Eigen::Vector3f &c : it->second)
                    {
                        float d2 = (c - q).squaredNorm();
                        if (d2 < out_d2)
                        {
                            out_d2 = d2;
                            out_pt = c;
                            found  = true;
                        }
                    }
                }
        if (!found) out_d2 = std::numeric_limits<float>::infinity();
    }

    void removeFar(const Eigen::Vector3f &robot, float r_max)
    {
        const float r2 = r_max * r_max;
        for (auto it = cells_.begin(); it != cells_.end();)
        {
            const KissVoxelKey &k = it->first;
            Eigen::Vector3f center((static_cast<float>(k.ix) + 0.5f) * voxel_,
                                   (static_cast<float>(k.iy) + 0.5f) * voxel_,
                                   (static_cast<float>(k.iz) + 0.5f) * voxel_);
            if ((center - robot).squaredNorm() > r2) it = cells_.erase(it);
            else
                ++it;
        }
    }
};

namespace {

std::vector<Eigen::Vector3f> voxelDownsample(const std::vector<Eigen::Vector3f> &in, float voxel)
{
    if (in.empty() || voxel <= 0.f) return {};
    std::unordered_map<KissVoxelKey, Eigen::Vector3f, KissVoxelKeyHash> one;
    one.reserve(in.size() / 4 + 8);
    for (const Eigen::Vector3f &p : in)
    {
        KissVoxelKey k = kissVoxelKey(p, voxel);
        if (one.find(k) == one.end()) one.emplace(k, p);
    }
    std::vector<Eigen::Vector3f> out;
    out.reserve(one.size());
    for (auto &e : one) out.push_back(e.second);
    return out;
}

void deskew(std::vector<Eigen::Vector3f> &pts, const std::vector<float> *rel_times,
            const Eigen::Vector3f &omega, const Eigen::Vector3f &vel, float sweep_dt)
{
    const int N = static_cast<int>(pts.size());
    if (N == 0) return;
    for (int i = 0; i < N; ++i)
    {
        float s = 0.f;
        if (rel_times && static_cast<int>(rel_times->size()) == N)
            s = (*rel_times)[i] * sweep_dt;
        else if (N > 1)
            s = (static_cast<float>(i) / static_cast<float>(N - 1)) * sweep_dt;
        else
            s = 0.f;
        pts[i] = SO3Exp(omega * s) * pts[i] + vel * s;
    }
}

Eigen::Matrix4f constantVelocityPred(const Eigen::Matrix4f &T_tm2, const Eigen::Matrix4f &T_tm1,
                                      float dt)
{
    if (dt < 1e-6f) return Eigen::Matrix4f::Identity();
    Eigen::Matrix3f R0 = T_tm2.block<3, 3>(0, 0);
    Eigen::Vector3f t0 = T_tm2.block<3, 1>(0, 3);
    Eigen::Matrix3f R1 = T_tm1.block<3, 3>(0, 0);
    Eigen::Vector3f t1 = T_tm1.block<3, 1>(0, 3);

    Eigen::Matrix3f Rrel = R0.transpose() * R1;
    Eigen::Vector3f trel = R0.transpose() * (t1 - t0);

    Eigen::Matrix4f Tpred = Eigen::Matrix4f::Identity();
    Tpred.block<3, 3>(0, 0) = Rrel;
    Tpred.block<3, 1>(0, 3) = trel;
    return Tpred;
}

Eigen::Matrix4f alignRobustGN(const KissICPVoxelMap &voxel_map, std::vector<Eigen::Vector3f> source_world,
                              float max_dist, float kernel_scale, float gamma, int max_iters, int &out_iters)
{
    const float max_d2 = max_dist * max_dist;
    Eigen::Matrix4f T_icp = Eigen::Matrix4f::Identity();
    out_iters = 0;

    for (int j = 0; j < max_iters; ++j)
    {
        Eigen::Matrix<float, 6, 6> JTJ = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> JTr = Eigen::Matrix<float, 6, 1>::Zero();
        int                         n_corr = 0;

        for (const Eigen::Vector3f &s : source_world)
        {
            Eigen::Vector3f q;
            float           d2 = 0.f;
            voxel_map.getClosest(s, q, d2);
            if (!(d2 < max_d2) || !std::isfinite(d2)) continue;

            Eigen::Vector3f r = s - q;
            const float     r2 = r.squaredNorm();
            // Avoid k→0 with r2→0 → 0/0 NaN in Geman–McClure weight
            const float k = std::max(kernel_scale, 1e-5f);
            const float denom = k + r2;
            const float w     = (k * k) / (denom * denom);

            Eigen::Matrix<float, 3, 6> J;
            J.setZero();
            J.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
            J.block<3, 3>(0, 3) = -hat(s);

            JTJ.noalias() += J.transpose() * w * J;
            JTr.noalias() += J.transpose() * w * r;
            ++n_corr;
        }

        if (n_corr < 3) break;

        Eigen::LDLT<Eigen::Matrix<float, 6, 6>> ldlt(JTJ);
        if (ldlt.info() != Eigen::Success) break;
        Eigen::Matrix<float, 6, 1> dx = ldlt.solve(-JTr);
        if (!dx.allFinite()) break;

        Eigen::Matrix4f dT = deltaSE3(dx);
        for (Eigen::Vector3f &s : source_world) s = transform3(dT, s);

        T_icp = mult44(dT, T_icp);
        if (!T_icp.allFinite())
        {
            T_icp   = Eigen::Matrix4f::Identity();
            out_iters = 0;
            break;
        }
        out_iters = j + 1;
        if (dx.norm() < gamma) break;
    }

    return T_icp;
}

} // namespace

KissICP::KissICP(KissICPConfig cfg) : cfg_(cfg), map_(std::make_unique<KissICPVoxelMap>())
{
    voxel_map_  = 0.01f * cfg_.r_max;
    voxel_merge_ = cfg_.alpha * voxel_map_;
    voxel_icp_   = cfg_.beta * voxel_map_;
    sigma_       = cfg_.tau0 / 3.f;
    map_->voxel_ = voxel_map_;
    map_->n_max_ = cfg_.N_max;
}

KissICP::~KissICP() = default;

void KissICP::reset()
{
    T_history_.clear();
    delta_samples_.clear();
    last_merge_.clear();
    sigma_ = cfg_.tau0 / 3.f;
    map_->clear();
}

KissICPResult KissICP::registerFrame(const std::vector<Eigen::Vector3f> &points_velo,
                                     const std::vector<float>          *rel_times)
{
    KissICPResult res;
    if (!std::isfinite(sigma_)) sigma_ = cfg_.tau0 / 3.f;
    res.sigma   = sigma_;
    res.tau     = 3.f * sigma_;
    if (res.tau < cfg_.tau0) res.tau = cfg_.tau0;

    std::vector<Eigen::Vector3f> pts;
    pts.reserve(points_velo.size());
    std::vector<float> rel_aligned;
    const std::vector<float> *rel_use = nullptr;
    if (rel_times && rel_times->size() == points_velo.size())
    {
        rel_aligned.reserve(points_velo.size());
        for (size_t i = 0; i < points_velo.size(); ++i)
        {
            if (!points_velo[i].allFinite()) continue;
            const float t = (*rel_times)[i];
            if (!std::isfinite(t)) continue;
            pts.push_back(points_velo[i]);
            rel_aligned.push_back(std::clamp(t, 0.f, 1.f));
        }
        if (!pts.empty() && rel_aligned.size() == pts.size()) rel_use = &rel_aligned;
    }
    else
    {
        for (const Eigen::Vector3f &p : points_velo)
            if (p.allFinite()) pts.push_back(p);
    }
    if (pts.empty()) return res;

    Eigen::Matrix4f T_tm1 = Eigen::Matrix4f::Identity();
    if (!T_history_.empty()) T_tm1 = T_history_.back();
    if (!T_tm1.allFinite())
    {
        T_history_.clear();
        T_tm1 = Eigen::Matrix4f::Identity();
    }

    Eigen::Matrix4f T_tm2 = T_tm1;
    if (T_history_.size() >= 2) T_tm2 = T_history_[T_history_.size() - 2];

    const float dt = cfg_.delta_t_sweep;
    Eigen::Matrix4f T_pred = constantVelocityPred(T_tm2, T_tm1, dt);

    Eigen::Vector3f omega = SO3Log(T_pred.block<3, 3>(0, 0)) / dt;
    Eigen::Vector3f vel   = T_pred.block<3, 1>(0, 3) / dt;
    if (T_history_.size() < 2u)
    {
        omega.setZero();
        vel.setZero();
        T_pred = Eigen::Matrix4f::Identity();
    }

    if (cfg_.use_deskew) deskew(pts, rel_use, omega, vel, dt);

    std::vector<Eigen::Vector3f> merge_cloud = voxelDownsample(pts, voxel_merge_);
    last_merge_                              = merge_cloud;
    std::vector<Eigen::Vector3f> icp_cloud   = voxelDownsample(merge_cloud, voxel_icp_);

    const Eigen::Matrix4f T_init = mult44(T_tm1, T_pred);

    std::vector<Eigen::Vector3f> source_world;
    source_world.reserve(icp_cloud.size());
    for (const Eigen::Vector3f &p : icp_cloud) source_world.push_back(transform3(T_init, p));

    const float tau   = res.tau;
    const float kappa = std::max(sigma_ / 3.f, 1e-5f);

    int             icp_used = 0;
    Eigen::Matrix4f T_icp    = Eigen::Matrix4f::Identity();
    if (!map_->empty())
    {
        T_icp = alignRobustGN(*map_, std::move(source_world), tau, kappa, cfg_.icp_gamma, cfg_.max_icp_iters,
                              icp_used);
        if (!T_icp.allFinite()) T_icp = Eigen::Matrix4f::Identity();
    }

    Eigen::Matrix4f T_t = mult44(T_icp, T_init);
    if (!T_t.allFinite()) T_t = T_init;

    if (map_->empty())
    {
        // first frame: no ICP yet, keep default tau/sigma
    }
    else
    {
        const float delta = deltaFromCorrection(T_icp, cfg_.r_max);
        if (std::isfinite(delta) && delta > cfg_.delta_min)
        {
            delta_samples_.push_back(delta);
            while (delta_samples_.size() > 500u) delta_samples_.erase(delta_samples_.begin());
        }
        if (!delta_samples_.empty())
        {
            double      sum   = 0.0;
            std::size_t nval = 0;
            for (float d : delta_samples_)
            {
                if (!std::isfinite(d)) continue;
                sum += static_cast<double>(d) * static_cast<double>(d);
                ++nval;
            }
            if (nval > 0u)
            {
                sigma_ = static_cast<float>(std::sqrt(sum / static_cast<double>(nval)));
                if (!std::isfinite(sigma_) || sigma_ < cfg_.tau0 / 3.f) sigma_ = cfg_.tau0 / 3.f;
            }
        }
        res.sigma = sigma_;
        res.tau   = std::max(3.f * sigma_, cfg_.tau0);
    }

    int ncorr = 0;
    if (!map_->empty())
    {
        const float tau_corr = res.tau;
        const float max_d2   = tau_corr * tau_corr;
        for (const Eigen::Vector3f &p : icp_cloud)
        {
            Eigen::Vector3f s = transform3(T_t, p);
            Eigen::Vector3f q;
            float           d2 = 0.f;
            map_->getClosest(s, q, d2);
            if (d2 < max_d2 && std::isfinite(d2)) ++ncorr;
        }
    }

    // Map update with merged cloud at T_t
    for (const Eigen::Vector3f &p : merge_cloud)
    {
        Eigen::Vector3f pw = transform3(T_t, p);
        map_->addPoint(pw);
    }

    Eigen::Vector3f robot = T_t.block<3, 1>(0, 3);
    map_->removeFar(robot, cfg_.r_max);

    T_history_.push_back(T_t);
    while (T_history_.size() > 2u) T_history_.erase(T_history_.begin());

    res.T_world_velo        = T_t;
    res.num_correspondences = ncorr;
    res.icp_iterations      = icp_used;
    return res;
}
