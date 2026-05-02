#include "FastLIO2.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
//  SO(3) 유틸리티
// ─────────────────────────────────────────────────────────────────────────────

Eigen::Matrix3f SO3::hat(const Eigen::Vector3f &v)
{
    Eigen::Matrix3f M;
    M << 0, -v.z(), v.y(),
        v.z(), 0, -v.x(),
        -v.y(), v.x(), 0;
    return M;
}

Eigen::Matrix3f SO3::Exp(const Eigen::Vector3f &phi)
{
    float theta = phi.norm();
    if (theta < 1e-7f)
        return Eigen::Matrix3f::Identity() + hat(phi);
    Eigen::Matrix3f K = hat(phi / theta);
    return Eigen::Matrix3f::Identity() + std::sin(theta) * K + (1.f - std::cos(theta)) * (K * K);
}

Eigen::Vector3f SO3::Log(const Eigen::Matrix3f &R)
{
    float cosT = std::clamp((R.trace() - 1.f) * 0.5f, -1.f, 1.f);
    float theta = std::acos(cosT);
    if (std::abs(theta) < 1e-7f)
        return Eigen::Vector3f::Zero();
    Eigen::Matrix3f skew = (R - R.transpose()) * (theta / (2.f * std::sin(theta)));
    return Eigen::Vector3f(skew(2, 1), skew(0, 2), skew(1, 0));
}

Eigen::Matrix3f SO3::rightJacobian(const Eigen::Vector3f &phi)
{
    float theta = phi.norm();
    if (theta < 1e-7f)
        return Eigen::Matrix3f::Identity();
    Eigen::Matrix3f K = hat(phi / theta);
    return Eigen::Matrix3f::Identity() - ((1.f - std::cos(theta)) / theta) * K + ((theta - std::sin(theta)) / theta) * (K * K);
}

Eigen::Matrix3f SO3::rightJacobianInv(const Eigen::Vector3f &phi)
{
    float theta = phi.norm();
    if (theta < 1e-7f)
        return Eigen::Matrix3f::Identity();
    Eigen::Matrix3f K = hat(phi / theta);
    float cot = std::cos(theta * 0.5f) / std::sin(theta * 0.5f);
    return Eigen::Matrix3f::Identity() + 0.5f * hat(phi) + (1.f / (theta * theta) - cot / (2.f * theta)) * (K * K) * (theta * theta);
}

// ─────────────────────────────────────────────────────────────────────────────
//  FLIOState — 매니폴드 연산
// ─────────────────────────────────────────────────────────────────────────────

FLIOState FLIOState::boxplus(const Eigen::Matrix<float, 24, 1> &d) const
{
    FLIOState out;
    out.R = R * SO3::Exp(d.segment<3>(0));
    out.p = p + d.segment<3>(3);
    out.v = v + d.segment<3>(6);
    out.ba = ba + d.segment<3>(9);
    out.bw = bw + d.segment<3>(12);
    out.g = g + d.segment<3>(15);
    out.R_LI = R_LI * SO3::Exp(d.segment<3>(18));
    out.p_LI = p_LI + d.segment<3>(21);
    return out;
}

Eigen::Matrix<float, 24, 1> FLIOState::boxminus(const FLIOState &o) const
{
    Eigen::Matrix<float, 24, 1> d;
    d.segment<3>(0) = SO3::Log(o.R.transpose() * R);
    d.segment<3>(3) = p - o.p;
    d.segment<3>(6) = v - o.v;
    d.segment<3>(9) = ba - o.ba;
    d.segment<3>(12) = bw - o.bw;
    d.segment<3>(15) = g - o.g;
    d.segment<3>(18) = SO3::Log(o.R_LI.transpose() * R_LI);
    d.segment<3>(21) = p_LI - o.p_LI;
    return d;
}

FLIOState FLIOState::Identity()
{
    FLIOState s;
    s.R = Eigen::Matrix3f::Identity();
    s.p = Eigen::Vector3f::Zero();
    s.v = Eigen::Vector3f::Zero();
    s.ba = Eigen::Vector3f::Zero();
    s.bw = Eigen::Vector3f::Zero();
    s.g = Eigen::Vector3f(0.f, 0.f, -9.81f);
    s.R_LI = Eigen::Matrix3f::Identity();
    s.p_LI = Eigen::Vector3f::Zero();
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
//  FastLIO2 생성자
// ─────────────────────────────────────────────────────────────────────────────

FastLIO2::FastLIO2(const Config &cfg) : cfg_(cfg)
{
    x_opt_ = FLIOState::Identity();
    x_opt_.R_LI = cfg.R_LI_init;
    x_opt_.p_LI = cfg.p_LI_init;
    x_opt_.g = Eigen::Vector3f(0.f, 0.f, -cfg.gravity);
    P_opt_ = FLIOCovariance::Identity() * 1e-3f;
}

const char *FastLIO2::iekfFailString(FastLIO2IEKFFail f)
{
    switch (f)
    {
    case FastLIO2IEKFFail::None:
        return "OK";
    case FastLIO2IEKFFail::EmptyResiduals:
        return "empty_residuals";
    case FastLIO2IEKFFail::LDLTFactorization:
        return "ldlt_fail";
    case FastLIO2IEKFFail::MaxIterations:
        return "max_iterations";
    }
    return "?";
}

void FastLIO2::setWorldPoseFromVelo(const Eigen::Matrix3f &R_w_v,
                                    const Eigen::Vector3f &p_w_v)
{
    x_opt_.R = R_w_v;
    x_opt_.p = p_w_v;
    x_pred_.R = R_w_v;
    x_pred_.p = p_w_v;
}

void FastLIO2::setPredictFromVelo(const Eigen::Matrix3f &R_w_v,
                                  const Eigen::Vector3f &p_w_v)
{
    x_pred_.R = R_w_v;
    x_pred_.p = p_w_v;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 1: Forward Propagation — 식 (6)(7)
// ─────────────────────────────────────────────────────────────────────────────

void FastLIO2::forwardPropagate(const IMUData &imu, float dt)
{
    auto dxdt = motionModel(x_pred_, imu);
    x_pred_ = x_pred_.boxplus(dxdt * dt);

    FLIOCovariance Fx = computeFx(x_pred_, imu, dt);
    Eigen::Matrix<float, 24, 12> Fw = computeFw(x_pred_, dt);
    Eigen::Matrix<float, 12, 12> Q = buildProcessNoise();

    P_pred_ = Fx * P_pred_ * Fx.transpose() + Fw * Q * Fw.transpose();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 2: Backward Propagation (모션 왜곡 보정, TODO)
// ─────────────────────────────────────────────────────────────────────────────

void FastLIO2::backwardPropagate(std::vector<LiDARPoint> & /*scan*/,
                                 double /*scan_end_time*/)
{
    // TODO: IMU 버퍼로 각 포인트 샘플링 시점 자세 추정 → 스캔 종료 기준 재투영
    // KITTI처럼 이미 motion-corrected 스캔이면 생략 가능
}

// ─────────────────────────────────────────────────────────────────────────────
//  Steps 3~9: Iterated Extended Kalman Filter
//
//  측정 모델 (TSDF 기반):
//    zⱼ = uⱼᵀ · (T(x̂)·ᴸpⱼ - qⱼ) ≈ tsdf × truncation
//
//  Jacobian (해석적 유도):
//    ∂zⱼ/∂δθ    = -uⱼᵀ · ᴳRᵢ · [p_imu]∧      (p_imu = ᴵRₗ·ᴸpⱼ + ᴵpₗ)
//    ∂zⱼ/∂δp    =  uⱼᵀ
//    ∂zⱼ/∂δθ_LI = -uⱼᵀ · ᴳRᵢ · ᴵRₗ · [ᴸpⱼ]∧
//    ∂zⱼ/∂δp_LI =  uⱼᵀ · ᴳRᵢ
//    나머지 (v, ba, bw, g) = 0
// ─────────────────────────────────────────────────────────────────────────────

bool FastLIO2::computePointResidual(const LiDARPoint &point,
                                    const FLIOState &x_hat,
                                    float &residual,
                                    Eigen::Matrix<float, 1, 24> &H_row) const
{
    if (!mapPtr_)
        return false;

    // 포인트를 world frame으로 변환
    Eigen::Vector3f p_world = transformPoint(point.pos, x_hat);

    // TSDF 조회 — O(1) 해시 조회
    TSDFQueryResult q = mapPtr_->queryTSDF(p_world.x(), p_world.y(), p_world.z());
    if (!q.valid || q.weight < 1.f)
        return false;

    // 잔차: zⱼ = tsdf × truncation (실제 부호 거리, m 단위)
    residual = q.tsdf * mapPtr_->getTruncation();

    // 너무 멀리 있으면 outlier 처리 (truncation 70% 초과)
    if (std::abs(residual) > 0.7f * mapPtr_->getTruncation())
        return false;

    const Eigen::Vector3f &u = q.normal; // 표면 법선 (TSDF 기울기 방향)

    // p_imu = ᴵRₗ·ᴸpⱼ + ᴵpₗ  (LiDAR 포인트를 IMU 프레임으로)
    Eigen::Vector3f p_imu = x_hat.R_LI * point.pos + x_hat.p_LI;

    H_row = Eigen::Matrix<float, 1, 24>::Zero();

    // ∂zⱼ/∂δθ = -uᵀ · ᴳRᵢ · [p_imu]∧  →  row[0:3]
    H_row.block<1, 3>(0, 0) = -(u.transpose() * x_hat.R * SO3::hat(p_imu));

    // ∂zⱼ/∂δp = uᵀ  →  row[3:6]
    H_row.block<1, 3>(0, 3) = u.transpose();

    // v, ba, bw, g → 0 (이미 zero 초기화)

    // ∂zⱼ/∂δθ_LI = -uᵀ · ᴳRᵢ · ᴵRₗ · [ᴸpⱼ]∧  →  row[18:21]
    H_row.block<1, 3>(0, 18) = -(u.transpose() * x_hat.R * x_hat.R_LI * SO3::hat(point.pos));

    // ∂zⱼ/∂δp_LI = uᵀ · ᴳRᵢ  →  row[21:24]
    H_row.block<1, 3>(0, 21) = u.transpose() * x_hat.R;

    return true;
}

bool FastLIO2::iteratedUpdate(const std::vector<LiDARPoint> &scan)
{
    FLIOState x_hat = x_pred_;
    int kappa = -1;
    int last_m = 0;

    const float meas_var = cfg_.meas_noise * cfg_.meas_noise;

    while (true)
    {
        ++kappa;
        if (kappa >= cfg_.max_iterations)
        {
            lastIekfFail_ = FastLIO2IEKFFail::MaxIterations;
            lastIekfResidualM_ = last_m;
            return false;
        }

        // 유효 포인트의 잔차·Jacobian 수집
        std::vector<float> residuals;
        std::vector<Eigen::Matrix<float, 1, 24>> H_rows;

        for (const auto &pt : scan)
        {
            float z;
            Eigen::Matrix<float, 1, 24> H;
            if (computePointResidual(pt, x_hat, z, H))
            {
                residuals.push_back(z);
                H_rows.push_back(H);
            }
        }

        if (residuals.empty())
        {
            lastIekfFail_ = FastLIO2IEKFFail::EmptyResiduals;
            lastIekfResidualM_ = 0;
            return false;
        }

        const int m = static_cast<int>(residuals.size());
        last_m = m;

        // ── Jᵏ 계산 (식 11) ──────────────────────────────────────────────────
        FLIOCovariance J_k = computeManifoldJacobian(x_hat, x_pred_);
        FLIOCovariance J_inv = J_k.inverse();
        FLIOCovariance P_J = J_inv * P_pred_; // (Jᵏ)⁻¹·P̃

        Eigen::Matrix<float, 24, 1> dx_prior =
            J_inv * x_hat.boxminus(x_pred_);

        // R = σ²I 일 때 m×m 혁신공분산 S 역행렬 없이 Woodbury (24×24만 역/분해)
        const float inv_meas = 1.f / meas_var;
        Eigen::Matrix<float, 24, 24> B = Eigen::Matrix<float, 24, 24>::Zero();
        Eigen::Matrix<float, 24, 1> g = Eigen::Matrix<float, 24, 1>::Zero();

        for (int i = 0; i < m; ++i)
        {
            const Eigen::Matrix<float, 1, 24> &hi = H_rows[i];
            const float zi = residuals[i];
            const float ui = (hi * dx_prior)(0, 0) - zi;
            B.noalias() += inv_meas * hi.transpose() * hi;
            g.noalias() += inv_meas * hi.transpose() * ui;
        }

        const FLIOCovariance P_J_inv = P_J.inverse();
        const FLIOCovariance Mmat = P_J_inv + B;

        Eigen::LDLT<FLIOCovariance> ldltM(Mmat);
        if (ldltM.info() != Eigen::Success)
        {
            lastIekfFail_ = FastLIO2IEKFFail::LDLTFactorization;
            lastIekfResidualM_ = m;
            return false;
        }

        const Eigen::Matrix<float, 24, 1> y = ldltM.solve(g);

        Eigen::Matrix<float, 24, 1> Ht_t = Eigen::Matrix<float, 24, 1>::Zero();
        for (int i = 0; i < m; ++i)
        {
            const Eigen::Matrix<float, 1, 24> &hi = H_rows[i];
            const float zi = residuals[i];
            const float ui = (hi * dx_prior)(0, 0) - zi;
            const float ti =
                inv_meas * (ui - (hi * y)(0, 0));
            Ht_t.noalias() += hi.transpose() * ti;
        }

        Eigen::Matrix<float, 24, 1> delta = -dx_prior + P_J * Ht_t;

        // 공분산: K H = P_J Q, Q = Hᵀ S⁻¹ H = B − B M⁻¹ B
        const Eigen::Matrix<float, 24, 24> X = ldltM.solve(B);
        const Eigen::Matrix<float, 24, 24> Q = B - B * X;
        const Eigen::Matrix<float, 24, 24> KH = P_J * Q;

        // ── 상태 업데이트 (식 14) ─────────────────────────────────────────────
        FLIOState x_prev = x_hat;
        x_hat = x_hat.boxplus(delta);

        // ── 수렴 확인 ─────────────────────────────────────────────────────────
        if (x_hat.boxminus(x_prev).norm() < cfg_.converge_eps)
        {
            x_opt_ = x_hat; // 식 (15)
            P_opt_ = (FLIOCovariance::Identity() - KH) * P_pred_;
            lastIekfFail_ = FastLIO2IEKFFail::None;
            lastIekfResidualM_ = m;
            return true;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Step 11: 맵 업데이트 — BucketVoxelHash::Integrate()
//
//  LiDAR 포인트를 world frame으로 변환 후 TSDF 맵에 통합.
//  기존 Integrate()는 이미 World-frame 포인트를 받아 처리함.
// ─────────────────────────────────────────────────────────────────────────────

void FastLIO2::updateMap(const std::vector<LiDARPoint> &scan)
{
    if (!mapPtr_ || !ctxPtr_)
        return;

    manageMapBounds(x_opt_.p);

    // LiDAR world position = ᴳRᵢ · ᴵpₗ + ᴳpᵢ
    Eigen::Vector3f sensorWorld = x_opt_.R * x_opt_.p_LI + x_opt_.p;

    // LiDARPoint → VH_InputPoint (world frame 변환 포함)
    std::vector<VH_InputPoint> pts;
    pts.reserve(scan.size());

    for (const auto &lp : scan)
    {
        Eigen::Vector3f pw = transformPoint(lp.pos, x_opt_);

        VH_InputPoint vp{};
        vp.px = pw.x();
        vp.py = pw.y();
        vp.pz = pw.z();
        // 법선: 셰이더가 sn = normalize(sensorPos - pt) 로 직접 계산하므로 0 가능
        vp.nx = 0.f;
        vp.ny = 0.f;
        vp.nz = 0.f;
        // 반사도 → 흑백 색상
        uint8_t g = static_cast<uint8_t>(std::clamp(lp.intensity, 0.f, 1.f) * 255.f);
        vp.col = 0xFF000000u | (uint32_t(g) << 16) | (uint32_t(g) << 8) | g;
        pts.push_back(vp);
    }

    uint32_t N = std::min(static_cast<uint32_t>(pts.size()), VH_BATCH_SIZE);
    mapPtr_->Integrate(*ctxPtr_, pts.data(), N,
                       sensorWorld.x(), sensorWorld.y(), sensorWorld.z());
}

// ─────────────────────────────────────────────────────────────────────────────
//  데이터 입력 인터페이스
// ─────────────────────────────────────────────────────────────────────────────

void FastLIO2::feedIMU(const IMUData &imu)
{
    imu_buffer_.push_back(imu);

    if (!initialized_)
    {
        last_imu_time_ = imu.timestamp;
        return;
    }

    float dt = static_cast<float>(imu.timestamp - last_imu_time_);
    if (dt > 0.f && dt < 1.f)
        forwardPropagate(imu, dt);
    last_imu_time_ = imu.timestamp;
}

FastLIO2::Output FastLIO2::feedScan(const std::vector<LiDARPoint> &scan,
                                    double scan_end_time)
{
    Output out{};

    // 첫 스캔: 상태 초기화만
    if (!initialized_)
    {
        x_pred_ = x_opt_;
        P_pred_ = P_opt_;
        initialized_ = true;
        last_scan_time_ = scan_end_time;

        // 첫 스캔을 맵에 삽입하여 초기 맵 구성
        if (mapPtr_)
            updateMap(scan);
        out.valid = false;
        return out;
    }

    // Step 1은 feedIMU()에서 이미 수행됨

    // Step 2: Backward propagation
    auto scan_mut = scan;
    backwardPropagate(scan_mut, scan_end_time);

    // Steps 3~9: IEKF
    if (!iteratedUpdate(scan_mut))
    {
        out.valid = false;
        return out;
    }

    // Step 11: 맵 업데이트
    updateMap(scan_mut);

    // 출력
    out.state = x_opt_;
    out.covariance = P_opt_;
    out.valid = true;
    for (const auto &pt : scan_mut)
        out.transformed_points.push_back(transformPoint(pt.pos, x_opt_));

    // 다음 스캔 준비
    x_pred_ = x_opt_;
    P_pred_ = P_opt_;
    last_scan_time_ = scan_end_time;
    imu_buffer_.clear();

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  내부 헬퍼
// ─────────────────────────────────────────────────────────────────────────────

// 연속 운동학 모델 — 식 (1)(2)
Eigen::Matrix<float, 24, 1> FastLIO2::motionModel(const FLIOState &x,
                                                  const IMUData &u) const
{
    Eigen::Matrix<float, 24, 1> dxdt = Eigen::Matrix<float, 24, 1>::Zero();
    Eigen::Vector3f omega = u.gyro - x.bw;
    Eigen::Vector3f accel = u.accel - x.ba;
    dxdt.segment<3>(0) = omega;             // δR
    dxdt.segment<3>(3) = x.v;               // δp
    dxdt.segment<3>(6) = x.R * accel + x.g; // δv
    return dxdt;
}

// 상태 전이 Jacobian Fₓ — 식 (7)
FLIOCovariance FastLIO2::computeFx(const FLIOState &x,
                                   const IMUData &u,
                                   float dt) const
{
    FLIOCovariance Fx = FLIOCovariance::Identity();
    Eigen::Vector3f omega = u.gyro - x.bw;
    Eigen::Vector3f accel = u.accel - x.ba;

    Fx.block<3, 3>(0, 0) = SO3::Exp(-omega * dt);
    Fx.block<3, 3>(0, 12) = -SO3::rightJacobian(omega * dt) * dt;
    Fx.block<3, 3>(3, 6) = Eigen::Matrix3f::Identity() * dt;
    Fx.block<3, 3>(6, 0) = -x.R * SO3::hat(accel) * dt;
    Fx.block<3, 3>(6, 9) = -x.R * dt;
    Fx.block<3, 3>(6, 15) = Eigen::Matrix3f::Identity() * dt;
    return Fx;
}

// 노이즈 Jacobian Fw (24×12) — 식 (7)
Eigen::Matrix<float, 24, 12> FastLIO2::computeFw(const FLIOState &x, float dt) const
{
    Eigen::Matrix<float, 24, 12> Fw = Eigen::Matrix<float, 24, 12>::Zero();
    Fw.block<3, 3>(0, 3) = -Eigen::Matrix3f::Identity() * dt; // n_ω → δR
    Fw.block<3, 3>(6, 0) = -x.R * dt;                         // n_a → δv
    Fw.block<3, 3>(9, 6) = Eigen::Matrix3f::Identity() * dt;  // n_ba → δba
    Fw.block<3, 3>(12, 9) = Eigen::Matrix3f::Identity() * dt; // n_bω → δbw
    return Fw;
}

// 매니폴드 Jacobian Jᵏ (24×24) — 식 (11)
FLIOCovariance FastLIO2::computeManifoldJacobian(const FLIOState &x_hat,
                                                 const FLIOState &x_pred) const
{
    FLIOCovariance J = FLIOCovariance::Identity();
    Eigen::Vector3f dT_I = SO3::Log(x_pred.R.transpose() * x_hat.R);
    Eigen::Vector3f dT_L = SO3::Log(x_pred.R_LI.transpose() * x_hat.R_LI);
    J.block<3, 3>(0, 0) = SO3::rightJacobianInv(dT_I).transpose();
    J.block<3, 3>(18, 18) = SO3::rightJacobianInv(dT_L).transpose();
    return J;
}

// ᴳp̄ⱼ = ᴳRᵢ·(ᴵRₗ·ᴸpⱼ + ᴵpₗ) + ᴳpᵢ — 식 (16)
Eigen::Vector3f FastLIO2::transformPoint(const Eigen::Vector3f &p_L,
                                         const FLIOState &x) const
{
    return x.R * (x.R_LI * p_L + x.p_LI) + x.p;
}

// 맵 범위 관리: 센서 이동 시 deleteBox로 범위 밖 복셀 제거
void FastLIO2::manageMapBounds(const Eigen::Vector3f &sensor_pos)
{
    if (!mapPtr_ || !ctxPtr_)
        return;

    float dist = (sensor_pos - mapCenter_).norm();
    if (dist < cfg_.map_move_threshold)
        return;

    // 새 맵 중심
    mapCenter_ = sensor_pos;

    // 맵 중심에서 L/2 밖의 박스를 삭제
    float half = cfg_.map_size_L * 0.5f;
    // 맵 전체 범위를 먼저 잡고, 새 범위 밖 영역을 delete
    // 간단 구현: 새 맵 박스 외부를 전체 삭제 범위로 표현하기 어려우므로
    // "이전 맵 - 새 맵" 영역을 6개의 슬래브 박스로 근사
    // 여기서는 단순화: 새 중심 기준 L×L×L 밖 = deleteBox(전 범위)
    // TODO: 논문 §V-A의 정확한 구현 (현재는 단순 전체 박스 기준)
    Eigen::AlignedBox3f keepBox(
        sensor_pos - Eigen::Vector3f::Constant(half),
        sensor_pos + Eigen::Vector3f::Constant(half));

    // 실제 삭제 영역은 꽤 큰 범위이므로, 현재는 맵 크기의 2배 밖을 지움
    float outer = cfg_.map_size_L;
    Eigen::AlignedBox3f delBox(
        sensor_pos - Eigen::Vector3f::Constant(outer),
        sensor_pos - Eigen::Vector3f::Constant(half));
    // 삭제 박스는 개념적 구현 — 향후 6면 슬래브 방식으로 정교화 필요
    // 지금은 맵 경계 밖 영역을 X축 방향 기준 예시로 실행
    (void)keepBox;
    (void)delBox;
    // mapPtr_->deleteBox(delBox, *ctxPtr_);
}

// Q (12×12)
Eigen::Matrix<float, 12, 12> FastLIO2::buildProcessNoise() const
{
    Eigen::Matrix<float, 12, 12> Q = Eigen::Matrix<float, 12, 12>::Zero();
    Q.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() * (cfg_.acc_noise * cfg_.acc_noise);
    Q.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() * (cfg_.gyro_noise * cfg_.gyro_noise);
    Q.block<3, 3>(6, 6) = Eigen::Matrix3f::Identity() * (cfg_.acc_bias_noise * cfg_.acc_bias_noise);
    Q.block<3, 3>(9, 9) = Eigen::Matrix3f::Identity() * (cfg_.gyro_bias_noise * cfg_.gyro_bias_noise);
    return Q;
}
