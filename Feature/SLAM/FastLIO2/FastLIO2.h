#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  FastLIO2.h — BucketVoxelHash 기반 FAST-LIO2 구현
//
//  논문: "FAST-LIO2: Fast Direct LiDAR-inertial Odometry" (Wei Xu et al., 2021)
//
//  맵 백엔드: BucketVoxelHash (IKDTree 대체)
//    - KNN 대신 queryTSDF() O(1) 조회로 잔차 계산
//    - Integrate()로 맵 업데이트
//    - deleteBox()로 맵 윈도잉
//
//  파이프라인 (Algorithm 1):
//    1. Forward Propagation  : IMU 적분 → 상태/공분산 전파  (식 6, 7)
//    2. Backward Propagation : 모션 왜곡 보정 (TODO)
//    3. Residual Computation : queryTSDF → zⱼ, Hⱼ           (식 9)
//    4. Iterated Update      : IEKF → 상태 업데이트          (식 14, 15)
//    5. Mapping              : Integrate → TSDF 맵 업데이트  (식 16)
// ─────────────────────────────────────────────────────────────────────────────

#include "../../DataStructure/VoxelHash/BucketVoxelHash.h"
#include "../../IFeature.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <vector>
#include <deque>
#include <memory>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
//  §1. 기본 데이터 타입
// ─────────────────────────────────────────────────────────────────────────────

// 상태 벡터 x ∈ M = SO(3)×ℝ¹⁵×SO(3)×ℝ³  (논문 §IV-A, dim = 24)
struct FLIOState
{
    Eigen::Matrix3f R;    // ᴳRᵢ — IMU 자세 (global)            SO(3)
    Eigen::Vector3f p;    // ᴳpᵢ — IMU 위치                     ℝ³
    Eigen::Vector3f v;    // ᴳvᵢ — IMU 속도                     ℝ³
    Eigen::Vector3f ba;   // bₐ  — 가속도계 바이어스             ℝ³
    Eigen::Vector3f bw;   // b_ω — 자이로 바이어스               ℝ³
    Eigen::Vector3f g;    // ᴳg  — 중력 벡터 (global)            ℝ³
    Eigen::Matrix3f R_LI; // ᴵRₗ — LiDAR→IMU 회전 외부 파라미터 SO(3)
    Eigen::Vector3f p_LI; // ᴵpₗ — LiDAR→IMU 이동 외부 파라미터 ℝ³

    // x ⊞ δx (SO(3) 부분은 Exp map)
    FLIOState boxplus(const Eigen::Matrix<float, 24, 1> &delta) const;

    // x ⊟ other → δx ∈ ℝ²⁴
    Eigen::Matrix<float, 24, 1> boxminus(const FLIOState &other) const;

    static FLIOState Identity();
};

using FLIOCovariance = Eigen::Matrix<float, 24, 24>;

// IMU 측정값
struct IMUData
{
    double          timestamp;
    Eigen::Vector3f accel;  // aₘ (m/s²)
    Eigen::Vector3f gyro;   // ωₘ (rad/s)
};

// LiDAR 포인트 (센서 프레임)
struct LiDARPoint
{
    Eigen::Vector3f pos;
    double          timestamp;
    float           intensity;
};

// ─────────────────────────────────────────────────────────────────────────────
//  §2. FastLIO2
// ─────────────────────────────────────────────────────────────────────────────

// 마지막 iteratedUpdate 실패 원인 (디버그 / ImGui)
enum class FastLIO2IEKFFail : std::uint8_t
{
    None = 0,
    EmptyResiduals,
    LDLTFactorization,
    MaxIterations,
};

class FastLIO2
{
public:
    struct Config
    {
        // IMU 노이즈 (연속 시간)
        float acc_noise       = 0.1f;
        float gyro_noise      = 0.01f;
        float acc_bias_noise  = 0.001f;
        float gyro_bias_noise = 0.0001f;

        // LiDAR-IMU 초기 외부 파라미터 (캘리브레이션 값)
        Eigen::Matrix3f R_LI_init = Eigen::Matrix3f::Identity();
        Eigen::Vector3f p_LI_init = Eigen::Vector3f::Zero();

        float gravity = 9.81f;

        // IEKF
        float converge_eps  = 0.001f;
        int   max_iterations = 30;

        // 측정 노이즈 분산 (TSDF 불확실성)
        float meas_noise = 0.01f;  // σ (m)

        // 맵 윈도잉: 센서 이동이 map_move_threshold 초과 시 deleteBox 실행
        float map_size_L          = 100.f;  // 로컬 맵 한 변 길이 (m)
        float map_move_threshold  = 5.f;    // 이동 임계값 (m)
    };

    struct Output
    {
        FLIOState      state;
        FLIOCovariance covariance;
        std::vector<Eigen::Vector3f> transformed_points;  // ᴳp̄ⱼ (식 16)
        bool           valid = false;
    };

    // ─────────────────────────────────────────────────────────────────────────

    explicit FastLIO2(const Config &cfg);

    // 반드시 feedScan 호출 전에 BucketVoxelHash 연결
    void setMap(BucketVoxelHash &map, VulkanContext &ctx)
    {
        mapPtr_ = &map;
        ctxPtr_ = &ctx;
    }

    // KITTI 데모: 상태의 R,p를 velodyne 바디의 world pose로 시드 (첫 스캔 전 x_opt_ 등)
    void setWorldPoseFromVelo(const Eigen::Matrix3f &R_w_v,
                              const Eigen::Vector3f &p_w_v);

    // 스캔 직전 예측: IMU 없을 때 외부 prior (예: GT)로 x_pred_ 정렬
    void setPredictFromVelo(const Eigen::Matrix3f &R_w_v,
                            const Eigen::Vector3f &p_w_v);

    // ── 데이터 입력 ──────────────────────────────────────────────────────────
    void   feedIMU(const IMUData &imu);
    Output feedScan(const std::vector<LiDARPoint> &scan, double scan_end_time);

    // ── 조회 ─────────────────────────────────────────────────────────────────
    const FLIOState      &getState()      const { return x_opt_; }
    const FLIOCovariance &getCovariance() const { return P_opt_; }
    bool                  isInitialized() const { return initialized_; }

    FastLIO2IEKFFail lastIEKFFail() const { return lastIekfFail_; }
    int              lastIEKFResidualCount() const { return lastIekfResidualM_; }
    static const char *iekfFailString(FastLIO2IEKFFail f);

private:
    // ── Algorithm 1 단계 ─────────────────────────────────────────────────────

    // Step 1: 식 (6)(7) — IMU 1회 → 상태/공분산 전파
    void forwardPropagate(const IMUData &imu, float dt);

    // Step 2: 모션 왜곡 보정 (TODO)
    void backwardPropagate(std::vector<LiDARPoint> &scan, double scan_end_time);

    // Steps 3~9: IEKF (수렴까지 반복)
    bool iteratedUpdate(const std::vector<LiDARPoint> &scan);

    // Step 11: BucketVoxelHash에 포인트 삽입 (식 16)
    void updateMap(const std::vector<LiDARPoint> &scan);

    // ── IEKF 헬퍼 ────────────────────────────────────────────────────────────

    // 연속 운동학 모델 — 식 (1)(2)
    Eigen::Matrix<float, 24, 1> motionModel(const FLIOState &x, const IMUData &u) const;

    // 상태 전이 Jacobian Fₓ — 식 (7)
    FLIOCovariance computeFx(const FLIOState &x, const IMUData &u, float dt) const;

    // 노이즈 Jacobian Fw (24×12) — 식 (7)
    Eigen::Matrix<float, 24, 12> computeFw(const FLIOState &x, float dt) const;

    // TSDF 기반 잔차 zⱼ + Jacobian Hⱼ 계산 — 식 (9)
    //   zⱼ  = tsdf × truncation  (표면까지 부호 거리)
    //   Hⱼ  = ∂zⱼ/∂x̃  (해석적 미분)
    bool computePointResidual(const LiDARPoint            &point,
                               const FLIOState             &x_hat,
                               float                       &residual,
                               Eigen::Matrix<float, 1, 24> &H_row) const;

    // 매니폴드 Jacobian Jᵏ (24×24) — 식 (11)
    FLIOCovariance computeManifoldJacobian(const FLIOState &x_hat,
                                            const FLIOState &x_pred) const;

    // ᴳp̄ⱼ = ᴳRᵢ·(ᴵRₗ·ᴸpⱼ + ᴵpₗ) + ᴳpᵢ — 식 (16)
    Eigen::Vector3f transformPoint(const Eigen::Vector3f &p_L,
                                    const FLIOState       &x) const;

    // 맵 범위 밖 복셀 제거 (deleteBox)
    void manageMapBounds(const Eigen::Vector3f &sensor_pos);

    // Q (12×12)
    Eigen::Matrix<float, 12, 12> buildProcessNoise() const;

    // ── 멤버 변수 ─────────────────────────────────────────────────────────────
    Config cfg_;

    FLIOState      x_opt_;   // 현재 최적 추정 (x̄ₖ)
    FLIOCovariance P_opt_;   // 현재 공분산 (P̄ₖ)

    FLIOState      x_pred_;  // Forward propagation 결과 (x̃ₖ)
    FLIOCovariance P_pred_;  // 전파 공분산 (P̃ₖ)

    std::deque<IMUData> imu_buffer_;
    double last_imu_time_  = -1.0;
    double last_scan_time_ = -1.0;

    // 맵 백엔드 (외부 소유, 포인터로 참조)
    BucketVoxelHash *mapPtr_ = nullptr;
    VulkanContext   *ctxPtr_ = nullptr;

    // 맵 이동 기준점 (deleteBox 판단용)
    Eigen::Vector3f mapCenter_ = Eigen::Vector3f::Zero();

    bool initialized_ = false;

    FastLIO2IEKFFail lastIekfFail_        = FastLIO2IEKFFail::None;
    int               lastIekfResidualM_ = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  §3. SO(3) 유틸리티
// ─────────────────────────────────────────────────────────────────────────────
namespace SO3
{
    Eigen::Matrix3f hat(const Eigen::Vector3f &v);
    Eigen::Matrix3f Exp(const Eigen::Vector3f &phi);
    Eigen::Vector3f Log(const Eigen::Matrix3f &R);
    Eigen::Matrix3f rightJacobian(const Eigen::Vector3f &phi);
    Eigen::Matrix3f rightJacobianInv(const Eigen::Vector3f &phi);
}
