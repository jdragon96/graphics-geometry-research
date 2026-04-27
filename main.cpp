#include "Application.h"
#include "Feature/TriangleFeature.h"
#include "Feature/BlinnPhongFeature.h"
#include "Feature/ComputeTest.h"
#include "Feature/TSDF.h"
#include "Feature/BucketedHash.h"
#include "Feature/VoxelHash/VoxelHash.h"
// #include "Feature/RecoveredPointCloud.h"  // 구 VoxelHash API 의존 → 재설계 후 재연결 필요

int main()
{
    Application app(800, 600, "Vision 3D");
    app.addFeature(std::make_unique<TriangleFeature>());
    app.addFeature(std::make_unique<BlinnPhongFeature>());
    app.addFeature(std::make_unique<ComputeTest>());
    app.addFeature(std::make_unique<TSDFFeature>());
    app.addFeature(std::make_unique<BucketedHash>());
    app.addFeature(std::make_unique<VoxelHashFeature>());
    // RecoveredPointCloudFeature: 구 API(snapshotRecentSamples 등)에 의존 → 신규 VoxelHash에서 제거됨
    app.run();
    return 0;
}
