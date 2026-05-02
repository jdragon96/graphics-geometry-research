#include "Application.h"
#include "Feature/Samples/TriangleFeature.h"
#include "Feature/Samples/BlinnPhongFeature.h"
#include "Feature/Samples/ComputeTest.h"
#include "Feature/DataStructure/VoxelHash/VoxelHash.h"
#include "Feature/DataStructure/VoxelHash/BucketVoxelHashFeature.h"
#include "Feature/DataStructure/GaussianSplatting/GaussianSplatting.h"
#include "Feature/SLAM/Viewer/ViewerFeature.h"
#include "Feature/SLAM/FastLIO2/FastLIO2Feature.h"
#include "Feature/SLAM/KissICP/KissICPFeature.h"

int main()
{
    Application app(800, 600, "Vision 3D");
    app.addFeature(std::make_unique<TriangleFeature>());
    app.addFeature(std::make_unique<BlinnPhongFeature>());
    app.addFeature(std::make_unique<ComputeTest>());
    app.addFeature(std::make_unique<VoxelHashFeature>());
    app.addFeature(std::make_unique<BucketVoxelHashFeature>());
    app.addFeature(std::make_unique<GaussianSplattingFeature>());
    app.addFeature(std::make_unique<KITTIViewerFeature>());
    app.addFeature(std::make_unique<FastLIO2Feature>());
    app.addFeature(std::make_unique<KissICPFeature>());
    app.run();
    return 0;
}
