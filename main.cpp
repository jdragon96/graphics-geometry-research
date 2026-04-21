#include "Application.h"
#include "Feature/TriangleFeature.h"
#include "Feature/BlinnPhongFeature.h"
#include "Feature/ComputeTest.h"
#include "Feature/TSDF.h"
#include "Feature/BucketedHash.h"
#include "Feature/VoxelHash.h"

int main()
{
    Application app(800, 600, "Vision 3D");
    app.addFeature(std::make_unique<TriangleFeature>());
    app.addFeature(std::make_unique<BlinnPhongFeature>());
    app.addFeature(std::make_unique<ComputeTest>());
    app.addFeature(std::make_unique<TSDFFeature>());
    app.addFeature(std::make_unique<BucketedHash>());
    app.addFeature(std::make_unique<VoxelHashFeature>());
    app.run();
    return 0;
}
