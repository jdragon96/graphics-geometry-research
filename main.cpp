#include "Application.h"
#include "Feature/TriangleFeature.h"
#include "Feature/BlinnPhongFeature.h"

int main() {
    Application app(800, 600, "Vision 3D");
    app.addFeature(std::make_unique<TriangleFeature>());
    app.addFeature(std::make_unique<BlinnPhongFeature>());
    app.run();
    return 0;
}
