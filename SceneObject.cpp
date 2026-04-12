#include "SceneObject.h"

SceneObject::SceneObject(std::vector<Vertex> vertices)
    : vertices(std::move(vertices)) {}
