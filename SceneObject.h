#pragma once

#include <Eigen/Core>
#include <vulkan/vulkan.h>
#include <vector>
#include <array>

struct Vertex
{
    Eigen::Vector3f position;
    Eigen::Vector3f color;
    Eigen::Vector3f normal;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription desc{};
        desc.binding = 0;
        desc.stride = sizeof(Vertex);
        desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return desc;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 3> attrs{};

        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)};
        attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
        attrs[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};

        return attrs;
    }
};

class SceneObject
{
public:
    std::vector<Vertex> vertices;

    SceneObject(std::vector<Vertex> vertices);
};
