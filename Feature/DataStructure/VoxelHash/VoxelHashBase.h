#pragma once

#include "../../Utilities.h"
#include <stdexcept>

class VoxelHashBase
{
public:
    void createBuf(
        VkPhysicalDevice &physicalDevice,
        VkDevice &device,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags props,
        VkBuffer &outBuffer,
        VkDeviceMemory &outMemory)
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bci, nullptr, &outBuffer) != VK_SUCCESS)
            throw std::runtime_error("VoxelHash: vkCreateBuffer");

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, outBuffer, &req);

        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(physicalDevice, req.memoryTypeBits, props);
        if (vkAllocateMemory(device, &ai, nullptr, &outMemory) != VK_SUCCESS)
            throw std::runtime_error("VoxelHash: vkAllocateMemory");

        vkBindBufferMemory(device, outBuffer, outMemory, 0);
    }
};