#include "Buffer.h"
#include "../../Utilities.h"

Buffer::Buffer() {}

void Buffer::Initialize(
    VkPhysicalDevice physicalDevice, // 핸들은 값으로 전달 가능
    VkDevice device,
    VkDeviceSize bytes,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags props,
    uint32_t bindingNumber)
{
    this->iBindingNumber = bindingNumber;
    this->storedDevice = device; // 해제를 위해 저장

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bci, nullptr, &bufferInstance) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: vkCreateBuffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, bufferInstance, &req);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    // findMemoryType은 Utilities.h에 정의되어 있다고 가정
    ai.memoryTypeIndex = findMemoryType(physicalDevice, req.memoryTypeBits, props);

    if (vkAllocateMemory(device, &ai, nullptr, &memoryInstance) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: vkAllocateMemory");

    vkBindBufferMemory(device, bufferInstance, memoryInstance, 0);
}

void Buffer::Clear()
{
    if (storedDevice != VK_NULL_HANDLE)
    {
        persistentUnmap();
        if (bufferInstance != VK_NULL_HANDLE)
            vkDestroyBuffer(storedDevice, bufferInstance, nullptr);
        if (memoryInstance != VK_NULL_HANDLE)
            vkFreeMemory(storedDevice, memoryInstance, nullptr);
    }
    bufferInstance = VK_NULL_HANDLE;
    memoryInstance = VK_NULL_HANDLE;
}

void Buffer::Barrier(VkCommandBuffer cmd, VkAccessFlags src, VkAccessFlags dst,
                     VkPipelineStageFlags srcS, VkPipelineStageFlags dstS)
{
    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = bufferInstance;
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 1, &b, 0, nullptr);
}