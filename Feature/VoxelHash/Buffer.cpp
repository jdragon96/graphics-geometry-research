#include "Buffer.h"
#include "../../Utilities.h"

Buffer::~Buffer() { Clear(); }

void Buffer::Initialize(
    VkPhysicalDevice physicalDevice, // 핸들은 값으로 전달 가능
    VkDevice device,
    VkDeviceSize bytes,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags props)
{
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
        if (bufferInstance != VK_NULL_HANDLE)
            vkDestroyBuffer(storedDevice, bufferInstance, nullptr);
        if (memoryInstance != VK_NULL_HANDLE)
            vkFreeMemory(storedDevice, memoryInstance, nullptr);
    }
    bufferInstance = VK_NULL_HANDLE;
    memoryInstance = VK_NULL_HANDLE;
}

template <typename T>
Buffer::ScopedMemoryGuard<T> Buffer::Access(VkDeviceSize bytes, uint32_t offset, VkMemoryMapFlags flags)
{
    void *mappedPtr = nullptr;
    if (vkMapMemory(storedDevice, memoryInstance, offset, bytes, flags, &mappedPtr) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: vkMapMemory failed");

    return ScopedMemoryGuard<T>(storedDevice, memoryInstance, static_cast<T *>(mappedPtr));
}