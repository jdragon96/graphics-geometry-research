#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <stdexcept>

#include "../../Utilities.h"

class Buffer
{
public:
    // RAII 기반 매핑 관리자
    template <typename T>
    class ScopedMemoryGuard
    {
    public:
        ScopedMemoryGuard(VkDevice device, VkDeviceMemory memory, T *ptr)
            : device(device), memory(memory), ptr(ptr) {}

        ~ScopedMemoryGuard()
        {
            if (ptr)
                vkUnmapMemory(device, memory);
        }

        // 복사 방지 (이동만 가능하게 설정하는 것이 안전)
        ScopedMemoryGuard(const ScopedMemoryGuard &) = delete;
        ScopedMemoryGuard &operator=(const ScopedMemoryGuard &) = delete;

        T *get() const { return ptr; }
        T &operator*() const { return *ptr; }
        T *operator->() const { return ptr; }

    private:
        VkDevice device;
        VkDeviceMemory memory;
        T *ptr;
    };

    // 소멸자에서 리소스 해제
    ~Buffer() { Clear(); }

    void Initialize(
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

    void Clear()
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

    VkBuffer GetBuffer() const { return bufferInstance; }
    uint32_t GetBindingNumber() { return iBindingNumber; }

    template <typename T>
    ScopedMemoryGuard<T> Access(VkDeviceSize bytes, uint32_t offset = 0, VkMemoryMapFlags flags = 0)
    {
        void *mappedPtr = nullptr;
        if (vkMapMemory(storedDevice, memoryInstance, offset, bytes, flags, &mappedPtr) != VK_SUCCESS)
            throw std::runtime_error("VoxelHash: vkMapMemory failed");

        return ScopedMemoryGuard<T>(storedDevice, memoryInstance, static_cast<T *>(mappedPtr));
    }

private:
    VkDevice storedDevice = VK_NULL_HANDLE;
    VkBuffer bufferInstance = VK_NULL_HANDLE;
    VkDeviceMemory memoryInstance = VK_NULL_HANDLE;
    uint32_t iBindingNumber = 0;
};