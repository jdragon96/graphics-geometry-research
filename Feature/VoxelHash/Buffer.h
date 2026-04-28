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
    Buffer();
    ~Buffer() { Clear(); }

    inline VkBuffer GetBuffer() const { return bufferInstance; }

    inline uint32_t GetBindingNumber() { return iBindingNumber; }

    void Initialize(
        VkPhysicalDevice physicalDevice, // 핸들은 값으로 전달 가능
        VkDevice device,
        VkDeviceSize bytes,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags props,
        uint32_t bindingNumber);

    void Clear();

    template <typename T>
    ScopedMemoryGuard<T> Access(VkDeviceSize bytes, uint32_t offset = 0, VkMemoryMapFlags flags = 0)
    {
        void *mappedPtr = nullptr;
        if (vkMapMemory(storedDevice, memoryInstance, offset, bytes, flags, &mappedPtr) != VK_SUCCESS)
            throw std::runtime_error("VoxelHash: vkMapMemory failed");

        return ScopedMemoryGuard<T>(storedDevice, memoryInstance, static_cast<T *>(mappedPtr));
    }

    void Barrier(VkCommandBuffer cmd, VkAccessFlags src, VkAccessFlags dst,
                 VkPipelineStageFlags srcS, VkPipelineStageFlags dstS);

private:
    VkDevice storedDevice = VK_NULL_HANDLE;
    VkBuffer bufferInstance = VK_NULL_HANDLE;
    VkDeviceMemory memoryInstance = VK_NULL_HANDLE;
    uint32_t iBindingNumber = 0;
};