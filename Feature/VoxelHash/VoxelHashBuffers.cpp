#include "VoxelHash.h"
#include "../../Utilities.h"
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
//  createBuf — Vulkan 버퍼 + 메모리 할당 헬퍼
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::createBuf(VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags props,
                                 VkBuffer &buf, VkDeviceMemory &mem)
{
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx_.device, &bci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: vkCreateBuffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx_.device, buf, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(ctx_.physicalDevice, req.memoryTypeBits, props);
    if (vkAllocateMemory(ctx_.device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: vkAllocateMemory");
    vkBindBufferMemory(ctx_.device, buf, mem, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
//  createBuffers — SoA 레이아웃, 11개 바인딩
//
//  bind  버퍼              속성           크기
//  ────  ──────────────    ─────────────  ──────────────────────
//   0    htBuf_            device-local   192 MB
//   1    posBuf_           host-coherent  BATCH × 16 B (vec4)
//   2    nrmBuf_           host-coherent  BATCH × 16 B (vec4)
//   3    colBuf_           host-coherent  BATCH ×  4 B (uint)
//   4    ctrBuf_           host-coherent    4 B
//   5    sortedPosBuf_     device-local   BATCH × 16 B
//   6    sortedNrmBuf_     device-local   BATCH × 16 B
//   7    sortedColBuf_     device-local   BATCH ×  4 B
//   8    histBuf_          device-local   NUM_BUCKETS × 4 B
//   9    blockSumsBuf_     device-local   SCAN_BLOCKS × 4 B
//  10    cellStartBuf_     device-local   NUM_BUCKETS × 4 B
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::createBuffers()
{
    constexpr auto DEVICE = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    constexpr auto HOST = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr auto SSBO = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    constexpr auto XFER = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    constexpr VkDeviceSize VEC4_BATCH = sizeof(float) * 4 * VH_BATCH_SIZE;
    constexpr VkDeviceSize COL_BATCH = sizeof(uint32_t) * VH_BATCH_SIZE;

    createBuf(sizeof(VH_Entry) * VH_TOTAL_ENTRIES, SSBO, DEVICE, htBuf_, htMem_);
    createBuf(VEC4_BATCH, SSBO, HOST, posBuf_, posMem_);
    createBuf(VEC4_BATCH, SSBO, HOST, nrmBuf_, nrmMem_);
    createBuf(COL_BATCH, SSBO, HOST, colBuf_, colMem_);
    createBuf(sizeof(uint32_t), SSBO | XFER, HOST, ctrBuf_, ctrMem_);
    createBuf(VEC4_BATCH, SSBO, DEVICE, sortedPosBuf_, sortedPosMem_);
    createBuf(VEC4_BATCH, SSBO, DEVICE, sortedNrmBuf_, sortedNrmMem_);
    createBuf(COL_BATCH, SSBO, DEVICE, sortedColBuf_, sortedColMem_);
    createBuf(sizeof(uint32_t) * VH_NUM_BUCKETS, SSBO | XFER, DEVICE, histBuf_, histMem_);
    createBuf(sizeof(uint32_t) * VH_SCAN_BLOCKS, SSBO, DEVICE, blockSumsBuf_, blockSumsMem_);
    createBuf(sizeof(uint32_t) * VH_NUM_BUCKETS, SSBO, DEVICE, cellStartBuf_, cellStartMem_);
}
