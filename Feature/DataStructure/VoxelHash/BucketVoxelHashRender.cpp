#include "BucketVoxelHash.h"
#include "../../Utilities.h"

void BucketVoxelHash::RenderVoxel(VkCommandBuffer cmd, const float *mvp)
{
    VHB_RenderPC pc{};
    for (int i = 0; i < 16; ++i)
    {
        pc.mvp[i] = mvp[i];
    }
    pc.voxelSize  = voxelSize_;
    pc.truncation = truncation_;
    pc.projectSurface = 0.0f;
    kVoxelRenderShader_.Bind(cmd);
    kVoxelRenderShader_.Push(cmd, VK_SHADER_STAGE_VERTEX_BIT, pc);

    // 복셀당 36 vertices (큐브 6면 × 2삼각형 × 3꼭짓점)
    // 빈 슬롯은 셰이더에서 NDC 밖으로 클리핑
    kVoxelRenderShader_.Draw(cmd, VH_TOTAL_ENTRIES * 36);
}

void BucketVoxelHash::RenderColor(VkCommandBuffer cmd, const float *mvp, bool useSlotPoint)
{
    VHB_RenderPC pc{};
    for (int i = 0; i < 16; ++i)
        pc.mvp[i] = mvp[i];
    pc.voxelSize = voxelSize_;
    pc.projectSurface = 0.0f;
    if (useSlotPoint)
    {
        kPointRenderShader_.Bind(cmd);
        kPointRenderShader_.Push(cmd, VK_SHADER_STAGE_VERTEX_BIT, pc);
        kPointRenderShader_.Draw(cmd, lastIntegratedCount_);
        return;
    }

    kColorRenderShader_.Bind(cmd);
    kColorRenderShader_.Push(cmd, VK_SHADER_STAGE_VERTEX_BIT, pc);
    // 슬롯 수만큼 버텍스 발행 — 빈 슬롯은 셰이더에서 클리핑
    kColorRenderShader_.Draw(cmd, VH_TOTAL_ENTRIES);
}

void BucketVoxelHash::RenderVoxelSlot(VkCommandBuffer cmd, const float *mvp)
{
    VHB_RenderPC pc{};
    for (int i = 0; i < 16; ++i)
        pc.mvp[i] = mvp[i];
    pc.voxelSize = voxelSize_;
    pc.projectSurface = 0.0f;
    kPointRenderShader_.Bind(cmd);
    kPointRenderShader_.Push(cmd, VK_SHADER_STAGE_VERTEX_BIT, pc);
    kPointRenderShader_.Draw(cmd, lastIntegratedCount_);
}


void BucketVoxelHash::RenderTSDF(VkCommandBuffer cmd, const float *mvp)
{
    VHB_RenderPC pc{};
    for (int i = 0; i < 16; ++i)
        pc.mvp[i] = mvp[i];
    pc.voxelSize = voxelSize_;
    pc.truncation = truncation_;
    pc.projectSurface = tsdfProjectToSurface_ ? 1.0f : 0.0f;

    kTSDFRenderShader_.Bind(cmd);
    kTSDFRenderShader_.Push(cmd, VK_SHADER_STAGE_VERTEX_BIT, pc);
    kTSDFRenderShader_.Draw(cmd, VH_TOTAL_ENTRIES);
}