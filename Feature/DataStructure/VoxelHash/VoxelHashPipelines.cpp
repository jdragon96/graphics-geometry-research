#include "VoxelHash.h"
#include <stdexcept>
#include <fstream>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  .spv 파일 로드 헬퍼
// ─────────────────────────────────────────────────────────────────────────────

static VkShaderModule loadSpv(VkDevice device, const std::string &path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("VoxelHash: cannot open shader: " + path);
    size_t size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint32_t> buf(size / 4);
    f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(size));

    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = size;
    ci.pCode = buf.data();
    VkShaderModule mod;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: vkCreateShaderModule: " + path);
    return mod;
}

// ─────────────────────────────────────────────────────────────────────────────
//  createDescriptors — 7개 바인딩 디스크립터 셋 구성
//
//  binding  버퍼              stage
//  ───────  ─────────────────  ──────────────────────────
//  0        htBuf_            COMPUTE + VERTEX
//  1        ptBuf_            COMPUTE
//  2        ctrBuf_           COMPUTE
//  3        sortedPtBuf_      COMPUTE + VERTEX
//  4        histBuf_          COMPUTE
//  5        blockSumsBuf_     COMPUTE
//  6        cellStartBuf_     COMPUTE
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::createDescriptors()
{
    constexpr int NUM_BINDINGS = 11;
    VkDescriptorSetLayoutBinding b[NUM_BINDINGS]{};
    for (int i = 0; i < NUM_BINDINGS; i++)
    {
        b[i].binding = static_cast<uint32_t>(i);
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // binding 0 (htBuf_), 5 (sortedPosBuf_): 렌더 셰이더도 접근
    b[0].stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;
    b[5].stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = NUM_BINDINGS;
    lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(ctx_.device, &lci, nullptr, &descLayout_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: descriptor layout");

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NUM_BINDINGS};
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(ctx_.device, &pci, nullptr, &descPool_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: descriptor pool");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descPool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &descLayout_;
    if (vkAllocateDescriptorSets(ctx_.device, &ai, &descSet_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: descriptor alloc");

    // binding 순서와 VoxelHash.h 주석 테이블이 1:1 대응
    VkDescriptorBufferInfo infos[NUM_BINDINGS] = {
        {htBuf_, 0, VK_WHOLE_SIZE},        //  0
        {posBuf_, 0, VK_WHOLE_SIZE},       //  1
        {nrmBuf_, 0, VK_WHOLE_SIZE},       //  2
        {colBuf_, 0, VK_WHOLE_SIZE},       //  3
        {ctrBuf_, 0, VK_WHOLE_SIZE},       //  4
        {sortedPosBuf_, 0, VK_WHOLE_SIZE}, //  5
        {sortedNrmBuf_, 0, VK_WHOLE_SIZE}, //  6
        {sortedColBuf_, 0, VK_WHOLE_SIZE}, //  7
        {histBuf_, 0, VK_WHOLE_SIZE},      //  8
        {blockSumsBuf_, 0, VK_WHOLE_SIZE}, //  9
        {cellStartBuf_, 0, VK_WHOLE_SIZE}, // 10
    };
    VkWriteDescriptorSet w[NUM_BINDINGS]{};
    for (int i = 0; i < NUM_BINDINGS; i++)
    {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = descSet_;
        w[i].dstBinding = static_cast<uint32_t>(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(ctx_.device, NUM_BINDINGS, w, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
//  createComputePipelines — .spv 파일에서 7개 컴퓨트 파이프라인 생성
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::createComputePipelines()
{
    // 최대 push constant = sizeof(VH_GatherPC) = 32 bytes
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lci.setLayoutCount = 1;
    lci.pSetLayouts = &descLayout_;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(ctx_.device, &lci, nullptr, &compLayout_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: compute layout");

    auto make = [&](const std::string &name) -> VkPipeline
    {
        VkShaderModule mod = loadSpv(ctx_.device, ctx_.basePath + "/shaders/" + name + ".spv");
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = compLayout_;
        VkPipeline p;
        if (vkCreateComputePipelines(ctx_.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p) != VK_SUCCESS)
            throw std::runtime_error("VoxelHash: compute pipeline: " + name);
        vkDestroyShaderModule(ctx_.device, mod, nullptr);
        return p;
    };

    clearPipe_ = make("vh_clear.comp");
    histogramPipe_ = make("vh_histogram.comp");
    prefixScanPipe_ = make("vh_prefix_scan.comp");
    scatterPipe_ = make("vh_scatter.comp");
    gatherPipe_ = make("vh_gather.comp");
    finalizePipe_ = make("vh_finalize.comp");
    countPipe_ = make("vh_count.comp");
}

// ─────────────────────────────────────────────────────────────────────────────
//  createRenderPipeline — 복셀 큐브 + 입력 포인트 오버레이 파이프라인
// ─────────────────────────────────────────────────────────────────────────────

void VoxelHashFeature::createRenderPipeline()
{
    VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VH_RenderPC)};
    VkPipelineLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lci.setLayoutCount = 1;
    lci.pSetLayouts = &descLayout_;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(ctx_.device, &lci, nullptr, &renderLayout_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: render layout");

    auto vMod = loadSpv(ctx_.device, ctx_.basePath + "/shaders/vh_render.vert.spv");
    auto fMod = loadSpv(ctx_.device, ctx_.basePath + "/shaders/vh_render.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_VERTEX_BIT, vMod, "main"},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, fMod, "main"},
    };
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                              nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkViewport vp{0, 0, (float)ctx_.extent.width, (float)ctx_.extent.height, 0, 1};
    VkRect2D sc{{0, 0}, ctx_.extent};
    VkPipelineViewportStateCreateInfo vps{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                          nullptr, 0, 1, &vp, 1, &sc};
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth = 1.f;
    raster.cullMode = VK_CULL_MODE_NONE;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                            nullptr, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              nullptr,
                                              0,
                                              VK_FALSE,
                                              {},
                                              1,
                                              &ba};

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vps;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &ds;
    pci.pColorBlendState = &blend;
    pci.layout = renderLayout_;
    pci.renderPass = ctx_.renderPass;
    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &pci, nullptr, &renderPipe_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: render pipeline");

    vkDestroyShaderModule(ctx_.device, vMod, nullptr);

    // 입력 포인트 오버레이 파이프라인 (같은 레이아웃, vert만 교체)
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    auto pvMod = loadSpv(ctx_.device, ctx_.basePath + "/shaders/vh_ptrender.vert.spv");
    stages[0].module = pvMod;
    if (vkCreateGraphicsPipelines(ctx_.device, VK_NULL_HANDLE, 1, &pci, nullptr, &ptRenderPipe_) != VK_SUCCESS)
        throw std::runtime_error("VoxelHash: ptRender pipeline");
    vkDestroyShaderModule(ctx_.device, pvMod, nullptr);
    vkDestroyShaderModule(ctx_.device, fMod, nullptr);
}
