#include "RenderingShader.h"

#include <fstream>
#include <stdexcept>

namespace
{
VkShaderModule loadSpv(VkDevice device, const std::string &path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("RenderingShader: cannot open " + path);

    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint32_t> buf(size / 4);
    f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(size));

    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = size;
    ci.pCode = buf.data();

    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("RenderingShader: vkCreateShaderModule");
    return mod;
}

VkShaderModule compileGlsl(VkDevice device, const std::string &src, shaderc_shader_kind kind)
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    opts.SetOptimizationLevel(shaderc_optimization_level_performance);

    auto result = compiler.CompileGlslToSpv(src, kind, "inline", opts);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        throw std::runtime_error("RenderingShader GLSL compile: " + result.GetErrorMessage());

    std::vector<uint32_t> spv(result.cbegin(), result.cend());

    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spv.size() * sizeof(uint32_t);
    ci.pCode = spv.data();

    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("RenderingShader: vkCreateShaderModule (GLSL)");
    return mod;
}
} // namespace

void RenderingShader::initDescAndLayout(VkDevice device,
                                        const std::vector<Binding> &bindings,
                                        uint32_t pushConstantBytes)
{
    storedDevice_ = device;
    pcBytes_ = pushConstantBytes;

    std::vector<VkDescriptorSetLayoutBinding> lb(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i)
    {
        lb[i].binding = bindings[i].slot;
        lb[i].descriptorType = bindings[i].type;
        lb[i].descriptorCount = 1;
        lb[i].stageFlags = bindings[i].stages;
    }

    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = static_cast<uint32_t>(lb.size());
    dlci.pBindings = lb.data();
    if (vkCreateDescriptorSetLayout(device, &dlci, nullptr, &descLayout_) != VK_SUCCESS)
        throw std::runtime_error("RenderingShader: vkCreateDescriptorSetLayout");

    std::vector<VkDescriptorPoolSize> ps;
    ps.reserve(bindings.size());
    for (const auto &binding : bindings)
    {
        bool found = false;
        for (auto &poolSize : ps)
        {
            if (poolSize.type == binding.type)
            {
                poolSize.descriptorCount += 1;
                found = true;
                break;
            }
        }
        if (!found)
            ps.push_back(VkDescriptorPoolSize{binding.type, 1});
    }

    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
    dpci.pPoolSizes = ps.empty() ? nullptr : ps.data();
    if (vkCreateDescriptorPool(device, &dpci, nullptr, &descPool_) != VK_SUCCESS)
        throw std::runtime_error("RenderingShader: vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = descPool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &descLayout_;
    if (vkAllocateDescriptorSets(device, &dsai, &descSet_) != VK_SUCCESS)
        throw std::runtime_error("RenderingShader: vkAllocateDescriptorSets");

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &descLayout_;

    VkPushConstantRange pcr{};
    if (pcBytes_ > 0)
    {
        pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr.offset = 0;
        pcr.size = pcBytes_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
    }

    if (vkCreatePipelineLayout(device, &plci, nullptr, &layout_) != VK_SUCCESS)
        throw std::runtime_error("RenderingShader: vkCreatePipelineLayout");
}

void RenderingShader::buildPipeline(VkDevice device,
                                    VkRenderPass renderPass,
                                    VkExtent2D extent,
                                    VkShaderModule vertMod,
                                    VkShaderModule fragMod,
                                    const PipelineOptions &options)
{
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = static_cast<uint32_t>(options.vertexBindings.size());
    vi.pVertexBindingDescriptions = options.vertexBindings.empty() ? nullptr : options.vertexBindings.data();
    vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(options.vertexAttributes.size());
    vi.pVertexAttributeDescriptions = options.vertexAttributes.empty() ? nullptr : options.vertexAttributes.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = options.topology;

    VkViewport vp{0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, extent};
    VkPipelineViewportStateCreateInfo vps{};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.pViewports = &vp;
    vps.scissorCount = 1;
    vps.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = options.polygonMode;
    raster.lineWidth = 1.f;
    raster.cullMode = options.cullMode;
    raster.frontFace = options.frontFace;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = options.depthTestEnable;
    depth.depthWriteEnable = options.depthWriteEnable;
    depth.depthCompareOp = options.depthCompareOp;
    depth.depthBoundsTestEnable = VK_FALSE;
    depth.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask      = 0xF;
    ba.blendEnable         = options.blendEnable;
    ba.srcColorBlendFactor = options.srcColorBlendFactor;
    ba.dstColorBlendFactor = options.dstColorBlendFactor;
    ba.colorBlendOp        = options.colorBlendOp;
    ba.srcAlphaBlendFactor = options.srcAlphaBlendFactor;
    ba.dstAlphaBlendFactor = options.dstAlphaBlendFactor;
    ba.alphaBlendOp        = options.alphaBlendOp;
    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &ba;

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vps;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &depth;
    pci.pColorBlendState = &blend;
    pci.layout = layout_;
    pci.renderPass = renderPass;
    pci.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline_) != VK_SUCCESS)
        throw std::runtime_error("RenderingShader: vkCreateGraphicsPipelines");
}

void RenderingShader::Initialize(
    VkDevice device,
    VkRenderPass renderPass,
    VkExtent2D extent,
    const std::string &vertSpvPath,
    const std::string &fragSpvPath,
    const std::vector<Binding> &bindings,
    const PipelineOptions &options,
    uint32_t pushConstantBytes)
{
    initDescAndLayout(device, bindings, pushConstantBytes);
    VkShaderModule vertMod = loadSpv(device, vertSpvPath);
    VkShaderModule fragMod = loadSpv(device, fragSpvPath);
    try
    {
        buildPipeline(device, renderPass, extent, vertMod, fragMod, options);
    }
    catch (...)
    {
        vkDestroyShaderModule(device, vertMod, nullptr);
        vkDestroyShaderModule(device, fragMod, nullptr);
        throw;
    }
    vkDestroyShaderModule(device, vertMod, nullptr);
    vkDestroyShaderModule(device, fragMod, nullptr);
}

void RenderingShader::InitializeGLSL(
    VkDevice device,
    VkRenderPass renderPass,
    VkExtent2D extent,
    const std::string &vertSrc,
    const std::string &fragSrc,
    const std::vector<Binding> &bindings,
    const PipelineOptions &options,
    uint32_t pushConstantBytes)
{
    initDescAndLayout(device, bindings, pushConstantBytes);
    VkShaderModule vertMod = compileGlsl(device, vertSrc, shaderc_glsl_vertex_shader);
    VkShaderModule fragMod = compileGlsl(device, fragSrc, shaderc_glsl_fragment_shader);
    try
    {
        buildPipeline(device, renderPass, extent, vertMod, fragMod, options);
    }
    catch (...)
    {
        vkDestroyShaderModule(device, vertMod, nullptr);
        vkDestroyShaderModule(device, fragMod, nullptr);
        throw;
    }
    vkDestroyShaderModule(device, vertMod, nullptr);
    vkDestroyShaderModule(device, fragMod, nullptr);
}

void RenderingShader::BindBuffer(uint32_t slot, VkBuffer buffer,
                                 VkDeviceSize offset, VkDeviceSize size, VkDescriptorType type)
{
    VkDescriptorBufferInfo info{buffer, offset, size};
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = descSet_;
    w.dstBinding = slot;
    w.descriptorCount = 1;
    w.descriptorType = type;
    w.pBufferInfo = &info;
    vkUpdateDescriptorSets(storedDevice_, 1, &w, 0, nullptr);
}

void RenderingShader::BindImage(uint32_t slot, VkImageView imageView, VkSampler sampler,
                                VkImageLayout layout, VkDescriptorType type)
{
    VkDescriptorImageInfo info{};
    info.imageLayout = layout;
    info.imageView = imageView;
    info.sampler = sampler;

    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = descSet_;
    w.dstBinding = slot;
    w.descriptorCount = 1;
    w.descriptorType = type;
    w.pImageInfo = &info;
    vkUpdateDescriptorSets(storedDevice_, 1, &w, 0, nullptr);
}

void RenderingShader::Bind(VkCommandBuffer cmd) const
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            layout_, 0, 1, &descSet_, 0, nullptr);
}

void RenderingShader::Draw(VkCommandBuffer cmd, uint32_t vertexCount,
                           uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance) const
{
    vkCmdDraw(cmd, vertexCount, instanceCount, firstVertex, firstInstance);
}

void RenderingShader::Clear()
{
    if (storedDevice_ == VK_NULL_HANDLE)
        return;

    if (pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(storedDevice_, pipeline_, nullptr);
    if (layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(storedDevice_, layout_, nullptr);
    if (descPool_ != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(storedDevice_, descPool_, nullptr);
    if (descLayout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(storedDevice_, descLayout_, nullptr);

    pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    descPool_ = VK_NULL_HANDLE;
    descSet_ = VK_NULL_HANDLE;
    descLayout_ = VK_NULL_HANDLE;
    storedDevice_ = VK_NULL_HANDLE;
}
