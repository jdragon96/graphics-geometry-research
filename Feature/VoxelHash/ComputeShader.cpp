#include "ComputeShader.h"

#include <fstream>
#include <stdexcept>
#include <vector>

// ── Shader module helpers ─────────────────────────────────────────────────────

static VkShaderModule loadSpv(VkDevice device, const std::string &path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("ComputeShader: cannot open " + path);

    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint32_t> buf(size / 4);
    f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(size));

    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = size;
    ci.pCode    = buf.data();

    VkShaderModule mod;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("ComputeShader: vkCreateShaderModule");
    return mod;
}

static VkShaderModule compileGlsl(VkDevice device, const std::string &src,
                                   shaderc_shader_kind kind)
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    opts.SetOptimizationLevel(shaderc_optimization_level_performance);

    auto result = compiler.CompileGlslToSpv(src, kind, "inline", opts);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        throw std::runtime_error("ComputeShader GLSL compile: " + result.GetErrorMessage());

    std::vector<uint32_t> spv(result.cbegin(), result.cend());

    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spv.size() * sizeof(uint32_t);
    ci.pCode    = spv.data();

    VkShaderModule mod;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("ComputeShader: vkCreateShaderModule (GLSL)");
    return mod;
}

// ── Common init helpers ───────────────────────────────────────────────────────

void ComputeShader::initDescAndLayout(VkDevice device,
                                      const std::vector<Binding> &bindings,
                                      uint32_t pushConstantBytes)
{
    storedDevice_ = device;
    pcBytes_      = pushConstantBytes;

    std::vector<VkDescriptorSetLayoutBinding> lb(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i)
    {
        lb[i].binding         = bindings[i].slot;
        lb[i].descriptorType  = bindings[i].type;
        lb[i].descriptorCount = 1;
        lb[i].stageFlags      = bindings[i].stages;
    }

    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = static_cast<uint32_t>(lb.size());
    dlci.pBindings    = lb.data();
    if (vkCreateDescriptorSetLayout(device, &dlci, nullptr, &descLayout_) != VK_SUCCESS)
        throw std::runtime_error("ComputeShader: vkCreateDescriptorSetLayout");

    std::vector<VkDescriptorPoolSize> ps(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i)
        ps[i] = {bindings[i].type, 1};

    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 1;
    dpci.poolSizeCount = static_cast<uint32_t>(ps.size());
    dpci.pPoolSizes    = ps.data();
    if (vkCreateDescriptorPool(device, &dpci, nullptr, &descPool_) != VK_SUCCESS)
        throw std::runtime_error("ComputeShader: vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = descPool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &descLayout_;
    if (vkAllocateDescriptorSets(device, &dsai, &descSet_) != VK_SUCCESS)
        throw std::runtime_error("ComputeShader: vkAllocateDescriptorSets");

    VkPipelineLayoutCreateInfo plci{};
    plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &descLayout_;

    VkPushConstantRange pcr{};
    if (pcBytes_ > 0)
    {
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset     = 0;
        pcr.size       = pcBytes_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
    }
    if (vkCreatePipelineLayout(device, &plci, nullptr, &layout_) != VK_SUCCESS)
        throw std::runtime_error("ComputeShader: vkCreatePipelineLayout");
}

void ComputeShader::buildPipeline(VkDevice device, VkShaderModule mod)
{
    VkComputePipelineCreateInfo cpci{};
    cpci.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module       = mod;
    cpci.stage.pName        = "main";
    cpci.layout             = layout_;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device, mod, nullptr);
        throw std::runtime_error("ComputeShader: vkCreateComputePipelines");
    }
    vkDestroyShaderModule(device, mod, nullptr);
}

// ── Public API ────────────────────────────────────────────────────────────────

void ComputeShader::Initialize(
    VkDevice device,
    const std::string &spvPath,
    const std::vector<Binding> &bindings,
    uint32_t pushConstantBytes)
{
    initDescAndLayout(device, bindings, pushConstantBytes);
    buildPipeline(device, loadSpv(device, spvPath));
}

void ComputeShader::InitializeGLSL(
    VkDevice device,
    const std::string &glslSrc,
    shaderc_shader_kind kind,
    const std::vector<Binding> &bindings,
    uint32_t pushConstantBytes)
{
    initDescAndLayout(device, bindings, pushConstantBytes);
    buildPipeline(device, compileGlsl(device, glslSrc, kind));
}

void ComputeShader::BindBuffer(uint32_t slot, VkBuffer buffer,
                               VkDeviceSize offset, VkDeviceSize size)
{
    VkDescriptorBufferInfo info{buffer, offset, size};
    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet          = descSet_;
    w.dstBinding      = slot;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo     = &info;
    vkUpdateDescriptorSets(storedDevice_, 1, &w, 0, nullptr);
}

void ComputeShader::Bind(VkCommandBuffer cmd) const
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout_, 0, 1, &descSet_, 0, nullptr);
}

void ComputeShader::Dispatch(VkCommandBuffer cmd,
                             uint32_t x, uint32_t y, uint32_t z) const
{
    vkCmdDispatch(cmd, x, y, z);
}

void ComputeShader::Clear()
{
    if (storedDevice_ == VK_NULL_HANDLE)
        return;

    if (pipeline_   != VK_NULL_HANDLE) vkDestroyPipeline(storedDevice_, pipeline_, nullptr);
    if (layout_     != VK_NULL_HANDLE) vkDestroyPipelineLayout(storedDevice_, layout_, nullptr);
    if (descPool_   != VK_NULL_HANDLE) vkDestroyDescriptorPool(storedDevice_, descPool_, nullptr);
    if (descLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(storedDevice_, descLayout_, nullptr);

    pipeline_     = VK_NULL_HANDLE;
    layout_       = VK_NULL_HANDLE;
    descPool_     = VK_NULL_HANDLE;
    descSet_      = VK_NULL_HANDLE;
    descLayout_   = VK_NULL_HANDLE;
    storedDevice_ = VK_NULL_HANDLE;
}
