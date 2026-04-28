#pragma once

#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>
#include <string>
#include <vector>
#include <stdexcept>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

static VkShaderModule loadSpv(VkDevice device, const std::string &path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("loadSpv: cannot open " + path);
    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint32_t> buf(size / 4);
    f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(size));
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = size;
    ci.pCode = buf.data();
    VkShaderModule mod;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("loadSpv: vkCreateShaderModule failed for " + path);
    return mod;
}

// GLSL 소스 문자열을 런타임에 SPIR-V로 컴파일하여 VkShaderModule을 반환합니다.
inline VkShaderModule compileGLSL(VkDevice device, const std::string &src, shaderc_shader_kind kind)
{
    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    opts.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

    auto result = compiler.CompileGlslToSpv(src, kind, "inline_shader", opts);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
        throw std::runtime_error("GLSL compile error:\n" + result.GetErrorMessage());

    std::vector<uint32_t> spv(result.cbegin(), result.cend());

    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spv.size() * sizeof(uint32_t);
    ci.pCode = spv.data();

    VkShaderModule mod;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("vkCreateShaderModule failed");
    return mod;
}

// physDev 의 메모리 타입 중 filter 비트와 props 플래그를 모두 만족하는 인덱스를 반환합니다.
inline uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t filter, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physDev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("no suitable memory type");
}
