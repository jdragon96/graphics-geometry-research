#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>

// Vulkan 리소스를 Feature에 전달하기 위한 컨텍스트
struct VulkanContext
{
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    uint32_t graphicsQueueFamily;
    VkCommandPool commandPool;
    VkRenderPass renderPass;
    VkExtent2D extent;
    std::string basePath; // 실행 파일이 있는 디렉토리
};

// 매 프레임 렌더링에 필요한 정보
struct RenderContext
{
    VkCommandBuffer commandBuffer;
    uint32_t imageIndex;
};

class IFeature
{
public:
    virtual ~IFeature() = default;

    virtual const char *name() const = 0;

    // 앱 시작 시 Vulkan 리소스 초기화
    virtual void onInit(const VulkanContext &ctx) = 0;

    // 렌더 패스 시작 전 컴퓨트 커맨드 기록 (optional)
    virtual void onCompute(VkCommandBuffer cmd) {}

    // 매 프레임 Vulkan 렌더링
    virtual void onRender(const RenderContext &ctx) = 0;

    // 매 프레임 ImGui UI 그리기
    virtual void onImGui() {}

    // 키 이벤트 (GLFW key/action/mods 값 그대로 전달)
    virtual void onKey(int key, int action, int mods) {}

    // 앱 종료 시 리소스 해제
    virtual void onCleanup() = 0;
};
