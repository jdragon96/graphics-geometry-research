#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "Feature/IFeature.h"

#include <string>
#include <vector>
#include <memory>
#include <optional>

// ── Key definitions ───────────────────────────────────────────────────────────

enum class Key {
    Unknown = -1,
    Space   = GLFW_KEY_SPACE,
    Escape  = GLFW_KEY_ESCAPE,
    Enter   = GLFW_KEY_ENTER,
    Tab     = GLFW_KEY_TAB,
    Left    = GLFW_KEY_LEFT, Right = GLFW_KEY_RIGHT,
    Up      = GLFW_KEY_UP,   Down  = GLFW_KEY_DOWN,
    A = GLFW_KEY_A, B = GLFW_KEY_B, C = GLFW_KEY_C, D = GLFW_KEY_D,
    E = GLFW_KEY_E, F = GLFW_KEY_F, G = GLFW_KEY_G, H = GLFW_KEY_H,
    I = GLFW_KEY_I, J = GLFW_KEY_J, K = GLFW_KEY_K, L = GLFW_KEY_L,
    M = GLFW_KEY_M, N = GLFW_KEY_N, O = GLFW_KEY_O, P = GLFW_KEY_P,
    Q = GLFW_KEY_Q, R = GLFW_KEY_R, S = GLFW_KEY_S, T = GLFW_KEY_T,
    U = GLFW_KEY_U, V = GLFW_KEY_V, W = GLFW_KEY_W, X = GLFW_KEY_X,
    Y = GLFW_KEY_Y, Z = GLFW_KEY_Z,
    F1 = GLFW_KEY_F1, F2 = GLFW_KEY_F2, F3 = GLFW_KEY_F3,
    F4 = GLFW_KEY_F4, F5 = GLFW_KEY_F5, F6 = GLFW_KEY_F6,
};

enum class KeyAction {
    Press   = GLFW_PRESS,
    Release = GLFW_RELEASE,
    Repeat  = GLFW_REPEAT,
};

// ── Application ───────────────────────────────────────────────────────────────

class Application {
public:
    Application(uint32_t width, uint32_t height, const std::string& title);
    ~Application();

    // Feature 등록 (run() 호출 전에 추가)
    void addFeature(std::unique_ptr<IFeature> feature);

    void run();

private:
    // ── Window ────────────────────────────────────────────────────────────────
    uint32_t    width_, height_;
    std::string title_;
    GLFWwindow* window_ = nullptr;

    // ── Vulkan core ───────────────────────────────────────────────────────────
    VkInstance       instance_       = VK_NULL_HANDLE;
    VkSurfaceKHR     surface_        = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          graphicsQueue_  = VK_NULL_HANDLE;
    VkQueue          presentQueue_   = VK_NULL_HANDLE;
    uint32_t         graphicsFamily_ = 0;

    // ── Swapchain ─────────────────────────────────────────────────────────────
    VkSwapchainKHR           swapchain_           = VK_NULL_HANDLE;
    std::vector<VkImage>     swapchainImages_;
    VkFormat                 swapchainFormat_;
    VkExtent2D               swapchainExtent_;
    std::vector<VkImageView> swapchainViews_;

    // ── Render pass / framebuffers ────────────────────────────────────────────
    VkRenderPass                 renderPass_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer>   framebuffers_;

    // ── Commands ──────────────────────────────────────────────────────────────
    VkCommandPool                commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;

    // ── Sync ──────────────────────────────────────────────────────────────────
    static constexpr int         MAX_FRAMES = 2;
    std::vector<VkSemaphore>     imageAvailableSems_;
    std::vector<VkSemaphore>     renderFinishedSems_;
    std::vector<VkFence>         inFlightFences_;
    uint32_t                     currentFrame_ = 0;

    // ── ImGui ─────────────────────────────────────────────────────────────────
    VkDescriptorPool imguiPool_ = VK_NULL_HANDLE;

    // ── Features ──────────────────────────────────────────────────────────────
    std::vector<std::unique_ptr<IFeature>> features_;
    int activeFeature_ = 0;

    // ── Internal helpers ──────────────────────────────────────────────────────
    struct QueueFamilies {
        std::optional<uint32_t> graphics;
        std::optional<uint32_t> present;
        bool isComplete() const { return graphics.has_value() && present.has_value(); }
    };

    struct SwapchainSupport {
        VkSurfaceCapabilitiesKHR        caps;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR>   presentModes;
    };

    void initWindow();
    void initVulkan();
    void initImGui();
    void mainLoop();
    void drawFrame();
    void cleanup();

    void createInstance();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void createRenderPass();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();

    QueueFamilies   findQueueFamilies(VkPhysicalDevice dev);
    SwapchainSupport querySwapchainSupport(VkPhysicalDevice dev);

    VulkanContext makeContext() const;

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);

    static void keyCallback(GLFWwindow* win, int key, int scancode, int action, int mods);
};
