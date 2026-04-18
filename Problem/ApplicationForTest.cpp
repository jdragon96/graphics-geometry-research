#include "ApplicationForTest.h"

#include <mach-o/dyld.h>
#include <filesystem>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <iostream>
#include <stdexcept>
#include <set>
#include <algorithm>
#include <limits>
#include <cstring>

// ── Constructor / Destructor ──────────────────────────────────────────────────

Application::Application(uint32_t width, uint32_t height, const std::string &title)
    : width_(width), height_(height), title_(title) {}

Application::~Application() {}

void Application::addFeature(std::unique_ptr<IFeature> feature)
{
    features_.push_back(std::move(feature));
}

// ── Run ───────────────────────────────────────────────────────────────────────

void Application::run()
{
    initWindow();
    initVulkan();
    initImGui();

    for (auto &f : features_)
        f->onInit(makeContext());

    mainLoop();

    vkDeviceWaitIdle(device_);

    for (auto &f : features_)
        f->onCleanup();

    cleanup();
}

// ── Window ────────────────────────────────────────────────────────────────────

void Application::initWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window_ = glfwCreateWindow(width_, height_, title_.c_str(), nullptr, nullptr);
    glfwSetWindowUserPointer(window_, this);
    glfwSetKeyCallback(window_, keyCallback);
}

void Application::keyCallback(GLFWwindow *win, int key, int /*scancode*/, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureKeyboard)
        return;

    auto *app = static_cast<Application *>(glfwGetWindowUserPointer(win));

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(win, GLFW_TRUE);

    // 숫자 키 1~9: feature 전환
    if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9 && action == GLFW_PRESS)
    {
        int idx = key - GLFW_KEY_1;
        if (idx < (int)app->features_.size())
            app->activeFeature_ = idx;
    }

    if (app->activeFeature_ < (int)app->features_.size())
        app->features_[app->activeFeature_]->onKey(key, action, mods);
}

// ── Vulkan init ───────────────────────────────────────────────────────────────

void Application::initVulkan()
{
    createInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createRenderPass();
    createFramebuffers();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
}

void Application::createInstance()
{
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = title_.c_str();
    appInfo.apiVersion = VK_API_VERSION_1_0;

    uint32_t glfwExtCount = 0;
    const char **glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char *> exts(glfwExts, glfwExts + glfwExtCount);
    exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &appInfo;
    ci.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ci.ppEnabledExtensionNames = exts.data();

    if (vkCreateInstance(&ci, nullptr, &instance_) != VK_SUCCESS)
        throw std::runtime_error("failed to create instance");
}

void Application::createSurface()
{
    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS)
        throw std::runtime_error("failed to create surface");
}

Application::QueueFamilies Application::findQueueFamilies(VkPhysicalDevice dev)
{
    QueueFamilies q;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> fams(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, fams.data());

    for (uint32_t i = 0; i < count; i++)
    {
        if (fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            q.graphics = i;
        }
        VkBool32 present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface_, &present);
        if (present)
        {
            q.present = i;
        }
        if (q.isComplete())
        {
            break;
        }
    }
    return q;
}

Application::SwapchainSupport Application::querySwapchainSupport(VkPhysicalDevice dev)
{
    SwapchainSupport s;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface_, &s.caps);

    uint32_t n;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface_, &n, nullptr);
    s.formats.resize(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface_, &n, s.formats.data());

    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface_, &n, nullptr);
    s.presentModes.resize(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface_, &n, s.presentModes.data());
    return s;
}

void Application::pickPhysicalDevice()
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (!count)
        throw std::runtime_error("no Vulkan GPU found");
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(instance_, &count, devs.data());

    for (auto &d : devs)
    {
        auto q = findQueueFamilies(d);
        auto sc = querySwapchainSupport(d);
        if (q.isComplete() && !sc.formats.empty() && !sc.presentModes.empty())
        {
            physicalDevice_ = d;
            graphicsFamily_ = q.graphics.value();
            break;
        }
    }
    if (physicalDevice_ == VK_NULL_HANDLE)
        throw std::runtime_error("no suitable GPU");
}

void Application::createLogicalDevice()
{
    auto q = findQueueFamilies(physicalDevice_);
    std::set<uint32_t> unique = {q.graphics.value(), q.present.value()};
    float prio = 1.0f;

    std::vector<VkDeviceQueueCreateInfo> qCIs;
    for (uint32_t fam : unique)
    {
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = fam;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        qCIs.push_back(qi);
    }

    std::vector<const char *> devExts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    uint32_t extCount;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> avail(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, avail.data());
    for (auto &e : avail)
        if (strcmp(e.extensionName, "VK_KHR_portability_subset") == 0)
            devExts.push_back("VK_KHR_portability_subset");

    VkPhysicalDeviceFeatures features{};
    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount = static_cast<uint32_t>(qCIs.size());
    ci.pQueueCreateInfos = qCIs.data();
    ci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
    ci.ppEnabledExtensionNames = devExts.data();
    ci.pEnabledFeatures = &features;

    if (vkCreateDevice(physicalDevice_, &ci, nullptr, &device_) != VK_SUCCESS)
        throw std::runtime_error("failed to create device");

    vkGetDeviceQueue(device_, q.graphics.value(), 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, q.present.value(), 0, &presentQueue_);
}

void Application::createSwapchain()
{
    auto sc = querySwapchainSupport(physicalDevice_);
    VkSurfaceFormatKHR fmt = sc.formats[0];
    for (auto &f : sc.formats)
    {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            fmt = f;
        }
    }

    VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;
    for (auto &m : sc.presentModes)
    {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            pm = m;
        }
    }

    VkExtent2D ext;
    if (sc.caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        ext = sc.caps.currentExtent;
    }
    else
    {
        int w, h;
        glfwGetFramebufferSize(window_, &w, &h);
        ext = {
            std::clamp((uint32_t)w, sc.caps.minImageExtent.width, sc.caps.maxImageExtent.width),
            std::clamp((uint32_t)h, sc.caps.minImageExtent.height, sc.caps.maxImageExtent.height)};
    }

    uint32_t imgCount = sc.caps.minImageCount + 1;
    if (sc.caps.maxImageCount > 0)
    {
        imgCount = std::min(imgCount, sc.caps.maxImageCount);
    }

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = surface_;
    ci.minImageCount = imgCount;
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = ext;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    auto q = findQueueFamilies(physicalDevice_);
    uint32_t qfis[] = {q.graphics.value(), q.present.value()};
    if (q.graphics != q.present)
    {
        ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices = qfis;
    }
    else
    {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    ci.preTransform = sc.caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = pm;
    ci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create swapchain");
    }

    vkGetSwapchainImagesKHR(device_, swapchain_, &imgCount, nullptr);
    swapchainImages_.resize(imgCount);
    vkGetSwapchainImagesKHR(device_, swapchain_, &imgCount, swapchainImages_.data());

    swapchainFormat_ = fmt.format;
    swapchainExtent_ = ext;
}

void Application::createImageViews()
{
    swapchainViews_.resize(swapchainImages_.size());
    for (size_t i = 0; i < swapchainImages_.size(); i++)
    {
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = swapchainImages_[i];
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = swapchainFormat_;
        ci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                         VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device_, &ci, nullptr, &swapchainViews_[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create image view");
    }
}

void Application::createRenderPass()
{
    VkAttachmentDescription att{};
    att.format = swapchainFormat_;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &ref;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = 1;
    ci.pAttachments = &att;
    ci.subpassCount = 1;
    ci.pSubpasses = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies = &dep;

    if (vkCreateRenderPass(device_, &ci, nullptr, &renderPass_) != VK_SUCCESS)
        throw std::runtime_error("failed to create render pass");
}

void Application::createFramebuffers()
{
    framebuffers_.resize(swapchainViews_.size());
    for (size_t i = 0; i < swapchainViews_.size(); i++)
    {
        VkFramebufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass = renderPass_;
        ci.attachmentCount = 1;
        ci.pAttachments = &swapchainViews_[i];
        ci.width = swapchainExtent_.width;
        ci.height = swapchainExtent_.height;
        ci.layers = 1;
        if (vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create framebuffer");
    }
}

void Application::createCommandPool()
{
    auto q = findQueueFamilies(physicalDevice_);
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = q.graphics.value();
    if (vkCreateCommandPool(device_, &ci, nullptr, &commandPool_) != VK_SUCCESS)
        throw std::runtime_error("failed to create command pool");
}

void Application::createCommandBuffers()
{
    commandBuffers_.resize(MAX_FRAMES);
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = commandPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = MAX_FRAMES;
    if (vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate command buffers");
}

void Application::createSyncObjects()
{
    imageAvailableSems_.resize(MAX_FRAMES);
    renderFinishedSems_.resize(MAX_FRAMES);
    inFlightFences_.resize(MAX_FRAMES);

    VkSemaphoreCreateInfo semCI{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fenCI{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES; i++)
    {
        if (vkCreateSemaphore(device_, &semCI, nullptr, &imageAvailableSems_[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device_, &semCI, nullptr, &renderFinishedSems_[i]) != VK_SUCCESS ||
            vkCreateFence(device_, &fenCI, nullptr, &inFlightFences_[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create sync objects");
    }
}

// ── ImGui ─────────────────────────────────────────────────────────────────────

void Application::initImGui()
{
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets = 1;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(device_, &ci, nullptr, &imguiPool_) != VK_SUCCESS)
        throw std::runtime_error("failed to create imgui descriptor pool");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window_, true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = instance_;
    initInfo.PhysicalDevice = physicalDevice_;
    initInfo.Device = device_;
    initInfo.QueueFamily = graphicsFamily_;
    initInfo.Queue = graphicsQueue_;
    initInfo.DescriptorPool = imguiPool_;
    initInfo.RenderPass = renderPass_;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = static_cast<uint32_t>(swapchainImages_.size());
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&initInfo);

    ImGui_ImplVulkan_CreateFontsTexture();
}

// ── Context ───────────────────────────────────────────────────────────────────
static std::string getExeDir()
{
    char buf[4096];
    uint32_t size = sizeof(buf);
    _NSGetExecutablePath(buf, &size);
    return std::filesystem::path(buf).parent_path().string();
}

VulkanContext Application::makeContext() const
{
    return {
        instance_,
        physicalDevice_,
        device_,
        graphicsQueue_,
        graphicsFamily_,
        commandPool_,
        renderPass_,
        swapchainExtent_,
        getExeDir(),
    };
}

// ── Main loop ─────────────────────────────────────────────────────────────────
void Application::mainLoop()
{
    while (!glfwWindowShouldClose(window_))
    {
        glfwPollEvents();
        drawFrame();
    }
}

void Application::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
{
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &begin);

    VkClearValue clear = {{{0.1f, 0.1f, 0.1f, 1.0f}}};
    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = renderPass_;
    rp.framebuffer = framebuffers_[imageIndex];
    rp.renderArea = {{0, 0}, swapchainExtent_};
    rp.clearValueCount = 1;
    rp.pClearValues = &clear;

    // 렌더 패스 전: 컴퓨트 디스패치 (컴퓨트 피처만 실행)
    if (activeFeature_ < (int)features_.size())
        features_[activeFeature_]->onCompute(cmd);

    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    RenderContext rc{cmd, imageIndex};
    if (activeFeature_ < (int)features_.size())
        features_[activeFeature_]->onRender(rc);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

void Application::drawFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Feature 선택 패널
    ImGui::Begin("Features");
    for (int i = 0; i < (int)features_.size(); i++)
    {
        bool active = (i == activeFeature_);
        if (active)
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.f));
        if (ImGui::Button(features_[i]->name()))
            activeFeature_ = i;
        if (active)
            ImGui::PopStyleColor();
        if (i + 1 < (int)features_.size())
            ImGui::SameLine();
    }
    ImGui::Text("1~9: switch feature  |  ESC: quit");
    ImGui::End();

    if (activeFeature_ < (int)features_.size())
        features_[activeFeature_]->onImGui();

    ImGui::Render();

    vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);

    uint32_t imageIndex;
    vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                          imageAvailableSems_[currentFrame_], VK_NULL_HANDLE, &imageIndex);

    vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
    recordCommandBuffer(commandBuffers_[currentFrame_], imageIndex);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &imageAvailableSems_[currentFrame_];
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffers_[currentFrame_];
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &renderFinishedSems_[currentFrame_];

    vkQueueSubmit(graphicsQueue_, 1, &submit, inFlightFences_[currentFrame_]);

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &renderFinishedSems_[currentFrame_];
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_;
    present.pImageIndices = &imageIndex;

    vkQueuePresentKHR(presentQueue_, &present);
    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES;
}

// ── Cleanup ───────────────────────────────────────────────────────────────────

void Application::cleanup()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(device_, imguiPool_, nullptr);

    for (int i = 0; i < MAX_FRAMES; i++)
    {
        vkDestroySemaphore(device_, imageAvailableSems_[i], nullptr);
        vkDestroySemaphore(device_, renderFinishedSems_[i], nullptr);
        vkDestroyFence(device_, inFlightFences_[i], nullptr);
    }
    vkDestroyCommandPool(device_, commandPool_, nullptr);
    for (auto fb : framebuffers_)
        vkDestroyFramebuffer(device_, fb, nullptr);
    vkDestroyRenderPass(device_, renderPass_, nullptr);
    for (auto iv : swapchainViews_)
        vkDestroyImageView(device_, iv, nullptr);
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    vkDestroyDevice(device_, nullptr);
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkDestroyInstance(instance_, nullptr);
    glfwDestroyWindow(window_);
    glfwTerminate();
}
