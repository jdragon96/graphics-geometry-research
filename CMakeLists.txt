cmake_minimum_required(VERSION 3.20)
project(Vision3D)

set(CMAKE_CXX_STANDARD 17)

# ── Paths ─────────────────────────────────────────────────────────────────────
set(VULKAN_SDK  "$ENV{HOME}/VulkanSDK/1.4.328.1/macOS")
set(GLFW_DIR    "/opt/homebrew/opt/glfw")
set(EIGEN_DIR   "/opt/homebrew/opt/eigen/include/eigen3")

find_library(VULKAN_LIB vulkan PATHS ${VULKAN_SDK}/lib REQUIRED)
find_library(GLFW_LIB   glfw   PATHS ${GLFW_DIR}/lib   REQUIRED)

# ── ImGui (FetchContent) ──────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.5
)
FetchContent_MakeAvailable(imgui)

set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

# ── Shader compilation ────────────────────────────────────────────────────────
set(GLSLANG    "${VULKAN_SDK}/bin/glslangValidator")
set(SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(SHADER_OUT "${CMAKE_BINARY_DIR}/shaders")

file(MAKE_DIRECTORY ${SHADER_OUT})

set(SHADERS
    ${SHADER_DIR}/triangle.vert
    ${SHADER_DIR}/triangle.frag
    ${SHADER_DIR}/blinnphong.vert
    ${SHADER_DIR}/blinnphong.frag
)

foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV "${SHADER_OUT}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT  ${SPV}
        COMMAND ${GLSLANG} -V ${SHADER} -o ${SPV}
        DEPENDS ${SHADER}
        COMMENT "Compiling ${SHADER_NAME}"
    )
    list(APPEND SPV_FILES ${SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SPV_FILES})

# ── Common include / link helper ──────────────────────────────────────────────
set(COMMON_INCLUDES
    ${VULKAN_SDK}/include
    ${GLFW_DIR}/include
    ${EIGEN_DIR}
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
set(COMMON_LIBS ${VULKAN_LIB} ${GLFW_LIB})
set(COMMON_RPATH "${VULKAN_SDK}/lib;${GLFW_DIR}/lib")

# ── Target 1: vulkan_triangle (main.cpp 보존용) ───────────────────────────────
# add_executable(vulkan_triangle
#     main.cpp
#     SceneObject.cpp
# )
# target_include_directories(vulkan_triangle PRIVATE ${VULKAN_SDK}/include ${GLFW_DIR}/include ${EIGEN_DIR})
# target_link_libraries(vulkan_triangle PRIVATE ${VULKAN_LIB} ${GLFW_LIB})
# set_target_properties(vulkan_triangle PROPERTIES BUILD_RPATH "${COMMON_RPATH}")
# add_dependencies(vulkan_triangle shaders)

# ── Target 2: vision3d (Application + Feature 시스템) ────────────────────────
add_executable(vision3d
    main.cpp
    Application.cpp
    SceneObject.cpp
    Feature/TriangleFeature.cpp
    Feature/BlinnPhongFeature.cpp
    ${IMGUI_SOURCES}
)
target_include_directories(vision3d PRIVATE ${COMMON_INCLUDES})
target_link_libraries(vision3d PRIVATE ${COMMON_LIBS})
set_target_properties(vision3d PROPERTIES BUILD_RPATH "${COMMON_RPATH}")
add_dependencies(vision3d shaders)
