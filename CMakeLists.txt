cmake_minimum_required(VERSION 3.20)
project(Vision3D)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Paths ─────────────────────────────────────────────────────────────────────
set(VULKAN_SDK  "$ENV{HOME}/VulkanSDK/1.4.328.1/macOS")
set(GLFW_DIR    "/opt/homebrew/opt/glfw")
set(EIGEN_DIR   "/opt/homebrew/opt/eigen/include/eigen3")

# ── OpenVDB (brew install openvdb) ────────────────────────────────────────────
set(OPENVDB_ROOT /opt/homebrew/opt/openvdb)
set(TBB_ROOT     /opt/homebrew/opt/tbb)

find_path  (OPENVDB_INCLUDE openvdb/openvdb.h PATHS ${OPENVDB_ROOT}/include REQUIRED)
find_library(OPENVDB_LIB   openvdb           PATHS ${OPENVDB_ROOT}/lib      REQUIRED)
find_library(TBB_LIB       tbb               PATHS ${TBB_ROOT}/lib          REQUIRED)

find_library(VULKAN_LIB  vulkan           PATHS ${VULKAN_SDK}/lib REQUIRED)
find_library(GLFW_LIB    glfw             PATHS ${GLFW_DIR}/lib   REQUIRED)
find_library(SHADERC_LIB shaderc_combined PATHS ${VULKAN_SDK}/lib REQUIRED)

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
    ${SHADER_DIR}/compute_wave.comp
    ${SHADER_DIR}/cube_wave.vert
    ${SHADER_DIR}/cube_wave.frag
    ${SHADER_DIR}/tsdf_voxel.vert
    ${SHADER_DIR}/tsdf_voxel.frag
    # ── VoxelHash shaders ──────────────────────────────────────────────────────
    ${SHADER_DIR}/VoxelHash/VoxelHash_Clear.comp
    ${SHADER_DIR}/VoxelHash/VoxelHash_RenderPoint.vert
    ${SHADER_DIR}/VoxelHash/VoxelHash_RenderPointBySample.vert
    ${SHADER_DIR}/VoxelHash/VoxelHash_RenderTSDFPoint.vert
    ${SHADER_DIR}/VoxelHash/VoxelHash_RenderPoint.frag
    ${SHADER_DIR}/VoxelHash/VoxelHash_RenderVoxel.vert
    ${SHADER_DIR}/VoxelHash/VoxelHash_RenderVoxel.frag
    ${SHADER_DIR}/VoxelHash/VoxelHash_UpdateHistogram.comp
    ${SHADER_DIR}/VoxelHash/VoxelHash_UpdateBucket.comp
    ${SHADER_DIR}/VoxelHash/VoxelHash_ReallocatePoint.comp
    ${SHADER_DIR}/VoxelHash/VoxelHash_UpdateTSDF.comp
    ${SHADER_DIR}/VoxelHash/VoxelHash_UpdateHashTable.comp
    ${SHADER_DIR}/VoxelHash/VoxelHash_CheckOccupiedVoxel.comp
    ${SHADER_DIR}/VoxelHash/VoxelHash_DeleteBox.comp
    ${SHADER_DIR}/vh_clear.comp
    ${SHADER_DIR}/vh_histogram.comp
    ${SHADER_DIR}/vh_prefix_scan.comp
    ${SHADER_DIR}/vh_scatter.comp
    ${SHADER_DIR}/vh_gather.comp
    ${SHADER_DIR}/vh_finalize.comp
    ${SHADER_DIR}/vh_count.comp
    ${SHADER_DIR}/vh_render.vert
    ${SHADER_DIR}/vh_render.frag
    ${SHADER_DIR}/vh_ptrender.vert
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
    ${OPENVDB_INCLUDE}
    ${TBB_ROOT}/include
)
set(COMMON_LIBS ${VULKAN_LIB} ${GLFW_LIB} ${OPENVDB_LIB} ${TBB_LIB} ${SHADERC_LIB})
set(COMMON_RPATH "${VULKAN_SDK}/lib;${GLFW_DIR}/lib;${OPENVDB_ROOT}/lib;${TBB_ROOT}/lib")

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
    # ── Samples ────────────────────────────────────────────────────────────────
    Feature/Samples/TriangleFeature.cpp
    Feature/Samples/BlinnPhongFeature.cpp
    Feature/Samples/ComputeTest.cpp
    # ── Acquisition ────────────────────────────────────────────────────────────
    Feature/Acquisition/SimpleLiDAR.cpp
    # ── Common helpers ─────────────────────────────────────────────────────────
    Feature/Common/Buffer.cpp
    Feature/Common/ComputeShader.cpp
    Feature/Common/RenderingShader.cpp
    # ── VoxelHash ──────────────────────────────────────────────────────────────
    Feature/DataStructure/VoxelHash/BucketVoxelHash.cpp
    Feature/DataStructure/VoxelHash/BucketVoxelHashRender.cpp
    Feature/DataStructure/VoxelHash/VoxelHash.cpp
    Feature/DataStructure/VoxelHash/VoxelHashBuffers.cpp
    Feature/DataStructure/VoxelHash/VoxelHashPipelines.cpp
    Feature/DataStructure/VoxelHash/VoxelHashSort.cpp
    Feature/DataStructure/VoxelHash/VoxelHashDispatch.cpp
    Feature/DataStructure/VoxelHash/VoxelHashSynth.cpp
    Feature/DataStructure/VoxelHash/VoxelHashImGui.cpp
    # ── GaussianSplatting ──────────────────────────────────────────────────────
    Feature/DataStructure/GaussianSplatting/GaussianSplatting.cpp
    # ── SLAM ───────────────────────────────────────────────────────────────────
    Feature/SLAM/FastLIO2/FastLIO2.cpp
    Feature/SLAM/KissICP/KissICP.cpp
    Feature/SLAM/Viewer/Viewer.cpp
    ${IMGUI_SOURCES}
)
target_include_directories(vision3d PRIVATE
    ${COMMON_INCLUDES}
    ${CMAKE_SOURCE_DIR}                        # Application.h, SceneObject.h 등
    ${CMAKE_SOURCE_DIR}/Feature                # "IFeature.h" 직접 참조용 (Samples 등)
    ${CMAKE_SOURCE_DIR}/Feature/DataStructure  # "../IFeature.h" 등 상대경로 해석용
)
target_link_libraries(vision3d PRIVATE ${COMMON_LIBS})
set_target_properties(vision3d PROPERTIES BUILD_RPATH "${COMMON_RPATH}")
add_dependencies(vision3d shaders)

# ── Target 3: vision3d_test (Problem Runner) ──────────────────────────────────
add_executable(vision3d_test
    main_forTest.cpp
    Problem/ApplicationForTest.cpp
    SceneObject.cpp
    ${IMGUI_SOURCES}
)
target_include_directories(vision3d_test PRIVATE ${COMMON_INCLUDES} ${CMAKE_SOURCE_DIR})
target_link_libraries(vision3d_test PRIVATE ${COMMON_LIBS})
set_target_properties(vision3d_test PROPERTIES BUILD_RPATH "${COMMON_RPATH}")
add_dependencies(vision3d_test shaders)
