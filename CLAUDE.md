# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
# Configure (first time or after CMakeLists.txt changes)
cmake -S . -B build

# Build (compiles shaders + binary)
cmake --build build

# Run
./build/vision3d
```

Shaders are compiled automatically as part of the build via `glslangValidator`. Compiled `.spv` files land in `build/shaders/`.

## Dependencies (macOS / Apple Silicon)

| Library | Path |
|---------|------|
| Vulkan SDK | `~/VulkanSDK/1.4.328.1/macOS` |
| GLFW | `/opt/homebrew/opt/glfw` |
| Eigen3 | `/opt/homebrew/opt/eigen` |
| ImGui | fetched automatically via FetchContent (v1.91.5) |

## Architecture

The project is a Vulkan renderer built around a **Feature plugin system**.

### Core classes

- **`Application`** — owns the Vulkan core (instance, device, swapchain, render pass, command buffers, sync objects) and the ImGui integration. Exposes `addFeature()` to register features before calling `run()`.
- **`IFeature`** (`Feature/IFeature.h`) — interface all features implement. Lifecycle callbacks:
  - `onInit(VulkanContext)` — allocate pipelines, buffers, descriptors
  - `onCompute(VkCommandBuffer)` — optional; called before the render pass for compute work
  - `onRender(RenderContext)` — record draw commands inside the active render pass
  - `onImGui()` — draw per-feature ImGui panels
  - `onKey(key, action, mods)` — GLFW key events forwarded from the app
  - `onCleanup()` — destroy all Vulkan resources
- **`VulkanContext`** (`Feature/IFeature.h`) — snapshot of core Vulkan handles passed to features at init. Includes `basePath` (directory of the executable, used to locate `.spv` shaders).
- **`SceneObject`** — thin wrapper around a `std::vector<Vertex>`. `Vertex` holds `position`, `color`, `normal` as `Eigen::Vector3f`.

### Adding a new Feature

1. Create `Feature/MyFeature.h` / `.cpp` inheriting from `IFeature`.
2. Implement all pure-virtual methods (`onInit`, `onRender`, `onCleanup`) and any optional ones needed.
3. Add the `.cpp` to the `vision3d` target in `CMakeLists.txt`.
4. Register with `app.addFeature(std::make_unique<MyFeature>())` in `main.cpp`.
5. If new shaders are needed, add `.vert`/`.frag`/`.comp` files to `shaders/` and list them in the `SHADERS` variable in `CMakeLists.txt`.

### Existing features

| Feature class | Shader files | Description |
|---------------|-------------|-------------|
| `TriangleFeature` | `triangle.vert/frag` | Basic colored triangle |
| `BlinnPhongFeature` | `blinnphong.vert/frag` | Blinn-Phong lit mesh |
| `ComputeTest` | `compute_wave.comp`, `cube_wave.vert/frag` | GPU compute → instanced cube wave grid |

### Frame loop (Application)

`drawFrame()` → acquire swapchain image → `recordCommandBuffer()`:
1. Calls `onCompute()` on the active feature (before render pass, with a pipeline barrier)
2. Begins render pass
3. Calls `onRender()` on the active feature
4. Calls `onImGui()` on all features (renders feature-selector + active feature panel)
5. Ends render pass → submit → present

Only one feature is active at a time (`activeFeature_` index), switchable via the ImGui sidebar.
