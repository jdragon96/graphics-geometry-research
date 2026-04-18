# Graphics Research Tools

## 1. 사용 방법

```bash
cmake --build build
./build/Vision3D

cmake --build build --target vision3d_test
./build/vision3d_test
```

## 2. 어플리케이션 구조

- 해당 Vulkan pipeline은 다음과 같이 구성한다.

```bash
  ┌─────────────────────────────────────────────────────────────────┐
  │                           main.cpp                              │
  │  app.addFeature(TriangleFeature)                                │
  │  app.addFeature(BlinnPhongFeature)   →  features_[] 에 등록     │
  │  app.addFeature(ComputeTest)                                    │
  └──────────────────────────┬──────────────────────────────────────┘
                             │ app.run()
                             ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                       Application::run()                        │
  │                                                                 │
  │  initWindow()  →  initVulkan()  →  initImGui()                  │
  │                                                                 │
  │  for (auto& f : features_)                                      │
  │      f->onInit(makeContext())   ◄── VulkanContext 전달           │
  │                                                                 │
  │  mainLoop()                                                     │
  │                                                                 │
  │  vkDeviceWaitIdle()                                             │
  │  for (auto& f : features_)                                      │
  │      f->onCleanup()                                             │
  └──────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                     mainLoop()  (매 프레임)                      │
  │                                                                 │
  │  glfwPollEvents()  →  drawFrame()                               │
  └──────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                        drawFrame()                              │
  │                                                                 │
  │  ① ImGui::NewFrame()                                            │
  │                                                                 │
  │  ② Feature 선택 패널 그리기 (버튼 1~9)                           │
  │       activeFeature_ 변경 가능                                   │
  │                                                                 │
  │  ③ features_[activeFeature_]->onImGui()                         │
  │                                                                 │
  │  ④ ImGui::Render()                                              │
  │                                                                 │
  │  ⑤ vkAcquireNextImageKHR()                                      │
  │                                                                 │
  │  ⑥ recordCommandBuffer()  ──────────────────────────────────┐   │
  │                                                             │   │
  │  ⑦ vkQueueSubmit()  →  vkQueuePresentKHR()                  │   │
  └─────────────────────────────────────────────────────────────┼───┘
                                                                │
                             ┌──────────────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                   recordCommandBuffer()                         │
  │                                                                 │
  │  vkBeginCommandBuffer()                                         │
  │                                                                 │
  │  features_[active]->onCompute(cmd)  ◄── 렌더패스 시작 전        │
  │          │                               (compute shader용)     │
  │          ▼                                                      │
  │  vkCmdBeginRenderPass()                                         │
  │                                                                 │
  │  features_[active]->onRender(RenderContext{cmd, imageIndex})    │
  │                                                                 │
  │  ImGui_ImplVulkan_RenderDrawData()                              │
  │                                                                 │
  │  vkCmdEndRenderPass()  →  vkEndCommandBuffer()                  │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │               키 입력 (GLFW 콜백)                                │
  │                                                                 │
  │  keyCallback()                                                  │
  │    ├─ ESC              →  window close                          │
  │    ├─ 1~9             →  activeFeature_ 변경                    │
  │    └─ 나머지           →  features_[active]->onKey(key,action)  │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │                    IFeature (인터페이스)                         │
  │                                                                 │
  │  onInit(VulkanContext)   파이프라인·버퍼·디스크립터 생성         │
  │  onCompute(cmd)          compute dispatch (optional)            │
  │  onRender(RenderContext) draw call 기록                         │
  │  onImGui()               패널 UI (optional)                     │
  │  onKey(key,action,mods)  키 처리 (optional)                     │
  │  onCleanup()             Vulkan 리소스 해제                      │
  │                                                                 │
  │    ▲              ▲                   ▲                         │
  │    │              │                   │                         │
  │ Triangle    BlinnPhong           ComputeTest                    │
  │ Feature      Feature          (compute_wave)                    │
  └─────────────────────────────────────────────────────────────────┘
```

## 3. Vulkan Graphics Pipeline

```bash
  Graphics Pipeline 구성 (onInit 시점)
  ═══════════════════════════════════════════════════════════════════

    STEP 1. 셰이더 로드
    ┌─────────────────────────────────────────────────────────────┐
    │  .spv 파일 읽기 (basePath + "/shaders/xxx.vert.spv")        │
    │         │                                                   │
    │         ▼                                                   │
    │  vkCreateShaderModule()  →  VkShaderModule (vert / frag)    │
    └─────────────────────────────────────────────────────────────┘
                             │
                             ▼
    STEP 2. ShaderStage 설정
    ┌─────────────────────────────────────────────────────────────┐
    │  VkPipelineShaderStageCreateInfo  stages[2]                 │
    │                                                             │
    │  stages[0]  stage = VERTEX_BIT    module = vertMod          │
    │  stages[1]  stage = FRAGMENT_BIT  module = fragMod          │
    └─────────────────────────────────────────────────────────────┘
                             │
                             ▼
    STEP 3. 고정 함수 스테이트 설정 (Fixed-Function States)
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  VertexInputState        ← Vertex::getBindingDescription()  │
    │  (버텍스 레이아웃)          Vertex::getAttributeDescriptions() │
    │    binding=0, stride=sizeof(Vertex)                         │
    │    location 0: position (vec3)                              │
    │    location 1: color    (vec3)                              │
    │    location 2: normal   (vec3)                              │
    │                                                             │
    │  InputAssemblyState      ← TRIANGLE_LIST                    │
    │  (프리미티브 종류)                                            │
    │                                                             │
    │  ViewportState           ← extent.width / extent.height     │
    │  (뷰포트 + 시저)                                             │
    │                                                             │
    │  RasterizationState      ← FILL / BACK_CULL                 │
    │  (래스터화 방식)             Triangle  : CLOCKWISE            │
    │                            BlinnPhong : COUNTER_CLOCKWISE    │
    │                                                             │
    │  MultisampleState        ← SAMPLE_COUNT_1 (MSAA 없음)       │
    │                                                             │
    │  ColorBlendState         ← 블렌딩 없음, RGBA 전체 write       │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
                             │
                             ▼
    STEP 4. PipelineLayout 생성
    ┌─────────────────────────────────────────────────────────────┐
    │  vkCreatePipelineLayout()  →  VkPipelineLayout              │
    │                                                             │
    │  TriangleFeature          BlinnPhongFeature                 │
    │  ┌──────────────────┐     ┌──────────────────────────────┐  │
    │  │ PushConstant     │     │ DescriptorSetLayout          │  │
    │  │ VERTEX_BIT       │     │ binding=0                    │  │
    │  │ size=float(scale)│     │ UNIFORM_BUFFER               │  │
    │  └──────────────────┘     │ VERTEX | FRAGMENT stage      │  │
    │                           └──────────────────────────────┘  │
    │                                    │                        │
    │                           DescriptorPool → DescriptorSet    │
    │                           vkUpdateDescriptorSets()          │
    │                           (UBO 버퍼 바인딩)                   │
    └─────────────────────────────────────────────────────────────┘
                             │
                             ▼
    STEP 5. Pipeline 생성
    ┌─────────────────────────────────────────────────────────────┐
    │  VkGraphicsPipelineCreateInfo                               │
    │  ┌────────────────┬────────────────────────────────────┐    │
    │  │ pStages        │ ShaderStage[2] (vert + frag)        │    │
    │  │ pVertexInput   │ VertexInputState                    │    │
    │  │ pInputAssembly │ InputAssemblyState                  │    │
    │  │ pViewportState │ ViewportState                       │    │
    │  │ pRasterization │ RasterizationState                  │    │
    │  │ pMultisample   │ MultisampleState                    │    │
    │  │ pColorBlend    │ ColorBlendState                     │    │
    │  │ layout         │ PipelineLayout                      │    │
    │  │ renderPass     │ ctx_.renderPass  ◄── App에서 전달    │    │
    │  └────────────────┴────────────────────────────────────┘    │
    │           │                                                 │
    │           ▼                                                 │
    │  vkCreateGraphicsPipelines()  →  VkPipeline                 │
    │                                                             │
    │  vkDestroyShaderModule()  ← 파이프라인 생성 후 즉시 해제      │
    └─────────────────────────────────────────────────────────────┘
                             │
                             ▼
    STEP 6. onRender()에서 사용
    ┌─────────────────────────────────────────────────────────────┐
    │  vkCmdBindPipeline(cmd, GRAPHICS, pipeline_)                │
    │  vkCmdBindDescriptorSets(...)   ← BlinnPhong만 (UBO)        │
    │  vkCmdPushConstants(...)        ← Triangle만 (scale)        │
    │  vkCmdBindVertexBuffers(...)                                │
    │  vkCmdBindIndexBuffer(...)      ← BlinnPhong만 (구체 mesh)   │
    │  vkCmdDraw() / vkCmdDrawIndexed()                           │
    └─────────────────────────────────────────────────────────────┘
```

# 2. Vulkan Workflow

## 2.1. Buffer 생성 프로세스

```bash
  Vulkan 버퍼 생성 일반 프로세스
  ═══════════════════════════════════════════════════════════════════

    공통 생성 흐름 (createBuffer 헬퍼가 담당)
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │  STEP 1. vkCreateBuffer()                                   │
    │  ┌──────────────────────────────────────────────┐           │
    │  │ VkBufferCreateInfo                           │           │
    │  │   size        = 버퍼 크기 (bytes)             │           │
    │  │   usage       = 용도 플래그 (아래 표 참고)     │           │
    │  │   sharingMode = EXCLUSIVE (단일 큐 패밀리)    │           │
    │  └──────────────────────────────────────────────┘           │
    │           │                                                 │
    │           ▼                                                 │
    │  STEP 2. vkGetBufferMemoryRequirements()                    │
    │          GPU가 실제로 요구하는 메모리 크기 / 정렬 조회        │
    │          → req.size, req.alignment, req.memoryTypeBits      │
    │                                                             │
    │           │                                                 │
    │           ▼                                                 │
    │  STEP 3. findMemoryType()                                   │
    │  ┌──────────────────────────────────────────────┐           │
    │  │ vkGetPhysicalDeviceMemoryProperties()        │           │
    │  │   req.memoryTypeBits  &  원하는 propertyFlags  │           │
    │  │   → 조건 맞는 메모리 타입 인덱스 반환                │           │
    │  └──────────────────────────────────────────────┘           │
    │                                                             │
    │           │                                                 │
    │           ▼                                                 │
    │  STEP 4. vkAllocateMemory()                                 │
    │  ┌──────────────────────────────────────────────┐           │
    │  │ VkMemoryAllocateInfo                         │           │
    │  │   allocationSize  = req.size                 │           │
    │  │   memoryTypeIndex = findMemoryType() 결과     │           │
    │  └──────────────────────────────────────────────┘           │
    │           → VkDeviceMemory 할당                              │
    │                                                             │
    │           │                                                 │
    │           ▼                                                 │
    │  STEP 5. vkBindBufferMemory()                               │
    │          VkBuffer ↔ VkDeviceMemory 연결                      │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    버퍼 종류별 usage / memory 플래그 비교
    ┌──────────────┬───────────────────────────────┬──────────────────────────────────────┐
    │  버퍼 종류    │  usage 플래그                 │  memory property 플래그              │
    ├──────────────┼───────────────────────────────┼──────────────────────────────────────┤
    │ Vertex       │ VERTEX_BUFFER_BIT             │ HOST_VISIBLE | HOST_COHERENT         │
    │ (큐브 mesh)  │                               │ (CPU에서 직접 쓰기 가능)              │
    ├──────────────┼───────────────────────────────┼──────────────────────────────────────┤
    │ Index        │ INDEX_BUFFER_BIT              │ HOST_VISIBLE | HOST_COHERENT         │
    │ (큐브 mesh)  │                               │                                      │
    ├──────────────┼───────────────────────────────┼──────────────────────────────────────┤
    │ UBO          │ UNIFORM_BUFFER_BIT            │ HOST_VISIBLE | HOST_COHERENT         │
    │ (매 프레임   │                               │ (매 프레임 CPU→GPU 업데이트)          │
    │  갱신)       │                               │ + vkMapMemory로 영구 매핑 유지        │
    ├──────────────┼───────────────────────────────┼──────────────────────────────────────┤
    │ SSBO         │ STORAGE_BUFFER_BIT            │ DEVICE_LOCAL                         │
    │ (compute     │ | VERTEX_BUFFER_BIT           │ (GPU 전용 고속 메모리)               │
    │  출력 겸     │                               │ CPU 접근 불가, GPU만 읽고 씀         │
    │  instance    │                               │                                      │
    │  버텍스)     │                               │                                      │
    └──────────────┴───────────────────────────────┴──────────────────────────────────────┘

    Vertex / Index 버퍼 데이터 업로드 (초기 1회)
    ┌─────────────────────────────────────────────────────────────┐
    │  vkMapMemory()    → CPU 가상 주소(void*) 획득               │
    │  memcpy()         → CPU 메모리에서 버퍼로 복사               │
    │  vkUnmapMemory()  → 매핑 해제                               │
    └─────────────────────────────────────────────────────────────┘

    UBO 업로드 (매 프레임)
    ┌─────────────────────────────────────────────────────────────┐
    │  vkMapMemory()  ← onInit 때 한 번만 호출, uboMapped_ 보관   │
    │                                                             │
    │  매 프레임 updateUBO() :                                    │
    │    memcpy(uboMapped_, &ubo, sizeof(ubo))  ← 언매핑 없이     │
    │                                             직접 덮어씀     │
    │  HOST_COHERENT 플래그 덕분에 flush 없이 GPU에 즉시 반영      │
    └─────────────────────────────────────────────────────────────┘

    SSBO 데이터 흐름 (매 프레임)
    ┌─────────────────────────────────────────────────────────────┐
    │  Compute Shader  →  SSBO 쓰기 (Particle 위치/색상 갱신)     │
    │       │                                                     │
    │       │  vkCmdPipelineBarrier()                             │
    │       │  SHADER_WRITE → VERTEX_ATTRIBUTE_READ              │
    │       ▼                                                     │
    │  Graphics Shader →  SSBO를 binding 1 (instance 버텍스)      │
    │                      로 읽기                                 │
    │                                                             │
    │  ※ CPU 개입 없음 — GPU 내부에서 compute → graphics 전달      │
    └─────────────────────────────────────────────────────────────┘

    메모리 타입 선택 기준
    ┌──────────────────────────┬──────────────────────────────────┐
    │  HOST_VISIBLE            │  CPU에서 vkMapMemory로 접근 가능  │
    │  HOST_COHERENT           │  flush 없이 GPU에 자동 반영       │
    │  DEVICE_LOCAL            │  GPU 전용, 최고 속도              │
    │  HOST_VISIBLE+DEVICE_LOCAL│ 일부 GPU(통합 메모리)에서 가능   │
    └──────────────────────────┴──────────────────────────────────┘

    onCleanup 해제 순서 (생성 역순)
    ┌─────────────────────────────────────────────────────────────┐
    │  vkDestroyBuffer()   → 버퍼 핸들 해제                       │
    │  vkFreeMemory()      → 메모리 해제                          │
    │                                                             │
    │  UBO는 vkUnmapMemory() 먼저 호출 후 해제                    │
    └─────────────────────────────────────────────────────────────┘

  핵심 포인트:
  - VkBuffer와 VkDeviceMemory는 분리 — 버퍼는 핸들일 뿐, 실제 메모리는 따로 할당 후 vkBindBufferMemory로 연결
  - SSBO만 DEVICE_LOCAL — CPU가 쓸 필요 없고 compute shader가 직접 쓰기 때문에 GPU 전용 고속 메모리 사용
  - UBO는 영구 매핑 — 매 프레임 map/unmap 하면 오버헤드가 생기므로 포인터를 멤버(uboMapped_)에 보관해두고 memcpy만 반복
```
