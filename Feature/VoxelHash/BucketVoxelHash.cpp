#include "BucketVoxelHash.h"

#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
//  Gather shader — GPU sort 이후 sorted position 기반 P2G TSDF 누산
//
//  binding  0: HT            (raw int[], read-write)
//  binding  5: sorted pts    (vec4[], readonly) ← scatter 출력
//  binding  8: histogram     (uint[], readonly) ← scatter 이후 = bucket별 점 수
//  binding 10: cell start    (uint[], readonly) ← prefix scan 출력
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char *kVHB_GatherComp = R"GLSL(
#version 450
layout(local_size_x = 32) in;

layout(push_constant) uniform PC {
    float sensorX, sensorY, sensorZ;
    float voxelSize;
    uint  _numPts;
    uint  _numBuckets;
    float truncation;
    float _maxW;
} pc;

#define STRIDE      12
#define F_KX         0
#define F_KY         1
#define F_KZ         2
#define F_ACCT       7
#define F_ACCW       8
#define F_ACCC       9
#define F_ACCN      10
#define EMPTY        0x7FFFFFFF
#define BUCKET_SIZE  4

layout(set=0, binding= 0)          buffer HT      { int  raw[]; } ht;
layout(set=0, binding= 5) readonly buffer SortPts  { vec4 pts[]; };
layout(set=0, binding= 8) readonly buffer Hist     { uint h[];   } hist;
layout(set=0, binding=10) readonly buffer CellStart{ uint d[];   } cellStart;

shared int   sKX[BUCKET_SIZE];
shared int   sKY[BUCKET_SIZE];
shared int   sKZ[BUCKET_SIZE];
shared float sScrT[32];
shared float sScrW[32];
shared float sFT[BUCKET_SIZE];
shared float sFW[BUCKET_SIZE];
shared int   sFC[BUCKET_SIZE];
shared int   sFN[BUCKET_SIZE];

uint packNorm(vec3 n) {
    float l = abs(n.x)+abs(n.y)+abs(n.z);
    n /= max(l, 1e-6);
    if (n.z < 0.0) {
        vec2 s = vec2(n.x>=0.0?1.0:-1.0, n.y>=0.0?1.0:-1.0);
        n.xy = (1.0-abs(n.yx))*s;
    }
    return (uint(clamp(n.x*.5+.5,0.,1.)*65535.)&0xFFFFu)
          |((uint(clamp(n.y*.5+.5,0.,1.)*65535.)&0xFFFFu)<<16u);
}
uint posCol(ivec3 k) {
    uint h = uint(k.x)*73856093u^uint(k.y)*19349663u^uint(k.z)*83492791u;
    float t = float(h&0xFFFFu)/65535.;
    return uint(clamp(abs(fract(t      )*6.-3.)-1.,0.,1.)*255.)
          |(uint(clamp(abs(fract(t+.333)*6.-3.)-1.,0.,1.)*255.)<<8u)
          |(uint(clamp(abs(fract(t+.667)*6.-3.)-1.,0.,1.)*255.)<<16u)
          |0xFF000000u;
}

void main() {
    uint bucket = gl_WorkGroupID.x;
    uint tid    = gl_LocalInvocationID.x;

    // Phase 1: 기존 슬롯 키 로드
    if (tid < uint(BUCKET_SIZE)) {
        int gb = int((bucket * uint(BUCKET_SIZE) + tid) * uint(STRIDE));
        sKX[tid] = ht.raw[gb + F_KX];
        sKY[tid] = ht.raw[gb + F_KY];
        sKZ[tid] = ht.raw[gb + F_KZ];
        sFT[tid] = 0.0; sFW[tid] = 0.0; sFC[tid] = 0; sFN[tid] = 0;
    }
    barrier();

    uint pStart = cellStart.d[bucket];
    uint pEnd   = pStart + hist.h[bucket];  // scatter 이후 hist[b] = bucket b의 점 수
    if (pStart >= pEnd) return;

    // Phase 2: thread 0 이 슬롯 할당 (workgroup 단독 소유 → 원자 불필요)
    if (tid == 0u) {
        for (uint i = pStart; i < pEnd; i++) {
            ivec3 key = ivec3(floor(pts[i].xyz / pc.voxelSize));
            bool found = false;
            for (int s = 0; s < BUCKET_SIZE; s++) {
                if (sKX[s]==key.x && sKY[s]==key.y && sKZ[s]==key.z) { found=true; break; }
            }
            if (!found) {
                for (int s = 0; s < BUCKET_SIZE; s++) {
                    if (sKX[s] == EMPTY) {
                        int gb = int((bucket*uint(BUCKET_SIZE)+uint(s))*uint(STRIDE));
                        ht.raw[gb+F_KX] = key.x;
                        ht.raw[gb+F_KY] = key.y;
                        ht.raw[gb+F_KZ] = key.z;
                        sKX[s]=key.x; sKY[s]=key.y; sKZ[s]=key.z;
                        break;
                    }
                }
            }
        }
    }
    barrier();

    // Phase 3: 32 thread 협력 gather
    vec3  sensor = vec3(pc.sensorX, pc.sensorY, pc.sensorZ);
    float myT[4]; float myW[4]; int myC[4]; int myN[4];
    myT[0]=myT[1]=myT[2]=myT[3]=0.0;
    myW[0]=myW[1]=myW[2]=myW[3]=0.0;
    myC[0]=myC[1]=myC[2]=myC[3]=0;
    myN[0]=myN[1]=myN[2]=myN[3]=0;

    for (uint i = pStart + tid; i < pEnd; i += 32u) {
        vec3  pt  = pts[i].xyz;
        ivec3 key = ivec3(floor(pt / pc.voxelSize));
        int s = -1;
        if      (sKX[0]==key.x&&sKY[0]==key.y&&sKZ[0]==key.z) s=0;
        else if (sKX[1]==key.x&&sKY[1]==key.y&&sKZ[1]==key.z) s=1;
        else if (sKX[2]==key.x&&sKY[2]==key.y&&sKZ[2]==key.z) s=2;
        else if (sKX[3]==key.x&&sKY[3]==key.y&&sKZ[3]==key.z) s=3;
        if (s < 0) continue;

        vec3  sn   = normalize(sensor - pt);
        vec3  vc   = (vec3(key)+0.5)*pc.voxelSize;
        float tsdf = clamp(dot(vc-pt, sn)/pc.truncation, -1.0, 1.0);
        float w    = max(dot(sn, normalize(sensor-vc)), 0.05);
        myT[s] += tsdf*w; myW[s] += w;
        myC[s]  = int(posCol(key)); myN[s] = int(packNorm(sn));
    }

    // Phase 4: 슬롯별 tree reduction
#define REDUCE_SLOT(S) \
    sScrT[tid]=myT[S]; sScrW[tid]=myW[S]; barrier(); \
    if(tid<16u){sScrT[tid]+=sScrT[tid+16u];sScrW[tid]+=sScrW[tid+16u];} barrier(); \
    if(tid< 8u){sScrT[tid]+=sScrT[tid+ 8u];sScrW[tid]+=sScrW[tid+ 8u];} barrier(); \
    if(tid< 4u){sScrT[tid]+=sScrT[tid+ 4u];sScrW[tid]+=sScrW[tid+ 4u];} barrier(); \
    if(tid< 2u){sScrT[tid]+=sScrT[tid+ 2u];sScrW[tid]+=sScrW[tid+ 2u];} barrier(); \
    if(tid< 1u){sScrT[0]+=sScrT[1];sScrW[0]+=sScrW[1];} barrier(); \
    if(tid==0u){sFT[S]=sScrT[0];sFW[S]=sScrW[0];}

    REDUCE_SLOT(0)
    REDUCE_SLOT(1)
    REDUCE_SLOT(2)
    REDUCE_SLOT(3)

    if (tid == 0u) {
        for (int s = 0; s < BUCKET_SIZE; s++) {
            if (myC[s] != 0) { sFC[s]=myC[s]; sFN[s]=myN[s]; }
        }
    }
    barrier();

    // Phase 5: 원자 없는 write-back (thread t → slot t)
    if (tid < uint(BUCKET_SIZE) && sFW[tid] > 0.0 && sKX[tid] != EMPTY) {
        int gb = int((bucket*uint(BUCKET_SIZE)+tid)*uint(STRIDE));
        ht.raw[gb + F_ACCT] = floatBitsToInt(sFT[tid]);
        ht.raw[gb + F_ACCW] = floatBitsToInt(sFW[tid]);
        ht.raw[gb + F_ACCC] = sFC[tid];
        ht.raw[gb + F_ACCN] = sFN[tid];
    }
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
//  Finalize shader
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char *kVHB_FinalizeComp = R"GLSL(
#version 450
layout(local_size_x = 64) in;
layout(push_constant) uniform PC {
    uint  totalEntries;
    uint  currentFrame;
    float _p0, _p1;
} pc;

#define STRIDE  12
#define F_KX     0
#define F_TSDF   3
#define F_W      4
#define F_COL    5
#define F_NORM   6
#define F_ACCT   7
#define F_ACCW   8
#define F_ACCC   9
#define F_ACCN  10
#define F_FTAG  11
#define EMPTY   0x7FFFFFFF

layout(set=0, binding=0) buffer HT { int raw[]; } ht;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.totalEntries) return;

    int b = int(i * uint(STRIDE));
    if (ht.raw[b + F_KX] == EMPTY) return;

    float accW = intBitsToFloat(ht.raw[b + F_ACCW]);
    if (accW <= 0.0) return;

    float accT = intBitsToFloat(ht.raw[b + F_ACCT]);
    float oldW = intBitsToFloat(ht.raw[b + F_W]);
    float oldT = intBitsToFloat(ht.raw[b + F_TSDF]);

    float newW = min(oldW + accW, 50.0);
    float newT = clamp((oldT * oldW + accT) / (oldW + accW), -1.0, 1.0);

    ht.raw[b + F_TSDF] = floatBitsToInt(newT);
    ht.raw[b + F_W]    = floatBitsToInt(newW);
    ht.raw[b + F_COL]  = ht.raw[b + F_ACCC];
    ht.raw[b + F_NORM] = ht.raw[b + F_ACCN];
    ht.raw[b + F_FTAG] = int(pc.currentFrame);

    ht.raw[b + F_ACCT] = 0;
    ht.raw[b + F_ACCW] = 0;
    ht.raw[b + F_ACCC] = 0;
    ht.raw[b + F_ACCN] = 0;
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
//  Count shader
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char *kVHB_CountComp = R"GLSL(
#version 450
layout(local_size_x = 64) in;
layout(push_constant) uniform PC { uint total; uint _f; float _0; float _1; } pc;

#define STRIDE 12
#define EMPTY  0x7FFFFFFF

layout(set=0, binding=0) readonly buffer HT  { int  raw[]; } ht;
layout(set=0, binding=1)          buffer Ctr { uint n;     } ctr;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    if (ht.raw[int(i * uint(STRIDE))] != EMPTY)
        atomicAdd(ctr.n, 1u);
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
//  Initialize
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::Initialize(VulkanContext &context)
{
    CreateBuffers(context);
    CreateClearShader(context);
    CreateHistogramShader(context);
    CreatePrefixScanShader(context);
    CreateScatterShader(context);
    CreateGatherShader(context);
    CreateFinalizeShader(context);
    CreateCountShader(context);
    Clear(context);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Clear / ResetHashTable
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::ResetHashTable(VulkanContext &context)
{
    struct ClearPC
    {
        uint32_t numberOfEntries, p0, p1, p2;
    };

    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = context.commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(context.device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    kClearShader.Bind(cmd);
    kClearShader.Push(cmd, ClearPC{VH_TOTAL_ENTRIES, 0, 0, 0});
    kClearShader.Dispatch(cmd, (VH_TOTAL_ENTRIES + 63u) / 64u);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(context.graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(context.graphicsQueue);
    vkFreeCommandBuffers(context.device, context.commandPool, 1, &cmd);
}

void BucketVoxelHash::Clear(VulkanContext &context)
{
    ResetHashTable(context);
}

// ─────────────────────────────────────────────────────────────────────────────
//  BufBarrier helper
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::BufBarrier(VkCommandBuffer cmd, VkBuffer buf,
                                 VkAccessFlags src, VkAccessFlags dst,
                                 VkPipelineStageFlags srcS, VkPipelineStageFlags dstS)
{
    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = buf;
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 1, &b, 0, nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Integrate — GPU sort(histogram → prefix_scan → scatter) + gather → finalize → count
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::Integrate(VulkanContext &context,
                                const VH_InputPoint *pts, uint32_t count,
                                float sensorX, float sensorY, float sensorZ)
{
    if (count == 0)
        return;
    count = std::min(count, VH_BATCH_SIZE);

    // ── 1. CPU 업로드: unsorted 포인트 → host-visible 버퍼 ──────────────────
    UpdatePosition(context, pts, count);
    UpdateNormal(context, pts, count);
    UpdateColor(context, pts, count);

    // ── 2. One-shot command buffer ────────────────────────────────────────────
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = context.commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(context.device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    const VH_SortPC sortPC{count, VH_NUM_BUCKETS, voxelSize_, 0.f};

    // ── Sort Pass 1: Histogram ────────────────────────────────────────────────
    vkCmdFillBuffer(cmd, kHistorgramBuffer.GetBuffer(), 0, VK_WHOLE_SIZE, 0);
    BufBarrier(cmd, kHistorgramBuffer.GetBuffer(),
               VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    kHistogramShader.Bind(cmd);
    kHistogramShader.Push(cmd, sortPC);
    kHistogramShader.Dispatch(cmd, (count + 63u) / 64u);

    BufBarrier(cmd, kHistorgramBuffer.GetBuffer(),
               VK_ACCESS_SHADER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // ── Sort Pass 2: Prefix Scan (3 passes) ───────────────────────────────────
    kPrefixScanShader.Bind(cmd);

    kPrefixScanShader.Push(cmd, VH_ScanPC{VH_NUM_BUCKETS, 0u, 0u, 0u});
    kPrefixScanShader.Dispatch(cmd, VH_SCAN_BLOCKS);
    BufBarrier(cmd, kCellStartBuffer.GetBuffer(),
               VK_ACCESS_SHADER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    BufBarrier(cmd, kBlockSummationBuffer.GetBuffer(),
               VK_ACCESS_SHADER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    kPrefixScanShader.Push(cmd, VH_ScanPC{VH_NUM_BUCKETS, 1u, 0u, 0u});
    kPrefixScanShader.Dispatch(cmd, 1u);
    BufBarrier(cmd, kBlockSummationBuffer.GetBuffer(),
               VK_ACCESS_SHADER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    kPrefixScanShader.Push(cmd, VH_ScanPC{VH_NUM_BUCKETS, 2u, 0u, 0u});
    kPrefixScanShader.Dispatch(cmd, VH_SCAN_BLOCKS);
    BufBarrier(cmd, kCellStartBuffer.GetBuffer(),
               VK_ACCESS_SHADER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // ── Sort Pass 3: Scatter (histogram 재초기화 후 offset 카운터로 재사용) ──
    vkCmdFillBuffer(cmd, kHistorgramBuffer.GetBuffer(), 0, VK_WHOLE_SIZE, 0);
    BufBarrier(cmd, kHistorgramBuffer.GetBuffer(),
               VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    kScatterShader.Bind(cmd);
    kScatterShader.Push(cmd, sortPC);
    kScatterShader.Dispatch(cmd, (count + 63u) / 64u);

    BufBarrier(cmd, kSortedPositionBuffer.GetBuffer(),
               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    BufBarrier(cmd, kHistorgramBuffer.GetBuffer(),
               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // ── Integrate ①: Gather — P2G TSDF 누산 ──────────────────────────────────
    kGatherShader.Bind(cmd);
    kGatherShader.Push(cmd, VH_GatherPC{
                                sensorX, sensorY, sensorZ,
                                voxelSize_, count, VH_NUM_BUCKETS,
                                truncation_, maxWeight_});
    kGatherShader.Dispatch(cmd, VH_NUM_BUCKETS);

    BufBarrier(cmd, kHashTableBuffer.GetBuffer(),
               VK_ACCESS_SHADER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // ── Integrate ②: Finalize — acc → tsdf/weight 커밋 ───────────────────────
    ++frameIndex_;
    kFinalizeShader.Bind(cmd);
    kFinalizeShader.Push(cmd, VH_FinalizePC{VH_TOTAL_ENTRIES, frameIndex_, 0.f, 0.f});
    kFinalizeShader.Dispatch(cmd, (VH_TOTAL_ENTRIES + 63u) / 64u);

    BufBarrier(cmd, kHashTableBuffer.GetBuffer(),
               VK_ACCESS_SHADER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // ── Integrate ③: Count ────────────────────────────────────────────────────
    vkCmdFillBuffer(cmd, kPatchCountBuffer.GetBuffer(), 0, sizeof(uint32_t), 0);
    BufBarrier(cmd, kPatchCountBuffer.GetBuffer(),
               VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    kCountShader.Bind(cmd);
    kCountShader.Push(cmd, VH_FinalizePC{VH_TOTAL_ENTRIES, frameIndex_, 0.f, 0.f});
    kCountShader.Dispatch(cmd, (VH_TOTAL_ENTRIES + 63u) / 64u);

    BufBarrier(cmd, kPatchCountBuffer.GetBuffer(),
               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(context.graphicsQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(context.graphicsQueue);
    vkFreeCommandBuffers(context.device, context.commandPool, 1, &cmd);
}

// ─────────────────────────────────────────────────────────────────────────────
//  CPU Upload helpers
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::UpdatePosition(VulkanContext &, const VH_InputPoint *pts, uint32_t count)
{
    auto m = kPositionBuffer.Access<float>(sizeof(float) * 4 * count);
    for (uint32_t i = 0; i < count; i++)
    {
        m.get()[i * 4 + 0] = pts[i].px;
        m.get()[i * 4 + 1] = pts[i].py;
        m.get()[i * 4 + 2] = pts[i].pz;
        m.get()[i * 4 + 3] = 1.f;
    }
}

void BucketVoxelHash::UpdateNormal(VulkanContext &, const VH_InputPoint *pts, uint32_t count)
{
    auto m = kNormalBuffer.Access<float>(sizeof(float) * 4 * count);
    for (uint32_t i = 0; i < count; i++)
    {
        m.get()[i * 4 + 0] = pts[i].nx;
        m.get()[i * 4 + 1] = pts[i].ny;
        m.get()[i * 4 + 2] = pts[i].nz;
        m.get()[i * 4 + 3] = 0.f;
    }
}

void BucketVoxelHash::UpdateColor(VulkanContext &, const VH_InputPoint *pts, uint32_t count)
{
    auto m = kColorBuffer.Access<uint32_t>(sizeof(uint32_t) * count);
    for (uint32_t i = 0; i < count; i++)
        m.get()[i] = pts[i].col;
}

// ─────────────────────────────────────────────────────────────────────────────
//  CreateClearShader
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::CreateClearShader(VulkanContext &context)
{
    kClearShader.Initialize(
        context.device,
        context.basePath + "/shaders/VoxelHash_Clear.comp.spv",
        {{0}},
        16);
    kClearShader.BindBuffer(0, kHashTableBuffer.GetBuffer());
}

// ─────────────────────────────────────────────────────────────────────────────
//  CreateHistogramShader — vh_histogram.comp.spv 로드
//  binding 1: unsorted positions, binding 8: histogram output
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::CreateHistogramShader(VulkanContext &context)
{
    kHistogramShader.Initialize(
        context.device,
        context.basePath + "/shaders/vh_histogram.comp.spv",
        {{1}, {8}},
        sizeof(VH_SortPC));
    kHistogramShader.BindBuffer(1, kPositionBuffer.GetBuffer());
    kHistogramShader.BindBuffer(8, kHistorgramBuffer.GetBuffer());
}

// ─────────────────────────────────────────────────────────────────────────────
//  CreatePrefixScanShader — vh_prefix_scan.comp.spv 로드
//  binding 8: hist, binding 9: blockSums, binding 10: cellStart (output)
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::CreatePrefixScanShader(VulkanContext &context)
{
    kPrefixScanShader.Initialize(
        context.device,
        context.basePath + "/shaders/vh_prefix_scan.comp.spv",
        {{8}, {9}, {10}},
        sizeof(VH_ScanPC));
    kPrefixScanShader.BindBuffer(8, kHistorgramBuffer.GetBuffer());
    kPrefixScanShader.BindBuffer(9, kBlockSummationBuffer.GetBuffer());
    kPrefixScanShader.BindBuffer(10, kCellStartBuffer.GetBuffer());
}

// ─────────────────────────────────────────────────────────────────────────────
//  CreateScatterShader — vh_scatter.comp.spv 로드
//  binding 1,2,3: unsorted, binding 5,6,7: sorted output
//  binding 8: offset counters (hist cleared), binding 10: cellStart
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::CreateScatterShader(VulkanContext &context)
{
    kScatterShader.Initialize(
        context.device,
        context.basePath + "/shaders/vh_scatter.comp.spv",
        {{1}, {2}, {3}, {5}, {6}, {7}, {8}, {10}},
        sizeof(VH_SortPC));
    kScatterShader.BindBuffer(1, kPositionBuffer.GetBuffer());
    kScatterShader.BindBuffer(2, kNormalBuffer.GetBuffer());
    kScatterShader.BindBuffer(3, kColorBuffer.GetBuffer());
    kScatterShader.BindBuffer(5, kSortedPositionBuffer.GetBuffer());
    kScatterShader.BindBuffer(6, kSortedNormalBuffer.GetBuffer());
    kScatterShader.BindBuffer(7, kSortedColorBuffer.GetBuffer());
    kScatterShader.BindBuffer(8, kHistorgramBuffer.GetBuffer());
    kScatterShader.BindBuffer(10, kCellStartBuffer.GetBuffer());
}

// ─────────────────────────────────────────────────────────────────────────────
//  CreateGatherShader — inline GLSL (GPU sorted 버퍼 기반)
//  binding 0: HT, 5: sorted pts, 8: hist(count per bucket), 10: cellStart
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::CreateGatherShader(VulkanContext &context)
{
    kGatherShader.InitializeGLSL(
        context.device,
        kVHB_GatherComp,
        shaderc_compute_shader,
        {{0}, {5}, {8}, {10}},
        sizeof(VH_GatherPC));
    kGatherShader.BindBuffer(0, kHashTableBuffer.GetBuffer());
    kGatherShader.BindBuffer(5, kSortedPositionBuffer.GetBuffer());
    kGatherShader.BindBuffer(8, kHistorgramBuffer.GetBuffer());
    kGatherShader.BindBuffer(10, kCellStartBuffer.GetBuffer());
}

// ─────────────────────────────────────────────────────────────────────────────
//  CreateFinalizeShader
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::CreateFinalizeShader(VulkanContext &context)
{
    kFinalizeShader.InitializeGLSL(
        context.device,
        kVHB_FinalizeComp,
        shaderc_compute_shader,
        {{0}},
        sizeof(VH_FinalizePC));
    kFinalizeShader.BindBuffer(0, kHashTableBuffer.GetBuffer());
}

// ─────────────────────────────────────────────────────────────────────────────
//  CreateCountShader
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::CreateCountShader(VulkanContext &context)
{
    kCountShader.InitializeGLSL(
        context.device,
        kVHB_CountComp,
        shaderc_compute_shader,
        {{0}, {1}},
        sizeof(VH_FinalizePC));
    kCountShader.BindBuffer(0, kHashTableBuffer.GetBuffer());
    kCountShader.BindBuffer(1, kPatchCountBuffer.GetBuffer());
}

// ─────────────────────────────────────────────────────────────────────────────
//  CreateBuffers
// ─────────────────────────────────────────────────────────────────────────────

void BucketVoxelHash::CreateBuffers(VulkanContext &context)
{
    constexpr auto HOST = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr auto DEVICE = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    constexpr auto XFER = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    constexpr auto SSBO = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    constexpr VkDeviceSize VEC4_BATCH = sizeof(float) * 4 * VH_BATCH_SIZE;
    constexpr VkDeviceSize COL_BATCH = sizeof(uint32_t) * VH_BATCH_SIZE;
    constexpr VkDeviceSize BUCKETS = sizeof(uint32_t) * VH_NUM_BUCKETS;
    constexpr VkDeviceSize SCAN_BLOCKS = sizeof(uint32_t) * VH_SCAN_BLOCKS;

    // 입력 버퍼 (host-visible)
    kHashTableBuffer.Initialize(context.physicalDevice, context.device,
                                sizeof(VH_Entry) * VH_TOTAL_ENTRIES, SSBO, HOST, 0);
    kPositionBuffer.Initialize(context.physicalDevice, context.device,
                               VEC4_BATCH, SSBO, HOST, 1);
    kNormalBuffer.Initialize(context.physicalDevice, context.device,
                             VEC4_BATCH, SSBO, HOST, 2);
    kColorBuffer.Initialize(context.physicalDevice, context.device,
                            COL_BATCH, SSBO, HOST, 3);
    kPatchCountBuffer.Initialize(context.physicalDevice, context.device,
                                 sizeof(uint32_t), SSBO | XFER, HOST, 4);
    kHistorgramBuffer.Initialize(context.physicalDevice, context.device,
                                 BUCKETS, SSBO | XFER, HOST, 8);
    kBlockSummationBuffer.Initialize(context.physicalDevice, context.device,
                                     SCAN_BLOCKS, SSBO | XFER, HOST, 9);

    // GPU sort 출력 버퍼 (device-local)
    kSortedPositionBuffer.Initialize(context.physicalDevice, context.device,
                                     VEC4_BATCH, SSBO, DEVICE, 5);
    kSortedNormalBuffer.Initialize(context.physicalDevice, context.device,
                                   VEC4_BATCH, SSBO, DEVICE, 6);
    kSortedColorBuffer.Initialize(context.physicalDevice, context.device,
                                  COL_BATCH, SSBO, DEVICE, 7);

    // 버킷 시작 오프셋 (prefix scan 출력)
    kCellStartBuffer.Initialize(context.physicalDevice, context.device,
                                BUCKETS, SSBO | XFER, HOST, 10);
}
