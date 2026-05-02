#include "GaussianSplatting.h"

#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
//  Inline GLSL shaders
// ─────────────────────────────────────────────────────────────────────────────

static const char *kVertSrc = R"glsl(
#version 450

struct GS_Gaussian {
    float px, py, pz, scaleX;
    float nx, ny, nz, scaleY;
    float opacity;
    uint  col;
    float pad0, pad1;
};
layout(set=0, binding=0, std430) readonly buffer SplatBuf { GS_Gaussian splats[]; };

layout(push_constant) uniform PC {
    mat4  mvp;
    vec3  camRight;
    float splatScale;
    vec3  camUp;
    float _pad;
} pc;

layout(location=0) out vec2  fragUV;
layout(location=1) out vec4  fragColor;
layout(location=2) out vec3  fragNormal;
layout(location=3) out float fragOpacity;

const vec2 kCorners[6] = vec2[](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0));

void main() {
    int si = gl_VertexIndex / 6;
    GS_Gaussian s = splats[si];

    vec2 c   = kCorners[gl_VertexIndex % 6];
    vec3 pos = vec3(s.px, s.py, s.pz)
             + pc.camRight * c.x * s.scaleX * pc.splatScale
             + pc.camUp    * c.y * s.scaleY * pc.splatScale;

    gl_Position = pc.mvp * vec4(pos, 1.0);
    fragUV      = c;
    fragNormal  = vec3(s.nx, s.ny, s.nz);
    fragOpacity = s.opacity;

    uint col = s.col;
    fragColor = vec4(float( col        & 0xFFu) / 255.0,
                     float((col >>  8) & 0xFFu) / 255.0,
                     float((col >> 16) & 0xFFu) / 255.0,
                     1.0);
}
)glsl";

static const char *kFragSrc = R"glsl(
#version 450

layout(location=0) in vec2  fragUV;
layout(location=1) in vec4  fragColor;
layout(location=2) in vec3  fragNormal;
layout(location=3) in float fragOpacity;

layout(location=0) out vec4 outColor;

void main() {
    float d2 = dot(fragUV, fragUV);
    if (d2 > 1.0) discard;

    float alpha = fragOpacity * exp(-3.0 * d2);
    if (alpha < 0.01) discard;

    vec3  L       = normalize(vec3(0.3, 1.0, 0.5));
    float diffuse = max(dot(normalize(fragNormal), L), 0.0);
    float shade   = 0.3 + 0.7 * diffuse;

    outColor = vec4(fragColor.rgb * shade, alpha);
}
)glsl";

// ─────────────────────────────────────────────────────────────────────────────

void GaussianSplatting::Initialize(const VulkanContext &ctx)
{
    constexpr auto HOST = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                        | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr auto SSBO = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    splatBuffer_.Initialize(
        ctx.physicalDevice, ctx.device,
        sizeof(GS_Gaussian) * GS_MAX_SPLATS,
        SSBO, HOST, /*bindingNum=*/0);

    RenderingShader::Binding binding{};
    binding.slot   = 0;
    binding.type   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.stages = VK_SHADER_STAGE_VERTEX_BIT;

    RenderingShader::PipelineOptions opts{};
    opts.topology            = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    opts.cullMode            = VK_CULL_MODE_NONE;
    opts.depthTestEnable     = VK_FALSE;
    opts.depthWriteEnable    = VK_FALSE;
    opts.blendEnable         = VK_TRUE;
    opts.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    opts.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;   // additive
    opts.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    opts.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    opts.colorBlendOp        = VK_BLEND_OP_ADD;
    opts.alphaBlendOp        = VK_BLEND_OP_ADD;

    shader_.InitializeGLSL(
        ctx.device, ctx.renderPass, ctx.extent,
        kVertSrc, kFragSrc,
        {binding}, opts,
        static_cast<uint32_t>(sizeof(GS_RenderPC)));

    shader_.BindBuffer(0, splatBuffer_.GetBuffer());
}

void GaussianSplatting::Integrate(const VH_InputPoint *pts, uint32_t count)
{
    if (count == 0 || splatCount_ >= GS_MAX_SPLATS) return;

    // 남은 슬롯만큼만 append
    uint32_t available = GS_MAX_SPLATS - splatCount_;
    count = std::min(count, available);

    // splatCount_ 위치부터 append (byte offset 사용)
    uint32_t byteOffset = static_cast<uint32_t>(sizeof(GS_Gaussian)) * splatCount_;
    auto mapped = splatBuffer_.Access<GS_Gaussian>(sizeof(GS_Gaussian) * count, byteOffset);
    GS_Gaussian *dst = mapped.get();

    for (uint32_t i = 0; i < count; ++i)
    {
        const VH_InputPoint &p = pts[i];
        dst[i].px      = p.px;
        dst[i].py      = p.py;
        dst[i].pz      = p.pz;
        dst[i].scaleX  = 0.03f;
        dst[i].nx      = p.nx;
        dst[i].ny      = p.ny;
        dst[i].nz      = p.nz;
        dst[i].scaleY  = 0.03f;
        dst[i].opacity = 0.8f;
        dst[i].col     = p.col;
        dst[i]._pad[0] = 0.f;
        dst[i]._pad[1] = 0.f;
    }
    // ScopedMemoryGuard 소멸 시 자동 unmap

    splatCount_ += count;
}

void GaussianSplatting::Render(VkCommandBuffer cmd, const GS_RenderPC &pc)
{
    if (splatCount_ == 0) return;
    shader_.Bind(cmd);
    shader_.Push(cmd, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
    shader_.Draw(cmd, splatCount_ * 6);
}
