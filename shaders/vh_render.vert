#version 450

// -----------------------------
// Constants / Layout
// -----------------------------
#define STRIDE  12
#define EMPTY   0x7FFFFFFF

#define F_KX     0
#define F_KY     1
#define F_KZ     2
#define F_TSDF   3
#define F_W      4
#define F_COL    5
#define F_NORM   6
#define F_FTAG  11

// -----------------------------
// Push constants
// -----------------------------
layout(push_constant) uniform PC {
    mat4  mvp;
    float voxelSize;
    uint  colorMode;
    uint  currentFrame;
    uint  highlightFrames;
} pc;

// -----------------------------
layout(set=0, binding=0) readonly buffer HT {
    int raw[];
} ht;

layout(location=0) out vec3  fragColor;
layout(location=1) out float fragHighlight;

// -----------------------------
// Cube geometry
// -----------------------------
const vec3 kCubeVerts[8] = vec3[8](
    vec3(-0.5,-0.5,-0.5), vec3( 0.5,-0.5,-0.5),
    vec3( 0.5, 0.5,-0.5), vec3(-0.5, 0.5,-0.5),
    vec3(-0.5,-0.5, 0.5), vec3( 0.5,-0.5, 0.5),
    vec3( 0.5, 0.5, 0.5), vec3(-0.5, 0.5, 0.5)
);

const int kIndices[36] = int[36](
    0,2,1, 0,3,2,
    4,5,6, 4,6,7,
    0,1,5, 0,5,4,
    3,7,6, 3,6,2,
    0,4,7, 0,7,3,
    1,2,6, 1,6,5
);

const vec3 kFaceNormals[6] = vec3[6](
    vec3( 0, 0,-1), vec3( 0, 0, 1),
    vec3( 0,-1, 0), vec3( 0, 1, 0),
    vec3(-1, 0, 0), vec3( 1, 0, 0)
);

// -----------------------------
// Utility functions
// -----------------------------
int baseIndex(int voxelIdx) {
    return voxelIdx * STRIDE;
}

bool isEmpty(int base) {
    return ht.raw[base + F_KX] == EMPTY;
}

vec3 voxelCenter(int base) {
    return (vec3(
        float(ht.raw[base + F_KX]),
        float(ht.raw[base + F_KY]),
        float(ht.raw[base + F_KZ])
    ) + 0.5) * pc.voxelSize;
}

float voxelTSDF(int base) {
    return intBitsToFloat(ht.raw[base + F_TSDF]);
}

float voxelWeight(int base) {
    return intBitsToFloat(ht.raw[base + F_W]);
}

uint voxelColor(int base) {
    return uint(ht.raw[base + F_COL]);
}

uint voxelNormal(int base) {
    return uint(ht.raw[base + F_NORM]);
}

int voxelFrame(int base) {
    return ht.raw[base + F_FTAG];
}

// -----------------------------
// Decoding
// -----------------------------
vec3 unpackColor(uint c) {
    return vec3(
        float((c >>  0) & 0xFFu),
        float((c >>  8) & 0xFFu),
        float((c >> 16) & 0xFFu)
    ) / 255.0;
}

vec3 unpackNormal(uint p) {
    float fx = float(p & 0xFFFFu)/65535.0*2.0-1.0;
    float fy = float((p>>16u)&0xFFFFu)/65535.0*2.0-1.0;

    vec3 n = vec3(fx, fy, 1.0 - abs(fx) - abs(fy));

    if (n.z < 0.0) {
        n.xy = vec2(
            n.x >= 0.0 ? (1.0 - abs(n.y)) : -(1.0 - abs(n.y)),
            n.y >= 0.0 ? (1.0 - abs(n.x)) : -(1.0 - abs(n.x))
        );
    }

    return normalize(n);
}

vec3 tsdfColor(float t) {
    return (t < 0.0)
        ? mix(vec3(0,0,1), vec3(1), t + 1.0)
        : mix(vec3(1), vec3(1,0,0), t);
}

// -----------------------------
// Color mode resolver
// -----------------------------
vec3 resolveColor(int base, float age, float highlightT)
{
    if (pc.colorMode == 0u) {
        return unpackColor(voxelColor(base));
    }
    else if (pc.colorMode == 1u) {
        return unpackNormal(voxelNormal(base)) * 0.5 + 0.5;
    }
    else if (pc.colorMode == 2u) {
        return tsdfColor(voxelTSDF(base));
    }
    else if (pc.colorMode == 3u) {
        float w = clamp(voxelWeight(base) / 100.0, 0.0, 1.0);
        return vec3(w);
    }
    else {
        float normAge = clamp(age / float(max(pc.highlightFrames * 4u, 1u)), 0.0, 1.0);
        return mix(vec3(1.0,0.2,0.0), vec3(0.0,0.3,1.0), normAge);
    }
}

// -----------------------------
// Main
// -----------------------------
void main()
{
    int voxelIdx = gl_VertexIndex / 36;
    int triIdx   = gl_VertexIndex % 36;
    int base     = baseIndex(voxelIdx);

    // Empty voxel → clip
    if (isEmpty(base)) {
        gl_Position   = vec4(10.0);
        fragColor     = vec3(0.0);
        fragHighlight = 0.0;
        return;
    }

    // -------------------------
    // Transform
    // -------------------------
    vec3 center = voxelCenter(base);

    int   frameTag = voxelFrame(base);
    float age      = float(int(pc.currentFrame) - frameTag);
    float highlightT = (frameTag >= 0) ? max(0.0, 1.0 - age / float(pc.highlightFrames)) : 0.0;
    float scale = mix(0.92, 1.08, highlightT);
    
    vec3 localPos = kCubeVerts[kIndices[triIdx]] * pc.voxelSize * scale;
    gl_Position   = pc.mvp * vec4(center + localPos, 1.0);

    // -------------------------
    // Color
    // -------------------------
    vec3 baseColor = resolveColor(base, age, highlightT);

    // -------------------------
    // Lighting
    // -------------------------
    vec3  lightDir = normalize(vec3(1.0, 2.0, 1.5));
    float diffuse  = max(dot(kFaceNormals[triIdx / 6], lightDir), 0.0);

    baseColor *= 0.35 + 0.65 * diffuse;

    // -------------------------
    // Highlight blend
    // -------------------------
    fragColor     = mix(baseColor, vec3(1.0, 0.9, 0.1), highlightT * 0.85);
    fragHighlight = highlightT;
}