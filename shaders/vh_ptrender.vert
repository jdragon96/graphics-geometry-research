// vh_ptrender.vert — 입력 포인트 클라우드 오버레이 (시안색 점)
// SoA: sortedPosBuf_(binding=5)에서 위치 읽기
#version 450

layout(push_constant) uniform PC {
    mat4  mvp;
    float voxelSize;
    uint  colorMode;
    uint  currentFrame;
    uint  highlightFrames;
} pc;

layout(set=0, binding=5) readonly buffer SortedPosBuf { vec4 p[]; } sortedPos;

layout(location=0) out vec3  fragColor;
layout(location=1) out float fragHighlight;

void main() {
    vec3 pos      = sortedPos.p[gl_VertexIndex].xyz;
    gl_Position   = pc.mvp * vec4(pos, 1.0);
    gl_PointSize  = 3.0;
    fragColor     = vec3(0.0, 1.0, 1.0);
    fragHighlight = 0.0;
}
