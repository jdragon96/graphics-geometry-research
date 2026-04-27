// vh_render.frag — 복셀 큐브 프래그먼트
#version 450
layout(location=0) in  vec3  fragColor;
layout(location=1) in  float fragHighlight;
layout(location=0) out vec4  outColor;
void main() {
    outColor = vec4(fragColor, 1.0);
}
