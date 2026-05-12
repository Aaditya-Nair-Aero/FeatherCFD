#version 330 core
flat in int v_layer;
out vec4 fragColor;

uniform sampler3D u_src;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    fragColor = texelFetch(u_src, texel, 0);
}
