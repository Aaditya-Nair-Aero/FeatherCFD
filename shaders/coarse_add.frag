#version 330 core
flat in int v_layer;
out float fragColor;

uniform sampler3D u_current;
uniform sampler3D u_update;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    float cur = texelFetch(u_current, texel, 0).x;
    float upd = texelFetch(u_update, texel, 0).x;
    fragColor = cur + upd;
}
