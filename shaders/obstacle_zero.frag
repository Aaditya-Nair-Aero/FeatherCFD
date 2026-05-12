#version 330 core
flat in int v_layer;
out vec4 fragColor;

uniform sampler3D u_pressure;
uniform usampler3D u_mask;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    uint obs = texelFetch(u_mask, texel, 0).x;
    if (obs > 0u) {
        fragColor = vec4(0.0);
    } else {
        fragColor = texelFetch(u_pressure, texel, 0);
    }
}
