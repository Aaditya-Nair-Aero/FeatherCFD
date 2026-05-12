#version 330 core
flat in int v_layer;
out float fragColor;

uniform sampler3D u_solution;
uniform sampler3D u_rhs;
uniform usampler3D u_obstacle;
uniform float u_h_sq;
uniform int u_parity;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    ivec3 sz = textureSize(u_solution, 0);

    if (texel.x <= 0 || texel.x >= sz.x - 1 ||
        texel.y <= 0 || texel.y >= sz.y - 1 ||
        texel.z <= 0 || texel.z >= sz.z - 1) {
        fragColor = 0.0;
        return;
    }

    if (texelFetch(u_obstacle, texel, 0).x > 0u) {
        fragColor = 0.0;
        return;
    }

    int cell_parity = (texel.x + texel.y + texel.z) & 1;
    if (cell_parity != u_parity) {
        fragColor = texelFetch(u_solution, texel, 0).x;
        return;
    }

    float rhs_val = texelFetch(u_rhs, texel, 0).x;
    float ec = texelFetch(u_solution, texel, 0).x;

    float eL = texelFetch(u_solution, texel + ivec3(-1, 0, 0), 0).x;
    float eR = texelFetch(u_solution, texel + ivec3( 1, 0, 0), 0).x;
    float eD = texelFetch(u_solution, texel + ivec3(0, -1, 0), 0).x;
    float eU = texelFetch(u_solution, texel + ivec3(0,  1, 0), 0).x;
    float eB = texelFetch(u_solution, texel + ivec3(0, 0, -1), 0).x;
    float eF = texelFetch(u_solution, texel + ivec3(0, 0,  1), 0).x;

    float e_new = (eL + eR + eD + eU + eB + eF - u_h_sq * rhs_val) / 6.0;
    fragColor = e_new;
}
