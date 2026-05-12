#version 330 core
flat in int v_layer;
out vec4 fragColor;

uniform sampler3D u_init;
uniform sampler3D u_kprev;
uniform float u_coeff;
uniform usampler3D u_obstacle;
uniform sampler3D u_sdf;
uniform vec3 u_inflow_vel;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);

    if (texel.x == 0) {
        fragColor = vec4(u_inflow_vel, 1.0);
        return;
    }
    if (texel.x == 191) {
        fragColor = texelFetch(u_init, texel - ivec3(1, 0, 0), 0);
        return;
    }
    if (texel.y <= 0 || texel.y >= 191 || texel.z <= 0 || texel.z >= 191) {
        fragColor = vec4(0.0);
        return;
    }

    float sdf = texelFetch(u_sdf, texel, 0).r;
    float alpha = smoothstep(-0.5, 0.5, sdf);
    if (alpha < 0.001) {
        fragColor = vec4(0.0);
        return;
    }

    vec3 u0 = texelFetch(u_init, texel, 0).xyz;
    vec3 uk = texelFetch(u_kprev, texel, 0).xyz;
    fragColor = vec4((u0 + u_coeff * uk) * alpha, 1.0);
}
