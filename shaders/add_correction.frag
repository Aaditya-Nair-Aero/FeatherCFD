#version 330 core
flat in int v_layer;
out float fragColor;

uniform sampler3D u_pressure;
uniform sampler3D u_correction;
uniform sampler3D u_sdf;
uniform float u_omega;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);

    if (texel.x == 0 || texel.x == 191 ||
        texel.y <= 0 || texel.y >= 191 ||
        texel.z <= 0 || texel.z >= 191) {
        fragColor = 0.0;
        return;
    }

    float sdf = texelFetch(u_sdf, texel, 0).r;
    float alpha = smoothstep(-0.5, 0.5, sdf);
    if (alpha < 0.001) {
        fragColor = 0.0;
        return;
    }

    float p_old = texelFetch(u_pressure, texel, 0).x;
    float corr = texelFetch(u_correction, texel, 0).x;
    float p_new = p_old + u_omega * corr;

    fragColor = mix(0.0, p_new, alpha);
}
