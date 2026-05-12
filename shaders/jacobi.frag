#version 330 core
flat in int v_layer;
out float fragColor;

uniform sampler3D u_pressure;
uniform sampler3D u_divergence;
uniform usampler3D u_obstacle;
uniform sampler3D u_sdf;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    
    // Dirichlet BC: Pressure = 0 at Outflow (X=191)
    if (texel.x >= 191) {
        fragColor = 0.0;
        return;
    }

    if (texel.x <= 0 ||
        texel.y <= 0 || texel.y >= 191 ||
        texel.z <= 0 || texel.z >= 191) {
        fragColor = 0.0;
        return;
    }

    // SDF obstacle: zero pressure inside
    float sdf = texelFetch(u_sdf, texel, 0).r;
    if (sdf < -0.5) {
        fragColor = 0.0;
        return;
    }
    float alpha = smoothstep(-0.5, 0.5, sdf);
    
    // Neumann boundary conditions: if neighbor is boundary or obstacle, use center value
    // For SDF obstacle, neighbor is inside if its SDF < 0
    ivec3 L = texel - ivec3(1, 0, 0);
    ivec3 R = texel + ivec3(1, 0, 0);
    ivec3 D = texel - ivec3(0, 1, 0);
    ivec3 U = texel + ivec3(0, 1, 0);
    ivec3 B = texel - ivec3(0, 0, 1);
    ivec3 F = texel + ivec3(0, 0, 1);
    
    float sdfL = (L.x > 0) ? texelFetch(u_sdf, L, 0).r : 1.0;
    float sdfR = (R.x < 191) ? texelFetch(u_sdf, R, 0).r : 1.0;
    float sdfD = (D.y > 0) ? texelFetch(u_sdf, D, 0).r : 1.0;
    float sdfU = (U.y < 191) ? texelFetch(u_sdf, U, 0).r : 1.0;
    float sdfB = (B.z > 0) ? texelFetch(u_sdf, B, 0).r : 1.0;
    float sdfF = (F.z < 191) ? texelFetch(u_sdf, F, 0).r : 1.0;
    
    float pL = (L.x <= 0 || sdfL < 0.0)   ? texelFetch(u_pressure, texel, 0).x : texelFetch(u_pressure, L, 0).x;
    float pR = (R.x >= 191) ? 0.0 : ((sdfR < 0.0) ? texelFetch(u_pressure, texel, 0).x : texelFetch(u_pressure, R, 0).x);
    float pD = (D.y <= 0 || sdfD < 0.0)   ? texelFetch(u_pressure, texel, 0).x : texelFetch(u_pressure, D, 0).x;
    float pU = (U.y >= 191 || sdfU < 0.0) ? texelFetch(u_pressure, texel, 0).x : texelFetch(u_pressure, U, 0).x;
    float pB = (B.z <= 0 || sdfB < 0.0)   ? texelFetch(u_pressure, texel, 0).x : texelFetch(u_pressure, B, 0).x;
    float pF = (F.z >= 191 || sdfF < 0.0) ? texelFetch(u_pressure, texel, 0).x : texelFetch(u_pressure, F, 0).x;
    
    float div = texelFetch(u_divergence, texel, 0).x;
    
    // Blend pressure: cut cells get reduced correction
    float p_new = (pL + pR + pU + pD + pF + pB - div) / 6.0;
    fragColor = mix(0.0, p_new, alpha);
}
