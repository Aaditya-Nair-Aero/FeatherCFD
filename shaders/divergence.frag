#version 330 core
flat in int v_layer;
out float fragColor;

uniform sampler3D u_velocity;
uniform usampler3D u_obstacle;
uniform sampler3D u_sdf;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    
    // Boundary: Zero divergence at domain boundaries
    if (texel.x <= 0 || texel.x >= 191 ||
        texel.y <= 0 || texel.y >= 191 ||
        texel.z <= 0 || texel.z >= 191) {
        fragColor = 0.0;
        return;
    }

    // SDF-based obstacle: zero divergence inside, computed at cut/fluid cells
    float sdf = texelFetch(u_sdf, texel, 0).r;
    if (sdf < -0.5) {
        fragColor = 0.0;
        return;
    }
    float alpha = smoothstep(-0.5, 0.5, sdf);
    
    vec3 uc = texelFetch(u_velocity, texel, 0).xyz;
    float ux_right = texelFetch(u_velocity, texel + ivec3(1, 0, 0), 0).x;
    float uy_up    = texelFetch(u_velocity, texel + ivec3(0, 1, 0), 0).y;
    float uz_front = texelFetch(u_velocity, texel + ivec3(0, 0, 1), 0).z;
    
    // Forward difference divergence: (u(i+1) - u(i)) per axis
    // Composes with backward-diff gradient to give compact Laplacian
    float div = (ux_right - uc.x) + (uy_up - uc.y) + (uz_front - uc.z);
    
    fragColor = div * alpha;
}
