#version 330 core
flat in int v_layer;
out vec4 fragColor;

uniform sampler3D u_velocity;
uniform sampler3D u_pressure;
uniform usampler3D u_obstacle;
uniform sampler3D u_sdf;
uniform vec3 u_inflow_vel;
uniform float u_dt;

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    
    // Inflow at X=0
    if (texel.x == 0) {
        fragColor = vec4(u_inflow_vel, 1.0);
        return;
    }
    
    // Outflow at X=191 (Convective: ∂u/∂t + Uc·∂u/∂x = 0)
    // Forward Euler: u_new = u_current - Uc*dt/dx * (u_current - u_left)
    if (texel.x == 191) {
        vec3 u_current = texelFetch(u_velocity, texel, 0).xyz;
        vec3 u_left = texelFetch(u_velocity, texel - ivec3(1, 0, 0), 0).xyz;
        float uc = u_inflow_vel.x;
        float cfl = min(uc * u_dt, 0.5);
        fragColor = vec4(u_current - cfl * (u_current - u_left), 1.0);
        return;
    }

    if (texel.y <= 0 || texel.y >= 191 ||
        texel.z <= 0 || texel.z >= 191) {
        fragColor = vec4(0.0);
        return;
    }

    // SDF obstacle check
    float sdf = texelFetch(u_sdf, texel, 0).r;
    float alpha = smoothstep(-0.5, 0.5, sdf);
    if (alpha < 0.001) {
        fragColor = vec4(0.0);
        return;
    }
    
    vec3 u = texelFetch(u_velocity, texel, 0).xyz;
    
    // Neumann condition with SDF-based neighbor checks
    ivec3 L = texel - ivec3(1, 0, 0);
    ivec3 R = texel + ivec3(1, 0, 0);
    ivec3 D = texel - ivec3(0, 1, 0);
    ivec3 U = texel + ivec3(0, 1, 0);
    ivec3 B = texel - ivec3(0, 0, 1);
    ivec3 F = texel + ivec3(0, 0, 1);
    
    float sdfL = (L.x > 0)   ? texelFetch(u_sdf, L, 0).r : 1.0;
    float sdfR = (R.x < 191) ? texelFetch(u_sdf, R, 0).r : 1.0;
    float sdfD = (D.y > 0)   ? texelFetch(u_sdf, D, 0).r : 1.0;
    float sdfU = (U.y < 191) ? texelFetch(u_sdf, U, 0).r : 1.0;
    float sdfB = (B.z > 0)   ? texelFetch(u_sdf, B, 0).r : 1.0;
    float sdfF = (F.z < 191) ? texelFetch(u_sdf, F, 0).r : 1.0;
    
    float pL = (L.x <= 0   || sdfL < 0.0) ? texelFetch(u_pressure, texel, 0).x : texelFetch(u_pressure, L, 0).x;
    float pR = (R.x >= 191) ? 0.0 : ((sdfR < 0.0) ? texelFetch(u_pressure, texel, 0).x : texelFetch(u_pressure, R, 0).x);
    float pD = (D.y <= 0   || sdfD < 0.0) ? texelFetch(u_pressure, texel, 0).x : texelFetch(u_pressure, D, 0).x;
    float pU = (U.y >= 191 || sdfU < 0.0) ? texelFetch(u_pressure, texel, 0).x : texelFetch(u_pressure, U, 0).x;
    float pB = (B.z <= 0   || sdfB < 0.0) ? texelFetch(u_pressure, texel, 0).x : texelFetch(u_pressure, B, 0).x;
    float pF = (F.z >= 191 || sdfF < 0.0) ? texelFetch(u_pressure, texel, 0).x : texelFetch(u_pressure, F, 0).x;
    
    // Backward difference gradient: (p(i) - p(i-1)) per axis
    // Composes with forward-diff divergence to give compact Laplacian
    float pc = texelFetch(u_pressure, texel, 0).x;
    vec3 grad_p = vec3(pc - pL, pc - pD, pc - pB);
    
    vec3 u_new = u - grad_p;
    
    fragColor = vec4(u_new * alpha, 1.0);
}
