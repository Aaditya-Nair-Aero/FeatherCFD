#version 330 core
flat in int v_layer;
out vec4 fragColor;

uniform sampler3D u_velocity;
uniform float u_dt;
uniform vec3 u_force_pos;
uniform vec3 u_force_dir;
uniform float u_force_radius;
uniform usampler3D u_obstacle;
uniform sampler3D u_sdf;
uniform vec3 u_inflow_vel;

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

    // No-slip walls (Y and Z domain boundaries)
    if (texel.y <= 0 || texel.y >= 191 ||
        texel.z <= 0 || texel.z >= 191) {
        fragColor = vec4(0.0);
        return;
    }

    // SDF-based obstacle: smooth sub-grid boundary
    float sdf = texelFetch(u_sdf, texel, 0).r;
    float alpha = smoothstep(-0.5, 0.5, sdf);
    if (alpha < 0.001) {
        fragColor = vec4(0.0);
        return;
    }
    
    vec3 u = texelFetch(u_velocity, texel, 0).xyz;
    
    vec3 cell_pos = vec3(texel) + 0.5;
    float dist = distance(cell_pos, u_force_pos);
    if (dist < u_force_radius) {
        float falloff = 1.0 - (dist / u_force_radius);
        u += u_dt * u_force_dir * falloff;
    }
    
    fragColor = vec4(u * alpha, 1.0);
}
