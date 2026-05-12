#version 330 core
flat in int v_layer;
out float fragColor;

uniform sampler3D u_velocity;
uniform sampler3D u_density;
uniform float u_dt;
uniform usampler3D u_obstacle;
uniform sampler3D u_sdf;
uniform vec3 u_inflow_vel;

// --- 1D WENO-5 interpolation for float ---
float weno5_1d_f(float f[5], float xi) {
    float p0 = f[0] * xi*(xi+1.0)*0.5 + f[1] * (-xi*(xi+2.0)) + f[2] * (xi+1.0)*(xi+2.0)*0.5;
    float p1 = f[1] * xi*(xi-1.0)*0.5 + f[2] * (1.0 - xi*xi) + f[3] * xi*(xi+1.0)*0.5;
    float p2 = f[2] * (xi-1.0)*(xi-2.0)*0.5 + f[3] * (-xi*(xi-2.0)) + f[4] * xi*(xi-1.0)*0.5;

    float d2a = f[0] - 2.0*f[1] + f[2];
    float d2b = f[1] - 2.0*f[2] + f[3];
    float d2c = f[2] - 2.0*f[3] + f[4];
    float beta0 = 1.083333333 * d2a*d2a + 0.25 * (f[0] - 4.0*f[1] + 3.0*f[2])*(f[0] - 4.0*f[1] + 3.0*f[2]);
    float beta1 = 1.083333333 * d2b*d2b + 0.25 * (f[1] - f[3])*(f[1] - f[3]);
    float beta2 = 1.083333333 * d2c*d2c + 0.25 * (3.0*f[2] - 4.0*f[3] + f[4])*(3.0*f[2] - 4.0*f[3] + f[4]);

    float d0_opt = (xi - 1.0)*(xi - 2.0) / 12.0;
    float d1_opt = (4.0 - xi*xi) / 6.0;
    float d2_opt = (xi + 1.0)*(xi + 2.0) / 12.0;

    float eps = 1e-6;
    float a0 = d0_opt / ((beta0 + eps)*(beta0 + eps));
    float a1 = d1_opt / ((beta1 + eps)*(beta1 + eps));
    float a2 = d2_opt / ((beta2 + eps)*(beta2 + eps));
    float sum_a = a0 + a1 + a2;

    return (a0 * p0 + a1 * p1 + a2 * p2) / sum_a;
}

float weno5_sample_f(sampler3D tex, vec3 pos) {
    ivec3 i = ivec3(floor(pos));
    vec3 xi = pos - vec3(i);

    float zvals[5];
    for (int kk = -2; kk <= 2; kk++) {
        float yvals[5];
        for (int jj = -2; jj <= 2; jj++) {
            float xvals[5];
            for (int ii = -2; ii <= 2; ii++) {
                ivec3 tc = clamp(i + ivec3(ii, jj, kk), ivec3(0), ivec3(191));
                xvals[ii + 2] = texelFetch(tex, tc, 0).x;
            }
            yvals[jj + 2] = weno5_1d_f(xvals, xi.x);
        }
        zvals[kk + 2] = weno5_1d_f(yvals, xi.y);
    }
    return weno5_1d_f(zvals, xi.z);
}

float tri_sample_f(sampler3D tex, vec3 pos) {
    ivec3 i = ivec3(floor(pos));
    vec3 xi = pos - vec3(i);
    ivec3 c000 = clamp(i + ivec3(0,0,0), ivec3(0), ivec3(191));
    ivec3 c100 = clamp(i + ivec3(1,0,0), ivec3(0), ivec3(191));
    ivec3 c010 = clamp(i + ivec3(0,1,0), ivec3(0), ivec3(191));
    ivec3 c110 = clamp(i + ivec3(1,1,0), ivec3(0), ivec3(191));
    ivec3 c001 = clamp(i + ivec3(0,0,1), ivec3(0), ivec3(191));
    ivec3 c101 = clamp(i + ivec3(1,0,1), ivec3(0), ivec3(191));
    ivec3 c011 = clamp(i + ivec3(0,1,1), ivec3(0), ivec3(191));
    ivec3 c111 = clamp(i + ivec3(1,1,1), ivec3(0), ivec3(191));
    float v000 = texelFetch(tex, c000, 0).x;
    float v100 = texelFetch(tex, c100, 0).x;
    float v010 = texelFetch(tex, c010, 0).x;
    float v110 = texelFetch(tex, c110, 0).x;
    float v001 = texelFetch(tex, c001, 0).x;
    float v101 = texelFetch(tex, c101, 0).x;
    float v011 = texelFetch(tex, c011, 0).x;
    float v111 = texelFetch(tex, c111, 0).x;
    float v00 = v000 + (v100 - v000) * xi.x;
    float v01 = v001 + (v101 - v001) * xi.x;
    float v10 = v010 + (v110 - v010) * xi.x;
    float v11 = v011 + (v111 - v011) * xi.x;
    float v0 = v00 + (v10 - v00) * xi.y;
    float v1 = v01 + (v11 - v01) * xi.y;
    return v0 + (v1 - v0) * xi.z;
}

float sample_density(vec3 pos) {
    // Use trilinear for density (passive scalar) to avoid WENO-5 zeroing
    // out the sharp interface at the injection plane.
    return tri_sample_f(u_density, pos);
}

vec3 sample_velocity_vec3(vec3 pos) {
    ivec3 i = ivec3(floor(pos));
    vec3 xi = pos - vec3(i);

    vec3 zvals[5];
    for (int kk = -2; kk <= 2; kk++) {
        vec3 yvals[5];
        for (int jj = -2; jj <= 2; jj++) {
            vec3 xvals[5];
            for (int ii = -2; ii <= 2; ii++) {
                ivec3 tc = clamp(i + ivec3(ii, jj, kk), ivec3(0), ivec3(191));
                xvals[ii + 2] = texelFetch(u_velocity, tc, 0).xyz;
            }
            // Reuse float WENO for each component
            float fx[5], fy[5], fz[5];
            for (int n = 0; n < 5; n++) { fx[n] = xvals[n].x; fy[n] = xvals[n].y; fz[n] = xvals[n].z; }
            yvals[jj + 2] = vec3(weno5_1d_f(fx, xi.x), weno5_1d_f(fy, xi.x), weno5_1d_f(fz, xi.x));
        }
        float fx[5], fy[5], fz[5];
        for (int n = 0; n < 5; n++) { fx[n] = yvals[n].x; fy[n] = yvals[n].y; fz[n] = yvals[n].z; }
        zvals[kk + 2] = vec3(weno5_1d_f(fx, xi.y), weno5_1d_f(fy, xi.y), weno5_1d_f(fz, xi.y));
    }
    float fx[5], fy[5], fz[5];
    for (int n = 0; n < 5; n++) { fx[n] = zvals[n].x; fy[n] = zvals[n].y; fz[n] = zvals[n].z; }
    return vec3(weno5_1d_f(fx, xi.z), weno5_1d_f(fy, xi.z), weno5_1d_f(fz, xi.z));
}

vec3 tri_sample_vel(vec3 pos) {
    ivec3 i = ivec3(floor(pos));
    vec3 xi = pos - vec3(i);
    ivec3 c000 = clamp(i + ivec3(0,0,0), ivec3(0), ivec3(191));
    ivec3 c100 = clamp(i + ivec3(1,0,0), ivec3(0), ivec3(191));
    ivec3 c010 = clamp(i + ivec3(0,1,0), ivec3(0), ivec3(191));
    ivec3 c110 = clamp(i + ivec3(1,1,0), ivec3(0), ivec3(191));
    ivec3 c001 = clamp(i + ivec3(0,0,1), ivec3(0), ivec3(191));
    ivec3 c101 = clamp(i + ivec3(1,0,1), ivec3(0), ivec3(191));
    ivec3 c011 = clamp(i + ivec3(0,1,1), ivec3(0), ivec3(191));
    ivec3 c111 = clamp(i + ivec3(1,1,1), ivec3(0), ivec3(191));
    vec3 v000 = texelFetch(u_velocity, c000, 0).xyz;
    vec3 v100 = texelFetch(u_velocity, c100, 0).xyz;
    vec3 v010 = texelFetch(u_velocity, c010, 0).xyz;
    vec3 v110 = texelFetch(u_velocity, c110, 0).xyz;
    vec3 v001 = texelFetch(u_velocity, c001, 0).xyz;
    vec3 v101 = texelFetch(u_velocity, c101, 0).xyz;
    vec3 v011 = texelFetch(u_velocity, c011, 0).xyz;
    vec3 v111 = texelFetch(u_velocity, c111, 0).xyz;
    vec3 v00 = v000 + (v100 - v000) * xi.x;
    vec3 v01 = v001 + (v101 - v001) * xi.x;
    vec3 v10 = v010 + (v110 - v010) * xi.x;
    vec3 v11 = v011 + (v111 - v011) * xi.x;
    vec3 v0 = v00 + (v10 - v00) * xi.y;
    vec3 v1 = v01 + (v11 - v01) * xi.y;
    return v0 + (v1 - v0) * xi.z;
}

vec3 sample_velocity(vec3 pos) {
    if (pos.x < 2.0 || pos.x > 189.0 || pos.y < 2.0 || pos.y > 189.0 || pos.z < 2.0 || pos.z > 189.0) {
        return tri_sample_vel(pos);
    }
    return sample_velocity_vec3(pos);
}

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    
    // Upstream smoke injection — rectangular slab (upstream of airfoil)
    // Span X=40-64 (all the way to the airfoil LE at X≈66) so the Courant-limited
    // semi-Lagrangian backtrace always has fresh density within reach.
    if (texel.x >= 40 && texel.x <= 64) {
        if (texel.y >= 50 && texel.y <= 140 && texel.z >= 20 && texel.z <= 170) {
            fragColor = 1.0;
            return;
        }
    }
    
    if (texel.x == 0) {
        fragColor = 0.0;
        return;
    }
    
    if (texel.x == 191) {
        float d_current = texelFetch(u_density, texel, 0).x;
        float d_left = texelFetch(u_density, texel - ivec3(1, 0, 0), 0).x;
        float uc = u_inflow_vel.x;
        float cfl = min(uc * u_dt, 0.5);
        fragColor = d_current - cfl * (d_current - d_left);
        return;
    }

    if (texel.y <= 0 || texel.y >= 191 || 
        texel.z <= 0 || texel.z >= 191) {
        fragColor = 0.0;
        return;
    }

    float sdf = texelFetch(u_sdf, texel, 0).r;
    if (sdf < -0.5) {
        fragColor = 0.0;
        return;
    }
    float alpha = smoothstep(-0.5, 0.5, sdf);
    
    vec3 u = texelFetch(u_velocity, texel, 0).xyz;
    
    // --- BFECC Advection for density ---
    // Use a moderate time-step multiplier (1.5× fluid CFL) to stay ahead of
    // the Courant barrier without amplifying spurious Y/Z components 3×.
    float dt_den = u_dt * 3.0;
    vec3 pos = vec3(texel) + 0.5;
    
    // 1. Semi-Lagrangian backward trace
    vec3 pos_fwd = pos - dt_den * u;
    float d_fwd = sample_density(pos_fwd);
    
    // 2. Forward trace using velocity at sample point
    vec3 u_fwd = sample_velocity(pos_fwd);
    vec3 pos_back = pos_fwd + dt_den * u_fwd;
    float d_back = sample_density(pos_back);
    
    // 3. Error correction
    float d_orig = texelFetch(u_density, texel, 0).x;
    float d_new  = d_fwd + 0.5 * (d_orig - d_back);
    
    // 4. Min-max clamping
    ivec3 st = ivec3(pos_fwd - 0.5);
    float n000 = texelFetch(u_density, clamp(st + ivec3(0,0,0), ivec3(0), ivec3(191)), 0).x;
    float n100 = texelFetch(u_density, clamp(st + ivec3(1,0,0), ivec3(0), ivec3(191)), 0).x;
    float n010 = texelFetch(u_density, clamp(st + ivec3(0,1,0), ivec3(0), ivec3(191)), 0).x;
    float n110 = texelFetch(u_density, clamp(st + ivec3(1,1,0), ivec3(0), ivec3(191)), 0).x;
    float n001 = texelFetch(u_density, clamp(st + ivec3(0,0,1), ivec3(0), ivec3(191)), 0).x;
    float n101 = texelFetch(u_density, clamp(st + ivec3(1,0,1), ivec3(0), ivec3(191)), 0).x;
    float n011 = texelFetch(u_density, clamp(st + ivec3(0,1,1), ivec3(0), ivec3(191)), 0).x;
    float n111 = texelFetch(u_density, clamp(st + ivec3(1,1,1), ivec3(0), ivec3(191)), 0).x;
    
    float d_min = min(min(min(n000, n100), min(n010, n110)), min(min(n001, n101), min(n011, n111)));
    float d_max = max(max(max(n000, n100), max(n010, n110)), max(max(n001, n101), max(n011, n111)));
    
    d_new = clamp(d_new, d_min, d_max);
    
    d_new *= 0.998 * alpha;
    
    fragColor = d_new;
}
