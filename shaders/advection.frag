#version 330 core
flat in int v_layer;
out vec4 fragColor;

uniform sampler3D u_velocity;
uniform float u_dt;
uniform usampler3D u_obstacle;
uniform sampler3D u_sdf;
uniform vec3 u_inflow_vel;
uniform float u_viscosity;
uniform float u_sgs_coeff;

// --- 1D WENO-5 interpolation for vec3 ---
// f[5] = values at {i-2, i-1, i, i+1, i+2}, xi in [0,1]
vec3 weno5_1d(vec3 f[5], float xi) {
    vec3 p0 = f[0] * xi*(xi+1.0)*0.5 + f[1] * (-xi*(xi+2.0)) + f[2] * (xi+1.0)*(xi+2.0)*0.5;
    vec3 p1 = f[1] * xi*(xi-1.0)*0.5 + f[2] * (1.0 - xi*xi) + f[3] * xi*(xi+1.0)*0.5;
    vec3 p2 = f[2] * (xi-1.0)*(xi-2.0)*0.5 + f[3] * (-xi*(xi-2.0)) + f[4] * xi*(xi-1.0)*0.5;

    vec3 d2a = f[0] - 2.0*f[1] + f[2];
    vec3 d2b = f[1] - 2.0*f[2] + f[3];
    vec3 d2c = f[2] - 2.0*f[3] + f[4];
    vec3 beta0 = 1.083333333 * d2a*d2a + 0.25 * (f[0] - 4.0*f[1] + 3.0*f[2])*(f[0] - 4.0*f[1] + 3.0*f[2]);
    vec3 beta1 = 1.083333333 * d2b*d2b + 0.25 * (f[1] - f[3])*(f[1] - f[3]);
    vec3 beta2 = 1.083333333 * d2c*d2c + 0.25 * (3.0*f[2] - 4.0*f[3] + f[4])*(3.0*f[2] - 4.0*f[3] + f[4]);

    float d0_opt = (xi - 1.0)*(xi - 2.0) / 12.0;
    float d1_opt = (4.0 - xi*xi) / 6.0;
    float d2_opt = (xi + 1.0)*(xi + 2.0) / 12.0;

    float eps = 1e-6;
    vec3 a0 = d0_opt / ((beta0 + eps)*(beta0 + eps));
    vec3 a1 = d1_opt / ((beta1 + eps)*(beta1 + eps));
    vec3 a2 = d2_opt / ((beta2 + eps)*(beta2 + eps));
    vec3 sum_a = a0 + a1 + a2;

    return (a0 * p0 + a1 * p1 + a2 * p2) / sum_a;
}

// --- 3D tensor-product WENO-5 for vec3 ---
vec3 weno5_sample(sampler3D tex, vec3 pos) {
    ivec3 i = ivec3(floor(pos));
    vec3 xi = pos - vec3(i);

    vec3 zvals[5];
    for (int kk = -2; kk <= 2; kk++) {
        vec3 yvals[5];
        for (int jj = -2; jj <= 2; jj++) {
            vec3 xvals[5];
            for (int ii = -2; ii <= 2; ii++) {
                ivec3 tc = clamp(i + ivec3(ii, jj, kk), ivec3(0), ivec3(191));
                xvals[ii + 2] = texelFetch(tex, tc, 0).xyz;
            }
            yvals[jj + 2] = weno5_1d(xvals, xi.x);
        }
        zvals[kk + 2] = weno5_1d(yvals, xi.y);
    }
    return weno5_1d(zvals, xi.z);
}

vec3 tri_sample(sampler3D tex, vec3 pos) {
    ivec3 i = ivec3(floor(pos));
    vec3 xi = pos - vec3(i);
    ivec3 c00 = clamp(i + ivec3(0,0,0), ivec3(0), ivec3(191));
    ivec3 c10 = clamp(i + ivec3(1,0,0), ivec3(0), ivec3(191));
    ivec3 c01 = clamp(i + ivec3(0,1,0), ivec3(0), ivec3(191));
    ivec3 c11 = clamp(i + ivec3(1,1,0), ivec3(0), ivec3(191));
    ivec3 c00_1 = clamp(i + ivec3(0,0,1), ivec3(0), ivec3(191));
    ivec3 c10_1 = clamp(i + ivec3(1,0,1), ivec3(0), ivec3(191));
    ivec3 c01_1 = clamp(i + ivec3(0,1,1), ivec3(0), ivec3(191));
    ivec3 c11_1 = clamp(i + ivec3(1,1,1), ivec3(0), ivec3(191));
    vec3 v000 = texelFetch(tex, c00, 0).xyz;
    vec3 v100 = texelFetch(tex, c10, 0).xyz;
    vec3 v010 = texelFetch(tex, c01, 0).xyz;
    vec3 v110 = texelFetch(tex, c11, 0).xyz;
    vec3 v001 = texelFetch(tex, c00_1, 0).xyz;
    vec3 v101 = texelFetch(tex, c10_1, 0).xyz;
    vec3 v011 = texelFetch(tex, c01_1, 0).xyz;
    vec3 v111 = texelFetch(tex, c11_1, 0).xyz;
    
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
        return tri_sample(u_velocity, pos);
    }
    return weno5_sample(u_velocity, pos);
}

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

    // SDF-based obstacle check
    float sdf = texelFetch(u_sdf, texel, 0).r;
    float alpha = smoothstep(-0.5, 0.5, sdf);
    if (alpha < 0.001) {
        fragColor = vec4(0.0);
        return;
    }
    
    vec3 u = texelFetch(u_velocity, texel, 0).xyz;
    
    // --- BFECC Advection with WENO-5 velocity sampling ---
    vec3 pos = vec3(texel) + 0.5;
    
    // 1. Semi-Lagrangian backward trace (WENO-5 for velocity)
    vec3 pos_fwd = pos - u_dt * u;
    vec3 u_fwd = sample_velocity(pos_fwd);
    
    // 2. Forward trace
    vec3 pos_back = pos_fwd + u_dt * u_fwd;
    vec3 u_back = sample_velocity(pos_back);
    
    // 3. Correction
    vec3 u_orig = texelFetch(u_velocity, texel, 0).xyz;
    vec3 u_new  = u_fwd + 0.5 * (u_orig - u_back);
    
    // 4. Clamping (BFECC anti-overshoot)
    vec3 u_min, u_max;
#if CLAMP_MODE == 0
    // Mode 0: Standard — clamp to 8 neighbors of back-trace origin
    ivec3 st = ivec3(pos_fwd - 0.5);
    vec3 n000 = texelFetch(u_velocity, clamp(st + ivec3(0,0,0), ivec3(0), ivec3(191)), 0).xyz;
    vec3 n100 = texelFetch(u_velocity, clamp(st + ivec3(1,0,0), ivec3(0), ivec3(191)), 0).xyz;
    vec3 n010 = texelFetch(u_velocity, clamp(st + ivec3(0,1,0), ivec3(0), ivec3(191)), 0).xyz;
    vec3 n110 = texelFetch(u_velocity, clamp(st + ivec3(1,1,0), ivec3(0), ivec3(191)), 0).xyz;
    vec3 n001 = texelFetch(u_velocity, clamp(st + ivec3(0,0,1), ivec3(0), ivec3(191)), 0).xyz;
    vec3 n101 = texelFetch(u_velocity, clamp(st + ivec3(1,0,1), ivec3(0), ivec3(191)), 0).xyz;
    vec3 n011 = texelFetch(u_velocity, clamp(st + ivec3(0,1,1), ivec3(0), ivec3(191)), 0).xyz;
    vec3 n111 = texelFetch(u_velocity, clamp(st + ivec3(1,1,1), ivec3(0), ivec3(191)), 0).xyz;
    u_min = min(min(min(n000, n100), min(n010, n110)), min(min(n001, n101), min(n011, n111)));
    u_max = max(max(max(n000, n100), max(n010, n110)), max(max(n001, n101), max(n011, n111)));
#elif CLAMP_MODE == 1
    // Mode 1: None — disable clamping (may cause instability)
    u_min = vec3(-1e10);
    u_max = vec3(1e10);
#elif CLAMP_MODE == 2
    // Mode 2: SDF-aware — skip obstacle-interior cells in stencil
    ivec3 st = ivec3(pos_fwd - 0.5);
    ivec3 offs[8] = ivec3[8](
        ivec3(0,0,0), ivec3(1,0,0), ivec3(0,1,0), ivec3(1,1,0),
        ivec3(0,0,1), ivec3(1,0,1), ivec3(0,1,1), ivec3(1,1,1)
    );
    u_min = vec3(1e10);
    u_max = vec3(-1e10);
    int valid = 0;
    for (int i = 0; i < 8; i++) {
        ivec3 p = clamp(st + offs[i], ivec3(0), ivec3(191));
        float s = texelFetch(u_sdf, p, 0).r;
        if (s > 0.0) {
            vec3 v = texelFetch(u_velocity, p, 0).xyz;
            u_min = min(u_min, v);
            u_max = max(u_max, v);
            valid++;
        }
    }
    if (valid == 0) { u_min = vec3(-1e10); u_max = vec3(1e10); }
#elif CLAMP_MODE == 3
    // Mode 3: Local — clamp to current cell's immediate neighbors (not back-trace)
    ivec3 bl = clamp(texel - ivec3(1,1,1), ivec3(0), ivec3(191));
    ivec3 tr = clamp(texel + ivec3(1,1,1), ivec3(0), ivec3(191));
    u_min = texelFetch(u_velocity, bl, 0).xyz;
    u_max = texelFetch(u_velocity, tr, 0).xyz;
    for (int k = bl.z; k <= tr.z; k++) {
    for (int j = bl.y; j <= tr.y; j++) {
    for (int i = bl.x; i <= tr.x; i++) {
        vec3 v = texelFetch(u_velocity, ivec3(i,j,k), 0).xyz;
        u_min = min(u_min, v);
        u_max = max(u_max, v);
    }}}
#endif
    u_new = clamp(u_new, u_min, u_max);

    // --- WALE SGS Model with SDF boundary awareness ---
    // If a neighbor is inside obstacle (sdf < 0), use current cell velocity
    // Zeroes gradient across boundary → prevents spurious SGS production at walls
    ivec3 iL = clamp(texel - ivec3(1,0,0), ivec3(0), ivec3(191));
    ivec3 iR = clamp(texel + ivec3(1,0,0), ivec3(0), ivec3(191));
    ivec3 iD = clamp(texel - ivec3(0,1,0), ivec3(0), ivec3(191));
    ivec3 iU = clamp(texel + ivec3(0,1,0), ivec3(0), ivec3(191));
    ivec3 iB = clamp(texel - ivec3(0,0,1), ivec3(0), ivec3(191));
    ivec3 iF = clamp(texel + ivec3(0,0,1), ivec3(0), ivec3(191));

    vec3 uL = texelFetch(u_velocity, iL, 0).xyz;
    if (texelFetch(u_sdf, iL, 0).r < 0.0) uL = u;
    vec3 uR = texelFetch(u_velocity, iR, 0).xyz;
    if (texelFetch(u_sdf, iR, 0).r < 0.0) uR = u;
    vec3 uD = texelFetch(u_velocity, iD, 0).xyz;
    if (texelFetch(u_sdf, iD, 0).r < 0.0) uD = u;
    vec3 uU = texelFetch(u_velocity, iU, 0).xyz;
    if (texelFetch(u_sdf, iU, 0).r < 0.0) uU = u;
    vec3 uB = texelFetch(u_velocity, iB, 0).xyz;
    if (texelFetch(u_sdf, iB, 0).r < 0.0) uB = u;
    vec3 uF = texelFetch(u_velocity, iF, 0).xyz;
    if (texelFetch(u_sdf, iF, 0).r < 0.0) uF = u;

    float gxx = (uR.x - uL.x) * 0.5;
    float gxy = (uU.x - uD.x) * 0.5;
    float gxz = (uF.x - uB.x) * 0.5;
    float gyx = (uR.y - uL.y) * 0.5;
    float gyy = (uU.y - uD.y) * 0.5;
    float gyz = (uF.y - uB.y) * 0.5;
    float gzx = (uR.z - uL.z) * 0.5;
    float gzy = (uU.z - uD.z) * 0.5;
    float gzz = (uF.z - uB.z) * 0.5;

    float Sxx = gxx;
    float Syy = gyy;
    float Szz = gzz;
    float Sxy = (gxy + gyx) * 0.5;
    float Sxz = (gxz + gzx) * 0.5;
    float Syz = (gyz + gzy) * 0.5;

    float S2 = Sxx*Sxx + Syy*Syy + Szz*Szz + 2.0*(Sxy*Sxy + Sxz*Sxz + Syz*Syz);

    float Gxx = gxx*gxx + gxy*gyx + gxz*gzx;
    float Gxy = gxx*gxy + gxy*gyy + gxz*gzy;
    float Gxz = gxx*gxz + gxy*gyz + gxz*gzz;
    float Gyx = gyx*gxx + gyy*gyx + gyz*gzx;
    float Gyy = gyx*gxy + gyy*gyy + gyz*gzy;
    float Gyz = gyx*gxz + gyy*gyz + gyz*gzz;
    float Gzx = gzx*gxx + gzy*gyx + gzz*gzx;
    float Gzy = gzx*gxy + gzy*gyy + gzz*gzy;
    float Gzz = gzx*gxz + gzy*gyz + gzz*gzz;

    float Gkk = Gxx + Gyy + Gzz;

    float Sdxx = Gxx - Gkk * 0.33333333;
    float Sdyy = Gyy - Gkk * 0.33333333;
    float Sdzz = Gzz - Gkk * 0.33333333;
    float Sdxy = (Gxy + Gyx) * 0.5;
    float Sdxz = (Gxz + Gzx) * 0.5;
    float Sdyz = (Gyz + Gzy) * 0.5;

    float Sd2 = Sdxx*Sdxx + Sdyy*Sdyy + Sdzz*Sdzz + 2.0*(Sdxy*Sdxy + Sdxz*Sdxz + Sdyz*Sdyz);

    float Cw = u_sgs_coeff;
    float delta = 1.0;
    float Cw2_d2 = Cw * Cw * delta * delta;
    float S2_safe = max(S2, 1e-20);
    float Sd2_safe = max(Sd2, 1e-20);
    float nu_t = Cw2_d2 * pow(Sd2_safe, 1.5) / (pow(S2_safe, 2.5) + pow(Sd2_safe, 1.25));

    // Viscous diffusion: u += dt * (nu_molecular + nu_sgs) * Laplacian(u)
    float lap_x = uL.x + uR.x + uD.x + uU.x + uB.x + uF.x - 6.0 * u_new.x;
    float lap_y = uL.y + uR.y + uD.y + uU.y + uB.y + uF.y - 6.0 * u_new.y;
    float lap_z = uL.z + uR.z + uD.z + uU.z + uB.z + uF.z - 6.0 * u_new.z;
    float total_nu = nu_t + u_viscosity;
    float visc_scale = min(u_dt * total_nu, 0.5);
    u_new += vec3(lap_x, lap_y, lap_z) * visc_scale;

    // No base dissipation
    fragColor = vec4(u_new * alpha, 1.0);
}
