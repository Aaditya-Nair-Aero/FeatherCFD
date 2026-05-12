#version 330 core

flat in int v_layer;
out vec4 fragColor1; // (rho, mom_x, mom_y, mom_z)
out vec4 fragColor2; // (E, 0, 0, 0)

uniform sampler3D u_U1; // (rho, mom_x, mom_y, mom_z)
uniform sampler3D u_U2; // (E)
uniform sampler3D u_U1_old; // U^n for RK stage combination
uniform sampler3D u_U2_old;
uniform usampler3D u_obstacle; // obstacle mask (1=obstacle)
uniform sampler3D u_sdf; // signed distance field for slip-wall reflection

uniform float u_dt;
uniform float u_dx;
uniform float u_gamma;
uniform int u_grid_size;

// SSP-RK3 stage coefficients
uniform float u_rk_a; // coeff for U_old
uniform float u_rk_b; // coeff for U_current
uniform float u_rk_c; // coeff for dt * L(U_current)

// Inflow / freestream conditions
uniform vec4 u_inflow1; // (rho, mom_x, mom_y, mom_z)
uniform vec4 u_inflow2; // (E, 0, 0, 0)
uniform vec4 u_inflow3; // (p_inf/(gamma-1), 0, 0, 0) internal energy for obstacles
// BC type: 0=reflective walls (Sod), 1=inflow/outflow in X (external)
uniform int u_bc_type;

// --- helper: read conserved vars at (x, y, layer) ---
vec4 get_U1(ivec3 t) { return texelFetch(u_U1, t, 0); }
float get_E(ivec3 t) { return texelFetch(u_U2, t, 0).r; }
vec4 get_U1_old(ivec3 t) { return texelFetch(u_U1_old, t, 0); }
float get_E_old(ivec3 t) { return texelFetch(u_U2_old, t, 0).r; }

// --- primitives from conserved ---
void cons_to_prim(vec4 U1, float E, float gamma,
                  out float rho, out float u, out float v, out float w,
                  out float p, out float c) {
    rho = max(U1.x, 1e-8);
    u = U1.y / rho;
    v = U1.z / rho;
    w = U1.w / rho;
    float ke = 0.5 * rho * (u*u + v*v + w*w);
    p = max((gamma - 1.0) * (E - ke), 1e-8);
    c = sqrt(gamma * p / rho);
}

// --- minmod limiter ---
float minmod(float a, float b) {
    if (a * b <= 0.0) return 0.0;
    return sign(a) * min(abs(a), abs(b));
}

// --- MUSCL reconstruction at face between cell i and i+1 ---
// Given cells i-1, i, i+1, i+2 in the stencil
void muscl_reconstruct(float q_imm1, float q_i, float q_ip1, float q_ip2,
                       out float q_L, out float q_R) {
    float dq_L = q_i - q_imm1;
    float dq_R = q_ip1 - q_i;
    float dq_R2 = q_ip2 - q_ip1;
    float slope_L = minmod(dq_L, dq_R);
    float slope_R = minmod(dq_R, dq_R2);
    q_L = q_i + 0.5 * slope_L;
    q_R = q_ip1 - 0.5 * slope_R;
}

// --- HLLC flux in a given direction ---
// For X-direction: normal = 0 (x), tangent1 = 1 (y), tangent2 = 2 (z)
// For Y-direction: normal = 1, tangent1 = 0, tangent2 = 2
// For Z-direction: normal = 2, tangent1 = 0, tangent2 = 1
void hllc_flux(float rhoL, float uL, float vL, float wL, float pL, float cL,
               float rhoR, float uR, float vR, float wR, float pR, float cR,
               float gamma,
               out float F_rho, out float F_mom_n, out float F_mom_t1, out float F_mom_t2,
               out float F_E) {
    // Wave speed estimates (Einfeldt)
    float S_L = min(uL - cL, uR - cR);
    float S_R = max(uL + cL, uR + cR);

    // Compute S_* (contact wave speed)
    float rho_uL = rhoL * uL;
    float rho_uR = rhoR * uR;
    float S_star = (pR - pL + rho_uL*(S_L - uL) - rho_uR*(S_R - uR))
                 / (rhoL*(S_L - uL) - rhoR*(S_R - uR));

    // Left flux
    float F_L_rho   = rho_uL;
    float F_L_mom_n = rho_uL * uL + pL;
    float F_L_mom_t1 = rho_uL * vL;
    float F_L_mom_t2 = rho_uL * wL;
    float F_L_E     = uL * (0.5*rhoL*(uL*uL+vL*vL+wL*wL) + pL*gamma/(gamma-1.0));

    // Right flux
    float F_R_rho   = rho_uR;
    float F_R_mom_n = rho_uR * uR + pR;
    float F_R_mom_t1 = rho_uR * vR;
    float F_R_mom_t2 = rho_uR * wR;
    float F_R_E     = uR * (0.5*rhoR*(uR*uR+vR*vR+wR*wR) + pR*gamma/(gamma-1.0));

    if (S_L >= 0.0) {
        F_rho = F_L_rho; F_mom_n = F_L_mom_n;
        F_mom_t1 = F_L_mom_t1; F_mom_t2 = F_L_mom_t2; F_E = F_L_E;
    } else if (S_star >= 0.0) {
        // Left star state
        float factor = rhoL * (S_L - uL) / (S_L - S_star);
        float U_star_rho = factor;
        float U_star_mom_n = factor * S_star;
        float U_star_mom_t1 = factor * vL;
        float U_star_mom_t2 = factor * wL;
        float U_star_E = factor * (pL/((gamma-1.0)*rhoL) + 0.5*(uL*uL+vL*vL+wL*wL)
                                   + (S_star-uL)*(S_star + pL/(rhoL*(S_L-uL))));

        F_rho = F_L_rho + S_L * (U_star_rho - rhoL);
        F_mom_n = F_L_mom_n + S_L * (U_star_mom_n - rho_uL);
        F_mom_t1 = F_L_mom_t1 + S_L * (U_star_mom_t1 - rhoL*vL);
        F_mom_t2 = F_L_mom_t2 + S_L * (U_star_mom_t2 - rhoL*wL);
        F_E = F_L_E + S_L * (U_star_E - (0.5*rhoL*(uL*uL+vL*vL+wL*wL) + pL/(gamma-1.0)));
    } else if (S_R >= 0.0) {
        // Right star state
        float factor = rhoR * (S_R - uR) / (S_R - S_star);
        float U_star_rho = factor;
        float U_star_mom_n = factor * S_star;
        float U_star_mom_t1 = factor * vR;
        float U_star_mom_t2 = factor * wR;
        float U_star_E = factor * (pR/((gamma-1.0)*rhoR) + 0.5*(uR*uR+vR*vR+wR*wR)
                                   + (S_star-uR)*(S_star + pR/(rhoR*(S_R-uR))));

        F_rho = F_R_rho + S_R * (U_star_rho - rhoR);
        F_mom_n = F_R_mom_n + S_R * (U_star_mom_n - rho_uR);
        F_mom_t1 = F_R_mom_t1 + S_R * (U_star_mom_t1 - rhoR*vR);
        F_mom_t2 = F_R_mom_t2 + S_R * (U_star_mom_t2 - rhoR*wR);
        F_E = F_R_E + S_R * (U_star_E - (0.5*rhoR*(uR*uR+vR*vR+wR*wR) + pR/(gamma-1.0)));
    } else {
        F_rho = F_R_rho; F_mom_n = F_R_mom_n;
        F_mom_t1 = F_R_mom_t1; F_mom_t2 = F_R_mom_t2; F_E = F_R_E;
    }
}

// --- Read neighbor with boundary condition ---
// Handles domain boundaries and solid walls
ivec3 clamp_t(ivec3 t) {
    return ivec3(clamp(t.x, 0, u_grid_size-1),
                 clamp(t.y, 0, u_grid_size-1),
                 clamp(t.z, 0, u_grid_size-1));
}

void read_neighbor(ivec3 t, bool is_solid_normal,
                   out vec4 U1, out float E) {
    ivec3 tc = clamp_t(t);
    U1 = get_U1(tc);
    E = get_E(tc);
    // Solid wall reflection for normal component
    if (is_solid_normal) {
        U1.y = -U1.y; // negate normal momentum
    }
}

void main() {
    ivec3 texel = ivec3(gl_FragCoord.xy, v_layer);
    int g = u_grid_size;

    // ============================================================
    // Read current state from B (evolving buffer)
    // ============================================================
    vec4 U1_c = get_U1(texel);
    float E_c = get_E(texel);

    // ============================================================
    // Read old state from A (U^n) for RK combination
    // ============================================================
    vec4 U1_old = get_U1_old(texel);
    float E_old = get_E_old(texel);

    // ============================================================
    // X-direction stencil: offsets -2, -1, 0, +1, +2
    // BC type: 0=reflective wall, 1=inflow(x<0)/outflow(x>=g)
    // ============================================================
    ivec3 t;
    vec4 U1[5]; float E[5];
    for (int d = -2; d <= 2; d++) {
        int idx = d + 2;
        t = ivec3(texel.x + d, texel.y, texel.z);
        if (t.x < 0) {
            if (u_bc_type > 0) {
                U1[idx] = u_inflow1;
                E[idx] = u_inflow2.r;
            } else {
                t.x = 0;
                U1[idx] = get_U1(t);
                U1[idx].y = -U1[idx].y;
                E[idx] = get_E(t);
            }
        } else if (t.x >= g) {
            if (u_bc_type > 0) {
                t.x = g - 1;
                U1[idx] = get_U1(t);
                E[idx] = get_E(t);
            } else {
                t.x = g - 1;
                U1[idx] = get_U1(t);
                U1[idx].y = -U1[idx].y;
                E[idx] = get_E(t);
            }
        } else {
            U1[idx] = get_U1(t);
            E[idx] = get_E(t);
        }
    }

    // MUSCL reconstruct in X
    // For face between cells [0](-2) and [1](-1): not needed (left of cell)
    // For face between cells [1](-1) and [2](0): left face of our cell F_{i-1/2}
    // For face between cells [2](0) and [3](+1): right face F_{i+1/2}
    // For face between cells [3](+1) and [4](+2): not needed (right of cell)

    // Face i-1/2: left state from cell i-1, right state from cell i
    float rho_Lx, u_Lx, v_Lx, w_Lx, p_Lx, c_Lx;
    float rho_Rx, u_Rx, v_Rx, w_Rx, p_Rx, c_Rx;
    float rho_m1, u_m1, v_m1, w_m1, p_m1, c_m1;
    float rho_0, u_0, v_0, w_0, p_0, c_0;
    float rho_1, u_1, v_1, w_1, p_1, c_1;
    float rho_2, u_2, v_2, w_2, p_2, c_2;

    cons_to_prim(U1[0], E[0], u_gamma, rho_m1, u_m1, v_m1, w_m1, p_m1, c_m1);
    cons_to_prim(U1[1], E[1], u_gamma, rho_0, u_0, v_0, w_0, p_0, c_0);
    cons_to_prim(U1[2], E[2], u_gamma, rho_1, u_1, v_1, w_1, p_1, c_1);
    cons_to_prim(U1[3], E[3], u_gamma, rho_2, u_2, v_2, w_2, p_2, c_2);

    // Face i-1/2: reconstruct
    float q_L_rho, q_R_rho, q_L_u, q_R_u, q_L_v, q_R_v, q_L_w, q_R_w, q_L_p, q_R_p;
    muscl_reconstruct(rho_m1, rho_0, rho_1, rho_2, q_L_rho, q_R_rho);
    muscl_reconstruct(u_m1, u_0, u_1, u_2, q_L_u, q_R_u);
    muscl_reconstruct(v_m1, v_0, v_1, v_2, q_L_v, q_R_v);
    muscl_reconstruct(w_m1, w_0, w_1, w_2, q_L_w, q_R_w);
    muscl_reconstruct(p_m1, p_0, p_1, p_2, q_L_p, q_R_p);

    // Sound speed for reconstructed states
    float c_Lx_f = sqrt(u_gamma * q_L_p / max(q_L_rho, 1e-8));
    float c_Rx_f = sqrt(u_gamma * q_R_p / max(q_R_rho, 1e-8));

    // HLLC flux at face i-1/2 (left face of cell)
    float F_L_rho, F_L_mom_x, F_L_mom_y, F_L_mom_z, F_L_E;
    hllc_flux(q_L_rho, q_L_u, q_L_v, q_L_w, q_L_p, c_Lx_f,
              q_R_rho, q_R_u, q_R_v, q_R_w, q_R_p, c_Rx_f,
              u_gamma,
              F_L_rho, F_L_mom_x, F_L_mom_y, F_L_mom_z, F_L_E);

    // Face i+1/2: reconstruct from cells 0, 1, 2, 3 (indices 1,2,3,4)
    cons_to_prim(U1[0], E[0], u_gamma, rho_m1, u_m1, v_m1, w_m1, p_m1, c_m1);
    cons_to_prim(U1[1], E[1], u_gamma, rho_0, u_0, v_0, w_0, p_0, c_0);
    // Already have rho_1, u_1, etc. from above (U1[2])
    // Already have rho_2, u_2, etc. from above (U1[3])

    float rho_3, u_3, v_3, w_3, p_3, c_3;
    cons_to_prim(U1[4], E[4], u_gamma, rho_3, u_3, v_3, w_3, p_3, c_3);

    muscl_reconstruct(rho_0, rho_1, rho_2, rho_3, q_L_rho, q_R_rho);
    muscl_reconstruct(u_0, u_1, u_2, u_3, q_L_u, q_R_u);
    muscl_reconstruct(v_0, v_1, v_2, v_3, q_L_v, q_R_v);
    muscl_reconstruct(w_0, w_1, w_2, w_3, q_L_w, q_R_w);
    muscl_reconstruct(p_0, p_1, p_2, p_3, q_L_p, q_R_p);

    c_Lx_f = sqrt(u_gamma * q_L_p / max(q_L_rho, 1e-8));
    c_Rx_f = sqrt(u_gamma * q_R_p / max(q_R_rho, 1e-8));

    float F_R_rho, F_R_mom_x, F_R_mom_y, F_R_mom_z, F_R_E;
    hllc_flux(q_L_rho, q_L_u, q_L_v, q_L_w, q_L_p, c_Lx_f,
              q_R_rho, q_R_u, q_R_v, q_R_w, q_R_p, c_Rx_f,
              u_gamma,
              F_R_rho, F_R_mom_x, F_R_mom_y, F_R_mom_z, F_R_E);

    // ============================================================
    // Y-direction stencil
    // ============================================================
    vec4 U1_y[5]; float E_y[5];
    for (int d = -2; d <= 2; d++) {
        int idx = d + 2;
        t = ivec3(texel.x, texel.y + d, texel.z);
        bool wall = (t.y < 0 || t.y >= g);
        t = clamp_t(t);
        U1_y[idx] = get_U1(t);
        E_y[idx] = get_E(t);
        if (wall && u_bc_type == 0) {
            U1_y[idx].z = -U1_y[idx].z;
        }
    }

    // MUSCL in Y — same pattern, using cells -2,-1,0,+1,+2 in Y
    // We need to use the Y neighbor data. For the HLLC flux in Y:
    // normal = Y direction (mom_z becomes the tangential, mom_y is normal)

    // Convert to primitives for Y-stencil at offsets -2,-1,0,+1,+2
    float rho_y_m1, u_y_m1, v_y_m1, w_y_m1, p_y_m1;
    float rho_y_0,  u_y_0,  v_y_0,  w_y_0,  p_y_0;
    float rho_y_1,  u_y_1,  v_y_1,  w_y_1,  p_y_1;
    float rho_y_2,  u_y_2,  v_y_2,  w_y_2,  p_y_2;
    float c_y;
    cons_to_prim(U1_y[0], E_y[0], u_gamma, rho_y_m1, u_y_m1, v_y_m1, w_y_m1, p_y_m1, c_y);
    cons_to_prim(U1_y[1], E_y[1], u_gamma, rho_y_0,  u_y_0,  v_y_0,  w_y_0,  p_y_0, c_y);
    cons_to_prim(U1_y[2], E_y[2], u_gamma, rho_y_1,  u_y_1,  v_y_1,  w_y_1,  p_y_1, c_y);
    cons_to_prim(U1_y[3], E_y[3], u_gamma, rho_y_2,  u_y_2,  v_y_2,  w_y_2,  p_y_2, c_y);

    // Face j-1/2: reconstruction
    float q_L_rho_y, q_R_rho_y, q_L_u_y, q_R_u_y, q_L_v_y, q_R_v_y, q_L_w_y, q_R_w_y, q_L_p_y, q_R_p_y;
    // For Y-direction flux: normal = v, tangents = u, w
    muscl_reconstruct(rho_y_m1, rho_y_0, rho_y_1, rho_y_2, q_L_rho_y, q_R_rho_y);
    muscl_reconstruct(v_y_m1, v_y_0, v_y_1, v_y_2, q_L_v_y, q_R_v_y); // normal
    muscl_reconstruct(u_y_m1, u_y_0, u_y_1, u_y_2, q_L_u_y, q_R_u_y); // tangent1
    muscl_reconstruct(w_y_m1, w_y_0, w_y_1, w_y_2, q_L_w_y, q_R_w_y); // tangent2
    muscl_reconstruct(p_y_m1, p_y_0, p_y_1, p_y_2, q_L_p_y, q_R_p_y);

    float c_Ly_f = sqrt(u_gamma * q_L_p_y / max(q_L_rho_y, 1e-8));
    float c_Ry_f = sqrt(u_gamma * q_R_p_y / max(q_R_rho_y, 1e-8));

    // HLLC in Y: normal=v, tangent1=u, tangent2=w
    float Fy_L_rho, Fy_L_mom_x, Fy_L_mom_y, Fy_L_mom_z, Fy_L_E;
    hllc_flux(q_L_rho_y, q_L_v_y, q_L_u_y, q_L_w_y, q_L_p_y, c_Ly_f,
              q_R_rho_y, q_R_v_y, q_R_u_y, q_R_w_y, q_R_p_y, c_Ry_f,
              u_gamma,
              Fy_L_rho, Fy_L_mom_y, Fy_L_mom_x, Fy_L_mom_z, Fy_L_E);
    // Fy_L_mom_x = tangent1 momentum (was mom_n=normal in hllc, we passed v as normal, u as t1)
    // So: hllc returns (F_rho, F_mom_n, F_mom_t1, F_mom_t2)
    // We passed: normal=v, t1=u, t2=w
    // So: Fy_L_mom_y = F_mom_n (normal = v momentum)
    //     Fy_L_mom_x = F_mom_t1 (tangent1 = u momentum)
    //     Fy_L_mom_z = F_mom_t2 (tangent2 = w momentum)

    // Face j+1/2: reconstruction
    float rho_y_3, u_y_3, v_y_3, w_y_3, p_y_3;
    cons_to_prim(U1_y[4], E_y[4], u_gamma, rho_y_3, u_y_3, v_y_3, w_y_3, p_y_3, c_y);

    muscl_reconstruct(rho_y_0, rho_y_1, rho_y_2, rho_y_3, q_L_rho_y, q_R_rho_y);
    muscl_reconstruct(v_y_0, v_y_1, v_y_2, v_y_3, q_L_v_y, q_R_v_y);
    muscl_reconstruct(u_y_0, u_y_1, u_y_2, u_y_3, q_L_u_y, q_R_u_y);
    muscl_reconstruct(w_y_0, w_y_1, w_y_2, w_y_3, q_L_w_y, q_R_w_y);
    muscl_reconstruct(p_y_0, p_y_1, p_y_2, p_y_3, q_L_p_y, q_R_p_y);

    c_Ly_f = sqrt(u_gamma * q_L_p_y / max(q_L_rho_y, 1e-8));
    c_Ry_f = sqrt(u_gamma * q_R_p_y / max(q_R_rho_y, 1e-8));

    float Fy_R_rho, Fy_R_mom_x, Fy_R_mom_y, Fy_R_mom_z, Fy_R_E;
    hllc_flux(q_L_rho_y, q_L_v_y, q_L_u_y, q_L_w_y, q_L_p_y, c_Ly_f,
              q_R_rho_y, q_R_v_y, q_R_u_y, q_R_w_y, q_R_p_y, c_Ry_f,
              u_gamma,
              Fy_R_rho, Fy_R_mom_y, Fy_R_mom_x, Fy_R_mom_z, Fy_R_E);

    // ============================================================
    // Z-direction stencil — Z=0 always reflective (wall), Z=top extrapolated
    // ============================================================
    vec4 U1_z[5]; float E_z[5];
    for (int d = -2; d <= 2; d++) {
        int idx = d + 2;
        t = ivec3(texel.x, texel.y, texel.z + d);
        bool wall = (t.z < 0);
        if (t.z >= g) {
            t.z = g - 1;
        } else if (t.z < 0) {
            t.z = 0;
        }
        U1_z[idx] = get_U1(t);
        E_z[idx] = get_E(t);
        if (wall) {
            U1_z[idx].w = -U1_z[idx].w; // reflect Z-momentum at bottom wall
        }
    }

    float rho_z_m1, u_z_m1, v_z_m1, w_z_m1, p_z_m1;
    float rho_z_0,  u_z_0,  v_z_0,  w_z_0,  p_z_0;
    float rho_z_1,  u_z_1,  v_z_1,  w_z_1,  p_z_1;
    float rho_z_2,  u_z_2,  v_z_2,  w_z_2,  p_z_2;
    float c_z;
    cons_to_prim(U1_z[0], E_z[0], u_gamma, rho_z_m1, u_z_m1, v_z_m1, w_z_m1, p_z_m1, c_z);
    cons_to_prim(U1_z[1], E_z[1], u_gamma, rho_z_0,  u_z_0,  v_z_0,  w_z_0,  p_z_0, c_z);
    cons_to_prim(U1_z[2], E_z[2], u_gamma, rho_z_1,  u_z_1,  v_z_1,  w_z_1,  p_z_1, c_z);
    cons_to_prim(U1_z[3], E_z[3], u_gamma, rho_z_2,  u_z_2,  v_z_2,  w_z_2,  p_z_2, c_z);

    // Face k-1/2
    float q_L_rho_z, q_R_rho_z, q_L_u_z, q_R_u_z, q_L_v_z, q_R_v_z, q_L_w_z, q_R_w_z, q_L_p_z, q_R_p_z;
    muscl_reconstruct(rho_z_m1, rho_z_0, rho_z_1, rho_z_2, q_L_rho_z, q_R_rho_z);
    muscl_reconstruct(w_z_m1, w_z_0, w_z_1, w_z_2, q_L_w_z, q_R_w_z); // normal
    muscl_reconstruct(u_z_m1, u_z_0, u_z_1, u_z_2, q_L_u_z, q_R_u_z); // t1
    muscl_reconstruct(v_z_m1, v_z_0, v_z_1, v_z_2, q_L_v_z, q_R_v_z); // t2
    muscl_reconstruct(p_z_m1, p_z_0, p_z_1, p_z_2, q_L_p_z, q_R_p_z);

    float c_Lz_f = sqrt(u_gamma * q_L_p_z / max(q_L_rho_z, 1e-8));
    float c_Rz_f = sqrt(u_gamma * q_R_p_z / max(q_R_rho_z, 1e-8));

    float Fz_L_rho, Fz_L_mom_x, Fz_L_mom_y, Fz_L_mom_z, Fz_L_E;
    hllc_flux(q_L_rho_z, q_L_w_z, q_L_u_z, q_L_v_z, q_L_p_z, c_Lz_f,
              q_R_rho_z, q_R_w_z, q_R_u_z, q_R_v_z, q_R_p_z, c_Rz_f,
              u_gamma,
              Fz_L_rho, Fz_L_mom_z, Fz_L_mom_x, Fz_L_mom_y, Fz_L_E);

    // Face k+1/2
    float rho_z_3, u_z_3, v_z_3, w_z_3, p_z_3;
    cons_to_prim(U1_z[4], E_z[4], u_gamma, rho_z_3, u_z_3, v_z_3, w_z_3, p_z_3, c_z);

    muscl_reconstruct(rho_z_0, rho_z_1, rho_z_2, rho_z_3, q_L_rho_z, q_R_rho_z);
    muscl_reconstruct(w_z_0, w_z_1, w_z_2, w_z_3, q_L_w_z, q_R_w_z);
    muscl_reconstruct(u_z_0, u_z_1, u_z_2, u_z_3, q_L_u_z, q_R_u_z);
    muscl_reconstruct(v_z_0, v_z_1, v_z_2, v_z_3, q_L_v_z, q_R_v_z);
    muscl_reconstruct(p_z_0, p_z_1, p_z_2, p_z_3, q_L_p_z, q_R_p_z);

    c_Lz_f = sqrt(u_gamma * q_L_p_z / max(q_L_rho_z, 1e-8));
    c_Rz_f = sqrt(u_gamma * q_R_p_z / max(q_R_rho_z, 1e-8));

    float Fz_R_rho, Fz_R_mom_x, Fz_R_mom_y, Fz_R_mom_z, Fz_R_E;
    hllc_flux(q_L_rho_z, q_L_w_z, q_L_u_z, q_L_v_z, q_L_p_z, c_Lz_f,
              q_R_rho_z, q_R_w_z, q_R_u_z, q_R_v_z, q_R_p_z, c_Rz_f,
              u_gamma,
              Fz_R_rho, Fz_R_mom_z, Fz_R_mom_x, Fz_R_mom_y, Fz_R_E);

    // ============================================================
    // Flux divergence: L(U) = -dF/dx - dG/dy - dH/dz
    // ============================================================
    float inv_dx = 1.0 / u_dx;
    float L_rho   = -(F_R_rho - F_L_rho   + Fy_R_rho - Fy_L_rho   + Fz_R_rho - Fz_L_rho) * inv_dx;
    float L_mom_x = -(F_R_mom_x - F_L_mom_x + Fy_R_mom_x - Fy_L_mom_x + Fz_R_mom_x - Fz_L_mom_x) * inv_dx;
    float L_mom_y = -(F_R_mom_y - F_L_mom_y + Fy_R_mom_y - Fy_L_mom_y + Fz_R_mom_y - Fz_L_mom_y) * inv_dx;
    float L_mom_z = -(F_R_mom_z - F_L_mom_z + Fy_R_mom_z - Fy_L_mom_z + Fz_R_mom_z - Fz_L_mom_z) * inv_dx;
    float L_E     = -(F_R_E - F_L_E   + Fy_R_E - Fy_L_E   + Fz_R_E - Fz_L_E) * inv_dx;

    // ============================================================
    // SSP-RK3 stage update:
    // U_new = u_rk_a * U_old + u_rk_b * U_current + u_rk_c * dt * L(U_current)
    // ============================================================
    vec4 U1_new;
    U1_new.x = u_rk_a * U1_old.x + u_rk_b * U1_c.x + u_rk_c * u_dt * L_rho;
    U1_new.y = u_rk_a * U1_old.y + u_rk_b * U1_c.y + u_rk_c * u_dt * L_mom_x;
    U1_new.z = u_rk_a * U1_old.z + u_rk_b * U1_c.z + u_rk_c * u_dt * L_mom_y;
    U1_new.w = u_rk_a * U1_old.w + u_rk_b * U1_c.w + u_rk_c * u_dt * L_mom_z;
    float E_new = u_rk_a * E_old + u_rk_b * E_c + u_rk_c * u_dt * L_E;

    // ============================================================
    // Positivity preservation
    // ============================================================
    float rho_new = max(U1_new.x, 1e-6);
    float px = (u_gamma - 1.0) * (E_new - 0.5 * (U1_new.y*U1_new.y + U1_new.z*U1_new.z + U1_new.w*U1_new.w) / max(rho_new, 1e-8));
    if (px < 1e-8) {
        // Reconstruct E from minimum pressure
        E_new = 1e-8 / (u_gamma - 1.0) + 0.5 * (U1_new.y*U1_new.y + U1_new.z*U1_new.z + U1_new.w*U1_new.w) / max(rho_new, 1e-8);
    }
    U1_new.x = rho_new;

    // ============================================================
    // Obstacle cells: slip wall (zero normal velocity, preserve tangential)
    // Uses SDF gradient to determine wall normal direction.
    // ============================================================
    uint obs = texelFetch(u_obstacle, texel, 0).r;
    if (obs > 0u) {
        float sx = texelFetch(u_sdf, ivec3(min(texel.x+1, u_grid_size-1), texel.y, texel.z), 0).r
                 - texelFetch(u_sdf, ivec3(max(texel.x-1, 0), texel.y, texel.z), 0).r;
        float sy = texelFetch(u_sdf, ivec3(texel.x, min(texel.y+1, u_grid_size-1), texel.z), 0).r
                 - texelFetch(u_sdf, ivec3(texel.x, max(texel.y-1, 0), texel.z), 0).r;
        float sz = texelFetch(u_sdf, ivec3(texel.x, texel.y, min(texel.z+1, u_grid_size-1)), 0).r
                 - texelFetch(u_sdf, ivec3(texel.x, texel.y, max(texel.z-1, 0)), 0).r;
        float nlen = sqrt(sx*sx + sy*sy + sz*sz);
        if (nlen > 1e-8) {
            sx /= nlen; sy /= nlen; sz /= nlen;
            float vn = U1_new.y * sx + U1_new.z * sy + U1_new.w * sz;
            U1_new.y -= vn * sx;
            U1_new.z -= vn * sy;
            U1_new.w -= vn * sz;
        }
        U1_new.x = u_inflow1.r;
        E_new = u_inflow3.r;
    }

    fragColor1 = U1_new;
    fragColor2 = vec4(E_new, 0.0, 0.0, 0.0);
}
