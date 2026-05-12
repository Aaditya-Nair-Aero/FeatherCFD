#version 330 core

in vec2 v_uv;
in vec3 v_ray_dir;

out vec4 fragColor;

uniform sampler3D u_volume;
uniform usampler3D u_obstacle;
uniform sampler3D u_sdf;

uniform vec3 u_cam_pos;
uniform mat4 u_inv_view_proj;
uniform int  u_viz_mode;

uniform float u_density_scale;
uniform float u_step_size;
uniform int   u_num_steps;
uniform int   u_grid_size;

const vec3 BOX_MIN = vec3(-1.0);
const vec3 BOX_MAX = vec3( 1.0);

vec3 nasa_rainbow(float t) {
    vec3 c0 = vec3(0.00, 0.00, 0.45);
    vec3 c1 = vec3(0.00, 0.55, 0.85);
    vec3 c2 = vec3(0.00, 0.75, 0.25);
    vec3 c3 = vec3(0.95, 0.85, 0.05);
    vec3 c4 = vec3(0.90, 0.15, 0.05);
    if (t < 0.25) return mix(c0, c1, t / 0.25);
    if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
    return mix(c3, c4, (t - 0.75) / 0.25);
}

vec3 compute_vorticity(vec3 uv) {
    float d = 1.0 / float(u_grid_size);
    vec3 v_px = texture(u_volume, uv + vec3( d, 0.0, 0.0)).xyz;
    vec3 v_nx = texture(u_volume, uv + vec3(-d, 0.0, 0.0)).xyz;
    vec3 v_py = texture(u_volume, uv + vec3(0.0,  d, 0.0)).xyz;
    vec3 v_ny = texture(u_volume, uv + vec3(0.0, -d, 0.0)).xyz;
    vec3 v_pz = texture(u_volume, uv + vec3(0.0, 0.0,  d)).xyz;
    vec3 v_nz = texture(u_volume, uv + vec3(0.0, 0.0, -d)).xyz;
    vec3 dv_dx = (v_px - v_nx) * 0.5 * float(u_grid_size);
    vec3 dv_dy = (v_py - v_ny) * 0.5 * float(u_grid_size);
    vec3 dv_dz = (v_pz - v_nz) * 0.5 * float(u_grid_size);
    return vec3(
        dv_dy.z - dv_dz.y,
        dv_dz.x - dv_dx.z,
        dv_dx.y - dv_dy.x
    );
}

vec2 intersect_box(vec3 ro, vec3 rd) {
    vec3 inv_rd = 1.0 / rd;
    vec3 t0 = (BOX_MIN - ro) * inv_rd;
    vec3 t1 = (BOX_MAX - ro) * inv_rd;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tNear = max(max(tmin.x, tmin.y), tmin.z);
    float tFar  = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(tNear, tFar);
}

void main() {
    vec3 rd = normalize(v_ray_dir);
    vec3 ro = u_cam_pos;

    vec2 t_hit = intersect_box(ro, rd);
    if (t_hit.y < t_hit.x || t_hit.y < 0.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float t_start = max(t_hit.x, 0.0);
    float t_end   = t_hit.y;
    float step = (t_end - t_start) / float(u_num_steps);
    step = max(step, u_step_size);

    vec3 vol_col = vec3(0.0);
    float vol_a = 0.0;
    float max_vort = 1.0;
    float t_first_obs = 1e10;

    float t = t_start + step * 0.5;

    for (int i = 0; i < u_num_steps; i++) {
        if (t > t_end) break;

        vec3 pos = ro + rd * t;
        vec3 uv  = pos * 0.5 + 0.5;

        if (texture(u_obstacle, uv).r > 0u) {
            if (t_first_obs > 1e9) t_first_obs = t;
            t += step;
            continue;
        }

        if (t_first_obs < 1e9) break;

        vec4 data = texture(u_volume, uv);

        float val = 0.0;
        vec3  col = vec3(0.0);
        float alpha_scale = 1.0;

        if (u_viz_mode == 3) {
            float mag = length(data.xyz);
            float t_col = clamp(mag * 0.5, 0.0, 1.0);
            col = nasa_rainbow(t_col);
            val = mag;
            alpha_scale = 4.0;

            float sdf_dist = texture(u_sdf, uv).r;
            float render_radius = float(u_grid_size) * 0.1;
            if (sdf_dist > render_radius) {
                t += step;
                continue;
            }
            float prox = 1.0 - clamp(sdf_dist / render_radius, 0.0, 1.0);
            alpha_scale *= prox;
        } else if (u_viz_mode == 4) {
            float mach = data.a;
            float t_col = clamp(mach * 1.5, 0.0, 1.0);
            col = nasa_rainbow(t_col);
            val = mach;
            alpha_scale = 4.0;
        } else if (u_viz_mode == 5) {
            float d = 1.0 / float(u_grid_size);
            float mach_c = data.a;
            float mach_px = texture(u_volume, uv + vec3( d, 0.0, 0.0)).a;
            float mach_nx = texture(u_volume, uv + vec3(-d, 0.0, 0.0)).a;
            float schlieren = abs(mach_px - mach_nx) * float(u_grid_size) * 5.0;
            float t_col = clamp(schlieren, 0.0, 1.0);
            col = nasa_rainbow(t_col);
            val = schlieren;
            alpha_scale = 6.0;
        } else if (u_viz_mode == 0) {
            float mag = length(data.xyz);
            float t_col = clamp(mag * 0.5, 0.0, 1.0);
            col = nasa_rainbow(t_col);
            val = mag;
            alpha_scale = 4.0;
        } else if (u_viz_mode == 1) {
            float den = data.a;
            col = mix(vec3(0.1, 0.2, 0.4), vec3(0.9, 0.9, 1.0), den);
            val = den;
            alpha_scale = 4.0;
        } else {
            vec3 vort = compute_vorticity(uv);
            float vmag = length(vort);
            max_vort = max(max_vort, vmag);
            float t_col = clamp(vmag / max_vort, 0.0, 1.0);
            col = nasa_rainbow(t_col);
            val = vmag;
            alpha_scale = 4.0;
        }

        if (val > 0.001 && alpha_scale > 0.001) {
            float alpha = clamp(val * u_density_scale * step * alpha_scale, 0.0, 0.15);
            vol_col += (1.0 - vol_a) * alpha * col;
            vol_a   += (1.0 - vol_a) * alpha;
        }

        t += step;
        if (vol_a > 0.995) break;
    }

    vec3 bg = vec3(0.015, 0.02, 0.04);
    vec3 final_col = vol_col + (1.0 - vol_a) * bg;

    if (t_first_obs < 1e9) {
        float obs_alpha = 0.7;
        vec3 obs_col = vec3(0.3, 0.3, 0.3);
        final_col = mix(final_col, obs_col, obs_alpha);
    }

    fragColor = vec4(final_col, 1.0);
}
