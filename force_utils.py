import numpy as np


def compute_lift_drag(pressure, mask, velocity=None, viscosity=0.0, inflow_vel=(1.0, 0.0, 0.0)):
    """Integrates pressure and viscous forces over obstacle surface.

    Returns (total_force[3], force_pressure[3], force_viscous[3])
    where indices are (Drag_X, Span_Y, Lift_Z).
    """
    force_p = np.zeros(3)
    force_v = np.zeros(3)
    fluid_mask = (mask == 0)
    all_surface_pressures = []

    dim_pairs = [
        (2, 1, 0),  # X
        (1, 1, 1),  # Y
        (0, 1, 0),  # Z
    ]

    for f_idx, (dim, pm_scale, _) in enumerate(dim_pairs):
        slices_p = [slice(None)] * 3
        slices_m = [slice(None)] * 3
        slices_p[dim] = slice(None, -1)
        slices_m[dim] = slice(1, None)

        shift_p = [slice(None)] * 3
        shift_m = [slice(None)] * 3
        shift_p[dim] = slice(1, None)
        shift_m[dim] = slice(None, -1)

        fluid_sl_p = tuple(slices_p)
        fluid_sl_m = tuple(slices_m)
        obst_sl_p = tuple(shift_p)
        obst_sl_m = tuple(shift_m)

        is_sp_raw = fluid_mask[obst_sl_m] & (mask[obst_sl_p] == 1)
        is_sm_raw = fluid_mask[obst_sl_p] & (mask[obst_sl_m] == 1)
        if is_sp_raw.any():
            all_surface_pressures.extend(pressure[obst_sl_m][is_sp_raw].tolist())
        if is_sm_raw.any():
            all_surface_pressures.extend(pressure[obst_sl_p][is_sm_raw].tolist())

    p_mean = np.mean(all_surface_pressures) if all_surface_pressures else 0.0

    for f_idx, (dim, pm_scale, _) in enumerate(dim_pairs):
        slices_p = [slice(None)] * 3
        slices_m = [slice(None)] * 3
        slices_p[dim] = slice(None, -1)
        slices_m[dim] = slice(1, None)

        shift_p = [slice(None)] * 3
        shift_m = [slice(None)] * 3
        shift_p[dim] = slice(1, None)
        shift_m[dim] = slice(None, -1)

        fluid_sl_p = tuple(slices_p)
        fluid_sl_m = tuple(slices_m)
        obst_sl_p = tuple(shift_p)
        obst_sl_m = tuple(shift_m)

        is_sp = fluid_mask[obst_sl_m] & (mask[obst_sl_p] == 1)
        if is_sp.any():
            press_contrib = np.sum(pressure[obst_sl_m][is_sp] - p_mean)
            force_p[f_idx] += press_contrib

        is_sm = fluid_mask[obst_sl_p] & (mask[obst_sl_m] == 1)
        if is_sm.any():
            press_contrib = np.sum(pressure[obst_sl_p][is_sm] - p_mean)
            force_p[f_idx] -= press_contrib

    if velocity is not None and viscosity > 0.0:
        vel = velocity
        nu = viscosity

        for f_idx in range(3):
            dim = [2, 1, 0][f_idx]
            vel_comp = f_idx

            slices_p = [slice(None)] * 3
            slices_m = [slice(None)] * 3
            slices_p[dim] = slice(None, -1)
            slices_m[dim] = slice(1, None)

            shift_p = [slice(None)] * 3
            shift_m = [slice(None)] * 3
            shift_p[dim] = slice(1, None)
            shift_m[dim] = slice(None, -1)

            obst_sl_m = tuple(slices_p)
            obst_sl_p = tuple(shift_p)
            obst_sl_p2 = tuple(slices_m)
            obst_sl_m2 = tuple(shift_m)

            is_sp = fluid_mask[obst_sl_m] & (mask[obst_sl_p] == 1)
            if is_sp.any():
                u_n = vel[..., vel_comp][obst_sl_m][is_sp]
                force_v[f_idx] += np.sum(2.0 * nu * u_n)
                for k in range(3):
                    if k != f_idx:
                        u_t = vel[..., k][obst_sl_m][is_sp]
                        force_v[k] += np.sum(nu * u_t)

            is_sm = fluid_mask[obst_sl_p2] & (mask[obst_sl_m2] == 1)
            if is_sm.any():
                u_n = vel[..., vel_comp][obst_sl_p2][is_sm]
                force_v[f_idx] += np.sum(2.0 * nu * u_n)
                for k in range(3):
                    if k != f_idx:
                        u_t = vel[..., k][obst_sl_p2][is_sm]
                        force_v[k] += np.sum(nu * u_t)

    total_force = force_p + force_v
    return total_force, force_p, force_v
