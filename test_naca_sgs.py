import moderngl
import numpy as np
import time

ctx = moderngl.create_standalone_context()
print(f"GPU: {ctx.info['GL_RENDERER']}")

from cfd_sim import CFD_System
from geometry_utils import create_naca_airfoil, create_naca_airfoil_sdf
from force_utils import compute_lift_drag

for name, sgs_coeff in [('SGS=0.325', 0.325), ('SGS=0', 0.0)]:
    cfd = CFD_System(ctx)
    cfd.use_vcycle = True
    cfd.vcycle_interval = 1
    cfd.sgs_coeff = sgs_coeff

    grid = cfd.grid_size
    U_inf = 2.0
    cfd.inflow_vel = (U_inf, 0.0, 0.0)
    for prog in [cfd.prog_forces, cfd.prog_advection, cfd.prog_projection, cfd.prog_advection_density]:
        if 'u_inflow_vel' in prog:
            prog['u_inflow_vel'].value = tuple(cfd.inflow_vel)

    cfd.set_reynolds(5000, char_length=60.0)

    chord = 60.0
    mask = create_naca_airfoil(grid, grid*0.5, chord, 0.15, 0, grid, aoa_deg=5.0)
    cfd.set_obstacle(mask)
    sdf = create_naca_airfoil_sdf(grid, grid*0.5, chord, 0.15, 0, grid, aoa_deg=5.0)
    cfd.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).astype(np.float16).tobytes())

    vel_init = np.full((grid, grid, grid, 4), 0, dtype=np.float16)
    vel_init[..., 0] = U_inf
    mask_exp = np.repeat(mask[..., np.newaxis], 4, axis=3)
    vel_init = np.where(mask_exp > 0, 0, vel_init)
    vel_init_t = np.transpose(vel_init, (2, 1, 0, 3))
    cfd.tex_velocity_A.write(vel_init_t.tobytes())
    cfd.tex_velocity_B.write(vel_init_t.tobytes())

    obs_data = cfd.tex_obstacle.read()
    obs_mask = np.frombuffer(obs_data, dtype='u1').reshape((grid, grid, grid))
    ref_area = chord * grid
    q_inf = 0.5 * U_inf * U_inf

    t_start = time.perf_counter()
    for step in range(1, 301):
        cfd.step(force_radius=0.0)
        if step % 50 == 0 or step <= 3:
            pres_raw = cfd.tex_pressure_A.read()
            p = np.frombuffer(pres_raw, dtype='f4').reshape((grid, grid, grid))
            force, fp, fv = compute_lift_drag(p, obs_mask, inflow_vel=(U_inf, 0.0, 0.0))
            Cl = force[2] / (q_inf * ref_area)
            Cd = force[0] / (q_inf * ref_area)
            print(f"  [{name}] step={step:4d}  Cd={Cd:.4f}  Cl={Cl:.4f}  "
                  f"p=[{p.min():.4f},{p.max():.4f}]")
    elapsed = time.perf_counter() - t_start
    print(f"  [{name}] Done: 300 steps in {elapsed:.1f}s")
