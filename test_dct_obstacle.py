import moderngl
import numpy as np
import time
import sys

ctx = moderngl.create_standalone_context()
print(f"GPU: {ctx.info['GL_RENDERER']}")

from cfd_sim import CFD_System
from geometry_utils import create_cylinder, create_cylinder_sdf

cfd = CFD_System(ctx)
cfd.use_dct = True
cfd.dct_post_smooth = 8

cfd.inflow_vel = (2.0, 0.0, 0.0)
for prog in [cfd.prog_forces, cfd.prog_advection, cfd.prog_projection, cfd.prog_advection_density]:
    if 'u_inflow_vel' in prog:
        prog['u_inflow_vel'].value = tuple(cfd.inflow_vel)

grid = cfd.grid_size
cfd.set_reynolds(1000, char_length=30.0)

print("Creating cylinder obstacle (D=30)...")
mask = create_cylinder(grid, grid/2, grid/2, 15.0, 1, grid-1)
cfd.set_obstacle(mask)
sdf = create_cylinder_sdf(grid, grid/2, grid/2, 15.0, 1, grid-1)
cfd.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).astype(np.float16).tobytes())

vel_init = np.full((grid, grid, grid, 4), 0, dtype=np.float16)
vel_init[..., 0] = 2.0
mask_exp = np.repeat(mask[..., np.newaxis], 4, axis=3)
vel_init = np.where(mask_exp > 0, 0, vel_init)
vel_init_t = np.transpose(vel_init, (2, 1, 0, 3))
cfd.tex_velocity_A.write(vel_init_t.tobytes())
cfd.tex_velocity_B.write(vel_init_t.tobytes())

obs_mask = mask

print(f"Grid: {grid}³, dt: {cfd.dt:.6f}, Re=1000, DCT iters: {cfd.dct_iterations}")

t_start = time.perf_counter()
for step in range(1, 51):
    t0 = time.perf_counter()
    cfd.step(force_radius=0.0)
    t1 = time.perf_counter()

    pres_raw = cfd.tex_pressure_A.read()
    pressure = np.frombuffer(pres_raw, dtype='f4').reshape((grid, grid, grid))
    p_min, p_max = float(pressure.min()), float(pressure.max())
    p_mean = float(pressure.mean())

    vel_raw = cfd.tex_velocity_A.read()
    vel = np.frombuffer(vel_raw, dtype=np.float16).reshape((grid, grid, grid, 4))
    speed = np.sqrt(vel[..., 0]**2 + vel[..., 1]**2 + vel[..., 2]**2)
    speed = np.nan_to_num(speed, nan=0.0)
    has_nan = np.any(np.isnan(vel.astype(np.float32)))
    has_inf = np.any(np.isinf(vel.astype(np.float32)))

    vel_f = vel[:, :, :, :3].astype(np.float64)
    div = (vel_f[1:-1,1:-1,2:,0] - vel_f[1:-1,1:-1,1:-1,0]) + \
          (vel_f[1:-1,2:,1:-1,1] - vel_f[1:-1,1:-1,1:-1,1]) + \
          (vel_f[2:,1:-1,1:-1,2] - vel_f[1:-1,1:-1,1:-1,2])
    mean_div = float(np.mean(np.abs(div)))
    max_div = float(np.max(np.abs(div)))

    print(f"  step={step:3d}  p=[{p_min:+.4f}, {p_max:+.4f}] mean={p_mean:+.6f}  "
          f"|u|_max={np.max(speed):.4f}  |div|_mean={mean_div:.6e} max={max_div:.4f}  "
          f"time={t1-t0:.3f}s  nan={has_nan}")

    if has_nan or has_inf or np.isnan(p_min):
        print("OVERFLOW!")
        sys.exit(1)

elapsed = time.perf_counter() - t_start
print(f"\nPassed: 50 steps in {elapsed:.1f}s ({elapsed/50:.3f}s/step)")
print(f"Final pressure: min={p_min:.4f}, max={p_max:.4f}, mean={p_mean:.6f}")
