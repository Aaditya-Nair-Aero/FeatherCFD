#!/usr/bin/env python3
"""Diagnose NACA lift with improved solver settings."""
import moderngl, numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_utils import create_naca_airfoil, create_naca_airfoil_sdf
from cfd_sim import CFD_System
from simulate_cfd import compute_lift_drag

ctx = moderngl.create_standalone_context()
import OpenGL.GL as gl; gl.glGetString(gl.GL_VERSION)

grid = 192; chord = 60.0; aoa = 5; U_inf = 2.0; Re = 5000
mask = create_naca_airfoil(grid, grid*0.5, chord, 0.12, 0, grid, aoa_deg=float(aoa))
sdf = create_naca_airfoil_sdf(grid, grid*0.5, chord, 0.12, 0, grid, aoa_deg=float(aoa))

cfd = CFD_System(ctx)
cfd.inflow_vel = (U_inf, 0.0, 0.0)
for prog in [cfd.prog_forces, cfd.prog_advection, cfd.prog_projection, cfd.prog_advection_density]:
    if 'u_inflow_vel' in prog:
        prog['u_inflow_vel'].value = tuple(cfd.inflow_vel)
cfd.set_obstacle(mask)
cfd.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).astype(np.float16).tobytes())
cfd.set_reynolds(Re, char_length=chord)

cfd.sgs_coeff = 0.0
cfd.jacobi_iters = 80
print(f'Config: sgs={cfd.sgs_coeff}, jacobi={cfd.jacobi_iters}, Re={Re}')

vel_init = np.full((grid, grid, grid, 4), 0, dtype=np.float16)
vel_init[..., 0] = U_inf
me = np.repeat(mask[..., np.newaxis], 4, axis=3)
vel_init = np.where(me > 0, 0, vel_init)
cfd.tex_velocity_A.write(np.transpose(vel_init, (2, 1, 0, 3)).tobytes())
cfd.tex_velocity_B.write(np.transpose(vel_init, (2, 1, 0, 3)).tobytes())

obs_data = cfd.tex_obstacle.read()
obs_mask = np.frombuffer(obs_data, dtype='u1').reshape((grid, grid, grid))

t0 = time.perf_counter()
for step in range(1, 301):
    cfd.step(force_radius=0.0)
    if step % 10 == 0:
        pres = np.frombuffer(cfd.tex_pressure_A.read(), dtype='f4').reshape((grid, grid, grid))
        vel = np.frombuffer(cfd.tex_velocity_A.read(), dtype=np.float16).reshape((grid, grid, grid, 4))
        ft, _, _ = compute_lift_drag(pres, obs_mask, velocity=vel[:,:,:,:3], viscosity=cfd.viscosity)
        S = chord * grid; q = 0.5 * U_inf**2
        Cl = ft[2] / (q * S)
        Cd = ft[0] / (q * S)
        print(f'  step={step:3d}: Cl={Cl:.4f}  Cd={Cd:.4f}  '
              f'ux_max={vel[:,:,:,0].max():.3f}  uz_range=[{vel[:,:,:,2].min():+.3f},{vel[:,:,:,2].max():+.3f}]  '
              f'p_range=[{pres.min():+.3f},{pres.max():+.3f}]  dt={cfd.dt:.5f}')

elapsed = time.perf_counter() - t0
print(f'\nTotal: {300} steps in {elapsed:.0f}s ({elapsed/300*1000:.0f}ms/step)')
