#!/usr/bin/env python3
"""NACA Cl measurement with per-step dt update."""
import moderngl, numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_utils import create_naca_airfoil, create_naca_airfoil_sdf
from cfd_sim import CFD_System
from simulate_cfd import compute_lift_drag

ctx = moderngl.create_standalone_context()
import OpenGL.GL as gl; gl.glGetString(gl.GL_VERSION)

grid = 192; chord = 60; le_pos = 48
U_inf = 2.0; aoa = 5; Re = 5000

mask = create_naca_airfoil(grid, le_pos, chord, 0.12, 0, grid, aoa_deg=float(aoa))
sdf = create_naca_airfoil_sdf(grid, le_pos, chord, 0.12, 0, grid, aoa_deg=float(aoa))
print(f'Obstacle cells: {mask.sum()}')

cfd = CFD_System(ctx)
cfd.inflow_vel = (U_inf, 0.0, 0.0)
for prog in [cfd.prog_forces, cfd.prog_advection, cfd.prog_projection, cfd.prog_advection_density]:
    if 'u_inflow_vel' in prog:
        prog['u_inflow_vel'].value = tuple(cfd.inflow_vel)
cfd.set_obstacle(mask)
cfd.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).astype(np.float16).tobytes())
cfd.set_reynolds(Re, char_length=chord)

cfd.sgs_coeff = 0.0; cfd.jacobi_iters = 120
cfd.cfl_target = 0.12; cfd.dt_update_interval = 1
cfd._update_dt()

vel_init = np.full((grid, grid, grid, 4), 0, dtype=np.float16)
vel_init[..., 0] = U_inf
me = np.repeat(mask[..., np.newaxis], 4, axis=3)
vel_init = np.where(me > 0, 0, vel_init)
cfd.tex_velocity_A.write(np.transpose(vel_init, (2, 1, 0, 3)).tobytes())
cfd.tex_velocity_B.write(np.transpose(vel_init, (2, 1, 0, 3)).tobytes())

obs_data = cfd.tex_obstacle.read()
obs_mask = np.frombuffer(obs_data, dtype='u1').reshape((grid, grid, grid))

t0 = time.perf_counter()
force_hist = []
for step in range(1, 1001):
    cfd.step(force_radius=0.0)
    if step % 25 == 0:
        pres = np.frombuffer(cfd.tex_pressure_A.read(), dtype='f4').reshape((grid, grid, grid))
        vel = np.frombuffer(cfd.tex_velocity_A.read(), dtype=np.float16).reshape((grid, grid, grid, 4))
        ft, fp, fv = compute_lift_drag(pres, obs_mask, velocity=vel[:,:,:,:3], viscosity=cfd.viscosity)
        S = chord * grid; q = 0.5 * U_inf**2
        Cl = ft[2] / (q * S); Cd = ft[0] / (q * S)
        ux_max = float(vel[:,:,:,0].max())
        force_hist.append((step, Cl, Cd))
        sim_time = step * cfd.dt
        if step % 100 == 0:
            print(f'step={step:3d} t≈{sim_time:.1f}: Cl={Cl:+.4f} Cd={Cd:.4f} ux_max={ux_max:.2f} dt={cfd.dt:.5f}')

elapsed = time.perf_counter() - t0
force_hist = np.array(force_hist)
# Average last 20 samples
Cl_avg = np.mean(force_hist[-20:, 1])
Cd_avg = np.mean(force_hist[-20:, 2])
print(f'\n1000 steps in {elapsed:.0f}s ({elapsed/1000*1000:.0f}ms/step)')
print(f'Last 500 steps Cl avg: {Cl_avg:.4f}, Cd avg: {Cd_avg:.4f}')
print(f'Expected: Cl~0.3-0.5 at 5° AoA, Re=5000')
print(f'Ratio to inviscid theory (Cl=0.55): {Cl_avg/0.55*100:.1f}%')
