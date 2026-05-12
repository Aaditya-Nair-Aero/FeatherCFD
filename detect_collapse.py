#!/usr/bin/env python3
"""Catch NACA collapse at the exact step it happens."""
import moderngl, numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_utils import create_naca_airfoil, create_naca_airfoil_sdf
from cfd_sim import CFD_System

ctx = moderngl.create_standalone_context()
import OpenGL.GL as gl; gl.glGetString(gl.GL_VERSION)

grid = 192; chord = 60; le_pos = 48
U_inf = 2.0; aoa = 5; Re = 5000

mask = create_naca_airfoil(grid, le_pos, chord, 0.12, 0, grid, aoa_deg=float(aoa))
sdf = create_naca_airfoil_sdf(grid, le_pos, chord, 0.12, 0, grid, aoa_deg=float(aoa))

cfd = CFD_System(ctx)
cfd.inflow_vel = (U_inf, 0.0, 0.0)
for prog in [cfd.prog_forces, cfd.prog_advection, cfd.prog_projection, cfd.prog_advection_density]:
    if 'u_inflow_vel' in prog:
        prog['u_inflow_vel'].value = tuple(cfd.inflow_vel)
cfd.set_obstacle(mask)
cfd.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).astype(np.float16).tobytes())
cfd.set_reynolds(Re, char_length=chord)

cfd.sgs_coeff = 0.0; cfd.jacobi_iters = 120
cfd.cfl_target = 0.12; cfd.dt_update_interval = 1  # Every step
cfd._update_dt()

vel_init = np.full((grid, grid, grid, 4), 0, dtype=np.float16)
vel_init[..., 0] = U_inf
me = np.repeat(mask[..., np.newaxis], 4, axis=3)
vel_init = np.where(me > 0, 0, vel_init)
cfd.tex_velocity_A.write(np.transpose(vel_init, (2, 1, 0, 3)).tobytes())
cfd.tex_velocity_B.write(np.transpose(vel_init, (2, 1, 0, 3)).tobytes())

prev_ux_max = U_inf
for step in range(1, 301):
    cfd.step(force_radius=0.0)
    
    vel = np.frombuffer(cfd.tex_velocity_A.read(), dtype=np.float16).reshape((grid, grid, grid, 4))
    ux_max = float(vel[:,:,:,0].max())
    uz_max = float(np.abs(vel[:,:,:,2]).max())
    
    if ux_max < prev_ux_max * 0.7 and step > 20:
        pres = np.frombuffer(cfd.tex_pressure_A.read(), dtype='f4').reshape((grid, grid, grid))
        print(f'\n*** COLLAPSE DETECTED at step {step} ***')
        print(f'  ux_max: {prev_ux_max:.3f} → {ux_max:.3f}')
        print(f'  uz_maxabs: {uz_max:.6f}')
        print(f'  p range: [{pres.min():+.3f}, {pres.max():+.3f}]')
        print(f'  dt: {cfd.dt:.6f}')
        
        print(f'  Velocity at X=0,Y=96,Z=96: {vel[96,96,0,:3]}')
        print(f'  Velocity at X=24,Y=96,Z=96: {vel[96,96,24,:3]}')
        print(f'  Velocity at X=48,Y=96,Z=96: {vel[96,96,48,:3]}')
        print(f'  Velocity at X=96,Y=96,Z=96: {vel[96,96,96,:3]}')
        print(f'  Velocity at X=144,Y=96,Z=96: {vel[96,96,144,:3]}')
        print(f'  Velocity at X=191,Y=96,Z=96: {vel[96,96,191,:3]}')
        
        div = np.frombuffer(cfd.tex_divergence.read(), dtype=np.float16).reshape((grid, grid, grid))
        print(f'  div range: [{div.min():+.6f}, {div.max():+.6f}]')
        print(f'  div non-zero: {(np.abs(div) > 1e-6).sum()}')
        break
    
    prev_ux_max = ux_max
    if step % 20 == 0:
        print(f'  step {step}: ux_max={ux_max:.3f} uz_maxabs={uz_max:.4f} dt={cfd.dt:.6f}')
