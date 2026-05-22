#!/usr/bin/env python3
"""100-step Vulkan CFD validation with NACA airfoil."""
import numpy as np
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vk_cfd.common.device import VKContext
from vk_cfd.sim.cfd_sim_vk import CFD_System_VK
from geometry_utils import create_naca_airfoil, create_naca_airfoil_sdf
from force_utils import compute_lift_drag

grid = 192
U_inf = 2.0
chord = 60.0

vk_ctx = VKContext()
cfd = CFD_System_VK(vk_ctx)
cfd.inflow_vel = (U_inf, 0.0, 0.0)
cfd.sgs_coeff = 0.0
cfd.viscosity = U_inf * chord / 5000.0  # Re=5000
cfd.use_dst = False
cfd.rbgs_iters = 60

ctr = (grid - 1) * 0.5
mask = create_naca_airfoil(grid, ctr, chord, 0.12, 0, grid, aoa_deg=5.0)
sdf = create_naca_airfoil_sdf(grid, ctr, chord, 0.12, 0, grid, aoa_deg=5.0)
cfd.set_obstacle(mask, sdf.astype(np.float16))

vel_init = np.full((grid, grid, grid, 4), 0, dtype=np.float16)
vel_init[..., 0] = U_inf
mask_exp = np.repeat(mask[..., np.newaxis], 4, axis=3)
vel_init = np.where(mask_exp > 0, 0, vel_init)
cfd.tex_velocity_A.upload(vel_init)
cfd.tex_velocity_B.upload(vel_init)

print(f"NACA0012 5deg AoA, Re=5000, grid={grid}^3")
print(f"  viscosity={cfd.viscosity:.6e}, rbgs={cfd.rbgs_iters}")
t0 = time.perf_counter()

for step in range(1, 101):
    cfd.step()

    if step % 10 == 0:
        vel = cfd.read_velocity()
        pres = cfd.read_pressure()
        
        v = vel[:,:,:,:3].astype(np.float64)
        div = (v[2:,1:-1,1:-1,0] - v[1:-1,1:-1,1:-1,0]) + \
              (v[1:-1,2:,1:-1,1] - v[1:-1,1:-1,1:-1,1]) + \
              (v[1:-1,1:-1,2:,2] - v[1:-1,1:-1,1:-1,2])
        mean_div = np.mean(np.abs(div))
        max_div = np.max(np.abs(div))
        
        print(f"  step={step:3d}: u_max={vel[:,:,:,0].max():.3f}  "
              f"|div|_mean={mean_div:.6f}  max={max_div:.4f}  "
              f"p=[{pres.min():+.3f},{pres.max():+.3f}]")

elapsed = time.perf_counter() - t0
print(f"\n100 steps in {elapsed:.1f}s ({elapsed/100*1000:.0f}ms/step)")

cfd.destroy()
vk_ctx.destroy()
