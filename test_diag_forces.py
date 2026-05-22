#!/usr/bin/env python3
"""Verify upload/download axis fix."""
import struct
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vk_cfd.common.device import VKContext
from vk_cfd.sim.cfd_sim_vk import CFD_System_VK

grid = 192

vk_ctx = VKContext()
cfd = CFD_System_VK(vk_ctx)

# Test upload/download roundtrip with known pattern
print("Test: upload pattern, dispatch copy, verify roundtrip\n")

# Create a pattern where (x, y, z) maps to unique value
vel_init = np.zeros((grid, grid, grid, 4), dtype=np.float16)
for x in range(grid):
    vel_init[x, :, :, 0] = x / 191.0  # x-velocity encodes x position

cfd.tex_velocity_A.upload(vel_init)
cfd.tex_velocity_B.upload(np.zeros((grid, grid, grid, 4), dtype=np.float16))

# Dispatch copy shader (copies A → B)
v2 = [cfd.tex_velocity_A.view, cfd.tex_velocity_B.view]
cfd._begin_cmd()
cfd._dispatch('copy', cfd._make_set('copy', v2))
cfd._barrier()
cfd._end_cmd()

vel = cfd.tex_velocity_B.download()
print(f"Readback shape: {vel.shape}")
print(f"Readback dtype: {vel.dtype}")

# Check various positions
print("\nPosition checks (expect x/191.0):")
all_ok = True
for x, y, z in [(0, 0, 0), (0, 50, 50), (50, 0, 50), (50, 50, 0), (100, 100, 100), (191, 191, 191)]:
    val = vel[x, y, z, 0]
    expected = x / 191.0
    ok = "OK" if np.isclose(val, expected, atol=1e-4) else "MISMATCH"
    if ok != "OK":
        all_ok = False
    print(f"  vel[{x:3d},{y:3d},{z:3d},0] = {val:.4f}  expected {expected:.4f}  [{ok}]")

# Check x=0 plane (should all be 0.0)
x0 = vel[0, :, :, 0]
print(f"\nx=0 plane: min={x0.min():.4f} max={x0.max():.4f} (expect 0.0)")
# Check z=0 plane (should have varying values)
z0 = vel[:, :, 0, 0]
print(f"z=0 plane: min={z0.min():.4f} max={z0.max():.4f} (expect 0.0..1.0)")

# Now test the forces shader x=0 inflow
print("\n\nTest: forces shader inflow BC on x=0 plane")
vel_init_zero = np.zeros((grid, grid, grid, 4), dtype=np.float16)
cfd.tex_velocity_A.upload(vel_init_zero)
cfd.tex_velocity_B.upload(vel_init_zero)

obs_init = np.zeros((grid, grid, grid), dtype=np.uint8)
sdf_init = np.full((grid, grid, grid), 10.0, dtype=np.float16)
cfd.tex_obstacle.upload(obs_init)
cfd.tex_sdf.upload(sdf_init)

pc64 = struct.pack('f', 0.16) + b'\x00' * 12
pc64 += b'\x00' * 16 + b'\x00' * 16
pc64 += struct.pack('fff', *[2.0, 0.0, 0.0]) + b'\x00' * 4

v4 = [cfd.tex_velocity_A.view, cfd.tex_velocity_B.view,
      cfd.tex_obstacle.view, cfd.tex_sdf.view]

cfd._begin_cmd()
cfd._dispatch('forces', cfd._make_set('forces', v4), (pc64, 64))
cfd._barrier()
cfd._end_cmd()

vel = cfd.tex_velocity_B.download()
x0 = vel[0, :, :, 0]  # x=0 plane, x-velocity
match = np.isclose(x0, 2.0, atol=1e-4)
n_match = match.sum()
n_total = x0.size
print(f"  x=0 plane cells with 2.0: {n_match} / {n_total}")
if n_match == n_total:
    print("  ✓ ALL x=0 cells have inflow velocity!")
else:
    print(f"  ✗ Only {n_match}/{n_total} cells match!")
    z_bad = [z for z in range(grid) if not np.all(match[:, z])]
    print(f"    Bad z-slices: {z_bad[:20]}...")

cfd.destroy()
vk_ctx.destroy()
