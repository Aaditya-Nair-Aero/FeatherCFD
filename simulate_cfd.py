"""
CFD Simulation with GUI Progress Window + live slice visualization.

Usage:
    __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
        python simulate_cfd.py [--steps 300]
"""
import moderngl
import numpy as np
import os
import time
import math
import argparse
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from geometry_utils import create_sphere, create_box, create_naca_airfoil, create_naca_airfoil_sdf, create_cylinder, create_cylinder_sdf


def generate_forces(step, grid_size):
    g = grid_size
    c = g / 2.0
    t = step * 0.05

    forces = []

    # Force 1: Orbiting vortex ring
    orbit_r = g * 0.25
    ox = c + orbit_r * math.cos(t)
    oy = c + 10.0 * math.sin(t * 1.7)
    oz = c + orbit_r * math.sin(t)
    dx = -math.sin(t) * 0.8
    dy = math.cos(t * 0.6) * 0.4
    dz = math.cos(t) * 0.8
    forces.append(((ox, oy, oz), (dx, dy, dz), 20.0))

    # Force 2: Counter-rotating vortex
    ox2 = c - orbit_r * math.cos(t * 0.8 + 1.0)
    oy2 = c - 15.0 * math.sin(t * 1.3)
    oz2 = c - orbit_r * math.sin(t * 0.8 + 1.0)
    dx2 = math.sin(t * 0.8 + 1.0) * 0.7
    dy2 = -math.cos(t * 0.5) * 0.5
    dz2 = -math.cos(t * 0.8 + 1.0) * 0.7
    forces.append(((ox2, oy2, oz2), (dx2, dy2, dz2), 18.0))

    # Force 3: Central upwelling pulse
    pulse = 0.6 + 0.4 * math.sin(t * 2.0)
    forces.append(((c, c * 0.4, c), (0.0, pulse, 0.0), 25.0))

    # Force 4: Diagonal shear (ramps in after step 40)
    if step > 40:
        shear = min(1.0, (step - 40) / 30.0)
        sx = c + 30.0 * math.sin(t * 0.3)
        sy = c + 30.0 * math.cos(t * 0.4)
        forces.append(((sx, sy, c),
                        (shear * 0.5, 0.0, shear * 0.5), 22.0))

    return forces


def compute_lift_drag(pressure, mask, velocity=None, viscosity=0.0, inflow_vel=(1.0, 0.0, 0.0)):
    """Integrates pressure and viscous forces over obstacle surface.
    
    pressure: pressure field [Z,Y,X] (numpy convention for axis labels)
    mask: obstacle mask [Z,Y,X] (0=fluid, 1=obstacle)
    velocity: velocity field [Z,Y,X,3] for viscous force (optional)
    viscosity: kinematic viscosity for viscous force
    inflow_vel: freestream velocity vector for dynamic pressure
    
    Returns (total_force[3], force_pressure[3], force_viscous[3])
    where indices are (Drag_X, Span_Y, Lift_Z).
    Also returns Cl, Cd coefficients as the 4th and 5th elements.
    """
    force_p = np.zeros(3)
    force_v = np.zeros(3)
    fluid_mask = (mask == 0)

    # Collect all surface face pressures for mean removal
    all_surface_pressures = []

    # Probes for each axis: dim0=Z, dim1=Y, dim2=X
    dim_pairs = [
        (2, 1, 0),  # X: dim2
        (1, 1, 1),  # Y: dim1
        (0, 1, 0),  # Z: dim0
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
        obst_sl_p  = tuple(shift_p)
        obst_sl_m  = tuple(shift_m)
        
        # Collect surface pressures for mean computation
        is_sp_raw = fluid_mask[obst_sl_m] & (mask[obst_sl_p] == 1)
        is_sm_raw = fluid_mask[obst_sl_p] & (mask[obst_sl_m] == 1)
        if is_sp_raw.any():
            all_surface_pressures.extend(pressure[obst_sl_m][is_sp_raw].tolist())
        if is_sm_raw.any():
            all_surface_pressures.extend(pressure[obst_sl_p][is_sm_raw].tolist())

    # Remove mean pressure to eliminate DC offset from non-converged solver
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
        obst_sl_p  = tuple(shift_p)
        obst_sl_m  = tuple(shift_m)
        
        # Plus direction: fluid at lower index, obstacle at higher index
        is_sp = fluid_mask[obst_sl_m] & (mask[obst_sl_p] == 1)
        if is_sp.any():
            press_contrib = np.sum(pressure[obst_sl_m][is_sp] - p_mean)
            force_p[f_idx] += press_contrib
            
        # Minus direction: fluid at higher index, obstacle at lower index
        is_sm = fluid_mask[obst_sl_p] & (mask[obst_sl_m] == 1)
        if is_sm.any():
            press_contrib = np.sum(pressure[obst_sl_p][is_sm] - p_mean)
            force_p[f_idx] -= press_contrib

    # Viscous contribution
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


class ProgressWindow:
    def __init__(self, total_steps, uinf=2.0):
        self.total = total_steps
        self._closed = False
        self._slice_data = None
        self._uinf = uinf

        self.root = tk.Tk()
        self.root.title("CFD Simulation Progress")
        self.root.geometry("720x680")
        self.root.minsize(600, 500)
        self.root.configure(bg="#1a1a2e")
        
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(1, lambda: self.root.attributes('-topmost', False))

        top = tk.Frame(self.root, bg="#1a1a2e")
        top.pack(fill="x", padx=10, pady=5)

        tk.Label(top, text="⚙  CFD Simulation Running",
                 font=("Helvetica", 14, "bold"), fg="#e0e0e0",
                 bg="#1a1a2e").pack()

        self.gpu_label = tk.Label(top, text="GPU: detecting...",
                                  font=("Helvetica", 9), fg="#7f8fa6",
                                  bg="#1a1a2e")
        self.gpu_label.pack()

        # Progress bar
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("custom.Horizontal.TProgressbar",
                        troughcolor='#16213e', background='#0f9b58',
                        darkcolor='#0f9b58', lightcolor='#34d399',
                        bordercolor='#16213e', thickness=22)

        self.progress = ttk.Progressbar(top, length=600,
                                         maximum=total_steps,
                                         style="custom.Horizontal.TProgressbar")
        self.progress.pack(pady=(8, 4))

        self.status = tk.Label(top, text="Initializing...",
                               font=("Courier", 10), fg="#a0cfff",
                               bg="#1a1a2e")
        self.status.pack()

        info_row = tk.Frame(top, bg="#1a1a2e")
        info_row.pack(fill="x")

        self.eta_label = tk.Label(info_row, text="",
                                   font=("Courier", 9), fg="#7f8fa6",
                                   bg="#1a1a2e")
        self.eta_label.pack(side="left", padx=5)

        self.force_label = tk.Label(info_row, text="Drag: 0.00 | Lift (Z): 0.00",
                                     font=("Courier", 10, "bold"), fg="#f1c40f",
                                     bg="#1a1a2e")
        self.force_label.pack(side="right", padx=5)

        self.fig = plt.Figure(figsize=(6, 4.5), dpi=100, facecolor="#1a1a2e")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#16213e")
        self.ax.set_xlabel("X (flow  →)", color="#7f8fa6", fontsize=8)
        self.ax.set_ylabel("Z", color="#7f8fa6", fontsize=8)
        self.ax.tick_params(colors="#7f8fa6", labelsize=7)
        # Placeholder until first update
        placeholder = np.zeros((192, 192))
        self.img = self.ax.imshow(placeholder, origin="lower", cmap="RdBu_r",
                                  aspect="auto", extent=[0, 192, 0, 192],
                                  vmin=-2, vmax=2)
        self.cbar = self.fig.colorbar(self.img, ax=self.ax, shrink=0.75,
                                      pad=0.02, ticks=[-2, -1, 0, 1, 2])
        self.cbar.set_label("u - U∞", color="#7f8fa6", fontsize=8)
        self.ax.set_title("Velocity perturbation  |  initializing...", color="#7f8fa6", fontsize=10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def set_gpu(self, name):
        if not self._closed:
            self.gpu_label.configure(text=f"GPU: {name}")

    def update(self, step, elapsed, frames_saved, forces=None):
        if self._closed:
            return
        self.progress['value'] = step
        pct = step / self.total * 100.0
        self.status.configure(
            text=f"Step {step}/{self.total}  ({pct:.0f}%)  |  "
                 f"Frames: {frames_saved}")
        
        if forces is not None:
            self.force_label.configure(
                text=f"Drag (X): {forces[0]:.2f} | Lift (Z): {forces[2]:.2f}")

        if step > 0:
            avg = elapsed / step
            eta = avg * (self.total - step)
            self.eta_label.configure(
                text=f"Elapsed: {elapsed:.0f}s  |  "
                     f"ETA: {eta:.0f}s  |  "
                     f"{avg:.2f}s/step")

    def update_slice(self, slice_data):
        if self._closed or slice_data is None:
            return
        try:
            self._slice_data = slice_data
            vlim = max(abs(slice_data.min()), abs(slice_data.max()), 0.5)
            self.img.set_data(slice_data)
            self.img.set_clim(-vlim, vlim)
            step_val = self.progress['value']
            title = f"u-({self._uinf:.1f})  step {step_val:.0f}  ∈[{slice_data.min():+.2f},{slice_data.max():+.2f}]"
            self.ax.set_title(title, color="#e0e0e0", fontsize=10)
            self.canvas.draw_idle()
        except Exception as e:
            print(f"  [slice error] {e}")

    def finish(self, total_frames, elapsed):
        if self._closed:
            return
        self.progress['value'] = self.total
        self.status.configure(text=f"✓  Done! {total_frames} frames saved",
                              fg="#34d399")
        self.eta_label.configure(text=f"Total: {elapsed:.1f}s")
        self.root.after(3000, self.root.destroy)

    def pump(self):
        """Process pending tk events (call from sim thread)."""
        if not self._closed:
            try:
                self.root.update()
            except tk.TclError:
                self._closed = True

    def _on_close(self):
        self._closed = True
        self.root.destroy()

    @property
    def closed(self):
        return self._closed


def main():
    parser = argparse.ArgumentParser(description="CFD Simulation")
    parser.add_argument('--mode', type=str, choices=['incompressible', 'compressible'], default='incompressible',
                        help='Flow solver mode (default: incompressible)')
    parser.add_argument('--steps', type=int, default=300,
                        help='Total simulation steps (default: 300)')
    parser.add_argument('--save-every', type=int, default=2,
                        help='Save snapshot every N steps (default: 2)')
    parser.add_argument('--obstacle', type=str, choices=['none', 'sphere', 'wing', 'cylinder'], default='none',
                        help='Type of obstacle to include (default: none)')
    parser.add_argument('--wind', type=float, default=0.0,
                        help='Wind speed along X axis (default: 0.0)')
    parser.add_argument('--mach', type=float, default=0.5,
                        help='Mach number for compressible mode (default: 0.5)')
    parser.add_argument('--no-forces', action='store_true',
                        help='Disable dynamic swirling forces')
    parser.add_argument('--re', type=float, default=None,
                        help='Target Reynolds number (sets viscosity = U*L/Re)')
    parser.add_argument('--chord', type=float, default=60.0,
                        help='Characteristic length in grid units (default: 60 for wing chord)')
    parser.add_argument('--viscosity', type=float, default=None,
                        help='Override kinematic viscosity directly (overrides --re)')
    parser.add_argument('--vcycle', action='store_true',
                        help='[EXPERIMENTAL] Use multigrid V-cycle (unstable with obstacles, off by default)')
    parser.add_argument('--vcycle-interval', type=int, default=5,
                        help='Run V-cycle every N steps (default: 5, 1=every step)')
    parser.add_argument('--cad', type=str, default=None,
                        help='Path to CAD mesh file (STL/OBJ). Overrides --obstacle.')
    parser.add_argument('--cad-chord', type=float, default=60.0,
                        help='Scale CAD mesh so longest axis = this many grid cells (default: 60)')
    parser.add_argument('--aoa', type=float, default=5.0,
                        help='Wing angle of attack in degrees (default: 5.0)')
    parser.add_argument('--cad-aoa', type=float, default=0.0,
                        help='Angle of attack in degrees (rotation around Y, default: 0)')
    parser.add_argument('--sod', action='store_true',
                        help='Use Sod shock tube initial condition (compressible mode)')
    parser.add_argument('--headless', action='store_true',
                        help='Disable GUI progress window (for non-interactive runs)')
    parser.add_argument('--out-dir', type=str, default='/home/aaditya/Downloads/tmp',
                        help='Output directory for frames (NVME SSD recommended)')
    parser.add_argument('--engine', type=str, choices=['opengl', 'vulkan'], default='opengl',
                        help='Rendering engine: opengl (current) or vulkan (in development)')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    for f in os.listdir(out_dir):
        if f.startswith('vel_') and f.endswith('.npy'):
            os.remove(os.path.join(out_dir, f))

    if args.engine == 'vulkan':
        _run_vulkan_incompressible(args)
        return

    win = None
    if not args.headless:
        win = ProgressWindow(args.steps, uinf=args.wind)
        win.pump()

    print("Creating OpenGL context...")
    ctx = moderngl.create_standalone_context()
    gpu_name = ctx.info['GL_RENDERER']
    print(f"  Renderer: {gpu_name}")
    if win is not None:
        win.set_gpu(gpu_name)
        win.pump()

    import OpenGL.GL as gl
    gl.glGetString(gl.GL_VERSION)

    if args.mode == 'compressible':
        _run_compressible(ctx, args, win)
    else:
        _run_incompressible(ctx, args, win)


def _run_vulkan_incompressible(args):
    from vk_cfd.common.device import VKContext
    from vk_cfd.sim.cfd_sim_vk import CFD_System_VK
    from geometry_utils import create_sphere, create_naca_airfoil, create_naca_airfoil_sdf, create_cylinder, create_cylinder_sdf
    from force_utils import compute_lift_drag

    grid = 192
    print("Creating Vulkan context...")
    vk_ctx = VKContext()
    cfd = CFD_System_VK(vk_ctx)
    cfd.inflow_vel = (args.wind, 0.0, 0.0)

    char_len = args.cad_chord if args.cad else args.chord
    if args.re is not None:
        cfd.set_reynolds(args.re, char_length=char_len)
    elif args.viscosity is not None:
        cfd.viscosity = args.viscosity
        print(f"  viscosity set to {cfd.viscosity:.6e}")

    print(f"Vulkan CFD System initialized. Grid: {grid}^3 | Wind: {args.wind}")

    mask = None
    if args.obstacle == 'sphere':
        ctr = (grid - 1) / 2.0
        mask = create_sphere(grid, (ctr, ctr, ctr), 25.0)
        x, y, z = np.meshgrid(np.linspace(0, grid-1, grid),
                               np.linspace(0, grid-1, grid),
                               np.linspace(0, grid-1, grid), indexing='ij')
        sdf = np.sqrt((x-ctr)**2 + (y-ctr)**2 + (z-ctr)**2) - 25.0
        cfd.set_obstacle(mask, sdf.astype(np.float16))
    elif args.obstacle == 'wing':
        ctr = (grid - 1) * 0.5
        mask = create_naca_airfoil(grid, ctr, 60.0, 0.15, 0, grid, aoa_deg=args.aoa)
        sdf = create_naca_airfoil_sdf(grid, ctr, 60.0, 0.15, 0, grid, aoa_deg=args.aoa)
        cfd.set_obstacle(mask, sdf.astype(np.float16))
    elif args.obstacle == 'cylinder':
        ctr = (grid - 1) / 2.0
        mask = create_cylinder(grid, ctr, ctr, 15.0, 1, grid-1)
        sdf = create_cylinder_sdf(grid, ctr, ctr, 15.0, 1, grid-1)
        cfd.set_obstacle(mask, sdf.astype(np.float16))

    if args.wind != 0.0:
        vel_init = np.full((grid, grid, grid, 4), 0, dtype=np.float16)
        vel_init[..., 0] = args.wind
        if args.obstacle != 'none' and mask is not None:
            mask_exp = np.repeat(mask[..., np.newaxis], 4, axis=3)
            vel_init = np.where(mask_exp > 0, 0, vel_init)
        cfd.tex_velocity_A.upload(vel_init)
        cfd.tex_velocity_B.upload(vel_init)
        print(f"  Velocity initialized to ({args.wind}, 0, 0) everywhere")

    print(f"Simulating {args.steps} steps...")
    saved = 0
    t_start = time.perf_counter()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    obs_mask = None
    if args.obstacle != 'none':
        obs_mask = mask

    forces_history = []

    for step in range(1, args.steps + 1):
        cfd.step()

        current_force = None
        if obs_mask is not None:
            pressure = cfd.read_pressure()
            vel_array = cfd.read_velocity()
            current_force, force_p, force_v = compute_lift_drag(
                pressure, obs_mask,
                velocity=vel_array[:, :, :, :3],
                viscosity=cfd.viscosity)
            forces_history.append(current_force)
            v = vel_array[:, :, :, :3].astype(np.float64)
            div = (v[2:, 1:-1, 1:-1, 0] - v[1:-1, 1:-1, 1:-1, 0]) + \
                  (v[1:-1, 2:, 1:-1, 1] - v[1:-1, 1:-1, 1:-1, 1]) + \
                  (v[1:-1, 1:-1, 2:, 2] - v[1:-1, 1:-1, 1:-1, 2])
            mean_div = np.mean(np.abs(div))
            max_div = np.max(np.abs(div))
            if step <= 5 or step % 50 == 0:
                print(f"  step={step:4d}  total=({current_force[0]:+9.1f}, {current_force[2]:+9.1f})  "
                      f"|div|_mean={mean_div:.6f}  max={max_div:.4f}")

        if step % args.save_every == 0:
            vel = cfd.read_velocity()
            den = cfd.read_density()
            rgba = np.zeros((grid, grid, grid, 4), dtype=np.float16)
            rgba[:, :, :, :3] = vel[:, :, :, :3]
            rgba[:, :, :, 3] = den if den.ndim == 3 else den[:, :, :, 0]
            fname = os.path.join(out_dir, f'vel_{saved:04d}.npy')
            np.save(fname, rgba)
            saved += 1

        elapsed = time.perf_counter() - t_start
        if step % 10 == 0:
            print(f"  step={step:4d}/{args.steps}  elapsed={elapsed:.1f}s  frames={saved}")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone! {saved} frames in {elapsed:.1f}s")
    if forces_history:
        np.save(os.path.join(out_dir, 'forces_history.npy'), np.array(forces_history))
    meta = {
        'grid_size': grid, 'total_frames': saved,
        'steps_per_frame': args.save_every,
        'total_sim_steps': args.steps, 'dt': cfd.dt, 'mode': 'incompressible',
        'engine': 'vulkan',
    }
    np.save(os.path.join(out_dir, 'metadata.npy'), meta)
    if obs_mask is not None:
        np.save(os.path.join(out_dir, 'obstacles.npy'), obs_mask)
    cfd.destroy()
    vk_ctx.destroy()


def _run_incompressible(ctx, args, win):
    from cfd_sim import CFD_System
    cfd = CFD_System(ctx)
    cfd.inflow_vel = (args.wind, 0.0, 0.0)
    for prog in [cfd.prog_forces, cfd.prog_advection, cfd.prog_projection, cfd.prog_advection_density]:
        if 'u_inflow_vel' in prog:
            prog['u_inflow_vel'].value = tuple(cfd.inflow_vel)

    char_len = args.cad_chord if args.cad else args.chord
    if args.re is not None:
        cfd.set_reynolds(args.re, char_length=char_len)
    elif args.viscosity is not None:
        cfd.viscosity = args.viscosity
        print(f"  viscosity set to {cfd.viscosity:.6e}")

    if args.vcycle:
        cfd.use_vcycle = True
        cfd.vcycle_interval = args.vcycle_interval
        print(f"  Using multigrid V-cycle solver (every {args.vcycle_interval} steps)")

    grid = cfd.grid_size
    print(f"CFD System initialized. Grid: {grid}^3 | Wind: {args.wind}")

    from geometry_utils import create_sphere, create_naca_airfoil, create_naca_airfoil_sdf, create_cylinder, create_cylinder_sdf
    from simulate_cfd import generate_forces
    from force_utils import compute_lift_drag

    mask = None
    if args.obstacle == 'sphere':
        print("Creating sphere obstacle...")
        ctr = (grid - 1) / 2.0
        mask = create_sphere(grid, (ctr, ctr, ctr), 25.0)
        cfd.set_obstacle(mask)
        x = np.linspace(0, grid-1, grid)
        y = np.linspace(0, grid-1, grid)
        z = np.linspace(0, grid-1, grid)
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        dist = np.sqrt((xv-ctr)**2 + (yv-ctr)**2 + (zv-ctr)**2) - 25.0
        sdf = dist.astype(np.float16)
        cfd.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).tobytes())
    elif args.obstacle == 'wing':
        wing_aoa = args.aoa
        ctr = (grid - 1) * 0.5
        print(f"Creating NACA airfoil ({wing_aoa}deg AoA, Centered) with SDF...")
        mask = create_naca_airfoil(grid, ctr, 60.0, 0.15, 0, grid, aoa_deg=wing_aoa)
        cfd.set_obstacle(mask)
        sdf = create_naca_airfoil_sdf(grid, ctr, 60.0, 0.15, 0, grid, aoa_deg=wing_aoa)
        cfd.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).astype(np.float16).tobytes())
    elif args.obstacle == 'cylinder':
        ctr = (grid - 1) / 2.0
        print("Creating cylinder obstacle (D=30, full span)...")
        mask = create_cylinder(grid, ctr, ctr, 15.0, 1, grid-1)
        cfd.set_obstacle(mask)
        sdf = create_cylinder_sdf(grid, ctr, ctr, 15.0, 1, grid-1)
        cfd.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).astype(np.float16).tobytes())

    if args.cad:
        print(f"Using CAD mesh: {args.cad}")
        cfd.set_cad_obstacle(args.cad, args.cad_chord, args.cad_aoa)

    if args.wind != 0.0:
        vel_init = np.full((grid, grid, grid, 4), 0, dtype=np.float16)
        vel_init[..., 0] = args.wind
        if args.obstacle != 'none' and mask is not None:
            mask_exp = np.repeat(mask[..., np.newaxis], 4, axis=3)
            vel_init = np.where(mask_exp > 0, 0, vel_init)
        vel_init_t = np.transpose(vel_init, (2, 1, 0, 3))
        cfd.tex_velocity_A.write(vel_init_t.tobytes())
        cfd.tex_velocity_B.write(vel_init_t.tobytes())
        cfd._update_dt()
        print(f"  Velocity initialized to ({args.wind}, 0, 0) everywhere (zero in obstacles), dt={cfd.dt:.4f}")

    print(f"Simulating {args.steps} steps...")
    saved = 0
    t_start = time.perf_counter()
    out_dir = args.out_dir

    obs_mask = None
    if args.obstacle != 'none':
        obs_data = cfd.tex_obstacle.read()
        obs_mask = np.frombuffer(obs_data, dtype='u1').reshape((grid, grid, grid))

    forces_history = []

    for step in range(1, args.steps + 1):
        if win is not None and win.closed:
            print("\nCancelled by user.")
            return

        if not args.no_forces:
            force_list = generate_forces(step, grid)
            pf = force_list[0]
            cfd.step(force_pos=pf[0], force_dir=pf[1], force_radius=pf[2])
            for pos, direction, radius in force_list[1:]:
                cfd.tex_velocity_A.use(location=0)
                cfd.prog_forces['u_velocity'].value = 0
                cfd.prog_forces['u_dt'].value = cfd.dt
                cfd.prog_forces['u_force_pos'].value = tuple(pos)
                cfd.prog_forces['u_force_dir'].value = tuple(direction)
                cfd.prog_forces['u_force_radius'].value = float(radius)
                cfd.render_pass(cfd.prog_forces, cfd.fbo_velocity_B)
                cfd.tex_velocity_A, cfd.tex_velocity_B = \
                    cfd.tex_velocity_B, cfd.tex_velocity_A
                cfd.fbo_velocity_A, cfd.fbo_velocity_B = \
                    cfd.fbo_velocity_B, cfd.fbo_velocity_A
        else:
            cfd.step(force_radius=0.0)

        current_force = None
        if obs_mask is not None:
            pres_raw = cfd.tex_pressure_A.read()
            pressure = np.frombuffer(pres_raw, dtype='f4').reshape((grid, grid, grid))
            vel_raw = cfd.tex_velocity_A.read()
            vel_array = np.frombuffer(vel_raw, dtype=np.float16).reshape((grid, grid, grid, 4))
            current_force, force_p, force_v = compute_lift_drag(pressure, obs_mask,
                                            velocity=vel_array[:,:,:,:3],
                                            viscosity=cfd.viscosity)
            forces_history.append(current_force)
            v = vel_array[:,:,:,:3].astype(np.float64)
            div = (v[1:-1,1:-1,2:,0] - v[1:-1,1:-1,1:-1,0]) + \
                  (v[1:-1,2:,1:-1,1] - v[1:-1,1:-1,1:-1,1]) + \
                  (v[2:,1:-1,1:-1,2] - v[1:-1,1:-1,1:-1,2])
            mean_div = np.mean(np.abs(div))
            max_div = np.max(np.abs(div))
            if step <= 5 or step % 50 == 0:
                print(f"  step={step:4d}  total=({current_force[0]:+9.1f}, {current_force[2]:+9.1f})  "
                      f"press=({force_p[0]:+9.1f}, {force_p[2]:+9.1f})  "
                      f"visc=({force_v[0]:+9.1f}, {force_v[2]:+9.1f})  "
                      f"|div|_mean={mean_div:.6f}  max={max_div:.4f}")

        if win is not None and obs_mask is not None and (step <= 5 or step % 2 == 0):
            mid_y = grid // 2
            # Show velocity perturbation: u - U_inf (reveals flow features immediately)
            u_pert = vel_array[:, mid_y, :, 0].T - args.wind
            print(f"  [slice] step={step} u_pert=[{u_pert.min():.3f},{u_pert.max():.3f}]")
            win.update_slice(u_pert)

        if step % args.save_every == 0:
            raw_vel = cfd.tex_velocity_A.read()
            raw_den = cfd.tex_density_A.read()
            vel = np.frombuffer(raw_vel, dtype=np.float16).reshape((grid, grid, grid, 4))
            den = np.frombuffer(raw_den, dtype=np.float16).reshape((grid, grid, grid, 1))
            rgba = np.zeros((grid, grid, grid, 4), dtype=np.float16)
            rgba[:, :, :, :3] = vel[:, :, :, :3]
            rgba[:, :, :, 3] = den[:, :, :, 0]
            fname = os.path.join(out_dir, f'vel_{saved:04d}.npy')
            np.save(fname, rgba)
            saved += 1

        elapsed = time.perf_counter() - t_start
        if win is not None:
            win.update(step, elapsed, saved, forces=current_force)
            win.pump()

    elapsed = time.perf_counter() - t_start
    print(f"\nDone! {saved} frames in {elapsed:.1f}s")
    if forces_history:
        np.save(os.path.join(out_dir, 'forces_history.npy'), np.array(forces_history))
    meta = {
        'grid_size': grid, 'total_frames': saved,
        'steps_per_frame': args.save_every,
        'total_sim_steps': args.steps, 'dt': cfd.dt, 'mode': 'incompressible',
    }
    np.save(os.path.join(out_dir, 'metadata.npy'), meta)
    obs_data = cfd.tex_obstacle.read()
    obs_array = np.frombuffer(obs_data, dtype='u1').reshape((grid, grid, grid))
    np.save(os.path.join(out_dir, 'obstacles.npy'), obs_array)
    if win is not None:
        win.finish(saved, elapsed)
        while not win.closed:
            win.pump()
            time.sleep(0.05)


def _run_compressible(ctx, args, win):
    from euler_solver import EulerSolver
    euler = EulerSolver(ctx)
    grid = euler.grid_size

    gamma = 1.4
    p_ref = 1.0 / gamma
    rho_ref = 1.0
    Mach = args.mach
    c_ref = 1.0  # sound speed in lattice units for p_ref, rho_ref
    u_ref = Mach * c_ref

    print(f"Euler solver initialized. Grid: {grid}^3 | Mach={Mach}")
    print(f"  gamma={gamma}, p_ref={p_ref:.4f}, rho_ref={rho_ref}, u_ref={u_ref:.4f}")
    print(f"  dt={euler.dt:.6f}")

    from geometry_utils import create_naca_airfoil, create_cylinder, create_sphere
    from scipy.ndimage import distance_transform_edt

    if args.sod:
        print("Using Sod shock tube initial condition...")
        euler.bc_type = 0  # Reflective walls for Sod tube
        euler.set_inflow(rho=rho_ref, u=u_ref, v=0.0, w=0.0, p=p_ref)
        euler.init_sod()
    else:
        euler.bc_type = 1  # Inflow/outflow for external flow
        euler.init_uniform(rho=rho_ref, u=u_ref, p=p_ref)

    mask = None
    if args.obstacle == 'sphere':
        ctr = (grid - 1) / 2.0
        print("Creating sphere obstacle...")
        mask = create_sphere(grid, (ctr, ctr, ctr), 25.0)
        euler.set_obstacle(mask)
    elif args.obstacle == 'wing':
        ctr = (grid - 1) * 0.5
        print("Creating NACA airfoil (5deg AoA)...")
        mask = create_naca_airfoil(grid, ctr, 60.0, 0.15, 0, grid, aoa_deg=5.0)
        euler.set_obstacle(mask)
    elif args.obstacle == 'cylinder':
        ctr = (grid - 1) / 2.0
        print("Creating cylinder obstacle...")
        mask = create_cylinder(grid, ctr, ctr, 15.0, 1, grid-1)
        euler.set_obstacle(mask)

    print(f"Simulating {args.steps} steps...")
    saved = 0
    t_start = time.perf_counter()
    out_dir = args.out_dir

    for step in range(1, args.steps + 1):
        if win is not None and win.closed:
            print("\nCancelled by user.")
            return

        euler.step()

        if step % max(1, args.steps // 20) == 0:
            mach = euler.compute_mach()
            print(f"  step={step:4d}  |M|_max={np.max(mach):.3f}  dt={euler.dt:.6f}")

        if step % args.save_every == 0:
            U1, E = euler.read_state()
            rho = U1[:, :, :, 0]
            u = np.where(rho > 1e-8, U1[:, :, :, 1] / rho, 0.0)
            v = np.where(rho > 1e-8, U1[:, :, :, 2] / rho, 0.0)
            w = np.where(rho > 1e-8, U1[:, :, :, 3] / rho, 0.0)
            ke = 0.5 * (U1[:,:,:,1]**2 + U1[:,:,:,2]**2 + U1[:,:,:,3]**2) / np.maximum(rho, 1e-8)
            p = np.maximum((gamma - 1.0) * (E[:,:,:,0] - ke), 1e-8)
            c = np.sqrt(gamma * p / np.maximum(rho, 1e-8))
            speed = np.sqrt(u*u + v*v + w*w)
            mach_field = speed / np.maximum(c, 1e-8)

            frame = np.zeros((grid, grid, grid, 4), dtype=np.float16)
            frame[:, :, :, 0] = u.astype(np.float16)
            frame[:, :, :, 1] = v.astype(np.float16)
            frame[:, :, :, 2] = w.astype(np.float16)
            frame[:, :, :, 3] = mach_field.astype(np.float16)
            fname = os.path.join(out_dir, f'vel_{saved:04d}.npy')
            np.save(fname, frame)
            saved += 1

        elapsed = time.perf_counter() - t_start
        if win is not None:
            current_force = None
            win.update(step, elapsed, saved, forces=current_force)
            win.pump()

    elapsed = time.perf_counter() - t_start
    print(f"\nDone! {saved} frames in {elapsed:.1f}s")
    meta = {
        'grid_size': grid, 'total_frames': saved,
        'steps_per_frame': args.save_every,
        'total_sim_steps': args.steps, 'dt': euler.dt, 'mode': 'compressible',
        'gamma': gamma, 'mach': Mach,
    }
    np.save(os.path.join(out_dir, 'metadata.npy'), meta)
    obs_data = euler.tex_obstacle.read()
    obs_array = np.frombuffer(obs_data, dtype='u1').reshape((grid, grid, grid))
    np.save(os.path.join(out_dir, 'obstacles.npy'), obs_array)
    if win is not None:
        win.finish(saved, elapsed)
        while not win.closed:
            win.pump()
            time.sleep(0.05)


if __name__ == '__main__':
    main()
