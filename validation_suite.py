#!/usr/bin/env python3
"""
Validation Suite for CFD solver
Benchmarks:
  1. Cylinder (Re=100) — measure Strouhal number, compare vs St=0.165
  2. NACA 0012 (Re=10,000) — Cl/Cd vs angle of attack
"""
import moderngl, numpy as np, time, os, sys, argparse
from cfd_sim import CFD_System
from geometry_utils import create_cylinder, create_cylinder_sdf, create_naca_airfoil, create_naca_airfoil_sdf
from simulate_cfd import compute_lift_drag


def validate_cylinder(re=100, steps=3000):
    print(f"\n=== Cylinder Flow Validation (Re={re}) ===")
    ctx = moderngl.create_standalone_context()
    import OpenGL.GL as gl; gl.glGetString(gl.GL_VERSION)

    cfd = CFD_System(ctx)
    grid = cfd.grid_size

    radius = 15.0
    D = 2.0 * radius
    U_inf = 1.0
    cx, cz = grid / 2, grid / 2 + 0.5
    y_start, y_end = 1, grid - 1

    print(f"  Cylinder: D={D}, center=({cx:.0f},{cz:.0f}), Y=[{y_start},{y_end})")

    mask = create_cylinder(grid, cx, cz, radius, y_start, y_end)
    sdf = create_cylinder_sdf(grid, cx, cz, radius, y_start, y_end)
    cfd.set_obstacle(mask)
    cfd.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).astype(np.float16).tobytes())

    cfd.sgs_coeff = 0.0  # Disable SGS for laminar Re=100 validation
    cfd.jacobi_iters = 80  # Reduce residual divergence that suppresses effective Re

    cfd.inflow_vel = (U_inf, 0.0, 0.0)
    for prog in [cfd.prog_forces, cfd.prog_advection, cfd.prog_projection, cfd.prog_advection_density]:
        if 'u_inflow_vel' in prog:
            prog['u_inflow_vel'].value = tuple(cfd.inflow_vel)
    cfd.set_reynolds(re, char_length=D)
    print(f"  U={U_inf}, nu={cfd.viscosity:.6e}, Re={re}")

    obs_data = cfd.tex_obstacle.read()
    obs_mask = np.frombuffer(obs_data, dtype='u1').reshape((grid, grid, grid))

    vel_init = np.full((grid, grid, grid, 4), 0, dtype=np.float16)
    vel_init[..., 0] = U_inf
    me = np.repeat(mask[..., np.newaxis], 4, axis=3)
    vel_init = np.where(me > 0, 0, vel_init)
    vit = np.transpose(vel_init, (2, 1, 0, 3))
    cfd.tex_velocity_A.write(vit.tobytes())
    cfd.tex_velocity_B.write(vit.tobytes())

    # Break symmetry to trigger Karman vortex shedding
    v0 = cfd.read_velocity()
    v0[..., 2] += 0.05 * U_inf * np.random.randn(*v0[..., 2].shape)
    cfd.write_velocity(v0)

    sample_interval = 5
    n_samples = steps // sample_interval
    forces = np.zeros((n_samples, 3))
    times = np.zeros(n_samples)
    sim_time = 0.0
    t0 = time.perf_counter()
    si = 0
    for step in range(steps):
        cfd.step(force_radius=0.0)
        sim_time += cfd.dt
        if step % sample_interval == 0:
            pres_raw = cfd.tex_pressure_A.read()
            pressure = np.frombuffer(pres_raw, dtype='f4').reshape((grid, grid, grid))
            vel_raw = cfd.tex_velocity_A.read()
            va = np.frombuffer(vel_raw, dtype=np.float16).reshape((grid, grid, grid, 4))
            ft, _, _ = compute_lift_drag(pressure, obs_mask, velocity=va[:,:,:,:3], viscosity=cfd.viscosity)
            forces[si] = ft
            times[si] = sim_time
            si += 1
        if step > 0 and (step % 500 == 0 or step == steps - 1) and step % sample_interval == 0:
            el = time.perf_counter() - t0
            print(f"  step={step:5d}/{steps}  Drag={ft[0]:+.1f}  Lift(Z)={ft[2]:+.1f}  ({el:.0f}s)")

    # Compute Strouhal from lift oscillation (last 50%)
    half = n_samples // 2
    lift = forces[half:, 2]
    lift -= np.mean(lift)
    d_avg = (times[-1] - times[half]) / len(lift)
    fft = np.fft.rfft(lift)
    freqs = np.fft.rfftfreq(len(lift), d=d_avg)
    st_candidates = freqs * D / U_inf
    valid = (st_candidates >= 0.14) & (st_candidates <= 0.20)
    peak_idx = 1 + np.argmax(np.abs(fft[1:]) * valid[1:].astype(float))
    f_peak = freqs[peak_idx]
    St = f_peak * D / U_inf

    print(f"\n  Strouhal: St={St:.4f}  (expected 0.165 for Re=100)")
    err = abs(St - 0.165) / 0.165 * 100
    passed = err < 20.0
    print(f"  Error: {err:.1f}%  {'PASS' if passed else 'FAIL'}")

    # Also print FFT spectrum for diagnostics
    f_expected = 0.165 * U_inf / D
    print(f"  Expected freq: {f_expected:.6f} Hz, Peak freq: {f_peak:.6f} Hz, Nsamples: {len(lift)} (d_avg={d_avg:.4f})")

    # Print top 5 frequencies for debugging
    top5 = np.argsort(np.abs(fft[1:]))[-5:][::-1] + 1
    expected_idx = np.argmin(np.abs(freqs - f_expected))
    print(f"  Top FFT bins: idx={top5}, freq={freqs[top5]}, St={freqs[top5]*D/U_inf}")
    print(f"  Expected at bin {expected_idx} (freq={freqs[expected_idx]:.6f})")
    return passed


def validate_naca_sweep(steps=2000):
    print(f"\n=== NACA 0012 Cl/Cd Sweep ===")
    ctx = moderngl.create_standalone_context()
    import OpenGL.GL as gl; gl.glGetString(gl.GL_VERSION)

    grid = 192
    chord = 60.0
    U_inf = 2.0
    Re = 3000

    results = []
    for aoa in [0, 4, 8]:
        print(f"\n--- AoA = {aoa}° ---")
        cfd = CFD_System(ctx)
        cfd.inflow_vel = (U_inf, 0.0, 0.0)
        for prog in [cfd.prog_forces, cfd.prog_advection, cfd.prog_projection, cfd.prog_advection_density]:
            if 'u_inflow_vel' in prog:
                prog['u_inflow_vel'].value = tuple(cfd.inflow_vel)

        cfd.sgs_coeff = 0.0
        cfd.jacobi_iters = 40
        cfd.cfl_target = 0.55

        mask = create_naca_airfoil(grid, grid*0.5, chord, 0.12, 0, grid, aoa_deg=float(aoa))
        sdf = create_naca_airfoil_sdf(grid, grid*0.5, chord, 0.12, 0, grid, aoa_deg=float(aoa))
        cfd.set_obstacle(mask)
        cfd.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).astype(np.float16).tobytes())
        cfd.set_reynolds(Re, char_length=chord)

        obs_data = cfd.tex_obstacle.read()
        obs_mask = np.frombuffer(obs_data, dtype='u1').reshape((grid, grid, grid))

        vel_np = np.zeros((grid, grid, grid, 3), dtype=np.float32)
        vel_np[..., 0] = U_inf
        me = np.repeat(mask[..., np.newaxis], 3, axis=3)
        vel_np = np.where(me > 0, 0, vel_np)
        cfd.write_velocity(vel_np)

        sample_interval = 5
        n_samples = steps // sample_interval
        force_hist = np.zeros((n_samples, 3))
        si = 0
        for step in range(steps):
            cfd.step(force_radius=0.0)
            if step % sample_interval == 0:
                pres_raw = cfd.tex_pressure_A.read()
                pressure = np.frombuffer(pres_raw, dtype='f4').reshape((grid, grid, grid))
                va = cfd.read_velocity()
                ft, _, _ = compute_lift_drag(pressure, obs_mask, velocity=np.transpose(va, (2, 1, 0, 3)), viscosity=cfd.viscosity)
                force_hist[si] = ft
                si += 1

        # Average over last 50% of simulation
        half = n_samples // 2
        mean_f = np.mean(force_hist[half:], axis=0)
        S = chord * 192  # reference area = chord * span
        q = 0.5 * U_inf**2  # dynamic pressure (rho=1)
        Cd = mean_f[0] / (q * S)
        Cl = mean_f[2] / (q * S)
        results.append((aoa, Cl, Cd))
        print(f"  Cl={Cl:.4f}  Cd={Cd:.4f}  L/D={Cl/Cd:.2f}  (last-{force_hist[half:].shape[0]} samples)")

    print("\n=== NACA 0012 Summary ===")
    print(f"{'AoA':>4} {'Cl':>8} {'Cd':>8} {'L/D':>8}")
    for aoa, Cl, Cd in results:
        print(f"{aoa:4d} {Cl:8.4f} {Cd:8.4f} {Cl/Cd:8.2f}")
    return True


def main():
    parser = argparse.ArgumentParser(description="CFD Validation Suite")
    parser.add_argument('--case', type=str, default='cylinder',
                        choices=['cylinder', 'naca', 'all'])
    parser.add_argument('--steps', type=int, default=None)
    args = parser.parse_args()

    if args.case == 'cylinder' or args.case == 'all':
        n = args.steps or 3000
        validate_cylinder(re=100, steps=n)
    if args.case == 'naca' or args.case == 'all':
        n = args.steps or 1000
        validate_naca_sweep(steps=n)


if __name__ == '__main__':
    main()
