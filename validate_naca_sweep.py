#!/usr/bin/env python3
"""
NACA 0012 sweep at Re=1000-5000 for incompressible solver.
Compares Cl/Cd against known low-Re NACA 0012 data.
"""
import moderngl, numpy as np, time, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_utils import create_naca_airfoil, create_naca_airfoil_sdf
from simulate_cfd import compute_lift_drag


def run_naca_case(ctx, U_inf, chord, Re, aoa, steps, sample_interval=5):
    from cfd_sim import CFD_System
    grid = 192
    
    cfd = CFD_System(ctx)
    cfd.inflow_vel = (U_inf, 0.0, 0.0)
    for prog in [cfd.prog_forces, cfd.prog_advection, cfd.prog_projection, cfd.prog_advection_density]:
        if 'u_inflow_vel' in prog:
            prog['u_inflow_vel'].value = tuple(cfd.inflow_vel)
    
    mask = create_naca_airfoil(grid, grid*0.5, chord, 0.12, 0, grid, aoa_deg=float(aoa))
    sdf = create_naca_airfoil_sdf(grid, grid*0.5, chord, 0.12, 0, grid, aoa_deg=float(aoa))
    cfd.set_obstacle(mask)
    cfd.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).astype(np.float16).tobytes())
    cfd.set_reynolds(Re, char_length=chord)
    
    obs_data = cfd.tex_obstacle.read()
    obs_mask = np.frombuffer(obs_data, dtype='u1').reshape((grid, grid, grid))
    
    vel_init = np.full((grid, grid, grid, 4), 0, dtype=np.float16)
    vel_init[..., 0] = U_inf
    me = np.repeat(mask[..., np.newaxis], 4, axis=3)
    vel_init = np.where(me > 0, 0, vel_init)
    vit = np.transpose(vel_init, (2, 1, 0, 3))
    cfd.tex_velocity_A.write(vit.tobytes())
    cfd.tex_velocity_B.write(vit.tobytes())
    
    n_samples = steps // sample_interval
    force_hist = np.zeros((n_samples, 3))
    si = 0
    for step in range(steps):
        cfd.step(force_radius=0.0)
        if step % sample_interval == 0:
            pres_raw = cfd.tex_pressure_A.read()
            pressure = np.frombuffer(pres_raw, dtype='f4').reshape((grid, grid, grid))
            vel_raw = cfd.tex_velocity_A.read()
            va = np.frombuffer(vel_raw, dtype=np.float16).reshape((grid, grid, grid, 4))
            ft, _, _ = compute_lift_drag(pressure, obs_mask, velocity=va[:,:,:,:3], viscosity=cfd.viscosity)
            force_hist[si] = ft
            si += 1
    
    n_avg = max(1, n_samples * 40 // 100)
    mean_f = np.mean(force_hist[-n_avg:], axis=0)
    
    S = chord * grid  # reference area
    q = 0.5 * U_inf**2
    Cd = mean_f[0] / (q * S)
    Cl = mean_f[2] / (q * S)
    
    return Cl, Cd, force_hist


def main():
    parser = argparse.ArgumentParser(description='NACA 0012 Re sweep')
    parser.add_argument('--steps', type=int, default=400, help='Steps per case')
    parser.add_argument('--re', type=str, default=None, help='Comma-separated Re values, e.g. 1000,5000')
    parser.add_argument('--aoa', type=str, default=None, help='Comma-separated AoA values, e.g. 0,5,10')
    parser.add_argument('--out-dir', type=str, default='/home/aaditya/Downloads/tmp')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    U_inf = 2.0
    chord = 60.0
    re_arg = args.re.split(',') if args.re else []
    AoA_arg = args.aoa.split(',') if args.aoa else []
    Re_list = [int(r) for r in re_arg] if re_arg else [5000]
    AoA_list = [int(a) for a in AoA_arg] if AoA_arg else [0, 5, 10]
    
    ctx = moderngl.create_standalone_context()
    import OpenGL.GL as gl; gl.glGetString(gl.GL_VERSION)
    
    all_results = {}  # Re -> [(aoa, Cl, Cd), ...]
    
    for Re in Re_list:
        print(f"\n{'='*60}")
        print(f"Reynolds number: Re = {Re}")
        print(f"{'='*60}")
        all_results[Re] = []
        
        for aoa in AoA_list:
            print(f"\n  --- AoA = {aoa}° ---")
            t0 = time.perf_counter()
            Cl, Cd, force_hist = run_naca_case(ctx, U_inf, chord, Re, aoa, args.steps)
            elapsed = time.perf_counter() - t0
            print(f"  Cl={Cl:.4f}  Cd={Cd:.4f}  L/D={Cl/Cd:.2f}  ({elapsed:.0f}s)")
            all_results[Re].append((aoa, Cl, Cd))
            
            np.savez(os.path.join(args.out_dir, f'naca_Re{Re}_aoa{aoa}.npz'),
                     forces=force_hist, Cl=Cl, Cd=Cd, Re=Re, aoa=aoa,
                     U_inf=U_inf, chord=chord)
    
    print(f"\n{'='*60}")
    print(f"NACA 0012 Sweep Summary")
    print(f"{'='*60}")
    header = f"{'Re':>5} {'AoA':>4} {'Cl':>8} {'Cd':>8} {'L/D':>8}"
    print(header)
    print('-' * len(header))
    for Re in Re_list:
        for aoa, Cl, Cd in all_results[Re]:
            ld = Cl / Cd if Cd > 1e-10 else 0.0
            print(f"{Re:5d} {aoa:4d} {Cl:8.4f} {Cd:8.4f} {ld:8.2f}")
    
    summary = {'Re_list': Re_list, 'AoA_list': AoA_list,
               'results': {str(Re): all_results[Re] for Re in Re_list}}
    np.save(os.path.join(args.out_dir, 'naca_sweep_summary.npy'), summary)
    print(f"\nSummary saved to: {os.path.join(args.out_dir, 'naca_sweep_summary.npy')}")


if __name__ == '__main__':
    main()
