#!/usr/bin/env python3
"""
Automated Validation Runner for GPU CFD Solver.
Runs all validation cases and produces a summary report.
"""
import moderngl, numpy as np, time, os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_sod_validation(ctx, steps=200):
    """Sod shock tube vs exact Riemann solution"""
    print("\n[1/4] Sod shock tube validation...")
    from euler_solver import EulerSolver
    euler = EulerSolver(ctx)
    euler.bc_type = 0
    euler.init_sod()
    
    for _ in range(steps):
        euler.step()
    
    U1, E = euler.read_state()
    rho = U1[:,:,:,0]
    u = np.where(rho > 1e-8, U1[:,:,:,1]/rho, 0.0)
    ke = 0.5 * (U1[:,:,:,1]**2 + U1[:,:,:,2]**2 + U1[:,:,:,3]**2) / np.maximum(rho, 1e-8)
    p = np.maximum((1.4-1.0)*(E[:,:,:,0]-ke), 1e-8)
    
    N = 192
    cy, cz = N//2, N//2
    rho_num = rho[cz, cy, :]
    u_num = u[cz, cy, :]
    p_num = p[cz, cy, :]
    
    t_sim = steps * euler.dt
    x_grid = np.arange(N, dtype=np.float64)
    
    from validate_sod import sod_exact
    rho_exact, u_exact, p_exact = sod_exact(x_grid, N/2, t_sim)
    
    rho_err = float(np.mean(np.abs(rho_num - rho_exact)) / np.mean(rho_exact))
    u_err = float(np.mean(np.abs(u_num - u_exact)) / (np.max(u_exact) - np.min(u_exact) + 1e-10))
    p_err = float(np.mean(np.abs(p_num - p_exact)) / np.mean(p_exact))
    
    return {
        'test': 'Sod shock tube',
        'steps': steps, 't_sim': t_sim, 'dt': euler.dt,
        'rho_error_pct': rho_err * 100,
        'u_error_pct': u_err * 100,
        'p_error_pct': p_err * 100,
        'passed': max(rho_err, p_err) < 0.15,  # <15% normalized error
        'stable': True,
    }


def run_compressible_obstacle(ctx, obs_type, Mach, steps=200):
    """Compressible flow over obstacle at given Mach"""
    from euler_solver import EulerSolver
    from geometry_utils import create_cylinder, create_sphere
    
    gamma = 1.4
    p_ref = 1.0 / gamma
    u_ref = Mach * 1.0
    
    euler = EulerSolver(ctx)
    euler.bc_type = 1
    euler.init_uniform(rho=1.0, u=u_ref, p=p_ref)
    
    if obs_type == 'cylinder':
        mask = create_cylinder(192, 192/2, 192/2, 15.0, 1, 191)
    elif obs_type == 'sphere':
        mask = create_sphere(192, (96, 96, 96), 25.0)
    elif obs_type == 'wedge':
        mask = np.zeros((192, 192, 192), dtype='u1')
        for x in range(192):
            ramp_y = 96 - int(x * np.tan(np.radians(15)))
            if ramp_y < 96:
                mask[ramp_y:96, :, x] = 1
    euler.set_obstacle(mask)
    
    for _ in range(steps):
        euler.step()
    
    U1, E = euler.read_state()
    rho = U1[:,:,:,0]
    has_nan = bool(np.any(np.isnan(rho)) or np.any(np.isinf(rho)))
    
    mach = euler.compute_mach()
    mmax = float(np.max(mach))
    
    return {
        'test': f'Compressible {obs_type} @ M={Mach}',
        'obstacle': obs_type, 'Mach': Mach,
        'steps': steps, 'dt_final': euler.dt,
        'mach_max': mmax,
        'stable': not has_nan,
        'passed': not has_nan and mmax > Mach * 0.9,
    }


def run_incompressible_naca(ctx, Re=5000, aoa=5, steps=200):
    """NACA 0012 lift/drag at given Re and AoA (diagnostic)"""
    from cfd_sim import CFD_System
    from geometry_utils import create_naca_airfoil, create_naca_airfoil_sdf
    from simulate_cfd import compute_lift_drag
    
    chord = 60.0
    U_inf = 2.0
    
    cfd = CFD_System(ctx)
    cfd.inflow_vel = (U_inf, 0.0, 0.0)
    for prog in [cfd.prog_forces, cfd.prog_advection, cfd.prog_projection, cfd.prog_advection_density]:
        if 'u_inflow_vel' in prog:
            prog['u_inflow_vel'].value = tuple(cfd.inflow_vel)
    
    mask = create_naca_airfoil(192, 96, chord, 0.12, 0, 192, aoa_deg=float(aoa))
    sdf = create_naca_airfoil_sdf(192, 96, chord, 0.12, 0, 192, aoa_deg=float(aoa))
    cfd.set_obstacle(mask)
    cfd.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).astype(np.float16).tobytes())
    cfd.set_reynolds(Re, char_length=chord)
    cfd.sgs_coeff = 0.0
    cfd.jacobi_iters = 120
    cfd.cfl_target = 0.12
    cfd.dt_update_interval = 1
    cfd._update_dt()
    
    obs_data = cfd.tex_obstacle.read()
    obs_mask = np.frombuffer(obs_data, dtype='u1').reshape((192, 192, 192))
    
    vel_init = np.full((192, 192, 192, 4), 0, dtype=np.float16)
    vel_init[..., 0] = U_inf
    me = np.repeat(mask[..., np.newaxis], 4, axis=3)
    vel_init = np.where(me > 0, 0, vel_init)
    cfd.tex_velocity_A.write(np.transpose(vel_init, (2, 1, 0, 3)).tobytes())
    cfd.tex_velocity_B.write(np.transpose(vel_init, (2, 1, 0, 3)).tobytes())
    
    force_hist = []
    for step in range(steps):
        cfd.step(force_radius=0.0)
        if step % 5 == 0:
            pres = np.frombuffer(cfd.tex_pressure_A.read(), dtype='f4').reshape((192, 192, 192))
            vel = np.frombuffer(cfd.tex_velocity_A.read(), dtype=np.float16).reshape((192, 192, 192, 4))
            ft, _, _ = compute_lift_drag(pres, obs_mask, velocity=vel[:,:,:,:3], viscosity=cfd.viscosity)
            force_hist.append(ft)
    
    mean_f = np.mean(force_hist[-20:], axis=0)
    S = chord * 192
    q = 0.5 * U_inf**2
    Cd = float(mean_f[0] / (q * S))
    Cl = float(mean_f[2] / (q * S))
    
    cl_expected = 0.55
    cl_ratio = Cl / cl_expected if cl_expected > 0 else 1.0
    
    return {
        'test': f'NACA 0012 @ Re={Re}, AoA={aoa}°',
        'Re': Re, 'AoA': aoa, 'steps': steps,
        'Cl': Cl, 'Cd': Cd, 'L/D': Cl/Cd if Cd > 1e-10 else 0.0,
        'stable': True,
        'passed': abs(cl_ratio) > 0.3,  # Should be at least 30% of expected
    }


def run_all(ctx, out_dir, sod_steps=200, comp_steps=100, naca_steps=100):
    results = []
    
    results.append(run_sod_validation(ctx, steps=sod_steps))
    print(f"  → {'PASS' if results[-1]['passed'] else 'FAIL'}  (ρ err={results[-1]['rho_error_pct']:.1f}%)")
    
    for obs in ['cylinder', 'sphere', 'wedge']:
        for Mach in [0.5, 1.5, 2.0, 3.0]:
            if obs == 'wedge' and Mach == 0.5:
                continue  # skip subsonic wedge (no shock forms)
            r = run_compressible_obstacle(ctx, obs, Mach, steps=comp_steps)
            results.append(r)
            status = 'PASS' if r['passed'] else 'FAIL'
            print(f"  → {status}  |M|_max={r['mach_max']:.2f}  dt={r['dt_final']:.4f}  "
                  f"{'☠ NaN' if not r['stable'] else ''}")
    
    r = run_incompressible_naca(ctx, Re=5000, aoa=5, steps=naca_steps)
    results.append(r)
    status = 'PASS' if r['passed'] else 'FAIL'
    print(f"  → {status}  Cl={r['Cl']:.4f}  Cd={r['Cd']:.4f}  L/D={r['L/D']:.2f}")
    
    return results


def generate_report(results, out_path):
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    lines = []
    lines.append("=" * 70)
    lines.append("GPU CFD SOLVER — VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Passed: {passed}/{total}  ({passed/total*100:.0f}%)")
    lines.append("")
    
    for r in results:
        status = "✓" if r['passed'] else "✗"
        name = r['test']
        lines.append(f"  [{status}] {name}")
        
        if 'rho_error_pct' in r:
            lines.append(f"         ρ err: {r['rho_error_pct']:.2f}%  "
                        f"u err: {r['u_error_pct']:.2f}%  "
                        f"p err: {r['p_error_pct']:.2f}%")
        if 'mach_max' in r:
            lines.append(f"         M_max={r['mach_max']:.3f}  dt={r['dt_final']:.5f}  "
                        f"stable={'yes' if r['stable'] else 'NO'}")
        if 'Cl' in r:
            lines.append(f"         Cl={r['Cl']:.4f}  Cd={r['Cd']:.4f}  L/D={r['L/D']:.2f}")
        lines.append("")
    
    lines.append("-" * 70)
    lines.append(f"Summary: {passed}/{total} tests passed")
    lines.append("=" * 70)
    
    report = '\n'.join(lines)
    print(report)
    
    if out_path:
        with open(out_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {out_path}")
    
    json_path = out_path.replace('.txt', '.json') if out_path else None
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"JSON results saved to: {json_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description='CFD Validation Runner')
    parser.add_argument('--sod-steps', type=int, default=200)
    parser.add_argument('--comp-steps', type=int, default=100)
    parser.add_argument('--naca-steps', type=int, default=500)
    parser.add_argument('--out-dir', type=str, default='/home/aaditya/Downloads/tmp')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"CFD Validation Runner")
    print(f"GPU: initializing...")
    ctx = moderngl.create_standalone_context()
    import OpenGL.GL as gl; gl.glGetString(gl.GL_VERSION)
    print(f"GPU: {ctx.info['GL_RENDERER']}")
    print(f"Grid: 192³")
    print(f"Sod steps: {args.sod_steps}  |  Compressible steps: {args.comp_steps}  |  NACA steps: {args.naca_steps}")
    
    t0 = time.perf_counter()
    results = run_all(ctx, args.out_dir,
                      sod_steps=args.sod_steps,
                      comp_steps=args.comp_steps,
                      naca_steps=args.naca_steps)
    elapsed = time.perf_counter() - t0
    
    report_path = os.path.join(args.out_dir, 'validation_report.txt')
    generate_report(results, report_path)
    
    print(f"\nTotal validation time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
