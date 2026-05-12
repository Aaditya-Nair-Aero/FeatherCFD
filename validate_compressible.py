#!/usr/bin/env python3
"""
Compressible Euler validation: cylinder/wedge at multiple Mach numbers.
Tests stability, shock structure, and max Mach convergence.
"""
import moderngl, numpy as np, time, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_utils import create_cylinder, create_sphere


def run_compressible_case(ctx, obstacle_type, Mach, steps=200, out_dir=None):
    from euler_solver import EulerSolver
    grid = 192
    gamma = 1.4
    p_ref = 1.0 / gamma
    rho_ref = 1.0
    c_ref = 1.0
    u_ref = Mach * c_ref
    
    euler = EulerSolver(ctx)
    euler.bc_type = 1
    euler.init_uniform(rho=rho_ref, u=u_ref, p=p_ref)
    
    if obstacle_type == 'cylinder':
        print(f"  Creating cylinder (D=30, spanwise)...")
        mask = create_cylinder(grid, grid/2, grid/2, 15.0, 1, grid-1)
        euler.set_obstacle(mask)
    elif obstacle_type == 'sphere':
        print(f"  Creating sphere (R=25)...")
        mask = create_sphere(grid, (grid/2, grid/2, grid/2), 25.0)
        euler.set_obstacle(mask)
    elif obstacle_type == 'wedge':
        print(f"  Creating wedge (15° ramp)...")
        mask = np.zeros((grid, grid, grid), dtype='u1')
        for x in range(grid):
            ramp_y = grid//2 - int(x * np.tan(np.radians(15)))
            if ramp_y < grid//2:
                mask[ramp_y:grid//2, :, x] = 1
        euler.set_obstacle(mask)
    
    mach_max_hist = []
    dt_hist = []
    t0 = time.perf_counter()
    
    for step in range(steps):
        euler.step()
        
        if step % 20 == 0 or step == steps - 1:
            mach = euler.compute_mach()
            mmax = float(np.max(mach))
            mach_max_hist.append((step, mmax))
            dt_hist.append((step, euler.dt))
            
            if step % 50 == 0:
                print(f"    step={step:4d}/{steps}  |M|_max={mmax:.3f}  dt={euler.dt:.6f}")
    
    elapsed = time.perf_counter() - t0
    
    U1, E = euler.read_state()
    rho = U1[:,:,:,0]
    u = np.where(rho > 1e-8, U1[:,:,:,1]/rho, 0.0)
    v = np.where(rho > 1e-8, U1[:,:,:,2]/rho, 0.0)
    w = np.where(rho > 1e-8, U1[:,:,:,3]/rho, 0.0)
    ke = 0.5 * (U1[:,:,:,1]**2 + U1[:,:,:,2]**2 + U1[:,:,:,3]**2) / np.maximum(rho, 1e-8)
    p = np.maximum((gamma - 1.0)*(E[:,:,:,0]-ke), 1e-8)
    c = np.sqrt(gamma * p / np.maximum(rho, 1e-8))
    speed = np.sqrt(u*u + v*v + w*w)
    mach_field = speed / np.maximum(c, 1e-8)
    
    result = {
        'obstacle': obstacle_type,
        'Mach': Mach,
        'steps': steps,
        'dt_final': euler.dt,
        'time_elapsed': elapsed,
        'mach_max_final': float(np.max(mach_field)),
        'mach_max_hist': np.array(mach_max_hist),
        'dt_hist': np.array(dt_hist),
        'valid': not (np.any(np.isnan(rho)) or np.any(np.isinf(rho))),
        'rho_min': float(np.min(rho)),
        'rho_max': float(np.max(rho)),
    }
    
    if out_dir and result['valid']:
        np.savez(os.path.join(out_dir, f'compressible_{obstacle_type}_M{Mach:.1f}.npz'),
                 rho=rho, u=u, v=v, w=w, p=p, mach=mach_field,
                 **{k: result[k] for k in ['obstacle', 'Mach', 'steps', 'dt_final']})
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Compressible Euler validation')
    parser.add_argument('--obstacle', type=str, default='cylinder',
                        choices=['cylinder', 'sphere', 'wedge'],
                        help='Obstacle type')
    parser.add_argument('--mach', type=str, default='0.5,1.5,2.0',
                        help='Comma-separated Mach numbers')
    parser.add_argument('--steps', type=int, default=200,
                        help='Steps per case')
    parser.add_argument('--out-dir', type=str, default='/home/aaditya/Downloads/tmp')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    Mach_list = [float(m) for m in args.mach.split(',')]
    obstacles = ['cylinder', 'sphere', 'wedge'] if args.obstacle == 'all' else [args.obstacle]
    
    ctx = moderngl.create_standalone_context()
    import OpenGL.GL as gl; gl.glGetString(gl.GL_VERSION)
    
    print(f"Compressible Euler Sweep")
    print(f"{'='*60}")
    
    all_results = []
    for obs in obstacles:
        for Mach in Mach_list:
            print(f"\nObstacle: {obs}, Mach={Mach}")
            print(f"{'-'*40}")
            t0 = time.perf_counter()
            result = run_compressible_case(ctx, obs, Mach, steps=args.steps, out_dir=args.out_dir)
            elapsed = time.perf_counter() - t0
            status = "OK" if result['valid'] else "FAIL (NaN/inf detected)"
            print(f"  Final |M|_max={result['mach_max_final']:.3f}  dt={result['dt_final']:.6f}  {status}")
            print(f"  ρ range: [{result['rho_min']:.4f}, {result['rho_max']:.4f}]  ({elapsed:.0f}s)")
            all_results.append(result)
    
    print(f"\n{'='*60}")
    print(f"Compressible Euler Summary")
    print(f"{'='*60}")
    header = f"{'Obstacle':>10} {'Mach':>5} {'Steps':>6} {'|M|_max':>8} {'dt':>8} {'Status':>8}"
    print(header)
    print('-' * len(header))
    for r in all_results:
        status = "OK" if r['valid'] else "FAIL"
        print(f"{r['obstacle']:>10} {r['Mach']:5.1f} {r['steps']:6d} {r['mach_max_final']:8.3f} {r['dt_final']:8.5f} {status:>8}")


if __name__ == '__main__':
    main()
