#!/usr/bin/env python3
"""
Sod shock tube validation:
Runs GPU Euler solver, extracts centerline, computes exact Riemann solution, plots comparison.
"""
import moderngl, numpy as np, time, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def sod_exact(x, x0, t, gamma=1.4):
    """Exact solution of Sod shock tube at position x and time t.
    
    Left:  rhoL=1.0, uL=0.0, pL=1.0
    Right: rhoR=0.125, uR=0.0, pR=0.1
    Diaphragm at x=x0.
    """
    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)
    
    p3_guess = 0.5 * (pL + pR)
    for _ in range(50):
        if p3_guess <= pL:
            A = 2.0 / ((gamma + 1.0) * rhoL)
            B = (gamma - 1.0) / (gamma + 1.0) * pL
            fL = np.sqrt(A / (p3_guess + B)) if p3_guess + B > 0 else 0.0
            fL *= (p3_guess - pL)
        else:
            A = 2.0 / ((gamma + 1.0) * rhoL)
            B = (gamma - 1.0) / (gamma + 1.0) * pL
            fL = np.sqrt(A / (pL + B)) * (p3_guess - pL) / np.sqrt(p3_guess / pL + B)
        
        A = 2.0 / ((gamma + 1.0) * rhoR)
        B = (gamma - 1.0) / (gamma + 1.0) * pR
        denom = p3_guess + B
        if denom <= 0:
            fR = 0.0
        else:
            fR = np.sqrt(A / denom) * (p3_guess - pR)
        
        p3_new = p3_guess - (fL + fR + uR - uL) / (1.0 / (rhoL * cL) + 1.0 / (rhoR * cR))
        if abs(p3_new - p3_guess) < 1e-12:
            break
        p3_guess = max(p3_new, 1e-12)
    
    p3 = p3_guess
    
    A = 2.0 / ((gamma + 1.0) * rhoR)
    B = (gamma - 1.0) / (gamma + 1.0) * pR
    u3 = uR + np.sqrt(A / (p3 + B)) * (p3 - pR)
    
    if p3 > pL:
        rho3L = rhoL * (p3/pL + (gamma-1)/(gamma+1)) / ((gamma-1)/(gamma+1) * p3/pL + 1)
    else:
        rho3L = rhoL * (p3/pL) ** (1.0/gamma)
    
    rho3R = rhoR * (p3/pR + (gamma-1)/(gamma+1)) / ((gamma-1)/(gamma+1) * p3/pR + 1)
    
    c3L = np.sqrt(gamma * p3 / rho3L)
    c3R = np.sqrt(gamma * p3 / rho3R)
    
    S_HL = uL - cL  # head of rarefaction
    S_TL = u3 - c3L  # tail of rarefaction
    
    vs = uR + cR * np.sqrt((gamma + 1) * p3 / (2 * gamma * pR) + (gamma - 1) / (2 * gamma))
    
    vc = u3
    
    out_rho = np.zeros_like(x, dtype=np.float64)
    out_u = np.zeros_like(x, dtype=np.float64)
    out_p = np.zeros_like(x, dtype=np.float64)
    
    xi = (x - x0) / t  # similarity variable
    
    for i, xi_i in enumerate(xi):
        if xi_i <= S_HL:
            out_rho[i] = rhoL
            out_u[i] = uL
            out_p[i] = pL
        elif xi_i <= S_TL:
            u_fan = 2.0/(gamma+1) * (cL + (gamma-1)/2 * uL + xi_i)
            c_fan = u_fan - xi_i
            out_u[i] = u_fan
            out_rho[i] = rhoL * (c_fan/cL) ** (2.0/(gamma-1))
            out_p[i] = pL * (c_fan/cL) ** (2.0*gamma/(gamma-1))
        elif xi_i <= vc:
            out_rho[i] = rho3L
            out_u[i] = u3
            out_p[i] = p3
        elif xi_i <= vs:
            out_rho[i] = rho3R
            out_u[i] = u3
            out_p[i] = p3
        else:
            out_rho[i] = rhoR
            out_u[i] = uR
            out_p[i] = pR
    
    return out_rho, out_u, out_p


def run_sod_simulation(steps=200, out_dir=None):
    ctx = moderngl.create_standalone_context()
    import OpenGL.GL as gl; gl.glGetString(gl.GL_VERSION)
    
    from euler_solver import EulerSolver
    euler = EulerSolver(ctx)
    euler.bc_type = 0
    euler.set_inflow(rho=1.0, u=0.0, v=0.0, w=0.0, p=1.0/1.4)
    euler.init_sod()
    
    t0 = time.perf_counter()
    for step in range(steps):
        euler.step()
        if step > 0 and step % 50 == 0:
            print(f"  step={step}/{steps}  dt={euler.dt:.6f}")
    elapsed = time.perf_counter() - t0
    
    U1, E = euler.read_state()
    rho = U1[:,:,:,0]
    u = np.where(rho > 1e-8, U1[:,:,:,1]/rho, 0.0)
    ke = 0.5 * (U1[:,:,:,1]**2 + U1[:,:,:,2]**2 + U1[:,:,:,3]**2) / np.maximum(rho, 1e-8)
    p = np.maximum((1.4-1.0)*(E[:,:,:,0]-ke), 1e-8)
    
    t_sim = steps * euler.dt  # approximate
    
    return {
        'rho': rho, 'u': u, 'p': p,
        'dt': euler.dt, 't_sim': t_sim, 'steps': steps,
        'elapsed': elapsed,
    }


def plot_sod_comparison(sol, x0=96, out_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    N = sol['rho'].shape[0]
    cy, cz = N//2, N//2
    
    rho_num = sol['rho'][cz, cy, :]
    u_num = sol['u'][cz, cy, :]
    p_num = sol['p'][cz, cy, :]
    t_sim = sol['t_sim']
    
    x_grid = np.arange(N, dtype=np.float64)
    rho_exact, u_exact, p_exact = sod_exact(x_grid, x0, t_sim)
    
    rho_err = np.mean(np.abs(rho_num - rho_exact)) / np.mean(rho_exact)
    u_err = np.mean(np.abs(u_num - u_exact)) / (np.max(u_exact) - np.min(u_exact))
    p_err = np.mean(np.abs(p_num - p_exact)) / np.mean(p_exact)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    labels = [
        (r'Density $\rho$', rho_num, rho_exact),
        (r'Velocity $u$', u_num, u_exact),
        (r'Pressure $p$', p_num, p_exact),
    ]
    errs = [rho_err, u_err, p_err]
    
    for ax, (title, num, exact), err in zip(axes, labels, errs):
        ax.plot(x_grid, num, 'b-', linewidth=1.5, label='Numerical (192³)')
        ax.plot(x_grid, exact, 'r--', linewidth=1.5, label='Exact Riemann')
        ax.set_xlabel('x (grid cells)')
        ax.set_ylabel(title)
        ax.set_title(f'{title}  (error={err*100:.1f}%)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, N-1)
    
    fig.suptitle(f'Sod Shock Tube — {sol["steps"]} steps, t$\\approx${t_sim:.2f}, dt={sol["dt"]:.4f}',
                 fontsize=13)
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {out_path}")
    else:
        plt.show()
    
    plt.close(fig)
    return {'rho_err': rho_err, 'u_err': u_err, 'p_err': p_err}


def main():
    parser = argparse.ArgumentParser(description='Sod shock tube validation')
    parser.add_argument('--steps', type=int, default=200, help='Simulation steps')
    parser.add_argument('--out-dir', type=str, default='/home/aaditya/Downloads/tmp',
                        help='Output directory for plot and data')
    parser.add_argument('--x0', type=float, default=96.0, help='Initial discontinuity position')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Running Sod simulation for {args.steps} steps...")
    sol = run_sod_simulation(steps=args.steps)
    print(f"  Steps: {sol['steps']}, dt={sol['dt']:.6f}, t_sim={sol['t_sim']:.2f}")
    print(f"  GPU time: {sol['elapsed']:.1f}s")
    
    np.savez(os.path.join(args.out_dir, 'sod_numerical.npz'),
             rho=sol['rho'], u=sol['u'], p=sol['p'],
             steps=sol['steps'], dt=sol['dt'], t_sim=sol['t_sim'])
    
    plot_path = os.path.join(args.out_dir, 'sod_validation.png')
    errs = plot_sod_comparison(sol, x0=args.x0, out_path=plot_path)
    
    print(f"\nNormalized errors:")
    print(f"  Density:   {errs['rho_err']*100:.2f}%")
    print(f"  Velocity:  {errs['u_err']*100:.2f}%")
    print(f"  Pressure:  {errs['p_err']*100:.2f}%")


if __name__ == '__main__':
    main()
