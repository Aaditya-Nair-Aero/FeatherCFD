#!/usr/bin/env python3
"""Rigorous validation: compression corner oblique shock angle vs θ-β-M theory."""
import moderngl, numpy as np, time, sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euler_solver import EulerSolver


def oblique_shock_theory(M, theta_deg, gamma=1.4):
    """Solve θ-β-M relation for shock angle β given M and θ.
    Returns weak shock angle in degrees, or 0 if no attached solution."""
    theta = math.radians(theta_deg)
    if theta < 1e-10:
        return math.degrees(math.asin(1.0 / M))

    def theta_of_beta(beta):
        """Return deflection angle θ for given shock angle β."""
        s = math.sin(beta)
        num = 2.0 / math.tan(beta) * (M**2 * s**2 - 1)
        den = M**2 * (gamma + math.cos(2*beta)) + 2
        return math.atan2(num, den)

    mu = math.asin(1.0 / M)
    if theta_deg >= 90:
        return 0.0

    phi = (math.sqrt(5) - 1) / 2
    a, b = mu + 0.001, math.pi / 2 - 0.001
    for _ in range(30):
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        if theta_of_beta(c) < theta_of_beta(d):
            a = c
        else:
            b = d
    beta_max = (a + b) / 2
    theta_max = theta_of_beta(beta_max)

    if theta > theta_max:
        return 0.0

    lo, hi = mu + 0.001, beta_max
    for _ in range(50):
        mid = (lo + hi) / 2
        if theta_of_beta(mid) < theta:
            lo = mid
        else:
            hi = mid

    return math.degrees((lo + hi) / 2)


def run_wedge_validation(Mach, theta_deg=10, steps=1000):
    grid = 192
    gamma = 1.4; p_ref = 1.0/gamma; c_ref = 1.0

    x0 = 48
    mask = np.zeros((grid, grid, grid), dtype='u1')
    for x in range(x0, grid):
        ramp_z = int((x - x0) * math.tan(math.radians(theta_deg)))
        if ramp_z < grid:
            mask[x, :, 1:ramp_z+1] = 1

    euler = EulerSolver(ctx)
    euler.bc_type = 1
    euler.cfl_target = 0.5
    euler.dt_update_interval = 5
    euler.init_uniform(rho=1.0, u=Mach*c_ref, p=p_ref)
    euler.set_obstacle(mask)

    for step in range(steps):
        euler.step()
        if step > 0 and step % 200 == 0:
            mach = euler.compute_mach()
            mmax = float(np.max(mach))
            print(f"    step={step}/{steps}  M_max={mmax:.3f}  dt={euler.dt:.5f}")

    U1, E = euler.read_state()
    rho = U1[:,:,:,0]
    mid_y = grid // 2

    has_nan = bool(np.any(np.isnan(rho)) or np.any(np.isinf(rho)))
    if has_nan:
        return {'Mach': Mach, 'theta': theta_deg, 'beta_numerical': 0,
                'beta_theory': 0, 'beta_error': 999, 'valid': False}

    mach = euler.compute_mach()
    mmax = float(np.max(mach))

    grad_z = np.abs(np.diff(rho[:, mid_y, :], axis=0))
    grad_z = np.pad(grad_z, ((0,1),(0,0)), mode='edge')

    points_shock = []
    for x in range(x0 + 10, min(x0 + 120, grid)):
        ramp_top = int((x - x0) * math.tan(math.radians(theta_deg)))
        search_start = max(ramp_top + 3, 3)
        if search_start >= grid - 3:
            continue
        z_line = grad_z[:, x]
        shock_z = np.argmax(z_line[search_start:]) + search_start
        if ramp_top + 2 < shock_z < grid - 2:
            points_shock.append((x, shock_z))

    beta_theory = oblique_shock_theory(Mach, theta_deg)

    if len(points_shock) < 5 or beta_theory < 1:
        return {
            'Mach': Mach, 'theta': theta_deg,
            'beta_numerical': 0, 'beta_theory': beta_theory,
            'beta_error': 999, 'rho_ratio': 0, 'rho_ratio_theory': 0,
            'mach_max': mmax,
            'valid': False,
        }

    xs = np.array([p[0] for p in points_shock])
    zs = np.array([p[1] for p in points_shock])
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, zs, rcond=None)[0]
    beta_numerical = math.degrees(math.atan(slope))

    x_sample = min(x0 + 80, grid - 10)
    shock_z_at_x = int(slope * x_sample + intercept)
    shock_z_at_x = max(5, min(shock_z_at_x, grid - 5))
    ramp_top_at_x = int((x_sample - x0) * math.tan(math.radians(theta_deg)))
    shock_start = max(ramp_top_at_x + 2, shock_z_at_x - 12)
    shock_end = min(grid - 1, shock_z_at_x + 12)
    rho_profile = rho[shock_start:shock_end, mid_y, x_sample]
    if len(rho_profile) < 4:
        rho_ratio = 1.0
    else:
        pre = float(np.max(rho_profile[:len(rho_profile)//2]))
        post = float(np.min(rho_profile[len(rho_profile)//2:]))
        rho_ratio = pre / max(post, 1e-10)
        if rho_ratio < 1.0:
            rho_ratio = 1.0 / max(rho_ratio, 1e-10)

    M_n = Mach * math.sin(math.radians(beta_theory))
    rho_ratio_theory = (gamma+1)*M_n**2 / ((gamma-1)*M_n**2 + 2)

    return {
        'Mach': Mach, 'theta': theta_deg,
        'beta_numerical': beta_numerical,
        'beta_theory': beta_theory,
        'beta_error': abs(beta_numerical - beta_theory),
        'rho_ratio': rho_ratio,
        'rho_ratio_theory': rho_ratio_theory,
        'mach_max': mmax,
        'valid': beta_numerical > 5,
    }


ctx = moderngl.create_standalone_context()
import OpenGL.GL as gl; gl.glGetString(gl.GL_VERSION)

theta_test = 10
print(f"Compression Corner Oblique Shock Validation (θ={theta_test}°)")
print(f"{'='*70}")
print(f"{'Mach':>5} {'β_num':>8} {'β_theory':>8} {'error°':>8} {'M_max':>6} {'ρ_ratio_num':>12} {'ρ_ratio_th':>12}")
print(f"{'-'*70}")

for Mach in [1.5, 2.0, 2.5, 3.0]:
    t0 = time.perf_counter()
    r = run_wedge_validation(Mach, theta_deg=theta_test, steps=1000)
    elapsed = time.perf_counter() - t0
    status = "OK" if r['valid'] else "FAIL"
    print(f"{r['Mach']:5.1f} {r['beta_numerical']:8.2f} {r['beta_theory']:8.2f} "
          f"{r['beta_error']:8.2f} {r.get('mach_max',0):6.2f} {r['rho_ratio']:12.4f} {r['rho_ratio_theory']:12.4f}  {status} ({elapsed:.0f}s)")
