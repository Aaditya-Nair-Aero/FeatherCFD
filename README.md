# FeatherCFD

Update- Vulkan Implementation completed

FeatherCFD is a GPU-native CFD framework for aerospace simulation and real-time volumetric flow visualization using Python, ModernGL, and OpenGL shaders. Features incompressible/compressible solvers, multigrid pressure projection, WENO-5 advection, CAD import, NACA airfoils, and aerodynamic validation tools.

Overview

FeatherCFD is a custom GPU-based CFD framework built from scratch using Python, ModernGL, and OpenGL shader compute pipelines. The solver simulates incompressible and compressible fluid flow entirely on the GPU using a 3D Eulerian grid architecture.

The project focuses on combining:

scientific simulation,
GPU compute,
aerospace engineering,
and interactive visualization

into a unified experimental CFD environment that runs on consumer hardware.

Features
Fluid Solvers
Incompressible Navier–Stokes solver
Compressible Euler solver
SSP-RK3 and RK4 time integration
Adaptive CFL timestepping
Numerical Methods
GPU multigrid pressure projection
Red-Black Gauss-Seidel smoothing
FFT/DST pressure solver paths
WENO-5 advection
BFECC dye transport
SDF embedded boundary handling
Aerodynamics
NACA airfoil generation
Adjustable angle of attack
Lift and drag force integration
Reynolds-number-based simulation control
Shock tube and wedge validation cases
Geometry Support
Sphere / cylinder / box obstacles
CAD mesh import (.stl, .obj)
GPU voxelization pipeline
Signed Distance Field obstacle generation
Visualization
Real-time volumetric raymarching
Velocity visualization
Dye/smoke rendering
Vorticity visualization
Mach number rendering
Schlieren-style density visualization
Architecture

FeatherCFD runs entirely on GPU fragment shader compute passes using layered rendering.

Pipeline overview:

Forces/BCs
    ↓
Advection
    ↓
Divergence
    ↓
Pressure Solve (Jacobi / Multigrid / FFT / DST)
    ↓
Projection
    ↓
Density Advection

Current grid:

192 × 192 × 192 (~7 million cells)

Hardware target:

RTX 3050 Laptop GPU (4 GB VRAM)
Project Goals

FeatherCFD is intended as:

an experimental CFD research environment,
a GPU simulation sandbox,
and a learning platform for numerical fluid dynamics and aerospace simulation.

Long-term roadmap includes:

immersed boundary methods,
higher-order compressible solvers,
CUDA/Vulkan backend,
adaptive mesh refinement,
and advanced turbulence modeling.
Current Known Issue

The volumetric visualizer currently contains a render-space transformation issue causing flow structures to visually drift downward in the viewer.

The underlying simulation fields and numerical results remain physically correct — the issue is isolated to visualization-space mapping and rendering transforms.

Example Run
python simulate_cfd.py \
    --steps 500 \
    --save-every 5 \
    --obstacle wing \
    --aoa 5 \
    --wind 2.0 \
    --vcycle
Dependencies

Core libraries:

moderngl
PyOpenGL
numpy
scipy
trimesh
glfw
moderngl_window
matplotlib
Validation

Current validation suite includes:

Sod shock tube
Oblique wedge shocks
Cylinder flow
NACA airfoil sweeps
Divergence stability checks
Boundary condition verification 

Update- Vulkan Implementation completed

Some utilities currently contain default output paths pointing to:

/home/aaditya/Downloads/tmp

These are primarily convenience defaults for development and can be overridden via CLI arguments or environment variables.

<img width="1920" height="1034" alt="Screenshot from 2026-05-12 11-31-38" src="https://github.com/user-attachments/assets/28597f62-dd6f-4eda-8f7d-36ac8c0c1e69" />
<img width="1920" height="1036" alt="Screenshot from 2026-05-12 11-30-02" src="https://github.com/user-attachments/assets/ffd5590a-1242-4315-8f02-06130c81645a" />
<img width="1920" height="1034" alt="Screenshot from 2026-05-12 11-28-31" src="https://github.com/user-attachments/assets/84f5d354-f4db-430a-92cc-79600a801184" />
<img width="1920" height="1034" alt="Screenshot from 2026-05-12 11-27-33" src="https://github.com/user-attachments/assets/d99d2fe6-798b-45c8-ae41-49a614d7be0a" />
<img width="1911" height="1032" alt="Screenshot from 2026-05-12 11-25-18" src="https://github.com/user-attachments/assets/b3ebaa00-51e9-4146-8322-e8bfa95e9c26" />

FeatherCFD Validation — All tests passing on RTX 3050, 192³ grid.


OpenGL Solver — Incompressible NS

  [1] validate_cfd.py
      Checks: divergence < 0.1, no-slip walls = 0 (exact), inflow BC = 1.0,
              no NaN/inf.
      Steps: 5
      Result: max div = 0.003590, wall velocity = 0.0 (exact)

  [2] validate_naca_sweep.py
      NACA 0012 sweep at Re = 1000–5000, AoA = 0°–10°.
      Reference: Sheldahl & Klimas (1981) SAND80-2114;
                 Jacobs & Sherman (1937) NACA Report 586.
      Steps: 400 per case
      Result: Cl and Cd stable across all Re/AoA.
      Lift-to-drag ratios within expected range for low-Re NACA 0012.

  [3] test_naca.py
      Long-duration NACA 0012 run.
      Steps: 500
      Result: No blowup, forces remain finite throughout.

  [4] test_naca_compare.py
      RBGS iterations vs V-cycle pressure solve.
      Reference: Internal benchmark.
      Steps: 300 per mode
      Result: Both solvers converge to comparable Cl.

  [5] test_naca_sgs.py
      Smagorinsky SGS model.
      Reference: Smagorinsky (1963); Lilly (1962).
      Steps: 300
      Result: Eddy viscosity active, solution remains stable.

  [6] test_naca_smoke.py
      Quick smoke test with GPU mask fix.
      Steps: 5
      Result: GPU obstacle mask cell count matches expectation.

  [7] test_fft.py
      FFT Poisson preconditioner.
      Steps: 30
      Result: Pressure solve converges, no divergence spikes.

  [8] test_dct_obstacle.py
      DST Poisson solve with obstacle.
      Steps: 50
      Result: Pressure field smooth, obstacle boundary conditions met.

  [9] test_clamp_modes.py
      Four advection clamp modes (0=std, 1=none, 2=SDF-aware, 3=local).
      Steps: 300 per mode
      Result: All modes produce finite lift.

OpenGL Solver — Compressible Euler


  [10] validate_sod.py
       Sod shock tube problem.
       Reference: Sod (1978) "A numerical study of gas dynamics";
                  Toro (1999) "Riemann Solvers and Numerical Methods".
       Initial conditions — Left: ρ=1.0, u=0.0, p=1.0;
                            Right: ρ=0.125, u=0.0, p=0.1.
       Comparison: exact Riemann solution (iterative solver, 50 iterations).
       Steps: variable (plots centerline at t=0.2).
       Result: ρ, u, p profiles match exact solution.
       Normalized error < 12%.

  [11] validate_wedge.py
       Compression corner oblique shock, θ = 10°.
       Reference: Anderson (2001) "Fundamentals of Aerodynamics";
                  NACA Report 1135 (θ-β-M relation).
       Mach sweep: 1.5, 2.0, 2.5, 3.0.
       Steps: 1000 per case.
       Result: Numerical shock angle matches θ-β-M theory.
       β error < 1° across all Mach numbers.

  [12] validate_compressible.py
       Compressible Euler obstacle sweep.
       Obstacles: cylinder, sphere, wedge.
       Mach: 0.5, 1.5, 2.0, 3.0.
       Steps: 200 per case.
       Result: Bow shocks form at all supersonic Mach numbers.
       Mach > inflow M upstream of obstacles, flow converges.
       No NaN/inf in any case.


Vulkan Compute Backend


  [13] test_diag_forces.py
       Vulkan upload/download axis mapping verification.
       Result: Byte data correctly transposed between CPU and GPU layout.

  [14] test_validate.py
       Full Vulkan CFD pipeline, NACA 0012 at AoA 5°.
       Steps: 100
       Result: Simulation runs without error.
       Forces remain finite, divergence bounded.

  [15] visualize_vk.py (standalone test)
       Vulkan raymarch viewer init + swapchain + frame upload.
       Frames: 150 (192³ RGBA16F)
       Result: Viewer initializes, swapchain created, textures uploaded,
               render loop runs at target framerate.


Validation Runner


  [16] run_validation.py
       Orchestrator: Sod shock tube (200 steps) +
                     compressible obstacles at M=0.5/1.5/2.0/3.0 (100 steps each) +
                     NACA 0012 incompressible at Re=5000, AoA=5° (100 steps).
       Reference: Combined references from [10], [11], [12], [2] above.
       Result: All cases pass.
       Report generated with per-case status.


License

MIT License

Disclaimer

FeatherCFD is an experimental research and educational project and is not validated for safety-critical engineering use.
