# FeatherCFD
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

Some utilities currently contain default output paths pointing to:

/home/aaditya/Downloads/tmp

These are primarily convenience defaults for development and can be overridden via CLI arguments or environment variables.

<img width="1920" height="1034" alt="Screenshot from 2026-05-12 11-31-38" src="https://github.com/user-attachments/assets/28597f62-dd6f-4eda-8f7d-36ac8c0c1e69" />
<img width="1920" height="1036" alt="Screenshot from 2026-05-12 11-30-02" src="https://github.com/user-attachments/assets/ffd5590a-1242-4315-8f02-06130c81645a" />
<img width="1920" height="1034" alt="Screenshot from 2026-05-12 11-28-31" src="https://github.com/user-attachments/assets/84f5d354-f4db-430a-92cc-79600a801184" />
<img width="1920" height="1034" alt="Screenshot from 2026-05-12 11-27-33" src="https://github.com/user-attachments/assets/d99d2fe6-798b-45c8-ae41-49a614d7be0a" />
<img width="1911" height="1032" alt="Screenshot from 2026-05-12 11-25-18" src="https://github.com/user-attachments/assets/b3ebaa00-51e9-4146-8322-e8bfa95e9c26" />


License

MIT License

Disclaimer

FeatherCFD is an experimental research and educational project and is not validated for safety-critical engineering use.
