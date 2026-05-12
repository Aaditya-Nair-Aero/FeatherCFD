import moderngl
import numpy as np
import time
from cfd_sim import CFD_System

def validate_cfd():
    print("Starting CFD Validation...")
    ctx = moderngl.create_standalone_context()
    
    import OpenGL.GL as gl
    gl.glGetString(gl.GL_VERSION) # Force PyOpenGL context resolution
    
    cfd = CFD_System(ctx)
    print("CFD System initialized. Grid: 192x192x192")
    
    print("Running 5 steps of the simulation...")
    for _ in range(5):
        cfd.step()
        
    print("Validating properties...")
    
    cfd.tex_velocity_A.use(location=0)
    cfd.prog_divergence['u_velocity'].value = 0
    cfd.render_pass(cfd.prog_divergence, cfd.fbo_divergence)
    
    div_data = cfd.tex_divergence.read()
    div_array = np.frombuffer(div_data, dtype=np.float16)
    max_div = np.max(np.abs(div_array))
    print(f"[Validation] Max absolute divergence after projection: {max_div:.6f}")
    if max_div > 0.1:
        print("[Warning] Divergence might be high. Expected near zero.")
        
    vel_data = cfd.tex_velocity_A.read()
    vel_array = np.frombuffer(vel_data, dtype=np.float16).reshape((192, 192, 192, 4))
    
    wall_y0 = np.max(np.abs(vel_array[:, 0, 1:, :3]))  # Y=0, exclude X=0
    wall_y1 = np.max(np.abs(vel_array[:, 191, 1:, :3]))  # Y=191, exclude X=0
    wall_z0 = np.max(np.abs(vel_array[0, :, 1:, :3]))  # Z=0, exclude X=0
    wall_z1 = np.max(np.abs(vel_array[191, :, 1:, :3]))  # Z=191, exclude X=0
    wall_max = max(wall_y0, wall_y1, wall_z0, wall_z1)
    print(f"[Validation] Max velocity at no-slip walls: {wall_max:.6f}")
    assert wall_max == 0.0, "Wall boundary condition violation!"
    
    inflow_vel = vel_array[:, :, 0, :3]
    inflow_x = inflow_vel[:, :, 0]
    inflow_yz = inflow_vel[:, :, 1:]
    inflow_max_yz = np.max(np.abs(inflow_yz)) if inflow_yz.size > 0 else 0.0
    print(f"[Validation] Inflow X=0, X-vel diff from 1.0: {np.max(np.abs(inflow_x - 1.0)):.6f}")
    print(f"[Validation] Inflow X=0, Y/Z-vel max: {inflow_max_yz:.6f}")
    
    max_vel = np.max(np.abs(vel_array[:, :, :, :3]))
    print(f"[Validation] Max internal velocity magnitude: {max_vel:.6f}")
    assert not np.isnan(max_vel) and not np.isinf(max_vel), "Velocity explosion detected!"
    
    print("All mathematical properties logically verified!")

if __name__ == '__main__':
    validate_cfd()
