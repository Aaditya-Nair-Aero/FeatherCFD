import moderngl
import numpy as np
import OpenGL.GL as gl

class CFD_System:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.grid_size = 192
        self.dt = 0.16
        self.viscosity = 1e-5

        self.tex_velocity_A = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 4, dtype='f2')
        self.tex_velocity_B = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 4, dtype='f2')
        self.tex_velocity_A.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_velocity_B.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_velocity_A.repeat_x = False
        self.tex_velocity_A.repeat_y = False
        self.tex_velocity_A.repeat_z = False
        self.tex_velocity_B.repeat_x = False
        self.tex_velocity_B.repeat_y = False
        self.tex_velocity_B.repeat_z = False

        self.tex_pressure_A = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='f4')
        self.tex_pressure_B = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='f4')
        self.tex_pressure_A.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex_pressure_B.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.tex_divergence = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='f2')
        self.tex_divergence.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.tex_density_A = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='f2')
        self.tex_density_B = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='f2')
        self.tex_density_A.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_density_B.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_density_A.write(np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float16).tobytes())
        self.tex_density_B.write(np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float16).tobytes())

        self.tex_obstacle = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='u1')
        self.tex_obstacle.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex_obstacle.write(np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype='u1').tobytes())

        self.tex_sdf = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='f2')
        self.tex_sdf.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex_sdf.write((np.ones((self.grid_size, self.grid_size, self.grid_size), dtype='f2') * 10.0).tobytes())

        self.mg_dummies = []

        self.fbo_velocity_A = self._setup_3d_fbo(self.tex_velocity_A, self.grid_size)
        self.fbo_velocity_B = self._setup_3d_fbo(self.tex_velocity_B, self.grid_size)
        self.fbo_pressure_A = self._setup_3d_fbo(self.tex_pressure_A, self.grid_size)
        self.fbo_pressure_B = self._setup_3d_fbo(self.tex_pressure_B, self.grid_size)
        self.fbo_divergence = self._setup_3d_fbo(self.tex_divergence, self.grid_size)
        self.fbo_density_A = self._setup_3d_fbo(self.tex_density_A, self.grid_size)
        self.fbo_density_B = self._setup_3d_fbo(self.tex_density_B, self.grid_size)

        self.mg_levels = 6
        self.mg_grid_sizes = [self.grid_size >> l for l in range(self.mg_levels + 1)]
        self.mg_dummies = []
        self.mg_corr_A = []
        self.mg_corr_B = []
        self.mg_aux = []
        self.mg_rhs = []
        self.mg_obstacle = []
        self.fbo_mg_corr_A = []
        self.fbo_mg_corr_B = []
        self.fbo_mg_aux = []
        self.fbo_mg_rhs = []

        for l in range(1, self.mg_levels + 1):
            N = self.mg_grid_sizes[l]
            ca = self.ctx.texture3d((N, N, N), 1, dtype='f4')
            cb = self.ctx.texture3d((N, N, N), 1, dtype='f4')
            aux = self.ctx.texture3d((N, N, N), 1, dtype='f4')
            rhs = self.ctx.texture3d((N, N, N), 1, dtype='f4')
            obs = self.ctx.texture3d((N, N, N), 1, dtype='u1')
            for t in [ca, cb, aux, rhs, obs]:
                t.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.mg_corr_A.append(ca)
            self.mg_corr_B.append(cb)
            self.mg_aux.append(aux)
            self.mg_rhs.append(rhs)
            self.mg_obstacle.append(obs)
            self.fbo_mg_corr_A.append(self._setup_3d_fbo(ca, N))
            self.fbo_mg_corr_B.append(self._setup_3d_fbo(cb, N))
            self.fbo_mg_aux.append(self._setup_3d_fbo(aux, N))
            self.fbo_mg_rhs.append(self._setup_3d_fbo(rhs, N))
            z = np.zeros((N, N, N), dtype='f4')
            ca.write(z.tobytes())
            cb.write(z.tobytes())
            aux.write(z.tobytes())
            rhs.write(z.tobytes())

        self.tex_fine_residual = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='f4')
        self.tex_fine_residual.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex_correction_fine = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='f4')
        self.tex_correction_fine.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbo_fine_residual = self._setup_3d_fbo(self.tex_fine_residual, self.grid_size)
        self.fbo_correction_fine = self._setup_3d_fbo(self.tex_correction_fine, self.grid_size)
        self.tex_fine_residual.write(np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype='f4').tobytes())
        self.tex_correction_fine.write(np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype='f4').tobytes())

        self.tex_velocity_init = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 4, dtype='f2')
        self.tex_velocity_init.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_velocity_init.repeat_x = False
        self.tex_velocity_init.repeat_y = False
        self.tex_velocity_init.repeat_z = False
        self.tex_k1 = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 4, dtype='f2')
        self.tex_k1.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_k1.repeat_x = False
        self.tex_k1.repeat_y = False
        self.tex_k1.repeat_z = False
        self.tex_k2 = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 4, dtype='f2')
        self.tex_k2.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_k2.repeat_x = False
        self.tex_k2.repeat_y = False
        self.tex_k2.repeat_z = False
        self.tex_k3 = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 4, dtype='f2')
        self.tex_k3.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_k3.repeat_x = False
        self.tex_k3.repeat_y = False
        self.tex_k3.repeat_z = False
        self.tex_k4 = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 4, dtype='f2')
        self.tex_k4.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_k4.repeat_x = False
        self.tex_k4.repeat_y = False
        self.tex_k4.repeat_z = False

        # FBOs for RK4 textures
        self.fbo_velocity_init = self._setup_3d_fbo(self.tex_velocity_init, self.grid_size)
        self.fbo_k1 = self._setup_3d_fbo(self.tex_k1, self.grid_size)
        self.fbo_k2 = self._setup_3d_fbo(self.tex_k2, self.grid_size)
        self.fbo_k3 = self._setup_3d_fbo(self.tex_k3, self.grid_size)
        self.fbo_k4 = self._setup_3d_fbo(self.tex_k4, self.grid_size)

        self.tex_stage_save = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 4, dtype='f2')
        self.tex_stage_save.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_stage_save.repeat_x = False
        self.tex_stage_save.repeat_y = False
        self.tex_stage_save.repeat_z = False
        self.fbo_stage_save = self._setup_3d_fbo(self.tex_stage_save, self.grid_size)

        self.prog_forces = self._load_program('shaders/forces.frag')
        self.prog_advection = self._load_program('shaders/advection.frag')
        self.prog_advection_density = self._load_program('shaders/advection_density.frag')
        self.prog_divergence = self._load_program('shaders/divergence.frag')
        self.prog_jacobi = self._load_program('shaders/jacobi.frag')
        self.prog_projection = self._load_program('shaders/projection.frag')

        self.prog_coarse_jacobi = self._load_program('shaders/coarse_jacobi.frag')
        self.prog_coarse_residual = self._load_program('shaders/coarse_residual.frag')
        self.prog_coarse_add = self._load_program('shaders/coarse_add.frag')
        self.prog_restrict = self._load_program('shaders/restrict.frag')
        self.prog_prolongate = self._load_program('shaders/prolongate.frag')
        self.prog_fine_residual = self._load_program('shaders/fine_residual.frag')
        self.prog_add_correction = self._load_program('shaders/add_correction.frag')
        self.prog_rbgs = self._load_program('shaders/rbgs.frag')
        self.prog_coarse_rbgs = self._load_program('shaders/coarse_rbgs.frag')
        self.prog_obstacle_zero = self._load_program('shaders/obstacle_zero.frag')
        self.prog_copy = self._load_program('shaders/copy_tex.frag')
        self.prog_rk4_setup = self._load_program('shaders/rk4_setup.frag')
        self.prog_rk4_save_k = self._load_program('shaders/rk4_save_k.frag')
        self.prog_rk4_combine = self._load_program('shaders/rk4_combine.frag')

        self.vao = self.ctx.vertex_array(self.prog_forces, [])

        for prog in [self.prog_forces, self.prog_advection, self.prog_advection_density,
                     self.prog_divergence, self.prog_jacobi, self.prog_projection,
                     self.prog_fine_residual, self.prog_add_correction,
                     self.prog_rbgs, self.prog_rk4_setup]:
            if 'u_obstacle' in prog:
                prog['u_obstacle'].value = 2
            if 'u_sdf' in prog:
                prog['u_sdf'].value = 3

        self.inflow_vel = (1.0, 0.0, 0.0)
        for prog in [self.prog_forces, self.prog_advection, self.prog_projection, self.prog_advection_density,
                     self.prog_rk4_setup]:
            if 'u_inflow_vel' in prog:
                prog['u_inflow_vel'].value = tuple(self.inflow_vel)

        self.Re = None
        self.char_length = 1.0
        self.use_vcycle = False
        self.sgs_coeff = 0.325
        self.vcycle_interval = 5
        self._vcycle_counter = 0

        self.jacobi_iters = 40
        self.cfl_target = 0.55
        self.dt_update_interval = 10
        self._dt_counter = 0
        self.use_fft = False
        self.fft_iterations = 8
        self.use_dct = False
        self.dct_iterations = 2
        self.dct_post_smooth = 10
        self.advection_clamp_mode = 0

    def read_velocity(self):
        vel_data = self.tex_velocity_A.read()
        vel = np.frombuffer(vel_data, dtype=np.float16).reshape(
            (self.grid_size, self.grid_size, self.grid_size, 4))
        return np.transpose(vel[..., :3], (2, 1, 0, 3)).astype(np.float32)

    def write_velocity(self, vel_np):
        vel4 = np.zeros((self.grid_size, self.grid_size, self.grid_size, 4), dtype=np.float16)
        vel4[..., :3] = vel_np[..., :3].astype(np.float16)
        self.tex_velocity_A.write(np.transpose(vel4, (2, 1, 0, 3)).tobytes())
        self.tex_velocity_B.write(np.transpose(vel4, (2, 1, 0, 3)).tobytes())

    def _read_max_velocity(self):
        vel_data = self.tex_velocity_A.read()
        vel = np.frombuffer(vel_data, dtype=np.float16).reshape(
            (self.grid_size, self.grid_size, self.grid_size, 4))
        speed = np.sqrt(vel[..., 0]**2 + vel[..., 1]**2 + vel[..., 2]**2)
        speed = np.nan_to_num(speed, nan=0.0, posinf=1.0, neginf=0.0)
        return float(np.max(speed))

    def _update_dt(self):
        max_u = self._read_max_velocity()
        max_u = max(max_u, float(np.linalg.norm(self.inflow_vel)))
        if max_u < 1e-10:
            max_u = 1.0
        dx = 1.0
        cfl = self.cfl_target
        nu = max(self.viscosity, 1e-10)
        dt_adv = cfl * dx / max_u
        dt_diff = cfl * dx * dx / (2.0 * nu)
        self.dt = min(dt_adv, dt_diff)

    def set_obstacle(self, mask_np):
        mask_np = np.ascontiguousarray(mask_np, dtype=np.uint8)
        self.tex_obstacle.write(np.transpose(mask_np, (2, 1, 0)).tobytes())
        for l in range(self.mg_levels):
            c = self.mg_grid_sizes[l + 1]
            s = 2 ** (l + 1)
            coarse = mask_np.reshape(c, s, c, s, c, s).any(axis=(1, 3, 5)).astype(np.uint8)
            self.mg_obstacle[l].write(np.transpose(coarse, (2, 1, 0)).tobytes())

    def set_reynolds(self, Re, char_length=None):
        if char_length is not None:
            self.char_length = char_length
        if Re <= 0:
            raise ValueError("Reynolds number must be positive")
        U = float(np.linalg.norm(self.inflow_vel))
        if U < 1e-10:
            print("Warning: inflow velocity near zero, Re not set")
            return
        self.viscosity = U * self.char_length / Re
        self.Re = Re
        self._update_dt()
        print(f"  Re={Re}: U={U:.3f}, L={self.char_length}, nu={self.viscosity:.6e}, dt={self.dt:.6f}")

    def set_cad_obstacle(self, mesh_path, cad_chord, aoa_deg=0.0):
        from cad_utils import load_mesh, prepare_mesh, scale_and_center, make_cad_obstacle, summary as cad_summary
        print(f"Loading CAD mesh: {mesh_path}")
        mesh = load_mesh(mesh_path)
        print(cad_summary(mesh))
        mesh = prepare_mesh(mesh, target_faces=50000)
        print(cad_summary(mesh))
        print(f"  Scaling to chord={cad_chord}, AoA={aoa_deg}°")
        scale_and_center(mesh, self.grid_size, cad_chord, aoa_deg)
        print(f"  Voxelizing {self.grid_size}³ ...")
        mask, sdf = make_cad_obstacle(mesh, self.grid_size)
        print(f"  Obstacle cells: {np.sum(mask)} / {self.grid_size**3}")
        self.set_obstacle(mask)
        self.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).astype(np.float16).tobytes())
        print("  CAD obstacle ready.")

    def set_advection_clamp_mode(self, mode):
        self.advection_clamp_mode = mode
        self.prog_advection = self._load_program(
            'shaders/advection.frag',
            defines={'CLAMP_MODE': mode}
        )
        for prog in [self.prog_advection]:
            if 'u_obstacle' in prog:
                prog['u_obstacle'].value = 2
            if 'u_sdf' in prog:
                prog['u_sdf'].value = 3
            if 'u_inflow_vel' in prog:
                prog['u_inflow_vel'].value = tuple(self.inflow_vel)
        print(f"  Advection clamp mode set to {mode}")

    def _setup_3d_fbo(self, tex, layer_size):
        dummy = self.ctx.texture((layer_size, layer_size), 4, dtype='f2')
        self.mg_dummies.append(dummy)
        fbo = self.ctx.framebuffer(color_attachments=[dummy])
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo.glo)
        gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, tex.glo, 0)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Incomplete FBO")
        return fbo

    def _load_program(self, frag_path, defines=None):
        with open('shaders/layered.vert', 'r') as f:
            vert_src = f.read()
        with open('shaders/layered.geom', 'r') as f:
            geom_src = f.read()
        with open(frag_path, 'r') as f:
            frag_src = f.read()
        if defines:
            define_str = '\n'.join(f'#define {k} {v}' for k, v in defines.items())
            frag_src = frag_src.replace('#version 330 core', f'#version 330 core\n{define_str}')
        prog = self.ctx.program(
            vertex_shader=vert_src,
            geometry_shader=geom_src,
            fragment_shader=frag_src
        )
        return prog

    def render_pass(self, prog, fbo):
        fbo.use()
        gl.glViewport(0, 0, self.grid_size, self.grid_size)
        self.vao = self.ctx.vertex_array(prog, [])
        self.vao.render(mode=moderngl.TRIANGLES, vertices=3, instances=self.grid_size)

    def render_pass_ex(self, prog, fbo, grid_n):
        fbo.use()
        gl.glViewport(0, 0, grid_n, grid_n)
        vao = self.ctx.vertex_array(prog, [])
        vao.render(mode=moderngl.TRIANGLES, vertices=3, instances=grid_n)
        gl.glViewport(0, 0, self.grid_size, self.grid_size)

    def _zero_obstacle_fine(self):
        self.tex_pressure_A.use(location=0)
        self.tex_obstacle.use(location=1)
        self.prog_obstacle_zero['u_pressure'].value = 0
        self.prog_obstacle_zero['u_mask'].value = 1
        self.render_pass(self.prog_obstacle_zero, self.fbo_pressure_B)
        self.tex_pressure_A, self.tex_pressure_B = self.tex_pressure_B, self.tex_pressure_A
        self.fbo_pressure_A, self.fbo_pressure_B = self.fbo_pressure_B, self.fbo_pressure_A

    def _zero_obstacle_coarse(self, level_idx):
        N = self.mg_grid_sizes[level_idx + 1]
        self.mg_corr_A[level_idx].use(location=0)
        self.mg_obstacle[level_idx].use(location=1)
        self.prog_obstacle_zero['u_pressure'].value = 0
        self.prog_obstacle_zero['u_mask'].value = 1
        self.render_pass_ex(self.prog_obstacle_zero, self.fbo_mg_corr_B[level_idx], N)
        self.mg_corr_A[level_idx], self.mg_corr_B[level_idx] = self.mg_corr_B[level_idx], self.mg_corr_A[level_idx]
        self.fbo_mg_corr_A[level_idx], self.fbo_mg_corr_B[level_idx] = self.fbo_mg_corr_B[level_idx], self.fbo_mg_corr_A[level_idx]

    def _fine_jacobi(self, n_iter):
        self.tex_divergence.use(location=1)
        self.tex_sdf.use(location=3)
        self.tex_obstacle.use(location=2)
        self.prog_jacobi['u_divergence'].value = 1
        self.prog_jacobi['u_pressure'].value = 0
        for _ in range(n_iter):
            self.tex_pressure_A.use(location=0)
            self.render_pass(self.prog_jacobi, self.fbo_pressure_B)
            self.tex_pressure_B.use(location=0)
            self.render_pass(self.prog_jacobi, self.fbo_pressure_A)

    def _fine_rbgs(self, n_iter):
        self.tex_divergence.use(location=1)
        self.tex_sdf.use(location=3)
        self.tex_obstacle.use(location=2)
        self.prog_rbgs['u_divergence'].value = 1
        self.prog_rbgs['u_pressure'].value = 0
        for _ in range(n_iter):
            self.tex_pressure_A.use(location=0)
            self.prog_rbgs['u_parity'].value = 0
            self.render_pass(self.prog_rbgs, self.fbo_pressure_B)
            self.tex_pressure_B.use(location=0)
            self.prog_rbgs['u_parity'].value = 1
            self.render_pass(self.prog_rbgs, self.fbo_pressure_A)

    def _compute_fine_residual(self):
        self.tex_divergence.use(location=1)
        self.tex_sdf.use(location=3)
        self.tex_obstacle.use(location=2)
        self.prog_fine_residual['u_divergence'].value = 1
        self.prog_fine_residual['u_pressure'].value = 0
        self.tex_pressure_A.use(location=0)
        self.render_pass(self.prog_fine_residual, self.fbo_fine_residual)

    def _coarse_jacobi(self, level_idx, n_iter):
        mg_level = level_idx + 1
        h_sq = 4.0 ** mg_level
        N = self.mg_grid_sizes[mg_level]
        omega = 0.5
        self.mg_obstacle[level_idx].use(location=2)
        self.prog_coarse_jacobi['u_obstacle'].value = 2
        self.prog_coarse_jacobi['u_omega'].value = omega
        for _ in range(n_iter):
            self.mg_corr_A[level_idx].use(location=0)
            self.mg_rhs[level_idx].use(location=1)
            self.prog_coarse_jacobi['u_solution'].value = 0
            self.prog_coarse_jacobi['u_rhs'].value = 1
            self.prog_coarse_jacobi['u_h_sq'].value = h_sq
            self.render_pass_ex(self.prog_coarse_jacobi, self.fbo_mg_corr_B[level_idx], N)
            self.mg_corr_B[level_idx].use(location=0)
            self.mg_rhs[level_idx].use(location=1)
            self.prog_coarse_jacobi['u_solution'].value = 0
            self.prog_coarse_jacobi['u_rhs'].value = 1
            self.prog_coarse_jacobi['u_h_sq'].value = h_sq
            self.render_pass_ex(self.prog_coarse_jacobi, self.fbo_mg_corr_A[level_idx], N)

    def _coarse_rbgs(self, level_idx, n_iter):
        mg_level = level_idx + 1
        h_sq = 4.0 ** mg_level
        N = self.mg_grid_sizes[mg_level]
        self.mg_obstacle[level_idx].use(location=2)
        self.prog_coarse_rbgs['u_obstacle'].value = 2
        for _ in range(n_iter):
            self.mg_corr_A[level_idx].use(location=0)
            self.mg_rhs[level_idx].use(location=1)
            self.prog_coarse_rbgs['u_solution'].value = 0
            self.prog_coarse_rbgs['u_rhs'].value = 1
            self.prog_coarse_rbgs['u_h_sq'].value = h_sq
            self.prog_coarse_rbgs['u_parity'].value = 0
            self.render_pass_ex(self.prog_coarse_rbgs, self.fbo_mg_corr_B[level_idx], N)
            self.mg_corr_B[level_idx].use(location=0)
            self.mg_rhs[level_idx].use(location=1)
            self.prog_coarse_rbgs['u_solution'].value = 0
            self.prog_coarse_rbgs['u_rhs'].value = 1
            self.prog_coarse_rbgs['u_h_sq'].value = h_sq
            self.prog_coarse_rbgs['u_parity'].value = 1
            self.render_pass_ex(self.prog_coarse_rbgs, self.fbo_mg_corr_A[level_idx], N)

    def _coarse_residual(self, level_idx):
        mg_level = level_idx + 1
        h_sq = 4.0 ** mg_level
        N = self.mg_grid_sizes[mg_level]
        self.mg_corr_A[level_idx].use(location=0)
        self.mg_rhs[level_idx].use(location=1)
        self.mg_obstacle[level_idx].use(location=2)
        self.prog_coarse_residual['u_solution'].value = 0
        self.prog_coarse_residual['u_rhs'].value = 1
        self.prog_coarse_residual['u_obstacle'].value = 2
        self.prog_coarse_residual['u_h_sq'].value = h_sq
        self.render_pass_ex(self.prog_coarse_residual, self.fbo_mg_corr_B[level_idx], N)
        return N

    def _restrict_fine_to_coarse(self):
        N_coarse = self.mg_grid_sizes[1]
        self.tex_fine_residual.use(location=0)
        self.prog_restrict['u_fine'].value = 0
        self.render_pass_ex(self.prog_restrict, self.fbo_mg_rhs[0], N_coarse)

    def _restrict_coarse(self, src_level_idx):
        src_mg_level = src_level_idx + 1
        dst_mg_level = src_mg_level + 1
        N_dst = self.mg_grid_sizes[dst_mg_level]
        self.mg_corr_B[src_level_idx].use(location=0)
        self.prog_restrict['u_fine'].value = 0
        self.render_pass_ex(self.prog_restrict, self.fbo_mg_rhs[src_level_idx + 1], N_dst)

    def _prolongate_to_fine(self):
        N_fine = self.grid_size
        self.mg_corr_A[0].use(location=0)
        self.prog_prolongate['u_coarse'].value = 0
        self.render_pass_ex(self.prog_prolongate, self.fbo_correction_fine, N_fine)

    def _prolongate_coarse(self, src_level_idx):
        dst_mg_level = src_level_idx + 1
        N_dst = self.mg_grid_sizes[dst_mg_level]
        self.mg_corr_A[src_level_idx + 1].use(location=0)
        self.prog_prolongate['u_coarse'].value = 0
        self.render_pass_ex(self.prog_prolongate, self.fbo_mg_aux[src_level_idx], N_dst)

    def _add_correction_fine(self, omega=0.7):
        self.tex_sdf.use(location=3)
        self.tex_obstacle.use(location=2)
        self.tex_pressure_A.use(location=0)
        self.tex_correction_fine.use(location=1)
        self.prog_add_correction['u_pressure'].value = 0
        self.prog_add_correction['u_correction'].value = 1
        self.prog_add_correction['u_omega'].value = omega
        self.render_pass(self.prog_add_correction, self.fbo_pressure_B)
        self.tex_pressure_A, self.tex_pressure_B = self.tex_pressure_B, self.tex_pressure_A
        self.fbo_pressure_A, self.fbo_pressure_B = self.fbo_pressure_B, self.fbo_pressure_A

    def _add_correction_coarse(self, level_idx):
        mg_level = level_idx + 1
        N = self.mg_grid_sizes[mg_level]
        self.mg_corr_A[level_idx].use(location=0)
        self.mg_aux[level_idx].use(location=1)
        self.prog_coarse_add['u_current'].value = 0
        self.prog_coarse_add['u_update'].value = 1
        self.render_pass_ex(self.prog_coarse_add, self.fbo_mg_corr_B[level_idx], N)
        self.mg_corr_A[level_idx], self.mg_corr_B[level_idx] = \
            self.mg_corr_B[level_idx], self.mg_corr_A[level_idx]
        self.fbo_mg_corr_A[level_idx], self.fbo_mg_corr_B[level_idx] = \
            self.fbo_mg_corr_B[level_idx], self.fbo_mg_corr_A[level_idx]

    def _clear_level(self, level_idx):
        N = self.mg_grid_sizes[level_idx + 1]
        z = np.zeros((N, N, N), dtype='f4')
        self.mg_corr_A[level_idx].write(z.tobytes())
        self.mg_corr_B[level_idx].write(z.tobytes())

    # ---- Multigrid V-cycle ----

    def _two_grid(self):
        """2-grid V-cycle: 192³ → 96³ → 192³ with RBGS smoothing."""
        n_pre = 5
        n_coarse = 80
        n_post = 10
        self._fine_rbgs(n_pre)
        self._compute_fine_residual()
        self._restrict_fine_to_coarse()
        self._clear_level(0)
        self._coarse_rbgs(0, n_coarse)
        self._prolongate_to_fine()
        self._add_correction_fine(omega=0.7)
        self._zero_obstacle_fine()
        self._fine_rbgs(n_post)

    def _full_vcycle(self):
        """Full V-cycle through all 6 multigrid levels."""
        n_pre = 3
        n_post = 5
        n_coarse = 80
        self._fine_rbgs(n_pre)
        self._compute_fine_residual()
        self._restrict_fine_to_coarse()
        for l in range(self.mg_levels):
            self._clear_level(l)
        for l in range(self.mg_levels - 1):
            self._coarse_rbgs(l, n_pre)
            self._coarse_residual(l)
            self._restrict_coarse(l)
        self._coarse_rbgs(self.mg_levels - 1, n_coarse)
        for l in range(self.mg_levels - 2, -1, -1):
            self._prolongate_coarse(l)
            self._add_correction_coarse(l)
            self._coarse_rbgs(l, n_post)
        self._prolongate_to_fine()
        self._add_correction_fine(omega=0.7)
        self._zero_obstacle_fine()
        self._fine_rbgs(n_post)

    def _laplacian_7pt(self, p, dx=1.0):
        """7-point Laplacian on CPU with proper BCs: Neumann at walls/inflow, Dirichlet at outflow."""
        N = p.shape[0]
        lap = np.zeros_like(p)
        # Interior: standard 7-point stencil
        lap[1:-1, 1:-1, 1:-1] = (
            p[:-2, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1] +
            p[1:-1, :-2, 1:-1] + p[1:-1, 2:, 1:-1] +
            p[1:-1, 1:-1, :-2] + p[1:-1, 1:-1, 2:] -
            6.0 * p[1:-1, 1:-1, 1:-1]
        ) / (dx * dx)
        # Inflow (X=0): Neumann ∂p/∂x=0 → ghost cell equals interior
        lap[0, 1:-1, 1:-1] = (
            p[0, 1:-1, 1:-1] + p[1, 1:-1, 1:-1] +
            p[0, :-2, 1:-1] + p[0, 2:, 1:-1] +
            p[0, 1:-1, :-2] + p[0, 1:-1, 2:] -
            6.0 * p[0, 1:-1, 1:-1]
        ) / (dx * dx)
        # Outflow (X=-1): Dirichlet p=0 at face between N-1 and N
        # Ghost at N satisfies: (p[-1] + ghost)/2 = 0 → ghost = -p[-1]
        lap[-1, 1:-1, 1:-1] = (
            p[-2, 1:-1, 1:-1] + (-p[-1, 1:-1, 1:-1]) +
            p[-1, :-2, 1:-1] + p[-1, 2:, 1:-1] +
            p[-1, 1:-1, :-2] + p[-1, 1:-1, 2:] -
            6.0 * p[-1, 1:-1, 1:-1]
        ) / (dx * dx)
        # Y-walls: Neumann ∂p/∂y=0
        lap[1:-1, 0, 1:-1] = (
            p[:-2, 0, 1:-1] + p[2:, 0, 1:-1] +
            p[1:-1, 0, 1:-1] + p[1:-1, 1, 1:-1] +
            p[1:-1, 0, :-2] + p[1:-1, 0, 2:] -
            6.0 * p[1:-1, 0, 1:-1]
        ) / (dx * dx)
        lap[1:-1, -1, 1:-1] = (
            p[:-2, -1, 1:-1] + p[2:, -1, 1:-1] +
            p[1:-1, -2, 1:-1] + p[1:-1, -1, 1:-1] +
            p[1:-1, -1, :-2] + p[1:-1, -1, 2:] -
            6.0 * p[1:-1, -1, 1:-1]
        ) / (dx * dx)
        # Z-walls: Neumann ∂p/∂z=0
        lap[1:-1, 1:-1, 0] = (
            p[:-2, 1:-1, 0] + p[2:, 1:-1, 0] +
            p[1:-1, :-2, 0] + p[1:-1, 2:, 0] +
            p[1:-1, 1:-1, 0] + p[1:-1, 1:-1, 1] -
            6.0 * p[1:-1, 1:-1, 0]
        ) / (dx * dx)
        lap[1:-1, 1:-1, -1] = (
            p[:-2, 1:-1, -1] + p[2:, 1:-1, -1] +
            p[1:-1, :-2, -1] + p[1:-1, 2:, -1] +
            p[1:-1, 1:-1, -2] + p[1:-1, 1:-1, -1] -
            6.0 * p[1:-1, 1:-1, -1]
        ) / (dx * dx)
        return lap

    def _laplacian_7pt_periodic(self, p, dx=1.0):
        """7-point Laplacian with periodic BCs (for FFT eigenvalue consistency)."""
        N = p.shape[0]
        lap = np.zeros_like(p)
        lap = (
            np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0) +
            np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1) +
            np.roll(p, 1, axis=2) + np.roll(p, -1, axis=2) -
            6.0 * p
        ) / (dx * dx)
        return lap

    def _fft_solve(self):
        """CPU FFT-based Poisson solver with IBM correction loop.

        Uses FFT (periodic) as preconditioner inside an iterative correction
        loop that enforces Neumann/Dirichlet BCs and obstacle masking.
        Under-relaxed for stability. Residual convergence checked.
        """
        from numpy.fft import fftn, ifftn
        N = self.grid_size
        n_iter = self.fft_iterations
        omega = 0.7  # under-relaxation factor
        tol = 1e-6   # relative residual tolerance

        div_data = self.tex_divergence.read()
        div = np.frombuffer(div_data, dtype=np.float16).reshape((N, N, N)).astype(np.float64)

        obs_data = self.tex_obstacle.read()
        obs = np.frombuffer(obs_data, dtype=np.uint8).reshape((N, N, N))
        obs_bool = obs > 0
        div[obs_bool] = 0.0

        # Warm-start from current GPU pressure
        p_data = self.tex_pressure_A.read()
        p = np.frombuffer(p_data, dtype=np.float32).reshape((N, N, N)).astype(np.float64)
        p[obs_bool] = 0.0

        # FFT eigenvalues of periodic 7-point Laplacian
        kx = np.fft.fftfreq(N) * 2.0 * np.pi
        ky = np.fft.fftfreq(N) * 2.0 * np.pi
        kz = np.fft.fftfreq(N) * 2.0 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        eig = 2.0 * (np.cos(KX) + np.cos(KY) + np.cos(KZ) - 3.0)
        eig[0, 0, 0] = 1.0  # DC mode handled separately via mean subtraction

        inflow = np.zeros((N, N, N), dtype=bool)
        inflow[0, :, :] = True
        outflow = np.zeros((N, N, N), dtype=bool)
        outflow[-1, :, :] = True
        walls = np.zeros((N, N, N), dtype=bool)
        walls[:, 0, :] = True; walls[:, -1, :] = True
        walls[:, :, 0] = True; walls[:, :, -1] = True

        fluid = ~obs_bool & ~outflow

        div_norm = max(np.linalg.norm(div[fluid]), 1e-30)

        for it in range(n_iter):
            lap = self._laplacian_7pt(p)
            r = div - lap
            r[obs_bool] = 0.0
            r[outflow] = 0.0
            r[walls] = 0.0

            r_norm = np.linalg.norm(r[fluid])
            if r_norm < tol * div_norm and it > 0:
                break

            # Make residual mean-zero (compatibility for Neumann BCs)
            r_mean = np.mean(r[fluid])
            r[fluid] -= r_mean

            R_hat = fftn(r)
            dp = np.real(ifftn(R_hat / eig))

            # Clamp dp to prevent blowup
            dp = np.clip(dp, -100.0, 100.0)

            # Enforce BCs and obstacles on correction
            dp[obs_bool] = -p[obs_bool]
            dp[outflow] = -p[outflow]
            dp[walls] = 0.0

            p += omega * dp
            p[obs_bool] = 0.0
            p[outflow] = 0.0

        p -= np.mean(p[fluid])
        p = np.clip(p, -1e3, 1e3)
        p_f32 = np.ascontiguousarray(p.astype(np.float32))
        self.tex_pressure_A.write(p_f32.tobytes())
        self.tex_pressure_B.write(p_f32.tobytes())

    def _dct_solve(self):
        """CPU DST-based Poisson solver + GPU RBGS post-smoothing.

        DST-II (Dirichlet p=0 at all faces) matches the RBGS shader BCs
        exactly, so DST provides an excellent single-shot preconditioner.
        GPU RBGS post-smoothing corrects obstacle-boundary errors.
        """
        from scipy.fft import dst, idst
        N = self.grid_size
        n_smooth = self.dct_post_smooth

        div_data = self.tex_divergence.read()
        div = np.frombuffer(div_data, dtype=np.float16).reshape((N, N, N)).astype(np.float64)

        obs_data = self.tex_obstacle.read()
        obs = np.frombuffer(obs_data, dtype=np.uint8).reshape((N, N, N))
        obs_bool = obs > 0
        div[obs_bool] = 0.0
        div[-1, :, :] = 0.0

        k = np.arange(N, dtype=np.float64) + 1.0
        eig1d = 2.0 * (np.cos(np.pi * k / N) - 1.0)
        eig = eig1d[:, None, None] + eig1d[None, :, None] + eig1d[None, None, :]
        eig_min_abs = np.min(np.abs(eig[eig < 0]))
        eig = np.where(np.abs(eig) < 1e-12, eig_min_abs * 1e-8, eig)

        coef = dst(div, type=2, axis=0, norm='ortho')
        coef = dst(coef, type=2, axis=1, norm='ortho')
        coef = dst(coef, type=2, axis=2, norm='ortho')
        coef /= eig
        p = idst(coef, type=2, axis=0, norm='ortho')
        p = idst(p, type=2, axis=1, norm='ortho')
        p = idst(p, type=2, axis=2, norm='ortho')

        p[obs_bool] = 0.0
        p[-1, :, :] = 0.0
        p = np.clip(p, -1e3, 1e3)

        p_f32 = np.ascontiguousarray(p.astype(np.float32))
        self.tex_pressure_A.write(p_f32.tobytes())
        self.tex_pressure_B.write(p_f32.tobytes())

        # GPU RBGS post-smoothing (matches DST BCs exactly)
        if n_smooth > 0:
            self._fine_rbgs(n_smooth)

    def _mg_solve(self):
        if self.use_dct:
            self._dct_solve()
        elif self.use_fft:
            self._fft_solve()
        elif self.use_vcycle:
            self._vcycle_counter += 1
            if self._vcycle_counter >= self.vcycle_interval:
                self._full_vcycle()
                self._vcycle_counter = 0
            else:
                self._fine_rbgs(self.jacobi_iters)
        else:
            self._fine_rbgs(self.jacobi_iters)

    def step(self, force_pos=(96.0, 96.0, 96.0), force_dir=(0.0, 1.0, 0.0), force_radius=10.0):
        self.tex_obstacle.use(location=2)
        self.tex_sdf.use(location=3)

        self._dt_counter += 1
        if self._dt_counter >= self.dt_update_interval:
            self._update_dt()
            self._dt_counter = 0

        # PASS 1: Add Forces (A -> B)
        self.tex_velocity_A.use(location=0)
        self.prog_forces['u_velocity'].value = 0
        self.prog_forces['u_dt'].value = self.dt
        self.prog_forces['u_force_pos'].value = tuple(force_pos)
        self.prog_forces['u_force_dir'].value = tuple(force_dir)
        self.prog_forces['u_force_radius'].value = float(force_radius)
        self.render_pass(self.prog_forces, self.fbo_velocity_B)

        # PASS 2: Advection (B -> A) with molecular + SGS viscosity
        self.tex_velocity_B.use(location=0)
        self.prog_advection['u_velocity'].value = 0
        self.prog_advection['u_dt'].value = self.dt
        self.prog_advection['u_viscosity'].value = self.viscosity
        self.prog_advection['u_sgs_coeff'].value = self.sgs_coeff
        self.render_pass(self.prog_advection, self.fbo_velocity_A)

        # PASS 3: Divergence (A -> Div)
        self.tex_velocity_A.use(location=0)
        self.prog_divergence['u_velocity'].value = 0
        self.render_pass(self.prog_divergence, self.fbo_divergence)

        # PASS 4: Pressure Solve (Multigrid V-cycle)
        self._mg_solve()

        # PASS 5: Projection (A + Pressure -> B)
        self.tex_velocity_A.use(location=0)
        self.tex_pressure_A.use(location=1)
        self.prog_projection['u_velocity'].value = 0
        self.prog_projection['u_pressure'].value = 1
        self.prog_projection['u_dt'].value = self.dt
        self.render_pass(self.prog_projection, self.fbo_velocity_B)

        # PASS 6: Density Advection
        self.tex_velocity_B.use(location=0)
        self.tex_density_A.use(location=1)
        self.prog_advection_density['u_velocity'].value = 0
        self.prog_advection_density['u_density'].value = 1
        self.prog_advection_density['u_dt'].value = self.dt
        self.render_pass(self.prog_advection_density, self.fbo_density_B)
        self.tex_density_A, self.tex_density_B = self.tex_density_B, self.tex_density_A
        self.fbo_density_A, self.fbo_density_B = self.fbo_density_B, self.fbo_density_A

        self.tex_velocity_A, self.tex_velocity_B = self.tex_velocity_B, self.tex_velocity_A
        self.fbo_velocity_A, self.fbo_velocity_B = self.fbo_velocity_B, self.fbo_velocity_A

    def step_rk4(self, force_pos=(96.0, 96.0, 96.0), force_dir=(0.0, 1.0, 0.0), force_radius=10.0):
        """Runge-Kutta 4th order time integration."""
        self._dt_counter += 1
        if self._dt_counter >= self.dt_update_interval:
            self._update_dt()
            self._dt_counter = 0

        # Save initial velocity u^n
        self.tex_velocity_A.use(location=0)
        init_data = self.tex_velocity_A.read()
        self.tex_velocity_init.write(init_data)

        self.tex_obstacle.use(location=2)
        self.tex_sdf.use(location=3)

        # Clear k accumulators
        zero = np.zeros((self.grid_size, self.grid_size, self.grid_size, 4), dtype='f2')
        self.tex_k1.write(zero.tobytes())
        self.tex_k2.write(zero.tobytes())
        self.tex_k3.write(zero.tobytes())
        self.tex_k4.write(zero.tobytes())

        # Helper: run one RK4 stage and save k_i
        def run_stage(coeff, k_prev_tex, k_prev_fbo, k_dst_fbo):
            # Setup: u_stage = u_init + coeff * k_prev -> B
            self.tex_velocity_init.use(location=0)
            k_prev_tex.use(location=1)
            self.prog_rk4_setup['u_init'].value = 0
            self.prog_rk4_setup['u_kprev'].value = 1
            self.prog_rk4_setup['u_coeff'].value = coeff
            self.render_pass(self.prog_rk4_setup, self.fbo_velocity_B)
            # Save u_stage (in B) to stage_save before forces overwrites it
            self.tex_velocity_B.use(location=0)
            self.prog_copy['u_src'].value = 0
            self.render_pass(self.prog_copy, self.fbo_stage_save)
            # Forces: B -> A
            self.tex_velocity_B.use(location=0)
            self.prog_forces['u_velocity'].value = 0
            self.prog_forces['u_dt'].value = self.dt
            self.prog_forces['u_force_pos'].value = tuple(force_pos)
            self.prog_forces['u_force_dir'].value = tuple(force_dir)
            self.prog_forces['u_force_radius'].value = float(force_radius)
            self.render_pass(self.prog_forces, self.fbo_velocity_A)
            # Advection: A -> B
            self.tex_velocity_A.use(location=0)
            self.prog_advection['u_velocity'].value = 0
            self.prog_advection['u_dt'].value = self.dt
            self.prog_advection['u_viscosity'].value = self.viscosity
            self.prog_advection['u_sgs_coeff'].value = self.sgs_coeff
            self.render_pass(self.prog_advection, self.fbo_velocity_B)
            # Save k = result(B) - u_stage(stage_save)
            self.tex_velocity_B.use(location=0)
            self.tex_stage_save.use(location=1)
            self.prog_rk4_save_k['u_result'].value = 0
            self.prog_rk4_save_k['u_input'].value = 1
            self.render_pass(self.prog_rk4_save_k, k_dst_fbo)

        # Stage 1: k1 = dt * RHS(u^n), coeff=0 (u_stage = u^n)
        run_stage(0.0, self.tex_k2, self.fbo_k2, self.fbo_k1)

        # Stage 2: k2 = dt * RHS(u^n + 0.5*k1)
        run_stage(0.5, self.tex_k1, self.fbo_k1, self.fbo_k2)

        # Stage 3: k3 = dt * RHS(u^n + 0.5*k2)
        run_stage(0.5, self.tex_k2, self.fbo_k2, self.fbo_k3)

        # Stage 4: k4 = dt * RHS(u^n + k3)
        run_stage(1.0, self.tex_k3, self.fbo_k3, self.fbo_k4)

        # Combine: u^{n+1} = u^n + (k1 + 2*k2 + 2*k3 + k4)/6 -> vel_A
        self.tex_velocity_init.use(location=0)
        self.tex_k1.use(location=1)
        self.tex_k2.use(location=2)
        self.tex_k3.use(location=3)
        self.tex_k4.use(location=4)
        self.prog_rk4_combine['u_init'].value = 0
        self.prog_rk4_combine['u_k1'].value = 1
        self.prog_rk4_combine['u_k2'].value = 2
        self.prog_rk4_combine['u_k3'].value = 3
        self.prog_rk4_combine['u_k4'].value = 4
        self.render_pass(self.prog_rk4_combine, self.fbo_velocity_A)

        # Divergence (A -> Div)
        self.tex_velocity_A.use(location=0)
        self.prog_divergence['u_velocity'].value = 0
        self.render_pass(self.prog_divergence, self.fbo_divergence)

        # Pressure Solve
        self._mg_solve()

        # Projection (A + Pressure -> B)
        self.tex_velocity_A.use(location=0)
        self.tex_pressure_A.use(location=1)
        self.prog_projection['u_velocity'].value = 0
        self.prog_projection['u_pressure'].value = 1
        self.prog_projection['u_dt'].value = self.dt
        self.render_pass(self.prog_projection, self.fbo_velocity_B)

        # Density Advection
        self.tex_velocity_B.use(location=0)
        self.tex_density_A.use(location=1)
        self.prog_advection_density['u_velocity'].value = 0
        self.prog_advection_density['u_density'].value = 1
        self.prog_advection_density['u_dt'].value = self.dt
        self.render_pass(self.prog_advection_density, self.fbo_density_B)
        self.tex_density_A, self.tex_density_B = self.tex_density_B, self.tex_density_A
        self.fbo_density_A, self.fbo_density_B = self.fbo_density_B, self.fbo_density_A

        # Swap velocity buffers for next step
        self.tex_velocity_A, self.tex_velocity_B = self.tex_velocity_B, self.tex_velocity_A
        self.fbo_velocity_A, self.fbo_velocity_B = self.fbo_velocity_B, self.fbo_velocity_A
