import moderngl
import numpy as np
import OpenGL.GL as gl

class EulerSolver:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.grid_size = 192
        self.gamma = 1.4
        self.dt = 0.1

        # 5 conserved variables packed into 2 textures:
        # tex_U1: (rho, mom_x, mom_y, mom_z) RGBA16F
        # tex_U2: (E) R16F
        # Each has A/B ping-pong buffers + a third "old" copy for RK

        def make_tex(components=4, dtype='f2'):
            tex = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), components, dtype=dtype)
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            tex.repeat_x = False
            tex.repeat_y = False
            tex.repeat_z = False
            return tex

        self.tex_U1 = make_tex(4)  # A buffer
        self.tex_U1_B = make_tex(4)  # B buffer
        self.tex_U1_old = make_tex(4)  # U^n copy

        self.tex_U2 = make_tex(1)  # A buffer
        self.tex_U2_B = make_tex(1)  # B buffer
        self.tex_U2_old = make_tex(1)  # U^n copy

        self.tex_obstacle = make_tex(1, 'u1')
        self.tex_obstacle.write(np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype='u1').tobytes())

        self.tex_sdf = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='f2')
        self.tex_sdf.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex_sdf.write((np.ones((self.grid_size, self.grid_size, self.grid_size), dtype='f2') * 10.0).tobytes())

        self.mg_dummies = []

        def make_fbo(tex):
            dummy = self.ctx.texture((self.grid_size, self.grid_size), 4, dtype='f2')
            self.mg_dummies.append(dummy)
            fbo = self.ctx.framebuffer(color_attachments=[dummy])
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo.glo)
            gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, tex.glo, 0)
            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("Incomplete FBO")
            return fbo

        self.fbo_U1 = make_fbo(self.tex_U1)
        self.fbo_U1_B = make_fbo(self.tex_U1_B)
        self.fbo_U1_old = make_fbo(self.tex_U1_old)
        self.fbo_U2 = make_fbo(self.tex_U2)
        self.fbo_U2_B = make_fbo(self.tex_U2_B)
        self.fbo_U2_old = make_fbo(self.tex_U2_old)

        # FBO for dual-color-attachment rendering (U1 + U2 at once)
        self.fbo_dual = self._make_dual_fbo(self.tex_U1_B, self.tex_U2_B)
        self.fbo_dual_A = self._make_dual_fbo(self.tex_U1, self.tex_U2)
        self.fbo_dual_old = self._make_dual_fbo(self.tex_U1_old, self.tex_U2_old)

        self.prog_update = self._load_program('shaders/euler_update.frag')

        self.U_inflow = np.array([1.0, 2.0, 0.0, 0.0, 0.0], dtype='f4')
        self.bc_type = 1  # 0=reflective walls, 1=inflow/outflow X
        self.cfl_target = 0.3
        self.dt_update_interval = 10
        self._dt_counter = 0

    def _make_dual_fbo(self, tex1, tex2):
        dummy1 = self.ctx.texture((self.grid_size, self.grid_size), 4, dtype='f2')
        dummy2 = self.ctx.texture((self.grid_size, self.grid_size), 1, dtype='f2')
        self.mg_dummies += [dummy1, dummy2]
        fbo = self.ctx.framebuffer(color_attachments=[dummy1, dummy2])
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo.glo)
        gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, tex1.glo, 0)
        gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, tex2.glo, 0)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Incomplete dual FBO")
        return fbo

    def _load_program(self, frag_path):
        with open('shaders/layered.vert', 'r') as f:
            vert_src = f.read()
        with open('shaders/layered.geom', 'r') as f:
            geom_src = f.read()
        with open(frag_path, 'r') as f:
            frag_src = f.read()
        prog = self.ctx.program(
            vertex_shader=vert_src,
            geometry_shader=geom_src,
            fragment_shader=frag_src
        )
        return prog

    def render_pass(self, prog, fbo):
        fbo.use()
        gl.glViewport(0, 0, self.grid_size, self.grid_size)
        vao = self.ctx.vertex_array(prog, [])
        vao.render(mode=moderngl.TRIANGLES, vertices=3, instances=self.grid_size)

    def render_pass_dual(self, prog, fbo):
        fbo.use()
        gl.glViewport(0, 0, self.grid_size, self.grid_size)
        bufs = gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1
        gl.glDrawBuffers(2, bufs)
        vao = self.ctx.vertex_array(prog, [])
        vao.render(mode=moderngl.TRIANGLES, vertices=3, instances=self.grid_size)
        gl.glDrawBuffers(1, bufs)  # reset

    def set_reynolds(self, Re, char_length):
        pass  # No viscosity in Euler

    def set_obstacle(self, mask):
        from scipy.ndimage import distance_transform_edt
        self.tex_obstacle.write(np.transpose(mask, (2, 1, 0)).tobytes())
        dist_out = distance_transform_edt(1 - mask).astype(np.float16)
        dist_in = distance_transform_edt(mask).astype(np.float16)
        sdf = dist_out - dist_in
        self.tex_sdf.write(np.transpose(sdf, (2, 1, 0)).tobytes())

    def set_inflow(self, rho=1.0, u=2.0, v=0.0, w=0.0, p=None):
        if p is None:
            p = 1.0 / self.gamma
        E = p / (self.gamma - 1.0) + 0.5 * rho * (u*u + v*v + w*w)
        # Round-trip through float16 so inflow matches texture storage
        f16 = np.array([rho, rho*u, rho*v, rho*w, E], dtype='f2')
        self.U_inflow = np.array(f16, dtype='f4')

    def init_uniform(self, rho=1.0, u=2.0, v=0.0, w=0.0, p=None):
        self.set_inflow(rho, u, v, w, p)
        arr1 = np.zeros((self.grid_size, self.grid_size, self.grid_size, 4), dtype='f2')
        arr2 = np.zeros((self.grid_size, self.grid_size, self.grid_size, 1), dtype='f2')
        arr1[:, :, :, 0] = self.U_inflow[0]
        arr1[:, :, :, 1] = self.U_inflow[1]
        arr1[:, :, :, 2] = self.U_inflow[2]
        arr1[:, :, :, 3] = self.U_inflow[3]
        arr2[:, :, :, 0] = self.U_inflow[4]
        self.tex_U1.write(np.transpose(arr1, (2, 1, 0, 3)).tobytes())
        self.tex_U2.write(np.transpose(arr2, (2, 1, 0, 3)).tobytes())
        self.tex_U1_B.write(np.transpose(arr1, (2, 1, 0, 3)).tobytes())
        self.tex_U2_B.write(np.transpose(arr2, (2, 1, 0, 3)).tobytes())

    def init_sod(self):
        """Sod shock tube initial condition along X axis"""
        g = self.grid_size
        half = g // 2
        gamma = self.gamma

        # Left state
        rho_L = 1.0
        u_L = 0.0
        p_L = 1.0
        # Right state
        rho_R = 0.125
        u_R = 0.0
        p_R = 0.1

        arr1 = np.zeros((g, g, g, 4), dtype='f2')
        arr2 = np.zeros((g, g, g, 1), dtype='f2')
        for x in range(g):
            if x < half:
                rho, u, p = rho_L, u_L, p_L
            else:
                rho, u, p = rho_R, u_R, p_R
            E = p / (gamma - 1.0) + 0.5 * rho * u * u
            arr1[x, :, :, 0] = rho
            arr1[x, :, :, 1] = rho * u
            arr1[x, :, :, 2] = 0.0
            arr1[x, :, :, 3] = 0.0
            arr2[x, :, :, 0] = E

        self.tex_U1.write(np.transpose(arr1, (2, 1, 0, 3)).tobytes())
        self.tex_U2.write(np.transpose(arr2, (2, 1, 0, 3)).tobytes())
        self.tex_U1_B.write(np.transpose(arr1, (2, 1, 0, 3)).tobytes())
        self.tex_U2_B.write(np.transpose(arr2, (2, 1, 0, 3)).tobytes())

    def _copy_to_old(self):
        """Copy U_A to U_old for RK stage combination"""
        data1 = self.tex_U1.read()
        data2 = self.tex_U2.read()
        self.tex_U1_old.write(data1)
        self.tex_U2_old.write(data2)

    def _swap(self):
        self.tex_U1, self.tex_U1_B = self.tex_U1_B, self.tex_U1
        self.tex_U2, self.tex_U2_B = self.tex_U2_B, self.tex_U2
        self.fbo_U1, self.fbo_U1_B = self.fbo_U1_B, self.fbo_U1
        self.fbo_U2, self.fbo_U2_B = self.fbo_U2_B, self.fbo_U2
        self.fbo_dual, self.fbo_dual_A = self.fbo_dual_A, self.fbo_dual

    def _run_stage(self, rk_a, rk_b, rk_c):
        """Run one SSP-RK3 stage"""
        p = self.prog_update
        p['u_dt'].value = self.dt
        p['u_dx'].value = 1.0
        p['u_gamma'].value = self.gamma
        p['u_grid_size'].value = self.grid_size
        p['u_rk_a'].value = rk_a
        p['u_rk_b'].value = rk_b
        p['u_rk_c'].value = rk_c
        p['u_bc_type'].value = self.bc_type

        inf1 = self.U_inflow
        p['u_inflow1'].value = (inf1[0], inf1[1], inf1[2], inf1[3])
        p['u_inflow2'].value = (inf1[4], 0.0, 0.0, 0.0)
        # Internal energy (p_inf/(gamma-1)) for obstacle cells
        e_int = inf1[4] - 0.5 * (inf1[1]*inf1[1] + inf1[2]*inf1[2] + inf1[3]*inf1[3]) / max(inf1[0], 1e-8)
        p['u_inflow3'].value = (float(e_int), 0.0, 0.0, 0.0)

        # Bind textures
        self.tex_U1_B.use(location=0)  # Current state (evolving)
        self.tex_U2_B.use(location=1)
        self.tex_U1_old.use(location=2)  # U^n
        self.tex_U2_old.use(location=3)
        self.tex_obstacle.use(location=4)
        self.tex_sdf.use(location=5)
        p['u_U1'].value = 0
        p['u_U2'].value = 1
        p['u_U1_old'].value = 2
        p['u_U2_old'].value = 3
        p['u_obstacle'].value = 4
        p['u_sdf'].value = 5

        # Render to dual FBO (both U1 and U2)
        self.render_pass_dual(p, self.fbo_dual)

    def step(self):
        """One full timestep with SSP-RK3"""
        # Adaptive CFL: update dt every N steps
        self._dt_counter += 1
        if self._dt_counter >= self.dt_update_interval:
            self._update_dt()
            self._dt_counter = 0
        # Copy U^n (in A) to old buffer for reference
        data1 = self.tex_U1.read()
        data2 = self.tex_U2.read()
        self.tex_U1_old.write(data1)
        self.tex_U2_old.write(data2)
        # Copy U^n to B as initial evolving state
        self.tex_U1_B.write(data1)
        self.tex_U2_B.write(data2)

        # Stage 1: U(1) = 1*U^n + 0*U^n + 1*dt*L(U^n)
        self._run_stage(1.0, 0.0, 1.0)

        # Stage 2: U(2) = 0.75*U^n + 0.25*U(1) + 0.25*dt*L(U(1))
        self._run_stage(0.75, 0.25, 0.25)

        # Stage 3: U^{n+1} = 1/3*U^n + 2/3*U(2) + 2/3*dt*L(U(2))
        self._run_stage(1.0/3.0, 2.0/3.0, 2.0/3.0)

        # Swap A and B so new state is in A
        self._swap()

    def _update_dt(self):
        """Adaptive dt based on CFL condition"""
        U1, E = self.read_state()
        rho = np.maximum(U1[:,:,:,0], 1e-8)
        u = U1[:,:,:,1] / rho
        v = U1[:,:,:,2] / rho
        w = U1[:,:,:,3] / rho
        ke = 0.5 * (U1[:,:,:,1]**2 + U1[:,:,:,2]**2 + U1[:,:,:,3]**2) / rho
        p = np.maximum((self.gamma - 1.0) * (E[:,:,:,0] - ke), 1e-8)
        c = np.sqrt(self.gamma * p / rho)
        speed = np.abs(u) + c
        max_speed = float(np.max(speed))
        if max_speed < 1e-10:
            max_speed = 1.0
        self.dt = self.cfl_target * 1.0 / max_speed

    def read_state(self):
        """Read back the conserved variables as numpy arrays"""
        raw1 = self.tex_U1.read()
        raw2 = self.tex_U2.read()
        U1 = np.frombuffer(raw1, dtype=np.float16).reshape((self.grid_size, self.grid_size, self.grid_size, 4))
        E = np.frombuffer(raw2, dtype=np.float16).reshape((self.grid_size, self.grid_size, self.grid_size, 1))
        return U1, E

    def compute_mach(self):
        """Compute Mach number field from current state"""
        U1, E = self.read_state()
        rho = np.maximum(U1[..., 0], 1e-8)
        u = U1[..., 1] / rho
        v = U1[..., 2] / rho
        w = U1[..., 3] / rho
        ke = 0.5 * (U1[..., 1]**2 + U1[..., 2]**2 + U1[..., 3]**2) / rho
        p = np.maximum((self.gamma - 1.0) * (E[..., 0] - ke), 1e-8)
        c = np.sqrt(self.gamma * p / rho)
        speed = np.sqrt(u*u + v*v + w*w)
        return speed / c
