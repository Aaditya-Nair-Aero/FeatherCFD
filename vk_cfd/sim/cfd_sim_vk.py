import struct
import numpy as np
import os
import vulkan as vk
from vulkan._vulkan import ffi

from ..common.device import VKContext
from ..common.texture import VKImage3D
from ..common.buffer import VKBuffer, create_staging_buffer
from ..common.shader import compile_glsl_to_spirv, create_shader_module, compile_file_to_spirv
from ..common.pipeline import VKComputePipeline
from ..common.descriptor import VKDescriptorManager

SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shaders')


class CFD_System_VK:
    def __init__(self, ctx: VKContext):
        self.vk = ctx
        self.grid_size = 192
        self.dt = 0.16
        self.inflow_vel = (2.0, 0.0, 0.0)
        self.sgs_coeff = 0.325
        self.viscosity = 1e-5
        self.use_dst = False
        self.rbgs_iters = 60

        self._create_textures()
        self._create_descriptors()
        self._create_pipelines()
        self._create_cmd_buffers()

    def _create_textures(self):
        N = self.grid_size
        self.tex_velocity_A = VKImage3D(self.vk, 'rgba16f', N)
        self.tex_velocity_B = VKImage3D(self.vk, 'rgba16f', N)
        self.tex_pressure_A = VKImage3D(self.vk, 'r32f', N)
        self.tex_pressure_B = VKImage3D(self.vk, 'r32f', N)
        self.tex_divergence = VKImage3D(self.vk, 'r16f', N)
        self.tex_density_A = VKImage3D(self.vk, 'r16f', N)
        self.tex_density_B = VKImage3D(self.vk, 'r16f', N)
        self.tex_obstacle = VKImage3D(self.vk, 'r8ui', N)
        self.tex_sdf = VKImage3D(self.vk, 'r16f', N)

        self.mg_pressure = [VKImage3D(self.vk, 'r32f', max(N >> i, 3)) for i in range(6)]
        self.mg_residual = [VKImage3D(self.vk, 'r32f', max(N >> i, 3)) for i in range(6)]

        for i in range(6):
            sz = max(N >> i, 3)
            self.mg_pressure[i] = VKImage3D(self.vk, 'r32f', sz)
            self.mg_residual[i] = VKImage3D(self.vk, 'r32f', sz)

    def _load_spirv(self, name):
        path = os.path.join(SHADER_DIR, name + '.spv')
        if not os.path.exists(path):
            glsl_path = os.path.join(SHADER_DIR, name + '.comp')
            if os.path.exists(glsl_path):
                spirv = compile_file_to_spirv(glsl_path, 'compute')
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(spirv)
            else:
                raise FileNotFoundError(f"Shader not found: {name}")
        with open(path, 'rb') as f:
            return f.read()

    def _create_descriptors(self):
        self.desc_mgr = VKDescriptorManager(self.vk.device)

        self.desc_layouts = {}
        self.desc_sets = {}

        # copy_tex: 2 bindings (in, out)
        self.desc_layouts['copy'] = self.desc_mgr.create_storage_image_layout(2)

        # forces: 4 bindings (vel_in, vel_out, obs, sdf)
        self.desc_layouts['forces'] = self.desc_mgr.create_storage_image_layout(4)

        # divergence: 4 bindings (vel, obs, sdf, div_out)
        self.desc_layouts['divergence'] = self.desc_mgr.create_storage_image_layout(4)

        # jacobi: 5 bindings (press, div, obs, sdf, press_out)
        self.desc_layouts['jacobi'] = self.desc_mgr.create_storage_image_layout(5)

        # projection: 5 bindings (vel, press, obs, sdf, vel_out)
        self.desc_layouts['projection'] = self.desc_mgr.create_storage_image_layout(5)

        # advection: 4 bindings (vel, vel_out, obs, sdf)
        self.desc_layouts['advection'] = self.desc_mgr.create_storage_image_layout(4)

        # advection_density: 5 bindings (vel, dens, dens_out, obs, sdf)
        self.desc_layouts['advection_density'] = self.desc_mgr.create_storage_image_layout(5)

        # rbgs: 5 bindings (press, div, obs, sdf, press_out)
        self.desc_layouts['rbgs'] = self.desc_mgr.create_storage_image_layout(5)

        # fine_residual: 5 bindings (press, div, obs, sdf, residual)
        self.desc_layouts['fine_residual'] = self.desc_mgr.create_storage_image_layout(5)

        # coarse_jacobi/rbgs/residual/add: 4 bindings
        self.desc_layouts['coarse_4'] = self.desc_mgr.create_storage_image_layout(4)
        # coarse_add: 3 bindings
        self.desc_layouts['coarse_3'] = self.desc_mgr.create_storage_image_layout(3)

        # restrict/prolongate: 2 bindings
        self.desc_layouts['mg_2'] = self.desc_mgr.create_storage_image_layout(2)

        # add_correction: 4 bindings (press, corr, sdf, out)
        self.desc_layouts['add_correction'] = self.desc_mgr.create_storage_image_layout(4)

        # obstacle_zero: 3 bindings
        self.desc_layouts['obstacle_zero'] = self.desc_mgr.create_storage_image_layout(3)

        # rk4_setup/save_k: 3 bindings
        self.desc_layouts['rk4_3'] = self.desc_mgr.create_storage_image_layout(3)

        # rk4_combine: 6 bindings
        self.desc_layouts['rk4_6'] = self.desc_mgr.create_storage_image_layout(6)

        # euler_update: 8 bindings
        self.desc_layouts['euler'] = self.desc_mgr.create_storage_image_layout(8)

    def _make_set(self, layout_name, views):
        return self.desc_mgr.create_descriptor_set(self.desc_layouts[layout_name], views)

    def _create_pipelines(self):
        pipe_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shaders')
        self.pipelines = {}

        shader_map = {
            'copy': ('copy_tex', 'copy', 0),
            'forces': ('forces', 'forces', 64),
            'divergence': ('divergence', 'divergence', 0),
            'jacobi': ('jacobi', 'jacobi', 0),
            'projection': ('projection', 'projection', 32),
            'advection': ('advection', 'advection', 64),
            'advection_density': ('advection_density', 'advection_density', 64),
            'rbgs': ('rbgs', 'rbgs', 16),
            'fine_residual': ('fine_residual', 'fine_residual', 0),
            'coarse_jacobi': ('coarse_jacobi', 'coarse_4', 4),
            'coarse_rbgs': ('coarse_rbgs', 'coarse_4', 8),
            'coarse_residual': ('coarse_residual', 'coarse_4', 0),
            'coarse_add': ('coarse_add', 'coarse_3', 0),
            'restrict': ('restrict', 'mg_2', 0),
            'prolongate': ('prolongate', 'mg_2', 0),
            'add_correction': ('add_correction', 'add_correction', 4),
            'obstacle_zero': ('obstacle_zero', 'obstacle_zero', 0),
            'rk4_setup': ('rk4_setup', 'rk4_3', 4),
            'rk4_save_k': ('rk4_save_k', 'rk4_3', 0),
            'rk4_combine': ('rk4_combine', 'rk4_6', 0),
            'euler_update': ('euler_update', 'euler', 128),
            'diag_x0': ('diag_x0', 'forces', 32),
            'diag_copy4': ('diag_copy4', 'forces', 64),
            'diag_uncond': ('diag_uncond', 'forces', 64),
            'diag_nopc': ('diag_nopc', 'forces', 0),
            'diag_2bind_pc': ('diag_2bind_pc', 'copy', 64),
            'diag_zcheck': ('diag_zcheck', 'copy', 0),
            'diag_zcheck4': ('diag_zcheck4', 'forces', 0),
            'diag_xcheck': ('diag_xcheck', 'copy', 0),
            'diag_xcheck4': ('diag_xcheck4', 'forces', 0),
            'diag_size': ('diag_size', 'copy', 0),
        }

        for name, (shader_name, layout_name, pc_size) in shader_map.items():
            spirv = self._load_spirv(shader_name)
            module = create_shader_module(self.vk.device, spirv)
            layout = self.desc_layouts.get(layout_name)
            self.pipelines[name] = VKComputePipeline(
                self.vk.device, module, layout, pc_size)
            vk.vkDestroyShaderModule(self.vk.device, module, None)

    def _create_cmd_buffers(self):
        self.cmd = self.vk.create_command_buffer()

    def _begin_cmd(self):
        self.vk.begin_command_buffer(self.cmd)

    def _end_cmd(self):
        self.vk.end_and_submit(self.cmd)
        self.cmd = self.vk.create_command_buffer()

    def _dispatch(self, pipe_name, descriptor_set, push_consts=None, grid_size=None):
        if grid_size is None:
            grid_size = self.grid_size
        groups = (grid_size + 7) // 8
        pipeline = self.pipelines[pipe_name]

        vk.vkCmdBindPipeline(self.cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline)
        vk.vkCmdBindDescriptorSets(self.cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline.layout, 0, 1, [descriptor_set], 0, None)
        if push_consts is not None:
            if isinstance(push_consts, tuple):
                data, size = push_consts
            else:
                data, size = push_consts, len(push_consts)
            if isinstance(data, bytes):
                data = ffi.new('char[]', data)
            vk.vkCmdPushConstants(self.cmd, pipeline.layout,
                                  vk.VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                  size, data)
        vk.vkCmdDispatch(self.cmd, groups, groups, groups)

    def _barrier(self, image=None):
        if image is not None:
            barrier = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=image,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            )
            vk.vkCmdPipelineBarrier(self.cmd,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, None, 0, None, 1, [barrier])
        else:
            mem_barrier = vk.VkMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
            )
            vk.vkCmdPipelineBarrier(self.cmd,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, [mem_barrier], 0, None, 0, None)

    def _dct_solve(self):
        """CPU DST-based Poisson solver + GPU RBGS post-smoothing.

        Uses DST-I on interior points (Dirichlet p=0 at domain boundaries),
        matching the finite-difference discretization of the pressure Poisson
        equation.
        """
        from scipy.fft import dst, idst
        N = self.grid_size
        n_smooth = 10
        M = N - 2  # interior points excluding boundary planes

        div_data = self.tex_divergence.download()
        div = div_data.astype(np.float64)

        obs_data = self.tex_obstacle.download()
        obs = obs_data.astype(bool)
        div[obs] = 0.0

        # Solve Poisson on interior: ∇²p = div, p=0 at boundaries
        # DST-I diagonalizes the (N-2)^3 interior Laplacian
        k = np.arange(M, dtype=np.float64) + 1.0
        eig1d = 2.0 * (np.cos(np.pi * k / (N - 1)) - 1.0)
        eig = eig1d[:, None, None] + eig1d[None, :, None] + eig1d[None, None, :]
        eig_min_abs = np.min(np.abs(eig[eig < 0]))
        eig = np.where(np.abs(eig) < 1e-12, eig_min_abs * 1e-8, eig)

        div_int = div[1:-1, 1:-1, 1:-1]
        coef = dst(div_int, type=1, axis=0, norm='ortho')
        coef = dst(coef, type=1, axis=1, norm='ortho')
        coef = dst(coef, type=1, axis=2, norm='ortho')
        coef /= eig
        p_int = idst(coef, type=1, axis=0, norm='ortho')
        p_int = idst(p_int, type=1, axis=1, norm='ortho')
        p_int = idst(p_int, type=1, axis=2, norm='ortho')

        # Reconstruct full pressure with Dirichlet BCs
        p = np.zeros((N, N, N), dtype=np.float64)
        p[1:-1, 1:-1, 1:-1] = p_int
        p[obs] = 0.0
        p = np.clip(p, -1e3, 1e3)

        p_f32 = np.ascontiguousarray(p.astype(np.float32))
        self.tex_pressure_A.upload(p_f32)
        self.tex_pressure_B.upload(p_f32)

        self._rbgs_smooth(n_smooth)

    def _rbgs_smooth(self, n_iter):
        self._begin_cmd()
        for _ in range(n_iter):
            v = [self.tex_pressure_A.view, self.tex_divergence.view,
                 self.tex_obstacle.view, self.tex_sdf.view,
                 self.tex_pressure_B.view]
            pc = struct.pack('if', 0, 1.0) + b'\x00' * 8
            self._dispatch('rbgs', self._make_set('rbgs', v), (pc, 16))
            self._barrier()

            v = [self.tex_pressure_B.view, self.tex_divergence.view,
                 self.tex_obstacle.view, self.tex_sdf.view,
                 self.tex_pressure_A.view]
            pc = struct.pack('if', 1, 1.0) + b'\x00' * 8
            self._dispatch('rbgs', self._make_set('rbgs', v), (pc, 16))
            self._barrier()
        self._end_cmd()

    def step(self):
        import struct

        self._begin_cmd()

        # PASS 1: Forces (A -> B)  pc_size=64
        v = [self.tex_velocity_A.view, self.tex_velocity_B.view,
             self.tex_obstacle.view, self.tex_sdf.view]
        pc = struct.pack('f', self.dt) + b'\x00' * 12
        pc += b'\x00' * 16 + b'\x00' * 16
        pc += struct.pack('fff', *self.inflow_vel) + b'\x00' * 4
        self._dispatch('forces', self._make_set('forces', v), (pc, 64))
        self._barrier()

        # PASS 2: Advection (B -> A)  pc_size=64
        v = [self.tex_velocity_B.view, self.tex_velocity_A.view,
             self.tex_obstacle.view, self.tex_sdf.view]
        pc = struct.pack('f', self.dt) + struct.pack('f', self.viscosity) + struct.pack('f', self.sgs_coeff) + b'\x00' * 4
        pc += struct.pack('fff', *self.inflow_vel) + b'\x00' * 36
        self._dispatch('advection', self._make_set('advection', v), (pc, 64))
        self._barrier()

        # PASS 3: Divergence (A -> Div)  pc_size=0
        v = [self.tex_velocity_A.view, self.tex_obstacle.view,
             self.tex_sdf.view, self.tex_divergence.view]
        self._dispatch('divergence', self._make_set('divergence', v))
        self._barrier()

        # Submit divergence compute for CPU readback
        self._end_cmd()

        # PASS 4: Pressure solve
        if self.use_dst:
            self._dct_solve()
        else:
            # GPU-only RBGS (fast, avoid slow CPU DST)
            self._rbgs_smooth(self.rbgs_iters)

        # PASS 5: Projection (A + Pressure -> B)  pc_size=32
        self._begin_cmd()
        v = [self.tex_velocity_A.view, self.tex_pressure_A.view,
             self.tex_obstacle.view, self.tex_sdf.view,
             self.tex_velocity_B.view]
        pc = struct.pack('f', self.dt) + b'\x00' * 12
        pc += struct.pack('fff', *self.inflow_vel) + b'\x00' * 4
        self._dispatch('projection', self._make_set('projection', v), (pc, 32))
        self._barrier()

        # PASS 6: Density advection (B velocity + A density -> B density)  pc_size=64
        v = [self.tex_velocity_B.view, self.tex_density_A.view,
             self.tex_density_B.view, self.tex_obstacle.view,
             self.tex_sdf.view]
        pc = struct.pack('f', self.dt) + b'\x00' * 60
        self._dispatch('advection_density', self._make_set('advection_density', v), (pc, 64))
        self._barrier()

        # End-of-step swaps (match OpenGL pattern)
        self.tex_velocity_A, self.tex_velocity_B = \
            self.tex_velocity_B, self.tex_velocity_A
        self.tex_density_A, self.tex_density_B = \
            self.tex_density_B, self.tex_density_A

        self._end_cmd()

    def set_obstacle(self, mask: np.ndarray, sdf: np.ndarray):
        self.tex_obstacle.upload(mask.astype(np.uint8))
        self.tex_sdf.upload(sdf.astype(np.float16))
        self.tex_pressure_A.upload(np.zeros((self.grid_size,) * 3, dtype=np.float32))
        self.tex_pressure_B.upload(np.zeros((self.grid_size,) * 3, dtype=np.float32))

    def set_reynolds(self, re, char_length=60.0):
        self.viscosity = self.inflow_vel[0] * char_length / max(re, 1.0)

    def read_velocity(self):
        return self.tex_velocity_A.download()

    def read_pressure(self):
        return self.tex_pressure_A.download()

    def read_divergence(self):
        return self.tex_divergence.download()

    def read_density(self):
        return self.tex_density_A.download()

    def read_obstacle(self):
        return self.tex_obstacle.download()

    def destroy(self):
        for tex in [self.tex_velocity_A, self.tex_velocity_B,
                     self.tex_pressure_A, self.tex_pressure_B,
                     self.tex_divergence, self.tex_density_A, self.tex_density_B,
                     self.tex_obstacle, self.tex_sdf]:
            tex.destroy()
        for mg_list in [self.mg_pressure, self.mg_residual]:
            for img in mg_list:
                img.destroy()
        for pipe in self.pipelines.values():
            pipe.destroy()
        self.desc_mgr.destroy()
