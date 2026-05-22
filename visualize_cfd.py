"""
3D Volumetric Raymarching Visualizer for pre-computed CFD data.

Streams frames from NVME SSD to GPU per-frame (no RAM preload).
Requires pre-computed frames in /home/aaditya/Downloads/tmp/

Controls:
    Left-click + drag : Orbit camera
    Scroll wheel      : Zoom in / out
    Space              : Pause / resume playback
    Left / Right arrow : Step through frames manually
    V                  : Toggle Velocity / Dye mode
    ESC                : Quit

Usage:
    CFD_DATA_DIR=/path/to/frames \
    __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
        python visualize_cfd.py
"""
import moderngl
import moderngl_window as mglw
import numpy as np
import math
import os
import glob
import sys
import time
from scipy.ndimage import distance_transform_edt


class CFDVisualizer(mglw.WindowConfig):
    title = "CFD 3D Fluid Visualizer"
    gl_version = (3, 3)
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True
    resource_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '.'))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print(f"GPU Vendor  : {self.ctx.info['GL_VENDOR']}")
        print(f"GPU Renderer: {self.ctx.info['GL_RENDERER']}")
        print()

        data_dir = os.environ.get('CFD_DATA_DIR', '/home/aaditya/Downloads/tmp')
        self.data_dir = data_dir
        self.live = os.environ.get('CFD_LIVE', '0') == '1'
        self._loaded_frames = 0
        self._live_latest = 0  # last uploaded frame index in live mode

        self.frame_files = sorted(glob.glob(os.path.join(data_dir, 'vel_*.npy')))
        if not self.frame_files and not self.live:
            raise FileNotFoundError(
                f"No simulation data found in {data_dir}/\n"
                "Run simulate_cfd.py first!"
            )

        if self.live and not self.frame_files:
            print(f"Live mode: waiting for simulation output in {data_dir}/ ...")
            while not self.frame_files:
                import time
                time.sleep(0.5)
                self.frame_files = sorted(glob.glob(os.path.join(data_dir, 'vel_*.npy')))
                # Also check for obstacles while waiting
                if os.path.exists(os.path.join(data_dir, 'obstacles.npy')):
                    print("  Obstacles found, waiting for first velocity frame...")
            print(f"  Found first frame. Streaming live from {data_dir}")

        n_frames = len(self.frame_files)
        print(f"Found {n_frames} frames. Streaming from {data_dir}")
        print(f"  (NVME SSD — expect ~30ms per frame load)")
        print()

        first = np.load(self.frame_files[0])
        self.grid_size = first.shape[0]
        print(f"  Frame data shape: {first.shape}, dtype: {first.dtype}")
        print(f"  Vel range: {first[:,:,:,:3].min():.3f} to {first[:,:,:,:3].max():.3f}")
        print(f"  Den range: {first[:,:,:,3].min():.3f} to {first[:,:,:,3].max():.3f}")

        obs_path = os.path.join(data_dir, 'obstacles.npy')
        if os.path.exists(obs_path):
            print("Loading obstacle mask...")
            obs_array = np.load(obs_path)
            self.tex_obstacle = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='u1')
            self.tex_obstacle.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.tex_obstacle.write(obs_array.tobytes())

            print("Computing SDF from obstacle mask...")
            dist_out = distance_transform_edt(1 - obs_array).astype(np.float16)
            dist_in = distance_transform_edt(obs_array).astype(np.float16)
            sdf = dist_out - dist_in
            sdf_range = np.max(np.abs(sdf))
            print(f"  SDF range: ±{sdf_range:.1f}")
            self.tex_sdf = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='f2')
            self.tex_sdf.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.tex_sdf.write(sdf.tobytes())
        else:
            self.tex_obstacle = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='u1')
            self.tex_obstacle.write(np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype='u1').tobytes())
            self.tex_sdf = self.ctx.texture3d((self.grid_size, self.grid_size, self.grid_size), 1, dtype='f2')
            self.tex_sdf.write(np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype='f2').tobytes())

        # Single GPU texture — updated each frame from RAM via PCIe (~3ms)
        self.tex_volume = self.ctx.texture3d(
            (self.grid_size, self.grid_size, self.grid_size),
            4, dtype='f2'
        )
        self.tex_volume.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_volume.repeat_x = False
        self.tex_volume.repeat_y = False
        self.tex_volume.repeat_z = False

        self._upload_frame(0)

        vert_path = os.path.join(self.resource_dir, 'shaders', 'raymarch.vert')
        frag_path = os.path.join(self.resource_dir, 'shaders', 'raymarch.frag')
        with open(vert_path) as f:
            vert_src = f.read()
        with open(frag_path) as f:
            frag_src = f.read()

        self.prog_raymarch = self.ctx.program(
            vertex_shader=vert_src,
            fragment_shader=frag_src,
        )
        self.vao_quad = self.ctx.vertex_array(self.prog_raymarch, [])

        self.prog_raymarch['u_volume'].value = 0
        self.prog_raymarch['u_obstacle'].value = 1
        self.prog_raymarch['u_sdf'].value = 2
        self.prog_raymarch['u_density_scale'].value = 35.0
        self.prog_raymarch['u_step_size'].value     = 0.006
        self.prog_raymarch['u_num_steps'].value      = 160
        self.prog_raymarch['u_grid_size'].value      = self.grid_size

        self.current_frame = 0
        self.last_loaded_frame = 0
        self.playing = True
        self.playback_speed = 15.0
        self.time_accum = 0.0
        self._loaded_frames = 1  # first frame already loaded above
        self.viz_mode = 0

        self.cam_dist  = 3.8
        self.cam_yaw   = -0.5
        self.cam_pitch = 0.3
        self.dragging  = False

        self.prog_raymarch['u_viz_mode'].value = self.viz_mode

        print(f"Ready! Streaming frames from NVME SSD.  Mode: Velocity (default)")
        print("  Mouse: orbit  |  Scroll: zoom  |  Space: pause  |  V: cycle viz mode")

    def _upload_frame(self, idx):
        idx = idx % len(self.frame_files)
        data = np.load(self.frame_files[idx])
        self.tex_volume.write(data.tobytes())
        self.last_loaded_frame = idx

    def _cam_pos(self):
        x = self.cam_dist * math.cos(self.cam_pitch) * math.sin(self.cam_yaw)
        y = self.cam_dist * math.sin(self.cam_pitch)
        z = self.cam_dist * math.cos(self.cam_pitch) * math.cos(self.cam_yaw)
        return np.array([x, y, z], dtype='f4')

    @staticmethod
    def _look_at(eye, target, up):
        f = target - eye
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        s_norm = np.linalg.norm(s)
        if s_norm < 1e-6:
            s = np.array([1.0, 0.0, 0.0], dtype='f4')
        else:
            s = s / s_norm
        u = np.cross(s, f)
        M = np.eye(4, dtype='f4')
        M[0, :3] = s
        M[1, :3] = u
        M[2, :3] = -f
        T = np.eye(4, dtype='f4')
        T[:3, 3] = -eye
        return M @ T

    @staticmethod
    def _perspective(fov_deg, aspect, near, far):
        f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
        M = np.zeros((4, 4), dtype='f4')
        M[0, 0] = f / aspect
        M[1, 1] = f
        M[2, 2] = (far + near) / (near - far)
        M[2, 3] = (2 * far * near) / (near - far)
        M[3, 2] = -1.0
        return M

    def on_mouse_press_event(self, x, y, button):
        if button == 1:
            self.dragging = True

    def on_mouse_release_event(self, x, y, button):
        if button == 1:
            self.dragging = False

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self.dragging:
            self.cam_yaw   += dx * 0.005
            self.cam_pitch += dy * 0.005
            self.cam_pitch = max(-1.2, min(1.2, self.cam_pitch))

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.cam_dist += y_offset * 0.25
        self.cam_dist = max(1.5, min(12.0, self.cam_dist))

    def on_key_event(self, key, action, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return
        keys = self.wnd.keys
        if key == keys.SPACE:
            self.playing = not self.playing
            state = ">> Playing" if self.playing else "|| Paused"
            print(f"[{state}] Frame {self.current_frame}/{len(self.frame_files)-1}")
        elif key == keys.V:
            self.viz_mode = (self.viz_mode + 1) % 6
            modes = ["Velocity", "Dye", "Vorticity", "Near Object", "Mach", "Schlieren"]
            print(f"Viz Mode: {modes[self.viz_mode]}")
            self.prog_raymarch['u_viz_mode'].value = self.viz_mode
        elif key == keys.RIGHT:
            self.current_frame = (self.current_frame + 1) % len(self.frame_files)
            self._upload_frame(self.current_frame)
        elif key == keys.LEFT:
            self.current_frame = (self.current_frame - 1) % len(self.frame_files)
            self._upload_frame(self.current_frame)

    def on_render(self, time_val, frame_time):
        # Live mode: check for new frames and show latest
        if self.live:
            new_files = sorted(glob.glob(os.path.join(self.data_dir, 'vel_*.npy')))
            if not new_files:
                return
            if len(new_files) > len(self.frame_files):
                self.frame_files = new_files
                print(f"  Live: {len(self.frame_files)} frames now available")
            latest = len(self.frame_files) - 1
            if latest != self._live_latest:
                self._live_latest = latest
                self.current_frame = latest
                self._upload_frame(latest)
        elif self.playing and len(self.frame_files) > 1:
            self.time_accum += frame_time
            frames_to_advance = int(self.time_accum * self.playback_speed)
            if frames_to_advance > 0:
                self.time_accum -= frames_to_advance / self.playback_speed
                n = len(self.frame_files)
                self.current_frame = (self.current_frame + frames_to_advance) % n
                self._upload_frame(self.current_frame)

        w, h = self.wnd.buffer_size
        aspect = w / h if h > 0 else 1.0
        eye = self._cam_pos()
        target = np.array([0.0, 0.0, 0.0], dtype='f4')
        up = np.array([0.0, 1.0, 0.0], dtype='f4')

        view = self._look_at(eye, target, up)
        proj = self._perspective(55.0, aspect, 0.1, 100.0)
        vp = proj @ view
        inv_vp = np.linalg.inv(vp)

        self.tex_volume.use(location=0)
        self.tex_obstacle.use(location=1)
        self.tex_sdf.use(location=2)

        self.prog_raymarch['u_volume'].value = 0
        self.prog_raymarch['u_obstacle'].value = 1
        self.prog_raymarch['u_sdf'].value = 2

        self.prog_raymarch['u_cam_pos'].value = tuple(eye)
        self.prog_raymarch['u_inv_view_proj'].write(inv_vp.T.astype('f4').tobytes())
        self.prog_raymarch['u_viz_mode'].value = self.viz_mode

        self.ctx.screen.use()
        if self.viz_mode == 0:
            self.ctx.clear(0.05, 0.01, 0.01)
        elif self.viz_mode == 1:
            self.ctx.clear(0.01, 0.01, 0.05)
        elif self.viz_mode == 2:
            self.ctx.clear(0.01, 0.05, 0.01)
        else:
            self.ctx.clear(0.02, 0.02, 0.02)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.vao_quad.render(mode=moderngl.TRIANGLES, vertices=6)


if __name__ == '__main__':
    CFDVisualizer.run()
