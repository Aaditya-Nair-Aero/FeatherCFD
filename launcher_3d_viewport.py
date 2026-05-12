"""
3D Viewport for emitter placement — matplotlib 3D.
No OpenGL profile issues. Embedded in tkinter window.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

DARK_BG = "#1a1a2e"
DARK_CARD = "#16213e"
DARK_FG = "#e0e0e0"
DARK_ACCENT = "#0f9b58"
GRID_COLOR = "#3a3a5a"


class EmitterViewport:
    def __init__(self, parent, grid_size=192, initial_emitters=None,
                 obstacle_type="NACA 4-Digit", naca_digits="0012",
                 callback=None):
        self.parent = parent
        self.grid_size = grid_size
        self.emitters = list(initial_emitters or [(0, grid_size//2, grid_size//2, 10)])
        self.selected = -1
        self.obstacle_type = obstacle_type
        self.naca_digits = naca_digits
        self.callback = callback
        self.dragging = False
        self._drag_idx = -1
        self._drag_was_dragged = False
        self._picked_this_click = False
        self._press_coords = None
        self.after_id = None

        self._build_ui()
        self._update_plot()

    def _build_ui(self):
        self.fig = plt.Figure(figsize=(8, 6), dpi=100,
                              facecolor=DARK_BG)
        self.fig.patch.set_facecolor(DARK_BG)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor(DARK_CARD)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("pick_event", self._on_pick)

        top = tk.Frame(self.parent, bg=DARK_BG)
        top.pack(fill="x", padx=5, pady=2)

        self.info_label = tk.Label(top, text=f"Emitters: {len(self.emitters)}",
                                   bg=DARK_BG, fg=DARK_FG,
                                   font=("Segoe UI", 10))
        self.info_label.pack(side="left", padx=5)

        ttk.Button(top, text="+ Add Emitter",
                   command=self._add_emitter, style="TButton",
                   width=15).pack(side="right", padx=2)
        ttk.Button(top, text="Done",
                   command=self._done, style="Accent.TButton",
                   width=10).pack(side="right", padx=2)
        ttk.Button(top, text="Delete Selected",
                   command=self._delete_selected, style="Danger.TButton",
                   width=15).pack(side="right", padx=2)

        instr = tk.Label(self.parent, text="Left-click to select emitter | Drag to move in X=0 plane | "
                         "'+ Add Emitter' to place new | 'Delete Selected' to remove",
                         bg=DARK_BG, fg="#7f8fa6", font=("Segoe UI", 9))
        instr.pack(fill="x", padx=5, pady=(0, 5))

    def _update_plot(self):
        self.ax.clear()
        g = self.grid_size

        self.ax.plot([0, 0, 0, 0, 0], [0, 0, g, g, 0], [0, g, g, 0, 0],
                     color=GRID_COLOR, linewidth=0.5)
        self.ax.plot([g, g, g, g, g], [0, 0, g, g, 0], [0, g, g, 0, 0],
                     color=GRID_COLOR, linewidth=0.5)
        for x in [0, g]:
            self.ax.plot([x, x], [0, g], [0, 0], color=GRID_COLOR, linewidth=0.5)
            self.ax.plot([x, x], [0, g], [g, g], color=GRID_COLOR, linewidth=0.5)
            self.ax.plot([x, x], [0, 0], [0, g], color=GRID_COLOR, linewidth=0.5)
            self.ax.plot([x, x], [g, g], [0, g], color=GRID_COLOR, linewidth=0.5)

        for i in range(5):
            frac = i / 4 * g
            self.ax.plot([0, 0], [frac, frac], [0, g], color=GRID_COLOR, linewidth=0.3, alpha=0.5)
            self.ax.plot([0, 0], [0, g], [frac, frac], color=GRID_COLOR, linewidth=0.3, alpha=0.5)

        arrow_len = g * 0.15
        self.ax.quiver(0, 0, 0, arrow_len, 0, 0, color="#e94560",
                       label="X (inflow)")
        self.ax.quiver(0, 0, 0, 0, arrow_len, 0, color="#0f9b58",
                       label="Y (span)")
        self.ax.quiver(0, 0, 0, 0, 0, arrow_len, color="#0f3460",
                       label="Z (lift)")

        self._draw_obstacle()

        emitter_colors = []
        for i, (ex, ey, ez, er) in enumerate(self.emitters):
            color = "#e94560" if i == self.selected else "#0f9b58"
            size = 60 if i == self.selected else 40
            self.ax.scatter([ex], [ey], [ez], c=color, s=size,
                           picker=5, alpha=0.9, edgecolors="white",
                           linewidths=1 if i == self.selected else 0.5)
            self.ax.quiver(ex, ey, ez, arrow_len*0.3, 0, 0,
                          color=color, alpha=0.7)
            self.ax.text(ex, ey, ez, f"  {i+1}", color="white", fontsize=8)

        self.ax.set_xlabel("X", color=DARK_FG)
        self.ax.set_ylabel("Y", color=DARK_FG)
        self.ax.set_zlabel("Z", color=DARK_FG)
        self.ax.set_title("Emitter Placement — X=0 plane (inflow face)",
                          color=DARK_FG, fontsize=12)

        self.ax.set_xlim(0, g)
        self.ax.set_ylim(0, g)
        self.ax.set_zlim(0, g)
        self.ax.tick_params(colors=DARK_FG)

        self.ax.view_init(elev=20, azim=-60)

        self.canvas.draw()

    def _draw_obstacle(self):
        g = self.grid_size
        otype = self.obstacle_type

        if "NACA" in otype:
            from launcher import generate_naca
            naca = self.naca_digits.strip()
            if len(naca) >= 4:
                try:
                    chord = 60
                    start_x = g * 0.35
                    zc = g / 2
                    x, yu, yl = generate_naca(naca)
                    xs = start_x + x * chord
                    for sy in [g*0.25, g*0.5, g*0.75]:
                        self.ax.plot(xs, [sy]*len(xs), zc + yu*chord,
                                    color="#5555aa", linewidth=1, alpha=0.6)
                        self.ax.plot(xs, [sy]*len(xs), zc + yl*chord,
                                    color="#5555aa", linewidth=1, alpha=0.6)
                    for xi, yupi, yloi in zip(xs, yu*chord, yl*chord):
                        self.ax.plot([xi, xi], [g*0.25, g*0.75],
                                    [zc+yupi, zc+yupi], color="#5555aa",
                                    linewidth=0.3, alpha=0.3)
                        self.ax.plot([xi, xi], [g*0.25, g*0.75],
                                    [zc+yloi, zc+yloi], color="#5555aa",
                                    linewidth=0.3, alpha=0.3)
                except Exception:
                    self._draw_bbox(60, 20)
        elif otype == "Cylinder":
            cx, cz = g/2, g/2
            r = 15
            theta = np.linspace(0, 2*np.pi, 32)
            for sy in [g*0.25, g*0.5, g*0.75]:
                self.ax.plot(cx + r*np.cos(theta), [sy]*32, cz + r*np.sin(theta),
                            color="#5555aa", linewidth=1, alpha=0.6)
        elif otype == "Sphere":
            self._draw_bbox(50, g/2)
        else:
            self._draw_bbox(40, g/2)

    def _draw_bbox(self, size, cx=None):
        g = self.grid_size
        cx = cx or g/2
        hs = size / 2
        corners = np.array([
            [cx-hs, g/2-hs, g/2-hs], [cx+hs, g/2-hs, g/2-hs],
            [cx+hs, g/2+hs, g/2-hs], [cx-hs, g/2+hs, g/2-hs],
            [cx-hs, g/2-hs, g/2+hs], [cx+hs, g/2-hs, g/2+hs],
            [cx+hs, g/2+hs, g/2+hs], [cx-hs, g/2+hs, g/2+hs],
        ])
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for i, j in edges:
            self.ax.plot(*zip(corners[i], corners[j]),
                        color="#5555aa", linewidth=1, alpha=0.6)

    def _on_pick(self, event):
        if event.ind is not None and len(event.ind) > 0:
            self._select_emitter(event.ind[0])
            self._drag_idx = event.ind[0]
            self._picked_this_click = True

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self.dragging = True
        self._picked_this_click = False
        self.after_id = self.parent.after(10, self._check_deselect)

    def _check_deselect(self):
        if not self._picked_this_click and self._drag_idx < 0:
            self._select_emitter(-1)

    def _on_release(self, event):
        self.dragging = False
        if not self._drag_was_dragged:
            self._drag_idx = -1
        self._drag_was_dragged = False

    def _on_motion(self, event):
        if not self.dragging or self._drag_idx < 0 or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._drag_was_dragged = True
        ex, ey, ez, er = self.emitters[self._drag_idx]
        new_y = max(0, min(self.grid_size, round(event.ydata)))
        self.emitters[self._drag_idx] = (0, new_y, ez, er)
        self._update_plot()

    def _select_emitter(self, idx):
        self.selected = idx
        self.info_label.config(text=f"Emitters: {len(self.emitters)}" +
                               (f" | Selected: {idx+1}" if idx >= 0 else ""))
        self._update_plot()

    def _add_emitter(self):
        g = self.grid_size
        self.emitters.append((0, g//2, g//2, 10))
        self._select_emitter(len(self.emitters) - 1)

    def _delete_selected(self):
        if self.selected >= 0 and len(self.emitters) > 1:
            self.emitters.pop(self.selected)
            self.selected = -1
            self._update_plot()
            self.info_label.config(text=f"Emitters: {len(self.emitters)}")

    def _done(self):
        if self.callback:
            self.callback(list(self.emitters))
        try:
            for widget in self.parent.winfo_children():
                widget.destroy()
        except tk.TclError:
            pass


def open_viewport(parent_frame, grid_size=192, initial_emitters=None,
                  obstacle_type="NACA 4-Digit", naca_digits="0012",
                  callback=None):
    """Open emitter viewport in the given tkinter frame."""
    viewport = EmitterViewport(parent_frame, grid_size, initial_emitters,
                               obstacle_type, naca_digits, callback)
    return viewport
