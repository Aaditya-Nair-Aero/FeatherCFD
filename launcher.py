"""
CFD Launcher — dark-themed GUI for setting up and running CFD simulations.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import math
import json
import subprocess
import os
import sys
import threading
import time
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────
DARK_BG = "#1a1a2e"
DARK_FG = "#e0e0e0"
DARK_CARD = "#16213e"
DARK_ACCENT = "#0f9b58"
DARK_ACCENT2 = "#e94560"
DARK_INPUT = "#0f3460"
DARK_BORDER = "#2a2a4a"

# ── NACA profile generator ─────────────────────────────────────────────
def naca_4_digit_coords(digits, n_points=200):
    """Returns (x, y_upper, y_lower) for a NACA 4-digit airfoil."""
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:]) / 100.0
    x = np.linspace(0, 1, n_points)
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    if m == 0:
        return x, yt, -yt
    yc = np.where(x < p, m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2 * (p - x), 2*m/(1-p)**2 * (p - x))
    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta); yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta); yl = yc - yt * np.cos(theta)
    return x, yu, yl


def naca_5_digit_coords(digits, n_points=200):
    """NACA 5-digit: LPQXX — L=lift, P=pos of max camber/20, Q=reflex, XX=thickness"""
    # Simplified: just thickness distribution with camber
    L = int(digits[0])
    P = int(digits[1])
    Q = int(digits[2])
    t = int(digits[3:]) / 100.0
    x = np.linspace(0, 1, n_points)
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    # Approximate mean line for 5-digit
    m = P / 20.0
    cl_max = L * 0.15
    if Q == 0:  # simple
        yc = np.where(x < m, cl_max/m**2 * (2*m*x - x**2) * (1 - x), 
                      cl_max/(1-m)**2 * (1 - 2*m + 2*m*x - x**2) * (1 - x))
    else:  # reflexed — skip for now
        yc = np.zeros_like(x)
    return x, yc + yt, yc - yt


def naca_6_digit_coords(digits, n_points=200):
    """NACA 6-series: 6AABBCC — simplified."""
    t = int(digits[4:]) / 100.0
    x = np.linspace(0, 1, n_points)
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    return x, yt, -yt


def generate_naca(digits, n_points=200):
    if len(digits) == 4:
        return naca_4_digit_coords(digits, n_points)
    elif len(digits) == 5:
        return naca_5_digit_coords(digits, n_points)
    else:
        x = np.linspace(0, 1, n_points)
        t = float(digits[-2:]) / 100.0
        yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
        return x, yt, -yt


def naca_description(digits):
    if len(digits) == 4:
        m = int(digits[0]) / 100.0
        p = int(digits[1]) / 10.0
        t = int(digits[2:]) / 100.0
        return f"NACA {digits}: {t*100:.0f}% thick, max camber {m*100:.0f}% at {p*100:.0f}% chord"
    elif len(digits) == 5:
        t = int(digits[3:]) / 100.0
        return f"NACA {digits}: {t*100:.0f}% thick, 5-digit series"
    else:
        return f"NACA {digits}"


# ── Object definitions ─────────────────────────────────────────────────
OBJECT_TYPES = {
    "NACA 4-Digit": {
        "variants": [f"{m:02d}{p:02d}" for m in range(0, 10) for p in range(0, 10)] +
                    [f"00{t:02d}" for t in range(6, 26)],
        "default": "0012",
        "custom": True,
        "generator": lambda d: naca_4_digit_coords(d),
        "description": naca_description,
    },
    "NACA 5-Digit": {
        "variants": [f"{L}{P}{Q}{t:02d}" for L in range(1, 5) for P in range(1, 10, 2) 
                     for Q in [0, 1] for t in range(12, 26, 2)],
        "default": "23012",
        "custom": True,
        "generator": lambda d: naca_5_digit_coords(d),
        "description": naca_description,
    },
    "NACA 6-Series": {
        "variants": [f"63{t:03d}" for t in range(12, 26, 2)],
        "default": "63012",
        "custom": True,
        "generator": lambda d: naca_6_digit_coords(d),
        "description": naca_description,
    },
    "Cylinder": {
        "variants": [],
        "default": None,
        "custom": False,
        "generator": None,
        "description": lambda _: "Circular cylinder, full span",
    },
    "Sphere": {
        "variants": [],
        "default": None,
        "custom": False,
        "generator": None,
        "description": lambda _: "Spherical obstacle",
    },
    "Custom .STL": {
        "variants": [],
        "default": None,
        "custom": False,
        "generator": None,
        "description": lambda _: "User-provided CAD mesh (STL/OBJ)",
    },
}


# ── Dark Theme ─────────────────────────────────────────────────────────
def setup_dark_theme():
    style = ttk.Style()
    style.theme_use("clam")

    style.configure(".", background=DARK_BG, foreground=DARK_FG,
                    fieldbackground=DARK_INPUT, selectbackground=DARK_ACCENT,
                    selectforeground="white", font=("Segoe UI", 10))
    style.configure("TLabel", background=DARK_BG, foreground=DARK_FG)
    style.configure("TFrame", background=DARK_BG)
    style.configure("TButton", background=DARK_ACCENT, foreground="white",
                    borderwidth=0, focusthroughcolor="none", focuscolor="none",
                    font=("Segoe UI", 10, "bold"))
    style.map("TButton", background=[("active", "#0d8a4f")])
    style.configure("Danger.TButton", background=DARK_ACCENT2,
                    font=("Segoe UI", 10, "bold"))
    style.map("Danger.TButton", background=[("active", "#d6384f")])
    style.configure("Accent.TButton", background=DARK_ACCENT,
                    font=("Segoe UI", 12, "bold"))
    style.map("Accent.TButton", background=[("active", "#0d8a4f")])
    style.configure("TEntry", fieldbackground=DARK_INPUT, foreground=DARK_FG,
                    borderwidth=1)
    style.configure("TCombobox", fieldbackground=DARK_INPUT, foreground=DARK_FG,
                    arrowcolor=DARK_FG)
    style.map("TCombobox", fieldbackground=[("readonly", DARK_INPUT)])
    style.configure("TCheckbutton", background=DARK_BG, foreground=DARK_FG)
    style.configure("TRadiobutton", background=DARK_BG, foreground=DARK_FG)
    style.configure("TScale", background=DARK_BG, troughcolor=DARK_CARD,
                    slidercolor=DARK_ACCENT)
    style.configure("TLabelframe", background=DARK_BG, foreground=DARK_FG,
                    bordercolor=DARK_BORDER)
    style.configure("TLabelframe.Label", background=DARK_BG, foreground=DARK_FG)
    style.configure("Horizontal.TProgressbar", troughcolor=DARK_CARD,
                    background=DARK_ACCENT)
    style.configure("Vertical.TScrollbar", background=DARK_CARD,
                    troughcolor=DARK_BG, arrowcolor=DARK_FG)

    return style


# ── Airfoil Preview Canvas ─────────────────────────────────────────────
class AirfoilPreview(tk.Canvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=DARK_BG, highlightthickness=1,
                         highlightbackground=DARK_BORDER, **kw)
        self.airfoil_data = None
        self.aoa = 0.0

    def draw_airfoil(self, x, yu, yl, aoa=0.0):
        self.delete("all")
        self.airfoil_data = (x, yu, yl)
        self.aoa = aoa

        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10:
            self.after(50, lambda: self.draw_airfoil(x, yu, yl, aoa))
            return

        pad = 20
        cx, cy = w / 2, h / 2
        scale = min((w - 2*pad) / 1.2, (h - 2*pad) / 0.8)

        angle = math.radians(aoa)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        def transform(xp, yp):
            rx = (xp - 0.5) * cos_a - yp * sin_a
            ry = (xp - 0.5) * sin_a + yp * cos_a
            return cx + rx * scale, cy - ry * scale

        pts_up = []
        pts_lo = []
        for xi, yui, yli in zip(x, yu, yl):
            pts_up.append(transform(xi, yui))
        for xi, yui, yli in reversed(list(zip(x, yu, yl))):
            pts_lo.append(transform(xi, yli))

        # Fill
        fill_coords = []
        for px, py in pts_up:
            fill_coords.extend([px, py])
        for px, py in pts_lo:
            fill_coords.extend([px, py])
        self.create_polygon(*fill_coords, fill=DARK_ACCENT, outline=DARK_FG,
                            width=1.5, stipple="")


# ── Main Application ───────────────────────────────────────────────────
class CFDLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CFD Launcher")
        self.root.geometry("1100x750")
        self.root.configure(bg=DARK_BG)
        self.root.minsize(900, 650)

        try:
            self.root.iconbitmap(default="")
        except:
            pass

        self.style = setup_dark_theme()

        # State
        self.state = {
            "object_type": "NACA 4-Digit",
            "naca_digits": "0012",
            "aoa": 5.0,
            "reynolds": 5000,
            "chord": 60,
            "grid_size": 192,
            "stl_path": "",
            "emitters": [(0, 96, 96, 10)],  # (x, y, z, radius)
            "mode": "incompressible",
            "mach": 0.5,
            "velocity": 2.0,
            "density": 1.0,
            "viscosity": None,
            "solver": "vcycle",
            "engine": "opengl",
            "steps": 500,
            "save_every": 2,
        }

        self.naca_var = tk.StringVar(value="0012")
        self.obj_type_var = tk.StringVar(value="NACA 4-Digit")
        self.obj_type_var.trace_add("write", self._on_obj_type_change)
        self.naca_var.trace_add("write", self._on_naca_change)

        self._build_ui()
        self._update_preview()

    # ── UI Builder ─────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=DARK_CARD, height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="CFD Simulation Launcher",
                 font=("Segoe UI", 16, "bold"), bg=DARK_CARD,
                 fg=DARK_ACCENT).pack(side="left", padx=20, pady=10)

        # Main content area — notebook for wizard steps
        self.notebook = ttk.Notebook(self.root, style="TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        style = ttk.Style()
        style.configure("TNotebook", background=DARK_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=DARK_CARD, foreground=DARK_FG,
                        padding=[12, 4], font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", DARK_ACCENT)],
                  foreground=[("selected", "white")])

        # Tab 1: Object Selection
        self.tab_obj = tk.Frame(self.notebook, bg=DARK_BG)
        self.notebook.add(self.tab_obj, text="  1. Select Object  ")
        self._build_tab_object()

        # Tab 2: Emitter Placement (placeholder)
        self.tab_emit = tk.Frame(self.notebook, bg=DARK_BG)
        self.notebook.add(self.tab_emit, text="  2. Emitters  ")
        self._build_tab_emitter()

        # Tab 3: Fluid Properties
        self.tab_fluid = tk.Frame(self.notebook, bg=DARK_BG)
        self.notebook.add(self.tab_fluid, text="  3. Fluid & Sim  ")
        self._build_tab_fluid()

        # Tab 4: Run
        self.tab_run = tk.Frame(self.notebook, bg=DARK_BG)
        self.notebook.add(self.tab_run, text="  4. Run  ")
        self._build_tab_run()

    def _build_tab_object(self):
        tab = self.tab_obj
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(0, weight=1)

        # Left panel — object selection
        left = tk.Frame(tab, bg=DARK_CARD, padx=15, pady=15)
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        tk.Label(left, text="Object Type", font=("Segoe UI", 12, "bold"),
                 bg=DARK_CARD, fg=DARK_FG).pack(anchor="w", pady=(0, 5))

        obj_menu = ttk.Combobox(left, textvariable=self.obj_type_var,
                                values=list(OBJECT_TYPES.keys()),
                                state="readonly", width=25)
        obj_menu.pack(fill="x", pady=(0, 15))

        # NACA controls
        self.naca_frame = tk.Frame(left, bg=DARK_CARD)
        self.naca_frame.pack(fill="x", pady=(0, 10))

        tk.Label(self.naca_frame, text="Profile Digits",
                 font=("Segoe UI", 10), bg=DARK_CARD,
                 fg=DARK_FG).pack(anchor="w")

        naca_entry = ttk.Entry(self.naca_frame, textvariable=self.naca_var,
                               width=15, font=("Courier", 12))
        naca_entry.pack(fill="x", pady=(3, 5))

        tk.Label(self.naca_frame, text="e.g. 0012, 23012, 63012",
                 font=("Segoe UI", 8), bg=DARK_CARD,
                 fg="#7f8fa6").pack(anchor="w")

        # NACA common presets
        self.presets_frame = tk.Frame(self.naca_frame, bg=DARK_CARD)
        self.presets_frame.pack(fill="x", pady=(5, 0))
        for preset in ["0006", "0012", "0015", "0018", "0021", "0025", "2412", "4412"]:
            btn = tk.Button(self.presets_frame, text=preset,
                           bg=DARK_INPUT, fg=DARK_FG,
                           relief="flat", padx=6, pady=1,
                           font=("Courier", 8),
                           command=lambda p=preset: self.naca_var.set(p))
            btn.pack(side="left", padx=1, pady=1)

        # AoA slider
        tk.Label(left, text="Angle of Attack",
                 font=("Segoe UI", 10), bg=DARK_CARD,
                 fg=DARK_FG).pack(anchor="w", pady=(10, 0))
        aoa_frame = tk.Frame(left, bg=DARK_CARD)
        aoa_frame.pack(fill="x")
        self.aoa_var = tk.DoubleVar(value=5.0)
        aoa_slider = ttk.Scale(aoa_frame, from_=-15, to=15,
                               variable=self.aoa_var, orient="horizontal",
                               command=lambda _: self._update_preview())
        aoa_slider.pack(side="left", fill="x", expand=True)
        self.aoa_label = tk.Label(aoa_frame, text="5.0°", width=5,
                                  bg=DARK_CARD, fg=DARK_FG)
        self.aoa_label.pack(side="right", padx=5)
        self.aoa_var.trace_add("write", lambda *a: self.aoa_label.configure(
            text=f"{self.aoa_var.get():.1f}°"))

        # Re & chord
        prop_frame = tk.Frame(left, bg=DARK_CARD)
        prop_frame.pack(fill="x", pady=(10, 0))
        tk.Label(prop_frame, text="Re", bg=DARK_CARD, fg=DARK_FG,
                 font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w")
        self.re_var = tk.StringVar(value="5000")
        ttk.Entry(prop_frame, textvariable=self.re_var, width=10).grid(
            row=0, column=1, padx=5)
        tk.Label(prop_frame, text="Chord", bg=DARK_CARD, fg=DARK_FG,
                 font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w")
        self.chord_var = tk.StringVar(value="60")
        ttk.Entry(prop_frame, textvariable=self.chord_var, width=10).grid(
            row=1, column=1, padx=5)

        # Grid size
        tk.Label(left, text="Grid Size", font=("Segoe UI", 10),
                 bg=DARK_CARD, fg=DARK_FG).pack(anchor="w", pady=(10, 0))
        self.grid_var = tk.StringVar(value="192")
        ttk.Combobox(left, textvariable=self.grid_var,
                     values=["96", "128", "192", "256"],
                     state="readonly", width=10).pack(anchor="w")

        # .STL path
        self.stl_frame = tk.Frame(left, bg=DARK_CARD)
        self.stl_path_var = tk.StringVar(value="")
        tk.Label(self.stl_frame, text="STL File", bg=DARK_CARD, fg=DARK_FG,
                 font=("Segoe UI", 10)).pack(anchor="w")
        stl_row = tk.Frame(self.stl_frame, bg=DARK_CARD)
        stl_row.pack(fill="x")
        ttk.Entry(stl_row, textvariable=self.stl_path_var, width=20).pack(
            side="left", fill="x", expand=True)
        ttk.Button(stl_row, text="Browse", command=self._browse_stl,
                   style="TButton", width=8).pack(side="right", padx=3)
        self.stl_frame.pack_forget()

        # Description
        self.desc_label = tk.Label(left, text="", font=("Segoe UI", 9),
                                   bg=DARK_CARD, fg="#7f8fa6", wraplength=250,
                                   justify="left")
        self.desc_label.pack(anchor="w", pady=(5, 0))

        # Right panel — preview
        right = tk.Frame(tab, bg=DARK_BG)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        preview_frame = tk.Frame(right, bg=DARK_CARD)
        preview_frame.pack(fill="both", expand=True)

        tk.Label(preview_frame, text="Airfoil Preview",
                 font=("Segoe UI", 12, "bold"), bg=DARK_CARD,
                 fg=DARK_FG).pack(anchor="nw", padx=10, pady=10)

        self.preview = AirfoilPreview(preview_frame, width=400, height=350)
        self.preview.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # 3D object info
        info_frame = tk.Frame(preview_frame, bg=DARK_CARD)
        info_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.info_text = tk.Text(info_frame, height=5, bg=DARK_CARD,
                                 fg=DARK_FG, relief="flat",
                                 font=("Segoe UI", 9), wrap="word")
        self.info_text.pack(fill="x")
        self.info_text.insert("1.0", "Select an object to see details.")

        # Next button at bottom right
        btn_frame = tk.Frame(tab, bg=DARK_BG)
        btn_frame.grid(row=1, column=0, columnspan=2, sticky="e", padx=10, pady=10)
        ttk.Button(btn_frame, text="Next →  Emitter Placement",
                   style="Accent.TButton",
                   command=self._go_emitter).pack(side="right")

    def _build_tab_emitter(self):
        tab = self.tab_emit
        tk.Label(tab, text="Click 'Open 3D Viewport' to place fluid emitters",
                 font=("Segoe UI", 14), bg=DARK_BG,
                 fg=DARK_FG).pack(pady=30)

        info = tk.Frame(tab, bg=DARK_CARD, padx=20, pady=20)
        info.pack(pady=10)
        tk.Label(info, text="Instructions:",
                 font=("Segoe UI", 11, "bold"), bg=DARK_CARD,
                 fg=DARK_FG).pack(anchor="w")
        for txt in [
            "+ Click to add fluid emitter on the inflow plane",
            "  Drag emitters to reposition (constrained to X=0 plane)",
            "  Each emitter injects fluid at 1/10th domain face size",
            "  Close the viewport window when done",
        ]:
            tk.Label(info, text=txt, font=("Segoe UI", 9), bg=DARK_CARD,
                     fg="#a0cfff").pack(anchor="w")

        btn_frame = tk.Frame(tab, bg=DARK_BG)
        btn_frame.pack(pady=20)
        ttk.Button(btn_frame, text="Open 3D Viewport",
                   style="Accent.TButton",
                   command=self._open_viewport).pack()

        self.emitter_count_label = tk.Label(
            tab, text=f"Emitters: {len(self.state['emitters'])}",
            font=("Segoe UI", 10), bg=DARK_BG, fg=DARK_FG)
        self.emitter_count_label.pack(pady=5)

        nav_frame = tk.Frame(tab, bg=DARK_BG)
        nav_frame.pack(side="bottom", fill="x", padx=10, pady=10)
        ttk.Button(nav_frame, text="← Back", style="TButton",
                   command=lambda: self.notebook.select(0)).pack(side="left")
        ttk.Button(nav_frame, text="Next →  Fluid Properties",
                   style="Accent.TButton",
                   command=lambda: self.notebook.select(2)).pack(side="right")

    def _build_tab_fluid(self):
        tab = self.tab_fluid
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)

        left = tk.Frame(tab, bg=DARK_CARD, padx=15, pady=15)
        left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        right = tk.Frame(tab, bg=DARK_CARD, padx=15, pady=15)
        right.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Left: Mode selection
        tk.Label(left, text="Flow Mode", font=("Segoe UI", 12, "bold"),
                 bg=DARK_CARD, fg=DARK_FG).pack(anchor="w", pady=(0, 10))
        self.mode_var = tk.StringVar(value="incompressible")
        for mode in ["incompressible", "compressible"]:
            ttk.Radiobutton(left, text=mode.capitalize(),
                           variable=self.mode_var, value=mode).pack(
                anchor="w", pady=2)

        tk.Label(left, text="", bg=DARK_CARD).pack(pady=5)

        # Mach
        tk.Label(left, text="Mach Number (compressible)",
                 font=("Segoe UI", 10), bg=DARK_CARD,
                 fg=DARK_FG).pack(anchor="w")
        self.mach_var = tk.StringVar(value="0.5")
        ttk.Entry(left, textvariable=self.mach_var, width=10).pack(anchor="w")

        # Velocity
        tk.Label(left, text="Flow Velocity (incompressible)",
                 font=("Segoe UI", 10), bg=DARK_CARD,
                 fg=DARK_FG).pack(anchor="w", pady=(5, 0))
        self.vel_var = tk.StringVar(value="2.0")
        ttk.Entry(left, textvariable=self.vel_var, width=10).pack(anchor="w")

        # Density
        tk.Label(left, text="Fluid Density",
                 font=("Segoe UI", 10), bg=DARK_CARD,
                 fg=DARK_FG).pack(anchor="w", pady=(5, 0))
        self.density_var = tk.StringVar(value="1.0")
        ttk.Entry(left, textvariable=self.density_var, width=10).pack(anchor="w")

        # Right: Solver settings
        tk.Label(right, text="Solver", font=("Segoe UI", 12, "bold"),
                 bg=DARK_CARD, fg=DARK_FG).pack(anchor="w", pady=(0, 10))
        self.solver_var = tk.StringVar(value="vcycle")
        for s in ["jacobi", "rbgs", "vcycle"]:
            ttk.Radiobutton(right, text=s.capitalize(),
                           variable=self.solver_var, value=s).pack(
                anchor="w", pady=2)

        tk.Label(right, text="", bg=DARK_CARD).pack(pady=5)
        tk.Label(right, text="Engine", font=("Segoe UI", 12, "bold"),
                 bg=DARK_CARD, fg=DARK_FG).pack(anchor="w", pady=(0, 10))
        self.engine_var = tk.StringVar(value="opengl")
        for eng in ["opengl", "vulkan"]:
            ttk.Radiobutton(right, text=eng.capitalize(),
                           variable=self.engine_var, value=eng).pack(
                anchor="w", pady=2)

        # Simulation parameters
        params = tk.Frame(right, bg=DARK_CARD)
        params.pack(fill="x", pady=10)

        tk.Label(params, text="Simulation Steps",
                 font=("Segoe UI", 10), bg=DARK_CARD,
                 fg=DARK_FG).grid(row=0, column=0, sticky="w")
        self.steps_var = tk.StringVar(value="500")
        ttk.Entry(params, textvariable=self.steps_var, width=10).grid(
            row=0, column=1, padx=5)

        tk.Label(params, text="Save Every N Steps",
                 font=("Segoe UI", 10), bg=DARK_CARD,
                 fg=DARK_FG).grid(row=1, column=0, sticky="w", pady=5)
        self.save_var = tk.StringVar(value="2")
        ttk.Entry(params, textvariable=self.save_var, width=10).grid(
            row=1, column=1, padx=5)

        # Viscosity override
        tk.Label(params, text="Viscosity (blank = auto from Re)",
                 font=("Segoe UI", 10), bg=DARK_CARD,
                 fg=DARK_FG).grid(row=2, column=0, sticky="w", pady=5)
        self.visc_var = tk.StringVar(value="")
        ttk.Entry(params, textvariable=self.visc_var, width=10).grid(
            row=2, column=1, padx=5)

        # Output directory
        tk.Label(params, text="Output Directory",
                 font=("Segoe UI", 10), bg=DARK_CARD,
                 fg=DARK_FG).grid(row=3, column=0, sticky="w", pady=5)
        self.outdir_var = tk.StringVar(value="/home/aaditya/Downloads/tmp")
        ttk.Entry(params, textvariable=self.outdir_var, width=25).grid(
            row=3, column=1, padx=5, columnspan=2)

        # Navigation
        nav = tk.Frame(tab, bg=DARK_BG)
        nav.grid(row=1, column=0, columnspan=2, sticky="e", padx=10, pady=10)
        ttk.Button(nav, text="← Back", style="TButton",
                   command=lambda: self.notebook.select(1)).pack(side="left")
        ttk.Button(nav, text="Next →  Run",
                   style="Accent.TButton",
                   command=lambda: self.notebook.select(3)).pack(side="right", padx=5)

    def _build_tab_run(self):
        tab = self.tab_run

        summary_frame = tk.Frame(tab, bg=DARK_CARD, padx=20, pady=20)
        summary_frame.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(summary_frame, text="Simulation Summary",
                 font=("Segoe UI", 14, "bold"), bg=DARK_CARD,
                 fg=DARK_FG).pack(anchor="w", pady=(0, 10))

        self.summary_text = tk.Text(summary_frame, height=15, bg=DARK_CARD,
                                    fg=DARK_FG, relief="flat",
                                    font=("Courier", 10), wrap="word")
        self.summary_text.pack(fill="both", expand=True)

        # Progress
        self.progress = ttk.Progressbar(tab, orient="horizontal",
                                        length=0, mode="determinate",
                                        style="Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=10, pady=(0, 5))
        self.progress_label = tk.Label(tab, text="", bg=DARK_BG, fg=DARK_FG,
                                       font=("Segoe UI", 9))
        self.progress_label.pack(pady=(0, 5))

        buttons = tk.Frame(tab, bg=DARK_BG)
        buttons.pack(fill="x", padx=10, pady=10)

        ttk.Button(buttons, text="← Back", style="TButton",
                   command=lambda: self.notebook.select(2)).pack(side="left")
        self.sim_btn = ttk.Button(buttons, text="▶  SIMULATE",
                                  style="Accent.TButton",
                                  command=self._run_simulation)
        self.sim_btn.pack(side="right", padx=5)
        self.viz_btn = ttk.Button(buttons, text="Visualize Results",
                                  style="TButton",
                                  command=self._launch_visualizer)
        self.viz_btn.pack(side="right", padx=5)

    # ── Callbacks ──────────────────────────────────────────────────────
    def _on_obj_type_change(self, *args):
        otype = self.obj_type_var.get()
        is_naca = otype.startswith("NACA")
        if is_naca:
            self.naca_frame.pack(fill="x", pady=(0, 10))
            self.stl_frame.pack_forget()
            if "4-Digit" in otype:
                self.naca_var.set("0012")
            elif "5-Digit" in otype:
                self.naca_var.set("23012")
            else:
                self.naca_var.set("63012")
        elif otype == "Custom .STL":
            self.naca_frame.pack_forget()
            self.stl_frame.pack(fill="x", pady=(0, 10))
        else:
            self.naca_frame.pack_forget()
            self.stl_frame.pack_forget()
        self.state["object_type"] = otype
        self._update_preview()

    def _on_naca_change(self, *args):
        self._update_preview()

    def _browse_stl(self):
        path = filedialog.askopenfilename(
            title="Select STL/OBJ File",
            filetypes=[("Mesh files", "*.stl *.obj"), ("All files", "*.*")])
        if path:
            self.stl_path_var.set(path)
            self.state["stl_path"] = path

    def _update_preview(self, *args):
        otype = self.obj_type_var.get()
        digits = self.naca_var.get().strip()
        aoa = self.aoa_var.get()

        info_lines = []
        info_lines.append(f"Object: {otype}")
        is_naca = otype.startswith("NACA")

        if is_naca and len(digits) >= 4:
            try:
                x, yu, yl = generate_naca(digits)
                self.preview.draw_airfoil(x, yu, yl, aoa)
                desc = naca_description(digits)
                info_lines.append(desc)
                camber = "Symmetric" if digits.startswith("00") else "Cambered"
                info_lines.append(f"Type: {camber}")
                info_lines.append(f"AoA: {aoa:.1f}°")
                t = int(digits[-2:]) / 100.0
                info_lines.append(f"Max Thickness: {t*100:.1f}% at 30% chord")
            except Exception as e:
                self.preview.delete("all")
                info_lines.append(f"Invalid profile: {e}")
        elif not is_naca:
            self.preview.delete("all")
            info_lines.append(OBJECT_TYPES[otype]["description"](None))
            if otype == "Cylinder":
                info_lines.append("Radius: 15 cells, full span along Y")
            elif otype == "Sphere":
                info_lines.append("Radius: 25 cells, centered")
            elif otype == "Custom .STL":
                path = self.stl_path_var.get() or "(not selected)"
                info_lines.append(f"STL: {path}")

        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", "\n".join(info_lines))

    def _go_emitter(self):
        # Save state from form
        self.state["naca_digits"] = self.naca_var.get()
        self.state["aoa"] = self.aoa_var.get()
        self.state["object_type"] = self.obj_type_var.get()
        try:
            self.state["reynolds"] = int(self.re_var.get())
            self.state["chord"] = int(self.chord_var.get())
            self.state["grid_size"] = int(self.grid_var.get())
        except ValueError:
            pass
        self.notebook.select(1)

    def _open_viewport(self):
        """Open the 3D emitter viewport in a new window."""
        try:
            from launcher_3d_viewport import open_viewport

            win = tk.Toplevel(self.root)
            win.title("CFD Emitter Placement")
            win.geometry("900x700")
            win.configure(bg=DARK_BG)
            win.minsize(700, 500)

            def on_done(emitters):
                self.state["emitters"] = emitters
                self.emitter_count_label.config(
                    text=f"Emitters: {len(emitters)}")
                win.destroy()

            open_viewport(
                win,
                grid_size=self.state["grid_size"],
                initial_emitters=self.state["emitters"],
                obstacle_type=self.state["object_type"],
                naca_digits=self.state["naca_digits"],
                callback=on_done,
            )
        except Exception as e:
            print(f"Viewport error: {e}")
            import traceback
            traceback.print_exc()

    def _run_simulation(self):
        self.state["mode"] = self.mode_var.get()
        self.state["mach"] = float(self.mach_var.get())
        self.state["velocity"] = float(self.vel_var.get())
        self.state["density"] = float(self.density_var.get())
        self.state["steps"] = int(self.steps_var.get())
        self.state["save_every"] = int(self.save_var.get())
        self.state["solver"] = self.solver_var.get()
        self.state["engine"] = self.engine_var.get()
        self.state["outdir"] = self.outdir_var.get()
        if self.visc_var.get().strip():
            self.state["viscosity"] = float(self.visc_var.get())
        else:
            self.state["viscosity"] = None

        # Build summary
        summary = json.dumps(self.state, indent=2)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", summary)

        self.sim_btn.config(state="disabled", text="Running...")
        self.progress["value"] = 0
        self.progress["maximum"] = self.state["steps"]

        # Run in thread
        thread = threading.Thread(target=self._sim_thread, daemon=True)
        thread.start()

    def _sim_thread(self):
        s = self.state
        cmd = [
            sys.executable, "simulate_cfd.py",
            "--mode", s["mode"],
            "--engine", s.get("engine", "opengl"),
            "--steps", str(s["steps"]),
            "--save-every", str(s["save_every"]),
            "--out-dir", s.get("outdir", "/home/aaditya/Downloads/tmp"),
        ]
        if s["mode"] == "compressible":
            cmd += ["--mach", str(s["mach"])]
        if s["mode"] == "incompressible":
            cmd += ["--wind", str(s["velocity"])]
        if s["viscosity"] is not None:
            cmd += ["--viscosity", str(s["viscosity"])]
        else:
            cmd += ["--re", str(s["reynolds"]), "--chord", str(s["chord"])]

        otype = s["object_type"]
        if "NACA" in otype:
            cmd += ["--obstacle", "wing"]
        elif otype == "Cylinder":
            cmd += ["--obstacle", "cylinder"]
        elif otype == "Sphere":
            cmd += ["--obstacle", "sphere"]
        elif otype == "Custom .STL" and s.get("stl_path"):
            cmd += ["--cad", s["stl_path"], "--cad-chord", str(s["chord"]),
                    "--cad-aoa", str(s["aoa"])]

        if s["solver"] == "vcycle":
            cmd += ["--vcycle", "--vcycle-interval", "1"]
        elif s["solver"] == "rbgs":
            cmd += ["--jacobi-iters", "80"]

        cmd += ["--no-forces"]

        print(f"Running: {' '.join(cmd)}")

        env = {**os.environ,
               "__NV_PRIME_RENDER_OFFLOAD": "1",
               "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
               "CFD_DATA_DIR": s.get("outdir", "/home/aaditya/Downloads/tmp"),
               "CFD_LIVE": "1"} if os.path.exists("/usr/bin/nvidia-smi") else None

        # Launch visualizer in parallel (unless Vulkan — no Vulkan viz yet)
        viz_proc = None
        if s.get("engine", "opengl") == "opengl":
            viz_cmd = [sys.executable, "visualize_cfd.py"]
            viz_proc = subprocess.Popen(
                viz_cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env,
            )

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env,
            )

            for line in proc.stdout:
                line = line.strip()
                if line:
                    if "step=" in line:
                        try:
                            parts = line.split()
                            step_part = [p for p in parts if p.startswith("step=")][0]
                            step_num = int(step_part.split("=")[1].split("/")[0])
                            self.root.after(0, lambda v=step_num: self.progress.configure(value=v))
                        except:
                            pass
                    self.root.after(0, lambda t=line: self._append_log(t))

            proc.wait()
            self.root.after(0, self._sim_done)
        except Exception as e:
            self.root.after(0, lambda: self._append_log(f"Error: {e}"))
            self.root.after(0, self._sim_done)
        finally:
            if viz_proc is not None:
                viz_proc.terminate()

    def _append_log(self, text):
        self.summary_text.insert(tk.END, f"\n{text}")
        self.summary_text.see(tk.END)

    def _sim_done(self):
        self.progress["value"] = self.progress["maximum"]
        self.sim_btn.config(state="normal", text="▶  SIMULATE")
        self._append_log("\n--- Simulation Complete ---")
        self.viz_btn.config(state="normal")

    def _launch_visualizer(self):
        cmd = [
            sys.executable, "visualize_cfd.py",
        ]
        env = {**os.environ, "CFD_DATA_DIR": self.state.get("outdir", "/home/aaditya/Downloads/tmp")}
        if os.path.exists("/usr/bin/nvidia-smi"):
            env.update({"__NV_PRIME_RENDER_OFFLOAD": "1",
                        "__GLX_VENDOR_LIBRARY_NAME": "nvidia"})
        subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), env=env)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = CFDLauncher()
    app.run()
