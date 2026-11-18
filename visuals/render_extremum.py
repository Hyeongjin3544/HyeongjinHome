"""
Manim 3D animations for Econometrics Visual Supplement 00 (Extremum Estimation):
  - ExtremumUlln3DScene        -> assets/media/extremum-ulln-consistency.mp4
  - ExtremumAsymptotic3DScene  -> assets/media/extremum-asymptotic-normality.mp4

Usage:
  .venv\\Scripts\\python.exe visuals\\render_extremum.py [--quality low|high]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    DR,
    LEFT,
    ORIGIN,
    PI,
    RIGHT,
    UL,
    UP,
    Create,
    DashedLine,
    Dot3D,
    FadeIn,
    FadeOut,
    Line,
    MathTex,
    ParametricFunction,
    Polygon,
    Text,
    ThreeDScene,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    smooth,
)

BG_HEX = "#0c0f14"
GRID_HEX = "#2b3547"
CYAN_HEX = "#38bdf8"
TEAL_HEX = "#2dd4bf"
INDIGO_HEX = "#6366f1"
INDIGO_LIGHT_HEX = "#818cf8"
AMBER_HEX = "#fbbf24"
ROSE_HEX = "#fb7185"
INK_HEX = "#f4f4f5"
MUTED_HEX = "#a1a1aa"

XY_SCALE = 1.20
Z_FLOOR = -1.24
Z_SCALE = 0.84


def pt3(t1: float, t2: float, q_val: float) -> np.ndarray:
    """Map parameter coordinates (t1, t2) and objective height q_val into 3D scene coords."""
    return np.array([t1 * XY_SCALE, t2 * XY_SCALE, Z_FLOOR + q_val * Z_SCALE], dtype=float)


def pill_tex(tex_str: str, pos: np.ndarray, color: str, font_size: int = 24) -> MathTex:
    lbl = MathTex(tex_str, font_size=font_size, color=color).move_to(pos)
    lbl.add_background_rectangle(color=BG_HEX, opacity=0.78, buff=0.05)
    return lbl


def make_hud_3d(scene: ThreeDScene, title_str: str, math_str: str, sub_str: str):
    title = Text(title_str, font="Segoe UI", font_size=22, weight="BOLD", color=INK_HEX)
    title_math = MathTex(math_str, font_size=23, color=INDIGO_LIGHT_HEX)
    title_row = VGroup(title, title_math).arrange(RIGHT, buff=0.16)
    title_row.to_corner(UL, buff=0.42)

    sub = MathTex(sub_str, font_size=20, color=MUTED_HEX)
    sub.next_to(title_row, DOWN, buff=0.14, aligned_edge=LEFT)
    scene.add_fixed_in_frame_mobjects(title_row, sub)
    return title_row, sub


def swap_sub_3d(
    scene: ThreeDScene,
    title_row: VGroup,
    old_sub: MathTex,
    new_tex: str,
    font_size: int = 20,
    color: str = MUTED_HEX,
) -> MathTex:
    scene.play(FadeOut(old_sub, shift=UP * 0.08, run_time=0.22, rate_func=smooth))
    scene.remove_fixed_in_frame_mobjects(old_sub)
    new_sub = MathTex(new_tex, font_size=font_size, color=color)
    new_sub.next_to(title_row, DOWN, buff=0.14, aligned_edge=LEFT)
    scene.add_fixed_in_frame_mobjects(new_sub)
    scene.play(FadeIn(new_sub, shift=UP * 0.08, run_time=0.26, rate_func=smooth))
    return new_sub


def make_dashed_3d(p0: np.ndarray, p1: np.ndarray, color: str, num_dashes: int = 8, stroke_width: float = 2.4) -> DashedLine:
    length = max(float(np.linalg.norm(p1 - p0)), 1e-4)
    return DashedLine(
        start=p0,
        end=p1,
        dash_length=length / (2.0 * num_dashes),
        dashed_ratio=0.55,
        color=color,
        stroke_width=stroke_width,
    )


def make_floor_and_axes(scene: ThreeDScene, r_max: float = 1.60, z_top: float = 2.45):
    """Build parameter space floor (z = 0 in objective units) + subtle coordinate frame."""
    c00 = pt3(-r_max, -r_max, 0.0)
    c10 = pt3(r_max, -r_max, 0.0)
    c11 = pt3(r_max, r_max, 0.0)
    c01 = pt3(-r_max, r_max, 0.0)
    floor = Polygon(
        c00,
        c10,
        c11,
        c01,
        fill_color="#111827",
        fill_opacity=0.65,
        stroke_color=GRID_HEX,
        stroke_width=1.4,
    )

    grid = VGroup()
    for g in [-1.2, -0.6, 0.0, 0.6, 1.2]:
        sw = 1.5 if abs(g) < 1e-6 else 0.9
        col = "#3f4c63" if abs(g) < 1e-6 else GRID_HEX
        grid.add(Line(pt3(-r_max, g, 0.0), pt3(r_max, g, 0.0), color=col, stroke_width=sw))
        grid.add(Line(pt3(g, -r_max, 0.0), pt3(g, r_max, 0.0), color=col, stroke_width=sw))

    z_axis = Line(pt3(-r_max, r_max, 0.0), pt3(-r_max, r_max, z_top), color="#475569", stroke_width=1.8)

    lbl_t1 = pill_tex(r"\theta_1", pt3(r_max + 0.22, -0.85, 0.03), INK_HEX, font_size=24)
    lbl_t2 = pill_tex(r"\theta_2", pt3(-1.35, -r_max - 0.20, 0.03), INK_HEX, font_size=24)
    lbl_q = pill_tex(r"Q(\theta)", pt3(-r_max, r_max, z_top + 0.22), INK_HEX, font_size=24)
    scene.add_fixed_orientation_mobjects(lbl_t1, lbl_t2, lbl_q)
    return VGroup(floor, grid, z_axis), VGroup(lbl_t1, lbl_t2, lbl_q)


def make_surface_mesh(
    fn: Callable[[float, float], float],
    r_max: float = 1.55,
    n_grid: int = 12,
    fill_color: str = INDIGO_HEX,
    fill_opacity: float = 0.20,
    stroke_color: str = INDIGO_LIGHT_HEX,
    stroke_width: float = 1.35,
) -> VGroup:
    """Fast translucent quad-mesh surface for crisp 60fps 3D animation."""
    xs = np.linspace(-r_max, r_max, n_grid + 1)
    ys = np.linspace(-r_max, r_max, n_grid + 1)
    pts = [[pt3(float(x), float(y), float(fn(float(x), float(y)))) for y in ys] for x in xs]

    quads = VGroup()
    for i in range(n_grid):
        for j in range(n_grid):
            q = Polygon(
                pts[i][j],
                pts[i + 1][j],
                pts[i + 1][j + 1],
                pts[i][j + 1],
                fill_color=fill_color,
                fill_opacity=fill_opacity,
                stroke_color=stroke_color,
                stroke_opacity=0.62,
                stroke_width=stroke_width,
            )
            quads.add(q)
    return quads


def make_wire_grid(
    fn: Callable[[float, float], float],
    r_max: float = 1.55,
    n_lines: int = 9,
    n_samples: int = 20,
    color: str = CYAN_HEX,
    stroke_width: float = 1.8,
) -> VGroup:
    """Clean wireframe surface using piecewise polylines (fast for always_redraw)."""
    coords = np.linspace(-r_max, r_max, n_lines)
    fine = np.linspace(-r_max, r_max, n_samples)
    wires = VGroup()
    for u in coords:
        curve1 = VMobject(color=color, stroke_width=stroke_width, stroke_opacity=0.82)
        curve1.set_points_as_corners([pt3(float(u), float(v), float(fn(float(u), float(v)))) for v in fine])
        curve2 = VMobject(color=color, stroke_width=stroke_width, stroke_opacity=0.82)
        curve2.set_points_as_corners([pt3(float(v), float(u), float(fn(float(v), float(u)))) for v in fine])
        wires.add(curve1, curve2)
    return wires


class ExtremumUlln3DScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_HEX
        self.set_camera_orientation(phi=64 * DEGREES, theta=-48 * DEGREES, zoom=0.96)

        title_row, sub_tex = make_hud_3d(
            self,
            "Extremum Estimator (3D):",
            r"\hat{\theta}_n = \arg\min_{\theta \in \Theta \subset \mathbb{R}^2} \hat{Q}_n(\theta)",
            r"\text{Identification: Population bowl } Q(\theta_1,\theta_2) \text{ has a unique global minimum at } \theta_0",
        )

        frame_grp, axis_lbls = make_floor_and_axes(self, r_max=1.60, z_top=2.55)

        t0_1, t0_2 = 0.0, 0.0

        def q_pop(t1: float, t2: float) -> float:
            # Anisotropic 3D convex bowl hovering at z_min = 0.72 above floor for clear drop-line separation
            u = 0.86 * (t1 - t0_1) - 0.50 * (t2 - t0_2)
            v = 0.50 * (t1 - t0_1) + 0.86 * (t2 - t0_2)
            return 0.72 + 0.48 * u * u + 0.22 * v * v

        pop_mesh = make_surface_mesh(
            q_pop,
            r_max=1.50,
            n_grid=12,
            fill_color=INDIGO_HEX,
            fill_opacity=0.22,
            stroke_color=INDIGO_LIGHT_HEX,
            stroke_width=1.15,
        )

        p_q0 = pt3(t0_1, t0_2, q_pop(t0_1, t0_2))
        p_t0 = pt3(t0_1, t0_2, 0.0)
        dot_q0 = Dot3D(p_q0, radius=0.075, color=AMBER_HEX, resolution=(8, 8))
        dot_t0 = Dot3D(p_t0, radius=0.075, color=AMBER_HEX, resolution=(8, 8))
        drop_t0 = make_dashed_3d(p_q0, p_t0, color=AMBER_HEX, num_dashes=8, stroke_width=2.4)
        lbl_t0 = pill_tex(r"\theta_0", p_t0 + np.array([0.24, -0.26, 0.04]), AMBER_HEX, font_size=25)
        lbl_q = pill_tex(r"Q(\theta)", pt3(1.42, 1.42, q_pop(1.42, 1.42) + 0.24), INDIGO_LIGHT_HEX, font_size=24)
        self.add_fixed_orientation_mobjects(lbl_t0, lbl_q)

        self.play(FadeIn(frame_grp), FadeIn(axis_lbls), run_time=0.6)
        self.play(FadeIn(pop_mesh), FadeIn(lbl_q), run_time=1.0, rate_func=smooth)
        self.play(Create(drop_t0), FadeIn(dot_q0), FadeIn(dot_t0), FadeIn(lbl_t0), run_time=0.6)
        self.wait(0.35)

        # Phase 2: Sample objective surface Q_hat_n(theta_1, theta_2) converging uniformly (ULLN)
        sub_tex = swap_sub_3d(
            self,
            title_row,
            sub_tex,
            r"\text{ULLN: } \sup_{\theta \in \Theta} |\hat{Q}_n(\theta) - Q(\theta)| \xrightarrow{p} 0 \;\Rightarrow\; \hat{\theta}_n \xrightarrow{p} \theta_0",
            font_size=20,
            color=INK_HEX,
        )

        def q_samp(t1: float, t2: float, amp: float) -> float:
            noise = (
                -0.55 * t1
                + 0.48 * t2
                + 0.22 * np.cos(2.3 * t1 - 0.6)
                + 0.18 * np.sin(2.5 * t2 + 0.4)
                + 0.12 * np.sin(2.0 * (t1 + t2))
            )
            return max(0.08, q_pop(t1, t2) + amp * noise)

        # Pre-compute argmin over dense 2D grid for smooth always_redraw tracking
        g1d = np.linspace(-1.40, 1.40, 45)
        G1, G2 = np.meshgrid(g1d, g1d, indexing="ij")

        def find_argmin_2d(amp: float) -> tuple[float, float, float]:
            vals = q_pop(G1, G2) + amp * (
                -0.55 * G1
                + 0.48 * G2
                + 0.22 * np.cos(2.3 * G1 - 0.6)
                + 0.18 * np.sin(2.5 * G2 + 0.4)
                + 0.12 * np.sin(2.0 * (G1 + G2))
            )
            idx = np.unravel_index(int(np.argmin(vals)), vals.shape)
            a1 = float(G1[idx])
            a2 = float(G2[idx])
            return a1, a2, float(q_samp(a1, a2, amp))

        amp_trk = ValueTracker(0.90)

        samp_wires = always_redraw(
            lambda: make_wire_grid(
                lambda x, y: q_samp(x, y, amp_trk.get_value()),
                r_max=1.52,
                n_lines=9,
                n_samples=19,
                color=CYAN_HEX,
                stroke_width=1.8,
            )
        )

        dot_qn = always_redraw(
            lambda: (
                lambda a1, a2, qv: Dot3D(pt3(a1, a2, qv), radius=0.075, color=CYAN_HEX, resolution=(8, 8))
            )(*find_argmin_2d(amp_trk.get_value()))
        )
        dot_tn = always_redraw(
            lambda: (
                lambda a1, a2, qv: Dot3D(pt3(a1, a2, 0.0), radius=0.075, color=CYAN_HEX, resolution=(8, 8))
            )(*find_argmin_2d(amp_trk.get_value()))
        )
        drop_tn = always_redraw(
            lambda: (
                lambda a1, a2, qv: make_dashed_3d(pt3(a1, a2, qv), pt3(a1, a2, 0.0), color=CYAN_HEX, num_dashes=7, stroke_width=2.3)
            )(*find_argmin_2d(amp_trk.get_value()))
        )

        n_badge = Text("n = 22", font="Consolas", font_size=24, color=TEAL_HEX).to_corner(DR, buff=0.50)
        n_badge._last_n = 22

        def update_n_badge(mob: Text):
            val = int(round(18.0 / max(amp_trk.get_value() ** 2, 0.036)))
            if getattr(mob, "_last_n", None) != val:
                mob._last_n = val
                fresh = Text(f"n = {val}", font="Consolas", font_size=24, color=TEAL_HEX).to_corner(DR, buff=0.50)
                mob.become(fresh)
                self.add_fixed_in_frame_mobjects(mob)

        n_badge.add_updater(update_n_badge)
        self.add_fixed_in_frame_mobjects(n_badge)

        a1_0, a2_0, qv_0 = find_argmin_2d(0.90)
        lbl_tn = pill_tex(r"\hat{\theta}_n", pt3(a1_0 - 0.26, a2_0 + 0.22, 0.05), CYAN_HEX, font_size=25)
        lbl_qn = pill_tex(r"\hat{Q}_n(\theta)", pt3(-1.45, -1.42, q_samp(-1.45, -1.42, 0.90) + 0.28), CYAN_HEX, font_size=24)
        self.add_fixed_orientation_mobjects(lbl_tn, lbl_qn)

        self.play(FadeIn(samp_wires), FadeIn(lbl_qn), FadeIn(drop_tn), FadeIn(dot_qn), FadeIn(dot_tn), FadeIn(lbl_tn), FadeIn(n_badge), run_time=0.8)
        self.wait(0.35)

        # Gentle 3D orbit while ULLN collapses sample surface onto population bowl
        self.begin_ambient_camera_rotation(rate=0.14)
        self.play(
            FadeOut(lbl_qn, run_time=0.4),
            FadeOut(lbl_tn, run_time=0.4),
            amp_trk.animate.set_value(0.05),
            run_time=3.8,
            rate_func=smooth,
        )
        self.stop_ambient_camera_rotation()
        self.wait(1.1)


class ExtremumAsymptotic3DScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_HEX
        self.set_camera_orientation(phi=62 * DEGREES, theta=-50 * DEGREES, zoom=0.96)

        title_row, sub_tex = make_hud_3d(
            self,
            "Asymptotic Normality (3D):",
            r"\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}(0, \, H^{-1}\Omega H^{-1})",
            r"\text{Anisotropic bowl: steep curvature } H_{11} \text{ vs flat valley } H_{22}",
        )

        frame_grp, axis_lbls = make_floor_and_axes(self, r_max=1.60, z_top=2.55)

        # Hessian diagonal in canonical coords: steep along theta_1 (h1 = 2.60), flat valley along theta_2 (h2 = 0.36)
        h1, h2 = 2.60, 0.36
        q_base_level = 0.68

        def q_aniso(t1: float, t2: float) -> float:
            return q_base_level + 0.5 * (h1 * t1 * t1 + h2 * t2 * t2) * 0.52

        # Population surface mesh + principal curvature slice curves (teal steep vs rose flat)
        aniso_mesh = make_surface_mesh(
            q_aniso,
            r_max=1.48,
            n_grid=12,
            fill_color=INDIGO_HEX,
            fill_opacity=0.20,
            stroke_color=GRID_HEX,
            stroke_width=1.05,
        )

        steep_slice = ParametricFunction(
            lambda t: pt3(t, 0.0, q_aniso(t, 0.0)),
            t_range=[-1.45, 1.45],
            color=TEAL_HEX,
            stroke_width=4.2,
        )
        flat_slice = ParametricFunction(
            lambda t: pt3(0.0, t, q_aniso(0.0, t)),
            t_range=[-1.45, 1.45],
            color=ROSE_HEX,
            stroke_width=4.2,
        )

        lbl_steep = pill_tex(r"\text{Steep } H_{11}", pt3(-1.46, 0.0, q_aniso(-1.46, 0.0) + 0.28), TEAL_HEX, font_size=22)
        lbl_flat = pill_tex(r"\text{Flat } H_{22}", pt3(0.0, 1.48, q_aniso(0.0, 1.48) + 0.25), ROSE_HEX, font_size=22)
        dot_t0 = Dot3D(pt3(0.0, 0.0, 0.0), radius=0.075, color=AMBER_HEX, resolution=(8, 8))
        lbl_t0 = pill_tex(r"\theta_0", pt3(-0.24, -0.24, 0.05), AMBER_HEX, font_size=24)
        self.add_fixed_orientation_mobjects(lbl_steep, lbl_flat, lbl_t0)

        self.play(FadeIn(frame_grp), FadeIn(axis_lbls), run_time=0.55)
        self.play(FadeIn(aniso_mesh), Create(steep_slice), Create(flat_slice), FadeIn(dot_t0), FadeIn(lbl_t0), FadeIn(lbl_steep), FadeIn(lbl_flat), run_time=1.1, rate_func=smooth)
        self.wait(0.35)

        # Phase 2: Isotropic score noise g_n = r*(cos a, sin a) tilts the bowl -> argmin traces elongated ellipse!
        sub_tex = swap_sub_3d(
            self,
            title_row,
            sub_tex,
            r"\text{Isotropic score tilt } \nabla\hat{Q}_n(\theta_0) \text{ maps via } H^{-1} \text{ into an elongated confidence ellipse}",
            font_size=19,
            color=INK_HEX,
        )

        score_rad = 0.34
        alpha_trk = ValueTracker(0.0)

        # effective curvatures in Q_aniso units are ke1 = h1*0.52, ke2 = h2*0.52
        ke1 = h1 * 0.52
        ke2 = h2 * 0.52

        def tilted_q(t1: float, t2: float, alpha: float) -> float:
            g1 = score_rad * np.cos(alpha)
            g2 = score_rad * np.sin(alpha)
            return max(0.08, q_aniso(t1, t2) + g1 * t1 + g2 * t2)

        def argmin_tilted(alpha: float) -> tuple[float, float, float]:
            g1 = score_rad * np.cos(alpha)
            g2 = score_rad * np.sin(alpha)
            th1 = float(np.clip(-g1 / ke1, -1.45, 1.45))
            th2 = float(np.clip(-g2 / ke2, -1.45, 1.45))
            return th1, th2, float(tilted_q(th1, th2, alpha))

        tilted_wires = always_redraw(
            lambda: make_wire_grid(
                lambda x, y: tilted_q(x, y, alpha_trk.get_value()),
                r_max=1.45,
                n_lines=7,
                n_samples=15,
                color=CYAN_HEX,
                stroke_width=1.45,
            )
        )
        dot_min_surf = always_redraw(
            lambda: (
                lambda a1, a2, qv: Dot3D(pt3(a1, a2, qv), radius=0.075, color=CYAN_HEX, resolution=(8, 8))
            )(*argmin_tilted(alpha_trk.get_value()))
        )
        dot_min_floor = always_redraw(
            lambda: (
                lambda a1, a2, qv: Dot3D(pt3(a1, a2, 0.0), radius=0.08, color=CYAN_HEX, resolution=(8, 8))
            )(*argmin_tilted(alpha_trk.get_value()))
        )
        drop_min = always_redraw(
            lambda: (
                lambda a1, a2, qv: make_dashed_3d(pt3(a1, a2, qv), pt3(a1, a2, 0.0), color=CYAN_HEX, num_dashes=6, stroke_width=2.2)
            )(*argmin_tilted(alpha_trk.get_value()))
        )

        # Parametric trail of confidence ellipse traced on the floor as alpha sweeps 0 -> 2*pi
        ellipse_trail = always_redraw(
            lambda: ParametricFunction(
                lambda a: pt3(-score_rad * np.cos(a) / ke1, -score_rad * np.sin(a) / ke2, 0.015),
                t_range=[0.0, max(0.02, alpha_trk.get_value())],
                color=AMBER_HEX,
                stroke_width=3.8,
            )
        )
        # Reference isotropic score circle on floor around theta_0 (scaled for comparison)
        score_circle = ParametricFunction(
            lambda a: pt3(score_rad * np.cos(a), score_rad * np.sin(a), 0.012),
            t_range=[0.0, 2 * PI],
            color=MUTED_HEX,
            stroke_width=1.6,
            stroke_opacity=0.65,
        )

        self.play(FadeIn(tilted_wires), FadeIn(score_circle), FadeIn(drop_min), FadeIn(dot_min_surf), FadeIn(dot_min_floor), FadeIn(ellipse_trail), run_time=0.65)
        self.begin_ambient_camera_rotation(rate=0.14)
        self.play(alpha_trk.animate.set_value(2 * PI), run_time=4.2, rate_func=smooth)
        self.stop_ambient_camera_rotation()

        lbl_ellipse = pill_tex(r"\mathrm{Cov}(\hat{\theta}_n) \propto H^{-1}\Omega H^{-1}", pt3(1.05, -1.25, 0.08), AMBER_HEX, font_size=22)
        self.add_fixed_orientation_mobjects(lbl_ellipse)
        self.play(FadeIn(lbl_ellipse), run_time=0.45)
        self.wait(1.3)


def render_all(quality: str, which: str = "all") -> None:
    root_dir = Path(__file__).resolve().parent.parent
    out_dir = root_dir / "assets" / "media"
    candidate_media = root_dir / "visuals" / "media"
    media_dir = (
        candidate_media
        if str(candidate_media).isascii()
        else Path(tempfile.gettempdir()) / "manim_extremum"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    winget_links = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links"
    ffmpeg_pkg_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    extra_bins = [str(winget_links)]
    if ffmpeg_pkg_root.exists():
        for bin_dir in ffmpeg_pkg_root.glob("Gyan.FFmpeg*/**/bin"):
            extra_bins.append(str(bin_dir))
    env["PATH"] = os.pathsep.join(extra_bins) + os.pathsep + env.get("PATH", "")

    q_flag = "-ql" if quality == "low" else "-qh"
    fps_flag = ["--fps", "15"] if quality == "low" else ["--fps", "60"]

    manim_exe = str(Path(sys.executable).parent / "manim.exe")
    script_path = str(Path(__file__).resolve())

    all_scenes = [
        ("ulln", "ExtremumUlln3DScene", "extremum-ulln-consistency.mp4"),
        ("asym", "ExtremumAsymptotic3DScene", "extremum-asymptotic-normality.mp4"),
    ]
    scenes = [
        (scene_name, out_name)
        for key, scene_name, out_name in all_scenes
        if which in ("all", key)
    ]

    for scene_name, out_name in scenes:
        cmd = [
            manim_exe,
            "render",
            q_flag,
            *fps_flag,
            "--media_dir",
            str(media_dir),
            "--output_file",
            out_name,
            script_path,
            scene_name,
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, env=env, check=True)

        matches = list(media_dir.glob(f"videos/**/{out_name}"))
        if not matches:
            raise FileNotFoundError(f"Rendered file not found for {out_name} under {media_dir}")
        latest = max(matches, key=lambda p: p.stat().st_mtime)
        dest = out_dir / out_name
        shutil.copy2(latest, dest)
        print(f"Copied {latest} -> {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 3D Extremum Estimation Manim videos")
    parser.add_argument("--quality", choices=["low", "high"], default="high")
    parser.add_argument("--scene", choices=["all", "ulln", "asym"], default="all")
    args = parser.parse_args()
    render_all(args.quality, args.scene)
