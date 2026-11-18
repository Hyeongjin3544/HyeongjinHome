"""
Manim 3D animation for Econometrics Visual Supplement 02 (IV & 2SLS):
  - IV2SLSProjectionScene -> assets/media/iv-2sls-projection.mp4

Usage:
  .venv\\Scripts\\python.exe visuals\\render_iv_2sls.py [--quality low|high]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from manim import (
    DEGREES,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UL,
    UP,
    Create,
    DashedLine,
    Dot3D,
    FadeIn,
    FadeOut,
    GrowFromPoint,
    Line,
    MathTex,
    Polygon,
    Text,
    ThreeDScene,
    VGroup,
    VMobject,
    smooth,
)

BG_HEX = "#0c0f14"
GRID_HEX = "#384152"
CYAN_HEX = "#38bdf8"
TEAL_HEX = "#2dd4bf"
INDIGO_HEX = "#6366f1"
INDIGO_LIGHT_HEX = "#818cf8"
AMBER_HEX = "#fbbf24"
ROSE_HEX = "#fb7185"
INK_HEX = "#f4f4f5"
MUTED_HEX = "#a1a1aa"

SCALE = 1.42


def mvec(v: np.ndarray) -> np.ndarray:
    """Scale a 3D math vector into scene coordinates."""
    return np.asarray(v, dtype=float) * SCALE


def make_right_angle_3d(
    corner: np.ndarray,
    dir1: np.ndarray,
    dir2: np.ndarray,
    size: float = 0.24,
    color: str = INK_HEX,
    stroke_width: float = 2.4,
) -> VMobject:
    """Create a 3D right-angle square marker at `corner` spanned by `dir1`, `dir2`."""
    d1 = np.asarray(dir1, dtype=float)
    d2 = np.asarray(dir2, dtype=float)
    d1 = d1 / max(np.linalg.norm(d1), 1e-8) * size
    d2 = d2 / max(np.linalg.norm(d2), 1e-8) * size
    p_a = corner + d1
    p_b = corner + d1 + d2
    p_c = corner + d2
    sq = VMobject(color=color, stroke_width=stroke_width)
    sq.set_points_as_corners([p_a, p_b, p_c])
    return sq


def make_dashed_3d(
    p0: np.ndarray,
    p1: np.ndarray,
    color: str,
    num_dashes: int = 10,
    stroke_width: float = 2.4,
) -> DashedLine:
    """Create a 3D dashed line segment from `p0` to `p1`."""
    length = max(float(np.linalg.norm(p1 - p0)), 1e-4)
    return DashedLine(
        start=p0,
        end=p1,
        dash_length=length / (2.0 * num_dashes),
        dashed_ratio=0.55,
        color=color,
        stroke_width=stroke_width,
    )


def make_label(tex_str: str, pos: np.ndarray, color: str, font_size: int = 25, pill: bool = False) -> MathTex:
    """Create a crisp 3D math label with optional dark backdrop pill."""
    lbl = MathTex(tex_str, font_size=font_size, color=color).move_to(pos)
    if pill:
        lbl.add_background_rectangle(color=BG_HEX, opacity=0.74, buff=0.05)
    return lbl


def make_vec3d(
    p0: np.ndarray,
    p1: np.ndarray,
    color: str,
    stroke_width: float = 5.4,
    tip_len: float = 0.22,
    tip_rad: float = 0.066,
) -> VGroup:
    """Lightweight 3D vector (Line shaft + 6-sided pyramid tip) for fast 60fps rendering."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    length = float(np.linalg.norm(v))
    if length < 1e-5:
        return VGroup()
    u = v / length
    tl = min(tip_len, 0.35 * length)
    tr = tip_rad * (tl / tip_len)
    base_pt = p1 - tl * u

    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    perp1 = np.cross(u, ref)
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(u, perp1)

    shaft = Line(p0, base_pt + 0.02 * u, color=color, stroke_width=stroke_width)
    n_sides = 6
    ring = [
        base_pt + tr * (np.cos(2 * np.pi * i / n_sides) * perp1 + np.sin(2 * np.pi * i / n_sides) * perp2)
        for i in range(n_sides)
    ]
    faces = [
        Polygon(
            p1,
            ring[i],
            ring[(i + 1) % n_sides],
            fill_color=color,
            fill_opacity=1.0,
            stroke_color=color,
            stroke_width=0.6,
        )
        for i in range(n_sides)
    ]
    return VGroup(shaft, *faces)


def make_hud_header(title_str: str, math_str: str, subtitle_str: str):
    """Build fixed-frame title + math badge + subtitle strip."""
    title = Text(title_str, font="Segoe UI", font_size=22, weight="BOLD", color=INK_HEX)
    title_math = MathTex(math_str, font_size=24, color=INDIGO_LIGHT_HEX)
    title_row = VGroup(title, title_math).arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
    title_row.to_corner(UL, buff=0.45)

    sub = MathTex(subtitle_str, font_size=21, color=MUTED_HEX)
    sub.next_to(title_row, DOWN, buff=0.16, aligned_edge=LEFT)
    return title_row, sub


def swap_hud_subtitle(
    scene: ThreeDScene,
    title_row: VGroup,
    old_sub: MathTex,
    new_tex: str,
    font_size: int = 21,
    color: str = MUTED_HEX,
) -> MathTex:
    """Smoothly cross-slide HUD subtitle in fixed screen space without 3D perspective bleed."""
    scene.play(FadeOut(old_sub, shift=UP * 0.08, run_time=0.24, rate_func=smooth))
    scene.remove_fixed_in_frame_mobjects(old_sub)
    scene.remove(old_sub)

    new_sub = MathTex(new_tex, font_size=font_size, color=color)
    new_sub.next_to(title_row, DOWN, buff=0.16, aligned_edge=LEFT)
    scene.add_fixed_in_frame_mobjects(new_sub)
    scene.play(FadeIn(new_sub, shift=UP * 0.08, run_time=0.28, rate_func=smooth))
    return new_sub


class IV2SLSProjectionScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_HEX

        # Exact R^3 geometry:
        # C(Z) is the z=0 horizontal plane; structural error u lies on +z so Z'u = 0.
        # Endogenous regressor x has a component in C(Z) plus endogeneity along +z (x'u != 0).
        x_hat_raw = np.array([1.65, 1.15, 0.0])  # P_Z x
        x_raw = np.array([1.65, 1.15, 1.38])
        u_raw = np.array([0.0, 0.0, 1.18])
        beta_true = 0.84
        y_raw = beta_true * x_raw + u_raw

        # OLS projection P_x y (biased upward because x'u > 0)
        b_ols = float(x_raw @ y_raw) / float(x_raw @ x_raw)
        y_ols_raw = b_ols * x_raw
        # 2SLS Stage 2 projection P_{x_hat} y = beta_true * x_hat
        b_2sls = float(x_hat_raw @ y_raw) / float(x_hat_raw @ x_hat_raw)
        y_2sls_raw = b_2sls * x_hat_raw

        world_offset = -mvec(np.array([1.05, 0.95, 1.12]))
        o_pt = world_offset
        x_pt = mvec(x_raw) + world_offset
        x_hat_pt = mvec(x_hat_raw) + world_offset
        u_pt = mvec(u_raw) + world_offset
        y_pt = mvec(y_raw) + world_offset
        y_ols_pt = mvec(y_ols_raw) + world_offset
        y_2sls_pt = mvec(y_2sls_raw) + world_offset

        theta_0 = -50 * DEGREES
        phi_0 = 66 * DEGREES
        self.set_camera_orientation(phi=phi_0, theta=theta_0, zoom=1.00)

        # Fixed HUD
        title_row, sub_tex = make_hud_header(
            "2SLS Projection Geometry:",
            r"\hat{\beta}_{\mathrm{2SLS}} = (\hat{x}'\hat{x})^{-1}\hat{x}'y",
            r"\text{Endogeneity: } x \not\perp u \;\Rightarrow\; \text{OLS projection } P_x y \text{ absorbs } u \; (\hat{\beta}_{\mathrm{OLS}} \neq \beta)",
        )
        self.add_fixed_in_frame_mobjects(title_row, sub_tex)

        # Instrument subspace C(Z) at z = o_pt[2]
        z0 = float(o_pt[2])
        ox, oy = float(o_pt[0]), float(o_pt[1])
        plane = Polygon(
            np.array([ox - 0.35, oy - 0.35, z0]),
            np.array([ox + 3.45, oy - 0.35, z0]),
            np.array([ox + 3.45, oy + 2.85, z0]),
            np.array([ox - 0.35, oy + 2.85, z0]),
            fill_color=TEAL_HEX,
            fill_opacity=0.15,
            stroke_color=TEAL_HEX,
            stroke_opacity=0.38,
            stroke_width=1.2,
        )
        grid_lines = VGroup()
        for g in [0.0, 0.9, 1.8, 2.7]:
            grid_lines.add(Line(np.array([ox - 0.25, oy + g, z0]), np.array([ox + 3.35, oy + g, z0]), color=TEAL_HEX, stroke_width=0.9))
        for g in [0.0, 1.0, 2.0, 3.0]:
            grid_lines.add(Line(np.array([ox + g, oy - 0.25, z0]), np.array([ox + g, oy + 2.75, z0]), color=TEAL_HEX, stroke_width=0.9))
        grid_lines.set_opacity(0.38)

        ax_z = Line(o_pt + np.array([0, 0, -0.25]), o_pt + np.array([0, 0, 3.55]), color=GRID_HEX, stroke_width=1.2)
        origin_dot = Dot3D(o_pt, radius=0.055, color=INK_HEX, resolution=(6, 6))
        lbl_cz = make_label(r"\mathcal{C}(Z) \perp u", np.array([ox + 2.70, oy + 2.35, z0 + 0.04]), TEAL_HEX, font_size=24, pill=True)
        self.add_fixed_orientation_mobjects(lbl_cz)
        self.add(plane, grid_lines, ax_z, origin_dot, lbl_cz)

        # Phase 1: Endogenous regressor x, error u, outcome y
        arr_u = make_vec3d(o_pt, u_pt, color=ROSE_HEX, stroke_width=4.6, tip_len=0.20, tip_rad=0.060)
        arr_x = make_vec3d(o_pt, x_pt, color=CYAN_HEX, stroke_width=5.4, tip_len=0.22, tip_rad=0.066)
        arr_y = make_vec3d(o_pt, y_pt, color=AMBER_HEX, stroke_width=5.8, tip_len=0.23, tip_rad=0.072)

        lbl_u = make_label(r"u", u_pt + np.array([-0.26, -0.06, 0.14]), ROSE_HEX, font_size=28)
        lbl_x = make_label(r"x", x_pt + np.array([-0.16, 0.14, 0.24]), CYAN_HEX, font_size=30)
        lbl_y = make_label(r"y = \beta x + u", y_pt + np.array([0.24, 0.10, 0.22]), AMBER_HEX, font_size=27, pill=True)
        self.add_fixed_orientation_mobjects(lbl_u, lbl_x, lbl_y)

        self.play(
            GrowFromPoint(arr_u, o_pt),
            GrowFromPoint(arr_x, o_pt),
            GrowFromPoint(arr_y, o_pt),
            run_time=1.5,
            rate_func=smooth,
        )
        self.play(FadeIn(lbl_u), FadeIn(lbl_x), FadeIn(lbl_y), run_time=0.45)

        # Show OLS biased projection P_x y
        ols_ray = make_dashed_3d(y_pt, y_ols_pt, color=ROSE_HEX, num_dashes=9, stroke_width=2.2)
        arr_y_ols = make_vec3d(o_pt, y_ols_pt, color=ROSE_HEX, stroke_width=5.0, tip_len=0.21, tip_rad=0.064)
        ra_ols = make_right_angle_3d(y_ols_pt, o_pt - y_ols_pt, y_pt - y_ols_pt, size=0.22, color=ROSE_HEX, stroke_width=2.2)
        lbl_y_ols = make_label(r"P_x y = \hat{\beta}_{\mathrm{OLS}} x", y_ols_pt + np.array([0.78, 0.18, -0.05]), ROSE_HEX, font_size=22, pill=True)
        self.add_fixed_orientation_mobjects(lbl_y_ols)

        self.play(
            Create(ols_ray),
            GrowFromPoint(arr_y_ols, o_pt),
            Create(ra_ols),
            FadeIn(lbl_y_ols),
            run_time=1.35,
            rate_func=smooth,
        )
        self.wait(0.45)

        # Phase 2: Stage 1 - Partial/Project x orthogonally onto C(Z) -> x_hat = P_Z x
        sub_s1 = swap_hud_subtitle(
            self,
            title_row,
            sub_tex,
            r"\text{Stage 1: Project } x \text{ onto } \mathcal{C}(Z) \;\Rightarrow\; \hat{x} = P_Z x \in \mathcal{C}(Z) \;\; (\hat{x} \perp u)",
            font_size=21,
            color=MUTED_HEX,
        )

        drop_x = make_dashed_3d(x_pt, x_hat_pt, color=TEAL_HEX, num_dashes=10, stroke_width=2.2)
        arr_x_hat = make_vec3d(o_pt, x_hat_pt, color=TEAL_HEX, stroke_width=5.6, tip_len=0.22, tip_rad=0.068)
        ra_xhat = make_right_angle_3d(x_hat_pt, o_pt - x_hat_pt, x_pt - x_hat_pt, size=0.22, color=TEAL_HEX, stroke_width=2.1)
        lbl_x_hat = make_label(r"\hat{x} = P_Z x", x_hat_pt + np.array([0.82, 0.14, -0.06]), TEAL_HEX, font_size=24, pill=True)
        self.add_fixed_orientation_mobjects(lbl_x_hat)

        # Dim OLS overshoot cleanly so Stage 1 & 2 stand out
        self.play(
            ols_ray.animate.set_opacity(0.25),
            arr_y_ols.animate.set_opacity(0.28),
            ra_ols.animate.set_opacity(0.25),
            lbl_y_ols.animate.set_opacity(0.35),
            Create(drop_x),
            GrowFromPoint(arr_x_hat, o_pt),
            run_time=1.55,
            rate_func=smooth,
        )
        self.play(Create(ra_xhat), FadeIn(lbl_x_hat), run_time=0.5)

        # Phase 3: Stage 2 - Project y onto x_hat -> beta_2SLS * x_hat = beta * x_hat
        swap_hud_subtitle(
            self,
            title_row,
            sub_s1,
            r"\text{Stage 2: Regress } y \text{ on } \hat{x} \;\Rightarrow\; \hat{\beta}_{\mathrm{2SLS}} = \frac{\hat{x}'y}{\hat{x}'\hat{x}} = \beta + \frac{\hat{x}'u}{\hat{x}'\hat{x}} = \beta",
            font_size=20,
            color=INK_HEX,
        )

        proj_2sls = make_dashed_3d(y_pt, y_2sls_pt, color=INDIGO_LIGHT_HEX, num_dashes=12, stroke_width=2.5)
        arr_y_2sls = make_vec3d(o_pt, y_2sls_pt, color=INDIGO_LIGHT_HEX, stroke_width=6.2, tip_len=0.24, tip_rad=0.074)
        ra_2sls = make_right_angle_3d(y_2sls_pt, o_pt - y_2sls_pt, y_pt - y_2sls_pt, size=0.25, color=INK_HEX, stroke_width=2.4)
        lbl_y_2sls = make_label(r"P_{\hat{x}} y = \hat{\beta}_{\mathrm{2SLS}} \hat{x}", y_2sls_pt + np.array([-0.38, -0.60, -0.22]), INDIGO_LIGHT_HEX, font_size=24, pill=True)
        self.add_fixed_orientation_mobjects(lbl_y_2sls)

        self.play(
            Create(proj_2sls),
            GrowFromPoint(arr_y_2sls, o_pt),
            run_time=1.55,
            rate_func=smooth,
        )
        self.play(Create(ra_2sls), FadeIn(lbl_y_2sls), run_time=0.65)

        # Slow gentle orbit arc (44 deg out & back) for 3D depth perception & seamless loop
        self.move_camera(theta=theta_0 + 44 * DEGREES, run_time=3.4, rate_func=smooth)
        self.wait(0.6)
        self.move_camera(theta=theta_0, run_time=3.4, rate_func=smooth)
        self.wait(0.4)


def render_all(quality: str) -> None:
    root_dir = Path(__file__).resolve().parent.parent
    out_dir = root_dir / "assets" / "media"
    candidate_media = root_dir / "visuals" / "media"
    media_dir = (
        candidate_media
        if str(candidate_media).isascii()
        else Path(tempfile.gettempdir()) / "manim_iv_2sls"
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
    out_name = "iv-2sls-projection.mp4"

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
        "IV2SLSProjectionScene",
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
    parser = argparse.ArgumentParser(description="Render IV/2SLS Manim video")
    parser.add_argument("--quality", choices=["low", "high"], default="high")
    args = parser.parse_args()
    render_all(args.quality)
