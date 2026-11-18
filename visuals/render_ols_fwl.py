"""
Manim 3D animations for Econometrics Visual Supplement 01:
  - OLSProjectionScene -> assets/media/ols-projection.mp4
  - FWLGeometryScene   -> assets/media/fwl-geometry.mp4

Usage:
  .venv\\Scripts\\python.exe visuals\\render_ols_fwl.py [--quality low|high]
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
    Create,
    DashedLine,
    Dot3D,
    FadeIn,
    FadeOut,
    GrowFromPoint,
    Line,
    MathTex,
    Polygon,
    ReplacementTransform,
    Text,
    ThreeDScene,
    VGroup,
    VMobject,
    smooth,
)

# Palette matching notes/econometrics-visual-ols-fwl.html
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

SCALE = 1.48


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


def make_label(tex_str: str, pos: np.ndarray, color: str, font_size: int = 26, pill: bool = False) -> MathTex:
    """Create a crisp 3D math label with optional dark backdrop pill."""
    lbl = MathTex(tex_str, font_size=font_size, color=color).move_to(pos)
    if pill:
        lbl.add_background_rectangle(color=BG_HEX, opacity=0.72, buff=0.05)
    return lbl


def make_vec3d(
    p0: np.ndarray,
    p1: np.ndarray,
    color: str,
    stroke_width: float = 5.4,
    tip_len: float = 0.23,
    tip_rad: float = 0.068,
) -> VGroup:
    """Lightweight, crisp 3D vector (Line shaft + 6-sided pyramid tip) for fast 60fps rendering."""
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
    title_math = MathTex(math_str, font_size=25, color=INDIGO_LIGHT_HEX)
    title_row = VGroup(title, title_math).arrange(RIGHT, buff=0.18, aligned_edge=DOWN)
    title_row.to_corner(UL, buff=0.45)

    sub = MathTex(subtitle_str, font_size=21, color=MUTED_HEX)
    sub.next_to(title_row, DOWN, buff=0.16, aligned_edge=LEFT)
    return title_row, sub


class OLSProjectionScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_HEX

        # Exact orthogonal projection geometry in R^3
        x1_raw = np.array([2.15, 0.35, 0.15])
        x2_raw = np.array([0.55, 2.05, 0.25])
        X = np.column_stack([x1_raw, x2_raw])
        b_true = np.array([0.92, 0.85])
        u_init = np.array([-0.26, -0.32, 1.82])
        Px = X @ np.linalg.inv(X.T @ X) @ X.T
        u_raw = u_init - Px @ u_init
        u_raw = u_raw * (1.72 / np.linalg.norm(u_raw))

        yhat_raw = X @ b_true
        y_raw = yhat_raw + u_raw
        b_hat = np.linalg.solve(X.T @ X, X.T @ y_raw)
        comp1_raw = b_hat[0] * x1_raw
        comp2_raw = b_hat[1] * x2_raw

        # Shift world coordinates so the 3D scene centers below the top HUD
        # while keeping camera frame_center at ORIGIN (required for fixed HUD alignment).
        world_offset = -mvec(np.array([1.05, 0.95, 1.15]))
        o_pt = world_offset
        x1 = mvec(x1_raw) + world_offset
        x2 = mvec(x2_raw) + world_offset
        yhat = mvec(yhat_raw) + world_offset
        y = mvec(y_raw) + world_offset
        u_vec = mvec(u_raw)
        comp1 = mvec(comp1_raw) + world_offset
        comp2 = mvec(comp2_raw) + world_offset

        x1_dir = x1 - o_pt
        x2_dir = x2 - o_pt

        theta_0 = -52 * DEGREES
        phi_0 = 67 * DEGREES
        self.set_camera_orientation(phi=phi_0, theta=theta_0, zoom=1.02)

        # Fixed HUD
        title_row, sub_tex = make_hud_header(
            "OLS Projection Geometry:",
            r"y = P_X y + M_X y \;\; (\hat{u} \perp \mathcal{C}(X))",
            r"\text{Step 1: Observation } y \in \mathbb{R}^n \text{ and column space } \mathcal{C}(X)=\mathrm{span}(x_1, x_2)",
        )
        self.add_fixed_in_frame_mobjects(title_row, sub_tex)

        # Subtle base axes & origin
        ax_x = Line(o_pt + mvec([-0.3, 0, 0]), o_pt + mvec([2.85, 0, 0]), color=GRID_HEX, stroke_width=1.4)
        ax_y = Line(o_pt + mvec([0, -0.3, 0]), o_pt + mvec([0, 2.75, 0]), color=GRID_HEX, stroke_width=1.4)
        ax_z = Line(o_pt + mvec([0, 0, -0.2]), o_pt + mvec([0, 0, 2.35]), color=GRID_HEX, stroke_width=1.4)
        origin_dot = Dot3D(o_pt, radius=0.055, color=INK_HEX, resolution=(6, 6))
        self.add(ax_x, ax_y, ax_z, origin_dot)

        # Column space plane C(X) as a single fast Polygon + crisp gridlines
        u0, u1 = -0.22, 1.24
        v0, v1 = -0.22, 1.24
        plane = Polygon(
            o_pt + u0 * x1_dir + v0 * x2_dir,
            o_pt + u1 * x1_dir + v0 * x2_dir,
            o_pt + u1 * x1_dir + v1 * x2_dir,
            o_pt + u0 * x1_dir + v1 * x2_dir,
            fill_color=INDIGO_HEX,
            fill_opacity=0.22,
            stroke_color=INDIGO_LIGHT_HEX,
            stroke_opacity=0.42,
            stroke_width=1.2,
        )

        grid_lines = VGroup()
        for g in [0.0, 0.5, 1.0]:
            grid_lines.add(
                Line(o_pt - 0.20 * x1_dir + g * x2_dir, o_pt + 1.22 * x1_dir + g * x2_dir, color=INDIGO_HEX, stroke_width=1.0),
                Line(o_pt + g * x1_dir - 0.20 * x2_dir, o_pt + g * x1_dir + 1.22 * x2_dir, color=INDIGO_HEX, stroke_width=1.0),
            )
        grid_lines.set_opacity(0.45)

        arr_x1 = make_vec3d(o_pt, x1, color=CYAN_HEX, stroke_width=5.2, tip_len=0.22, tip_rad=0.065)
        arr_x2 = make_vec3d(o_pt, x2, color=TEAL_HEX, stroke_width=5.2, tip_len=0.22, tip_rad=0.065)

        lbl_x1 = make_label("x_1", x1 + np.array([0.20, -0.24, -0.02]), CYAN_HEX, font_size=28, pill=True)
        lbl_x2 = make_label("x_2", x2 + np.array([-0.24, 0.22, 0.04]), TEAL_HEX, font_size=28, pill=True)
        lbl_cx = make_label(r"\mathcal{C}(X)", o_pt + 1.18 * x1_dir - 0.14 * x2_dir + np.array([0.0, 0.0, -0.24]), INDIGO_LIGHT_HEX, font_size=26, pill=True)
        self.add_fixed_orientation_mobjects(lbl_x1, lbl_x2, lbl_cx)

        # Phase 1: basis vectors & span plane
        self.play(
            GrowFromPoint(arr_x1, o_pt),
            GrowFromPoint(arr_x2, o_pt),
            FadeIn(plane),
            FadeIn(grid_lines),
            run_time=1.5,
            rate_func=smooth,
        )
        self.play(FadeIn(lbl_x1), FadeIn(lbl_x2), FadeIn(lbl_cx), run_time=0.45)

        # Phase 2: target vector y
        arr_y = make_vec3d(o_pt, y, color=AMBER_HEX, stroke_width=6.0, tip_len=0.24, tip_rad=0.075)
        lbl_y = make_label("y", y + np.array([0.08, 0.08, 0.24]), AMBER_HEX, font_size=32)
        self.add_fixed_orientation_mobjects(lbl_y)

        self.play(GrowFromPoint(arr_y, o_pt), run_time=1.3, rate_func=smooth)
        self.play(FadeIn(lbl_y), run_time=0.4)

        # Update HUD subtitle for Phase 3 (projection drop)
        sub_step2 = MathTex(
            r"\text{Step 2: Minimize } \|y - X\beta\|_2^2 \;\Leftrightarrow\; \text{Orthogonal drop onto } \mathcal{C}(X) \; (X'\hat{u}=0)",
            font_size=21,
            color=MUTED_HEX,
        )
        sub_step2.next_to(title_row, DOWN, buff=0.16, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(sub_step2)
        self.remove_fixed_in_frame_mobjects(sub_tex)
        self.play(FadeOut(sub_tex, run_time=0.25), FadeIn(sub_step2, run_time=0.35))

        # Phase 3: orthogonal drop from y to yhat + uhat + parallelogram decomposition
        proj_ray = make_dashed_3d(y, yhat, color=ROSE_HEX, num_dashes=10, stroke_width=2.5)
        arr_yhat = make_vec3d(o_pt, yhat, color=INDIGO_LIGHT_HEX, stroke_width=6.0, tip_len=0.24, tip_rad=0.075)
        arr_u = make_vec3d(yhat, y, color=ROSE_HEX, stroke_width=5.4, tip_len=0.22, tip_rad=0.068)
        right_angle = make_right_angle_3d(yhat, o_pt - yhat, u_vec, size=0.28, color=INK_HEX, stroke_width=2.4)

        par_1 = make_dashed_3d(comp1, yhat, color=TEAL_HEX, num_dashes=8, stroke_width=1.8)
        par_2 = make_dashed_3d(comp2, yhat, color=CYAN_HEX, num_dashes=8, stroke_width=1.8)

        lbl_yhat = make_label(r"\hat{y} = P_X y = X\hat{\beta}", yhat + np.array([0.32, 0.22, -0.32]), INDIGO_LIGHT_HEX, font_size=24, pill=True)
        lbl_u = make_label(r"\hat{u} = M_X y \perp \mathcal{C}(X)", yhat + 0.56 * u_vec + np.array([1.24, 0.36, 0.08]), ROSE_HEX, font_size=24, pill=True)
        self.add_fixed_orientation_mobjects(lbl_yhat, lbl_u)

        self.play(
            Create(proj_ray),
            GrowFromPoint(arr_yhat, o_pt),
            run_time=1.5,
            rate_func=smooth,
        )
        self.play(
            GrowFromPoint(arr_u, yhat),
            Create(right_angle),
            Create(par_1),
            Create(par_2),
            FadeIn(lbl_yhat),
            FadeIn(lbl_u),
            run_time=1.0,
            rate_func=smooth,
        )

        # Phase 4: Pythagorean decomposition HUD + slow gentle orbit arc (46 deg out & back)
        sub_step3 = MathTex(
            r"\text{Pythagorean Decomposition: } \|y\|_2^2 = \|P_X y\|_2^2 + \|M_X y\|_2^2 \;\; (\mathrm{TSS}=\mathrm{ESS}+\mathrm{RSS})",
            font_size=21,
            color=INK_HEX,
        )
        sub_step3.next_to(title_row, DOWN, buff=0.16, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(sub_step3)
        self.remove_fixed_in_frame_mobjects(sub_step2)
        self.play(FadeOut(sub_step2, run_time=0.25), FadeIn(sub_step3, run_time=0.4))

        self.move_camera(theta=theta_0 + 46 * DEGREES, run_time=3.4, rate_func=smooth)
        self.wait(0.6)
        self.move_camera(theta=theta_0, run_time=3.4, rate_func=smooth)
        self.wait(0.4)


class FWLGeometryScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_HEX

        # Control X1 vertical so C(X1)^perp is the clean z=0 horizontal plane
        x1_raw = np.array([0.0, 0.0, 2.05])
        x2_raw = np.array([1.62, 0.34, 1.32])
        M1 = np.eye(3) - np.outer(x1_raw, x1_raw) / float(x1_raw @ x1_raw)
        x2_tilde_raw = M1 @ x2_raw

        b1, b2 = 0.65, 1.36
        u_init = np.array([-0.26, 1.32, 0.0])
        X = np.column_stack([x1_raw, x2_raw])
        u_raw = u_init - X @ np.linalg.solve(X.T @ X, X.T @ u_init)
        u_raw = u_raw * (1.25 / np.linalg.norm(u_raw))

        y_raw = b1 * x1_raw + b2 * x2_raw + u_raw
        y_tilde_raw = M1 @ y_raw
        b2_hat = float(x2_tilde_raw @ y_tilde_raw) / float(x2_tilde_raw @ x2_tilde_raw)
        yhat_fwl_raw = b2_hat * x2_tilde_raw

        world_offset = -mvec(np.array([1.00, 0.85, 1.25]))
        o_pt = world_offset
        x1 = mvec(x1_raw) + world_offset
        x2 = mvec(x2_raw) + world_offset
        x2_tilde = mvec(x2_tilde_raw) + world_offset
        y = mvec(y_raw) + world_offset
        y_tilde = mvec(y_tilde_raw) + world_offset
        yhat_fwl = mvec(yhat_fwl_raw) + world_offset

        theta_0 = -48 * DEGREES
        phi_0 = 66 * DEGREES
        self.set_camera_orientation(phi=phi_0, theta=theta_0, zoom=1.02)

        # Fixed HUD
        title_row, sub_tex = make_hud_header(
            "Frisch–Waugh–Lovell Theorem:",
            r"\text{Residual-on-Residual Projection}",
            r"\text{Full Regression: } y = X_1\hat{\beta}_1 + X_2\hat{\beta}_2 + \hat{u} \;\; (\text{correlated } X_1, X_2)",
        )
        self.add_fixed_in_frame_mobjects(title_row, sub_tex)

        # Orthogonal subspace plane C(X1)^perp (z = world_offset[2])
        z0 = float(o_pt[2])
        ox, oy = float(o_pt[0]), float(o_pt[1])
        plane = Polygon(
            np.array([ox - 0.4, oy - 0.4, z0]),
            np.array([ox + 3.8, oy - 0.4, z0]),
            np.array([ox + 3.8, oy + 3.1, z0]),
            np.array([ox - 0.4, oy + 3.1, z0]),
            fill_color=TEAL_HEX,
            fill_opacity=0.16,
            stroke_color=TEAL_HEX,
            stroke_opacity=0.38,
            stroke_width=1.2,
        )
        grid_lines = VGroup()
        for g in [0.0, 1.0, 2.0, 3.0]:
            grid_lines.add(Line(np.array([ox - 0.3, oy + g, z0]), np.array([ox + 3.7, oy + g, z0]), color=TEAL_HEX, stroke_width=1.0))
        for g in [0.0, 1.0, 2.0, 3.0]:
            grid_lines.add(Line(np.array([ox + g, oy - 0.3, z0]), np.array([ox + g, oy + 3.0, z0]), color=TEAL_HEX, stroke_width=1.0))
        grid_lines.set_opacity(0.40)

        origin_dot = Dot3D(o_pt, radius=0.055, color=INK_HEX, resolution=(6, 6))
        lbl_plane = make_label(r"\mathcal{C}(X_1)^\perp", np.array([ox - 0.15, oy + 2.55, z0 + 0.06]), TEAL_HEX, font_size=25, pill=True)
        self.add_fixed_orientation_mobjects(lbl_plane)
        self.add(plane, grid_lines, origin_dot, lbl_plane)

        # Stage 0: Show X1, X2, y
        arr_x1 = make_vec3d(o_pt, x1, color=CYAN_HEX, stroke_width=5.4, tip_len=0.22, tip_rad=0.068)
        arr_x2 = make_vec3d(o_pt, x2, color=TEAL_HEX, stroke_width=5.0, tip_len=0.21, tip_rad=0.065)
        arr_y = make_vec3d(o_pt, y, color=AMBER_HEX, stroke_width=5.8, tip_len=0.23, tip_rad=0.070)

        lbl_x1 = make_label("X_1", x1 + np.array([-0.26, -0.08, 0.22]), CYAN_HEX, font_size=28)
        lbl_x2 = make_label("X_2", x2 + np.array([0.16, 0.0, 0.20]), TEAL_HEX, font_size=27)
        lbl_y = make_label("y", y + np.array([0.12, 0.05, 0.22]), AMBER_HEX, font_size=30)
        self.add_fixed_orientation_mobjects(lbl_x1, lbl_x2, lbl_y)

        self.play(
            GrowFromPoint(arr_x1, o_pt),
            GrowFromPoint(arr_x2, o_pt),
            GrowFromPoint(arr_y, o_pt),
            run_time=1.5,
            rate_func=smooth,
        )
        self.play(FadeIn(lbl_x1), FadeIn(lbl_x2), FadeIn(lbl_y), run_time=0.45)

        # Stage 1: Partial out X1 -> project X2 and y vertically onto C(X1)^perp
        sub_s1 = MathTex(
            r"\text{Stage 1 (Partial Out } X_1\text{): } \tilde{X}_2 = M_1 X_2,\;\; \tilde{y} = M_1 y \in \mathcal{C}(X_1)^\perp",
            font_size=21,
            color=MUTED_HEX,
        )
        sub_s1.next_to(title_row, DOWN, buff=0.16, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(sub_s1)
        self.remove_fixed_in_frame_mobjects(sub_tex)
        self.play(FadeOut(sub_tex, run_time=0.25), FadeIn(sub_s1, run_time=0.35))

        drop_x2 = make_dashed_3d(x2, x2_tilde, color=TEAL_HEX, num_dashes=9, stroke_width=2.0)
        drop_y = make_dashed_3d(y, y_tilde, color=AMBER_HEX, num_dashes=11, stroke_width=2.0)

        arr_x2_t = make_vec3d(o_pt, x2_tilde, color=TEAL_HEX, stroke_width=5.4, tip_len=0.21, tip_rad=0.068)
        arr_y_t = make_vec3d(o_pt, y_tilde, color=AMBER_HEX, stroke_width=5.4, tip_len=0.21, tip_rad=0.068)
        ra_x2 = make_right_angle_3d(x2_tilde, o_pt - x2_tilde, x1 - o_pt, size=0.22, color=MUTED_HEX, stroke_width=2.0)
        ra_y = make_right_angle_3d(y_tilde, o_pt - y_tilde, x1 - o_pt, size=0.22, color=MUTED_HEX, stroke_width=2.0)

        lbl_x2_t = make_label(r"\tilde{X}_2 = M_1 X_2", 0.42 * x2_tilde + 0.58 * o_pt + np.array([-0.82, -0.88, -0.18]), TEAL_HEX, font_size=24, pill=True)
        lbl_y_t = make_label(r"\tilde{y} = M_1 y", y_tilde + np.array([0.70, 0.42, 0.16]), AMBER_HEX, font_size=24, pill=True)
        self.add_fixed_orientation_mobjects(lbl_x2_t, lbl_y_t)

        ghost_x2 = arr_x2.copy().set_opacity(0.32)
        ghost_y = arr_y.copy().set_opacity(0.32)
        self.add(ghost_x2, ghost_y)

        self.play(
            Create(drop_x2),
            Create(drop_y),
            ReplacementTransform(arr_x2, arr_x2_t),
            ReplacementTransform(arr_y, arr_y_t),
            FadeOut(lbl_x2),
            FadeOut(lbl_y),
            run_time=2.0,
            rate_func=smooth,
        )
        self.play(
            Create(ra_x2),
            Create(ra_y),
            FadeIn(lbl_x2_t),
            FadeIn(lbl_y_t),
            run_time=0.55,
        )

        # Stage 2: Residual-on-residual projection inside C(X1)^perp
        sub_s2 = MathTex(
            r"\text{Stage 2: Regress } \tilde{y} \text{ on } \tilde{X}_2 \;\Rightarrow\; \hat{\beta}_2^{\mathrm{FWL}} = (\tilde{X}_2'\tilde{X}_2)^{-1}\tilde{X}_2'\tilde{y} \equiv \hat{\beta}_2^{\mathrm{Full}}",
            font_size=20,
            color=INK_HEX,
        )
        sub_s2.next_to(title_row, DOWN, buff=0.16, aligned_edge=LEFT)
        self.add_fixed_in_frame_mobjects(sub_s2)
        self.remove_fixed_in_frame_mobjects(sub_s1)
        self.play(FadeOut(sub_s1, run_time=0.25), FadeIn(sub_s2, run_time=0.4))

        proj_ray = make_dashed_3d(y_tilde, yhat_fwl, color=ROSE_HEX, num_dashes=10, stroke_width=2.4)
        arr_yhat_fwl = make_vec3d(o_pt, yhat_fwl, color=INDIGO_LIGHT_HEX, stroke_width=6.0, tip_len=0.23, tip_rad=0.074)
        arr_u = make_vec3d(yhat_fwl, y_tilde, color=ROSE_HEX, stroke_width=5.4, tip_len=0.21, tip_rad=0.068)
        ra_fwl = make_right_angle_3d(yhat_fwl, o_pt - yhat_fwl, y_tilde - yhat_fwl, size=0.25, color=INK_HEX, stroke_width=2.4)

        lbl_yhat_fwl = make_label(r"\hat{\beta}_2 \tilde{X}_2", yhat_fwl + np.array([0.24, -0.44, -0.06]), INDIGO_LIGHT_HEX, font_size=24, pill=True)
        lbl_u = make_label(r"\hat{u} = M_X y", 0.5 * (yhat_fwl + y_tilde) + np.array([0.52, -0.26, -0.14]), ROSE_HEX, font_size=24, pill=True)
        self.add_fixed_orientation_mobjects(lbl_yhat_fwl, lbl_u)

        self.play(
            Create(proj_ray),
            GrowFromPoint(arr_yhat_fwl, o_pt),
            run_time=1.5,
            rate_func=smooth,
        )
        self.play(
            GrowFromPoint(arr_u, yhat_fwl),
            Create(ra_fwl),
            FadeIn(lbl_yhat_fwl),
            FadeIn(lbl_u),
            run_time=0.95,
            rate_func=smooth,
        )

        # Slow gentle orbit arc (44 deg out & back) for 3D depth perception & seamless loop
        self.move_camera(theta=theta_0 + 44 * DEGREES, run_time=3.4, rate_func=smooth)
        self.wait(0.6)
        self.move_camera(theta=theta_0, run_time=3.4, rate_func=smooth)
        self.wait(0.4)


def render_all(quality: str, which: str = "all") -> None:
    root_dir = Path(__file__).resolve().parent.parent
    out_dir = root_dir / "assets" / "media"
    # TeX Live on Windows can choke on non-ASCII characters (e.g. Korean folder names)
    # in --output-directory; fall back to an ASCII temp dir when needed.
    candidate_media = root_dir / "visuals" / "media"
    media_dir = (
        candidate_media
        if str(candidate_media).isascii()
        else Path(tempfile.gettempdir()) / "manim_ols_fwl"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    # Ensure WinGet FFmpeg is reachable even in fresh shells
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
        ("ols", "OLSProjectionScene", "ols-projection.mp4"),
        ("fwl", "FWLGeometryScene", "fwl-geometry.mp4"),
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
    parser = argparse.ArgumentParser(description="Render OLS & FWL Manim videos")
    parser.add_argument("--quality", choices=["low", "high"], default="high")
    parser.add_argument("--scene", choices=["all", "ols", "fwl"], default="all")
    args = parser.parse_args()
    render_all(args.quality, args.scene)
