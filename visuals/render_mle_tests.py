"""
Manim animations for Econometrics Visual Supplement 04 (MLE & Wald/LR/LM):
  - MLETrinityScene       -> assets/media/mle-trinity-geometry.mp4
  - MLEProjection3DScene  -> assets/media/mle-2d-projection.mp4

Usage:
  .venv\\Scripts\\python.exe visuals\\render_mle_tests.py [--quality low|high]
"""

from __future__ import annotations

import argparse
import math
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
    Axes,
    Create,
    DashedLine,
    Dot,
    Dot3D,
    FadeIn,
    FadeOut,
    Line,
    MathTex,
    Polygon,
    Scene,
    Text,
    ThreeDScene,
    Transform,
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

XY_SCALE = 1.25
Z_FLOOR = -1.20
Z_SCALE = 1.05


def pt3(t1: float, t2: float, z_val: float) -> np.ndarray:
    return np.array([t1 * XY_SCALE, t2 * XY_SCALE, Z_FLOOR + z_val * Z_SCALE], dtype=float)


def pill_tex(tex_str: str, pos: np.ndarray, color: str, font_size: int = 24) -> MathTex:
    lbl = MathTex(tex_str, font_size=font_size, color=color).move_to(pos)
    lbl.add_background_rectangle(color=BG_HEX, opacity=0.78, buff=0.05)
    return lbl


def make_hud(title_str: str, math_str: str, sub_str: str):
    title = Text(title_str, font="Segoe UI", font_size=22, weight="BOLD", color=INK_HEX)
    title_math = MathTex(math_str, font_size=23, color=INDIGO_LIGHT_HEX)
    title_row = VGroup(title, title_math).arrange(RIGHT, buff=0.16)
    title_row.to_corner(UL, buff=0.42)

    sub = MathTex(sub_str, font_size=20, color=MUTED_HEX)
    sub.next_to(title_row, DOWN, buff=0.14, aligned_edge=LEFT)
    return title_row, sub


def swap_sub(scene: Scene, title_row: VGroup, old_sub: MathTex, new_tex: str, font_size: int = 20, color: str = MUTED_HEX) -> MathTex:
    scene.play(FadeOut(old_sub, shift=UP * 0.08, run_time=0.22, rate_func=smooth))
    new_sub = MathTex(new_tex, font_size=font_size, color=color)
    new_sub.next_to(title_row, DOWN, buff=0.14, aligned_edge=LEFT)
    scene.play(FadeIn(new_sub, shift=UP * 0.08, run_time=0.25, rate_func=smooth))
    return new_sub


def swap_sub_3d(scene: ThreeDScene, title_row: VGroup, old_sub: MathTex, new_tex: str, font_size: int = 20, color: str = MUTED_HEX) -> MathTex:
    scene.play(FadeOut(old_sub, shift=UP * 0.08, run_time=0.22, rate_func=smooth))
    scene.remove_fixed_in_frame_mobjects(old_sub)
    new_sub = MathTex(new_tex, font_size=font_size, color=color)
    new_sub.next_to(title_row, DOWN, buff=0.14, aligned_edge=LEFT)
    scene.add_fixed_in_frame_mobjects(new_sub)
    scene.play(FadeIn(new_sub, shift=UP * 0.08, run_time=0.25, rate_func=smooth))
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


def make_surface_mesh(
    fn: Callable[[float, float], float],
    r_max: float = 1.45,
    n_grid: int = 12,
    fill_color: str = INDIGO_HEX,
    fill_opacity: float = 0.22,
    stroke_color: str = INDIGO_LIGHT_HEX,
    stroke_width: float = 1.2,
) -> VGroup:
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
                stroke_opacity=0.60,
                stroke_width=stroke_width,
            )
            quads.add(q)
    return quads


class MLETrinityScene(Scene):
    def construct(self):
        self.camera.background_color = BG_HEX

        title_row, sub_tex = make_hud(
            "Likelihood Trinity:",
            r"W_n \approx \mathrm{LR}_n \approx \mathrm{LM}_n \xrightarrow{d} \chi^2_r",
            r"\text{Three geometric views on } \ell_n(\theta): \text{ horizontal (Wald), vertical (LR), and slope (LM)}",
        )
        self.add(title_row, sub_tex)

        ax = Axes(
            x_range=[-1.15, 1.35, 0.5],
            y_range=[-3.6, 0.5, 1.0],
            x_length=9.2,
            y_length=4.7,
            axis_config={"color": MUTED_HEX, "stroke_width": 1.6, "include_tip": True},
        ).shift(DOWN * 0.55)

        x_lbl = MathTex(r"\theta", font_size=26, color=INK_HEX).next_to(ax.x_axis, RIGHT, buff=0.14)
        y_lbl = MathTex(r"\ell_n(\theta) - \ell_n(\hat{\theta})", font_size=23, color=INK_HEX).next_to(ax.y_axis, UP, buff=0.12)

        skew_tr = ValueTracker(0.0)   # 0 = quadratic, 1 = skewed exponential-family
        curv_tr = ValueTracker(2.6)   # curvature at peak I_n
        theta0_tr = ValueTracker(0.82)

        def eval_ll(t: float) -> float:
            c = curv_tr.get_value()
            w = skew_tr.get_value()
            q = -0.5 * c * t * t
            mu = max(0.22, 1.0 + 0.65 * t)
            # Match 2nd derivative at t=0: d^2/dt^2 [-ln(1+at) - 1/(1+at) + 1] at t=0 is -a^2
            # So scaling by c / (0.65^2) matches exact peak curvature c!
            sk = (c / (0.65 * 0.65)) * (-math.log(mu) - 1.0 / mu + 1.0)
            return (1.0 - w) * q + w * sk

        def eval_score(t: float) -> float:
            c = curv_tr.get_value()
            w = skew_tr.get_value()
            dq = -c * t
            mu = max(0.22, 1.0 + 0.65 * t)
            dsk = (c / 0.65) * (1.0 - mu) / (mu * mu)
            return (1.0 - w) * dq + w * dsk

        def eval_wald_parabola(t: float) -> float:
            c = curv_tr.get_value()
            return -0.5 * c * t * t

        def valid_x_bounds():
            # Keep plot strictly within [-1.04, 1.25] AND eval_ll(t) >= -3.48 so curve stays inside axes
            lo = -1.04
            while lo < -0.05 and eval_ll(lo) < -3.48:
                lo += 0.02
            hi = 1.25
            while hi > 0.05 and eval_ll(hi) < -3.48:
                hi -= 0.02
            return [lo, hi]

        ll_curve = always_redraw(
            lambda: ax.plot(
                eval_ll,
                x_range=valid_x_bounds(),
                color=INDIGO_LIGHT_HEX,
                stroke_width=3.6,
            )
        )
        wald_parab = always_redraw(
            lambda: DashedLine(ax.c2p(0, 0), ax.c2p(0, 0), color=AMBER_HEX)
            if skew_tr.get_value() < 0.02
            else ax.plot(
                eval_wald_parabola,
                x_range=[-1.02, 1.25],
                color=AMBER_HEX,
                stroke_width=2.0,
            ).set_stroke(opacity=0.65)
        )

        dot_hat = Dot(ax.c2p(0.0, 0.0), color=TEAL_HEX, radius=0.085)
        lbl_hat = pill_tex(r"\hat{\theta}", ax.c2p(0.11, -0.22), TEAL_HEX, font_size=23)

        self.play(FadeIn(ax), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.55)
        self.play(Create(ll_curve), FadeIn(dot_hat), FadeIn(lbl_hat), run_time=1.0, rate_func=smooth)

        # Null point & 3 geometric tests
        dot_0 = always_redraw(lambda: Dot(ax.c2p(theta0_tr.get_value(), eval_ll(theta0_tr.get_value())), color=CYAN_HEX, radius=0.085))
        lbl_0 = always_redraw(
            lambda: pill_tex(
                r"\theta_0",
                ax.c2p(theta0_tr.get_value() + (-0.11 if theta0_tr.get_value() < 0 else 0.11), -0.18),
                CYAN_HEX,
                font_size=22,
            )
        )

        # 1. Wald: horizontal bracket above peak line y=0
        wald_bar = always_redraw(
            lambda: Line(
                ax.c2p(0.0, 0.22),
                ax.c2p(theta0_tr.get_value(), 0.22),
                color=AMBER_HEX,
                stroke_width=3.2,
            )
        )
        wald_tick_l = always_redraw(lambda: Line(ax.c2p(0.0, 0.14), ax.c2p(0.0, 0.30), color=AMBER_HEX, stroke_width=2.4))
        wald_tick_r = always_redraw(lambda: Line(ax.c2p(theta0_tr.get_value(), 0.14), ax.c2p(theta0_tr.get_value(), 0.30), color=AMBER_HEX, stroke_width=2.4))
        wald_lbl = always_redraw(
            lambda: pill_tex(
                r"\text{Wald: } \hat{\mathcal{I}}_n(\hat{\theta}-\theta_0)^2",
                ax.c2p(theta0_tr.get_value() * 0.5 + (-0.18 if theta0_tr.get_value() < 0 else 0.18), 0.50),
                AMBER_HEX,
                font_size=20,
            )
        )

        # 2. LR: vertical drop between peak (0) and curve at theta0
        lr_line = always_redraw(
            lambda: Line(
                ax.c2p(theta0_tr.get_value(), 0.0),
                ax.c2p(theta0_tr.get_value(), max(-3.50, eval_ll(theta0_tr.get_value()))),
                color=INDIGO_LIGHT_HEX,
                stroke_width=3.6,
            )
        )
        lr_lbl = always_redraw(
            lambda: pill_tex(
                r"\tfrac{1}{2}\mathrm{LR} = \ell_n(\hat{\theta})-\ell_n(\theta_0)",
                ax.c2p(theta0_tr.get_value() + (0.48 if theta0_tr.get_value() >= 0 else -0.52), max(-3.1, eval_ll(theta0_tr.get_value()) * 0.52)),
                INDIGO_LIGHT_HEX,
                font_size=20,
            )
        )

        # 3. LM: tangent line at theta0
        def make_tangent():
            t0 = theta0_tr.get_value()
            y0 = eval_ll(t0)
            s0 = eval_score(t0)
            dt = 0.24
            p1 = ax.c2p(t0 - dt, max(-3.50, min(0.42, y0 - s0 * dt)))
            p2 = ax.c2p(t0 + dt, max(-3.50, min(0.42, y0 + s0 * dt)))
            return Line(p1, p2, color=CYAN_HEX, stroke_width=3.2)

        lm_tangent = always_redraw(make_tangent)
        lm_lbl = always_redraw(
            lambda: pill_tex(
                r"\text{LM: } S_n(\theta_0) = \ell_n'(\theta_0)",
                ax.c2p(
                    theta0_tr.get_value() + (0.42 if theta0_tr.get_value() < 0 else 0.0),
                    max(-3.3, eval_ll(theta0_tr.get_value()) - (0.18 if theta0_tr.get_value() < 0 else 0.36)),
                ),
                CYAN_HEX,
                font_size=20,
            )
        )

        self.play(FadeIn(dot_0), FadeIn(lbl_0), run_time=0.45)
        self.play(Create(wald_bar), FadeIn(wald_tick_l), FadeIn(wald_tick_r), FadeIn(wald_lbl), run_time=0.7)
        self.play(Create(lr_line), FadeIn(lr_lbl), run_time=0.7)
        self.play(Create(lm_tangent), FadeIn(lm_lbl), run_time=0.7)
        self.wait(0.6)

        # Curvature grows (larger sample size / sharper peak) -> W = LR = LM stay equal
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{Higher curvature } \mathcal{I}_n \text{ (larger } n\text{): all three scale proportionally and } W = \mathrm{LR} = \mathrm{LM}",
            font_size=20,
            color=TEAL_HEX,
        )
        self.play(curv_tr.animate.set_value(4.4), run_time=1.4, rate_func=smooth)
        self.wait(0.4)
        self.play(curv_tr.animate.set_value(2.6), run_time=1.0, rate_func=smooth)

        # Morph to skewed likelihood -> divergence!
        self.add(wald_parab)
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{Finite-sample skewness: Wald uses the quadratic parabola at } \hat{\theta} \text{, so } W \neq \mathrm{LR} \neq \mathrm{LM}",
            font_size=20,
            color=AMBER_HEX,
        )
        self.play(skew_tr.animate.set_value(1.0), run_time=1.5, rate_func=smooth)
        self.wait(0.6)

        # Slide null hypothesis to steep left wall (theta0 < 0): LM > LR > W
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{Steep left side (}\theta_0 < \hat{\theta}\text{): slope } S_n(\theta_0) \text{ grows fastest } \Rightarrow \mathrm{LM} > \mathrm{LR} > W",
            font_size=20,
            color=CYAN_HEX,
        )
        self.play(theta0_tr.animate.set_value(-0.68), run_time=2.2, rate_func=smooth)
        self.wait(1.4)


class MLEProjection3DScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_HEX
        self.set_camera_orientation(phi=62 * DEGREES, theta=-52 * DEGREES, zoom=1.18)

        title_row, sub_tex = make_hud(
            "Constrained MLE (3D):",
            r"\tilde{\theta} = \arg\max_{R(\theta)=r} \ell_n(\theta)",
            r"\text{Unrestricted MLE } \hat{\theta} \text{ sits at the peak; } H_0: \theta_1+\theta_2=r \text{ slices the surface}",
        )
        self.add_fixed_in_frame_mobjects(title_row, sub_tex)

        r_max = 1.45
        floor = Polygon(
            pt3(-r_max, -r_max, 0.0),
            pt3(r_max, -r_max, 0.0),
            pt3(r_max, r_max, 0.0),
            pt3(-r_max, r_max, 0.0),
            fill_color="#111827",
            fill_opacity=0.65,
            stroke_color=GRID_HEX,
            stroke_width=1.4,
        )
        grid = VGroup()
        for g in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            col = "#3f4c63" if abs(g) < 1e-6 else GRID_HEX
            grid.add(Line(pt3(-r_max, g, 0.0), pt3(r_max, g, 0.0), color=col, stroke_width=1.0))
            grid.add(Line(pt3(g, -r_max, 0.0), pt3(g, r_max, 0.0), color=col, stroke_width=1.0))

        t_hat = np.array([0.62, 0.52])
        # Fisher info with correlation rho = 0.55
        i11, i22, i12 = 1.95, 1.05, 0.55 * math.sqrt(1.95 * 1.05)
        det_i = i11 * i22 - i12 * i12
        inv_i = np.array([[i22, -i12], [-i12, i11]]) / det_i
        info = np.array([[i11, i12], [i12, i22]])

        def ll_height(t1: float, t2: float) -> float:
            d = np.array([t1 - t_hat[0], t2 - t_hat[1]])
            return max(0.06, 1.55 - 0.36 * float(d @ info @ d))

        hill = make_surface_mesh(
            ll_height,
            r_max=r_max,
            n_grid=13,
            fill_color=INDIGO_HEX,
            fill_opacity=0.22,
            stroke_color=INDIGO_LIGHT_HEX,
            stroke_width=1.05,
        )

        # Unrestricted summit & floor point
        p_hat_top = pt3(t_hat[0], t_hat[1], ll_height(t_hat[0], t_hat[1]))
        p_hat_fl = pt3(t_hat[0], t_hat[1], 0.0)
        dot_hat_top = Dot3D(p_hat_top, radius=0.075, color=TEAL_HEX, resolution=(8, 8))
        dot_hat_fl = Dot3D(p_hat_fl, radius=0.065, color=TEAL_HEX, resolution=(8, 8))
        drop_hat = make_dashed_3d(p_hat_top, p_hat_fl, color=TEAL_HEX, num_dashes=7)
        lbl_hat = pill_tex(r"\hat{\theta}", p_hat_top + np.array([0.18, 0.16, 0.22]), TEAL_HEX, font_size=24)
        self.add_fixed_orientation_mobjects(lbl_hat)

        self.play(FadeIn(floor), FadeIn(grid), run_time=0.55)
        self.play(Create(hill), run_time=1.3, rate_func=smooth)
        self.play(FadeIn(dot_hat_top), FadeIn(dot_hat_fl), Create(drop_hat), FadeIn(lbl_hat), run_time=0.7)

        # Solve constrained MLE for t1 + t2 = r_const
        r_const = 0.20
        G = np.array([1.0, 1.0])
        gap = float(G @ t_hat) - r_const
        g_inv_g = float(G @ inv_i @ G)
        t_tilde = t_hat - (inv_i @ G) * (gap / g_inv_g)
        t_euc = t_hat - 0.5 * gap * G

        # Constraint wall & ridge
        xs_c = np.linspace(max(-r_max, r_const - r_max), min(r_max, r_const + r_max), 34)
        ridge_pts = [pt3(float(x), float(r_const - x), ll_height(float(x), float(r_const - x))) for x in xs_c]
        floor_c_pts = [pt3(float(x), float(r_const - x), 0.0) for x in xs_c]

        ridge_curve = VMobject(color=AMBER_HEX, stroke_width=4.0)
        ridge_curve.set_points_as_corners(ridge_pts)
        floor_c_line = Line(floor_c_pts[0], floor_c_pts[-1], color=AMBER_HEX, stroke_width=2.6)

        wall_quads = VGroup()
        for idx in range(len(xs_c) - 1):
            wall_quads.add(
                Polygon(
                    floor_c_pts[idx],
                    floor_c_pts[idx + 1],
                    ridge_pts[idx + 1],
                    ridge_pts[idx],
                    fill_color=AMBER_HEX,
                    fill_opacity=0.16,
                    stroke_opacity=0.0,
                )
            )

        p_til_top = pt3(t_tilde[0], t_tilde[1], ll_height(t_tilde[0], t_tilde[1]))
        p_til_fl = pt3(t_tilde[0], t_tilde[1], 0.0)
        p_til_at_hat = pt3(t_tilde[0], t_tilde[1], ll_height(t_hat[0], t_hat[1]))

        dot_til_top = Dot3D(p_til_top, radius=0.075, color=AMBER_HEX, resolution=(8, 8))
        dot_til_fl = Dot3D(p_til_fl, radius=0.068, color=AMBER_HEX, resolution=(8, 8))
        drop_til = make_dashed_3d(p_til_top, p_til_fl, color=AMBER_HEX, num_dashes=6)
        lr_bridge = make_dashed_3d(p_hat_top, p_til_at_hat, color=INDIGO_LIGHT_HEX, num_dashes=7)
        lr_drop = Line(p_til_at_hat, p_til_top, color=INDIGO_LIGHT_HEX, stroke_width=3.6)

        lbl_til = pill_tex(r"\tilde{\theta}", p_til_fl + np.array([-0.26, -0.24, 0.08]), AMBER_HEX, font_size=24)
        lbl_lr3d = pill_tex(r"\tfrac{1}{2}\mathrm{LR}", (p_til_at_hat + p_til_top) * 0.5 + np.array([-0.44, 0.12, 0.08]), INDIGO_LIGHT_HEX, font_size=21)
        self.add_fixed_orientation_mobjects(lbl_til, lbl_lr3d)

        sub_tex = swap_sub_3d(
            self,
            title_row,
            sub_tex,
            r"\text{Constraint slice } R(\theta)=r \text{ cuts the hill; } \tilde{\theta} \text{ maximizes the ridge and } \mathrm{LR}=2[\ell(\hat{\theta})-\ell(\tilde{\theta})]",
            font_size=19,
            color=AMBER_HEX,
        )
        self.play(FadeIn(wall_quads), Create(floor_c_line), Create(ridge_curve), run_time=1.1, rate_func=smooth)
        self.play(FadeIn(dot_til_top), FadeIn(dot_til_fl), Create(drop_til), FadeIn(lbl_til), run_time=0.6)
        self.play(Create(lr_bridge), Create(lr_drop), FadeIn(lbl_lr3d), run_time=0.75)

        # Draw tangent ellipse passing through tilde + I-projection vs Euclidean projection on floor
        d_til = t_tilde - t_hat
        q_level = float(d_til @ info @ d_til)
        ell_pts = []
        for k in range(65):
            ang = (k / 64.0) * 2.0 * PI
            u = np.array([math.cos(ang), math.sin(ang)])
            rad = math.sqrt(q_level / float(u @ info @ u))
            p = t_hat + rad * u
            ell_pts.append(pt3(float(p[0]), float(p[1]), 0.02))
        floor_ellipse = VMobject(color=INDIGO_LIGHT_HEX, stroke_width=2.6)
        floor_ellipse.set_points_as_corners(ell_pts)

        p_euc_fl = pt3(t_euc[0], t_euc[1], 0.02)
        proj_euc = make_dashed_3d(p_hat_fl, p_euc_fl, color=MUTED_HEX, num_dashes=6, stroke_width=2.0)
        proj_info = Line(p_hat_fl, p_til_fl, color=TEAL_HEX, stroke_width=3.0)
        dot_euc = Dot3D(p_euc_fl, radius=0.055, color=MUTED_HEX, resolution=(8, 8))

        # Score vector at tilde (Euclidean perpendicular to R(theta)=r!)
        score_dir = np.array([1.0, 1.0]) / math.sqrt(2.0) * 0.55
        p_score_end = pt3(t_tilde[0] + score_dir[0], t_tilde[1] + score_dir[1], 0.02)
        score_vec = Line(p_til_fl, p_score_end, color=CYAN_HEX, stroke_width=3.6)
        lbl_score = pill_tex(r"\nabla\ell(\tilde{\theta}) \perp R", p_score_end + np.array([0.24, 0.22, 0.06]), CYAN_HEX, font_size=20)
        self.add_fixed_orientation_mobjects(lbl_score)

        sub_tex = swap_sub_3d(
            self,
            title_row,
            sub_tex,
            r"\text{On the floor: } \tilde{\theta} \text{ is the } \mathcal{I}\text{-metric projection of } \hat{\theta} \text{ (not Euclidean), while } \nabla\ell(\tilde{\theta}) \perp \text{constraint}",
            font_size=19,
            color=CYAN_HEX,
        )
        self.play(Create(floor_ellipse), Create(proj_euc), FadeIn(dot_euc), Create(proj_info), run_time=1.0)
        self.play(Create(score_vec), FadeIn(lbl_score), run_time=0.65)

        self.begin_ambient_camera_rotation(rate=0.22)
        self.wait(3.2)
        self.stop_ambient_camera_rotation()
        self.wait(0.6)


def render_all(quality: str, which: str = "all") -> None:
    root_dir = Path(__file__).resolve().parent.parent
    out_dir = root_dir / "assets" / "media"
    candidate_media = root_dir / "visuals" / "media"
    media_dir = (
        candidate_media
        if str(candidate_media).isascii()
        else Path(tempfile.gettempdir()) / "manim_mle_tests"
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
        ("trinity", "MLETrinityScene", "mle-trinity-geometry.mp4"),
        ("proj3d", "MLEProjection3DScene", "mle-2d-projection.mp4"),
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
    parser = argparse.ArgumentParser(description="Render MLE & Wald/LR/LM Manim videos")
    parser.add_argument("--quality", choices=["low", "high"], default="high")
    parser.add_argument("--scene", choices=["all", "trinity", "proj3d"], default="all")
    args = parser.parse_args()
    render_all(args.quality, args.scene)
