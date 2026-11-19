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

XY_SCALE = 1.28
Z_FLOOR = -0.86
Z_SCALE = 1.06


def pt3(t1: float, t2: float, z_val: float) -> np.ndarray:
    return np.array([t1 * XY_SCALE, t2 * XY_SCALE, Z_FLOOR + z_val * Z_SCALE], dtype=float)


def make_arrow_3d(p0: np.ndarray, p1: np.ndarray, color: str, stroke_width: float = 4.2, tip_len: float = 0.20, tip_rad: float = 0.065) -> VGroup:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    length = float(np.linalg.norm(v))
    if length < 1e-4:
        return VGroup()
    u = v / length
    tl = min(tip_len, 0.38 * length)
    tr = tip_rad * (tl / tip_len)
    base_pt = p1 - tl * u
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    perp1 = np.cross(u, ref)
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(u, perp1)
    shaft = Line(p0, base_pt + 0.015 * u, color=color, stroke_width=stroke_width)
    ring = [base_pt + tr * (math.cos(2 * PI * k / 6) * perp1 + math.sin(2 * PI * k / 6) * perp2) for k in range(6)]
    cones = VGroup(*[
        Polygon(ring[k], ring[(k + 1) % 6], p1, fill_color=color, fill_opacity=0.95, stroke_width=0.0)
        for k in range(6)
    ])
    return VGroup(shaft, cones)


def make_right_angle_floor(corner: np.ndarray, d1: np.ndarray, d2: np.ndarray, size: float = 0.17, color: str = CYAN_HEX) -> VMobject:
    u1 = d1 / max(1e-6, float(np.linalg.norm(d1))) * size
    u2 = d2 / max(1e-6, float(np.linalg.norm(d2))) * size
    sq = VMobject(color=color, stroke_width=2.2)
    sq.set_points_as_corners([corner + u1, corner + u1 + u2, corner + u2])
    return sq


def pill_tex(tex_str: str, pos: np.ndarray, color: str, font_size: int = 24, use_bg: bool = True) -> MathTex:
    lbl = MathTex(tex_str, font_size=font_size, color=color).move_to(pos)
    if use_bg:
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
            r"\text{Population curve } \mathbb{E}[\ell(\theta)] \text{ peaks at } \theta_0; \text{ sample noise shifts } \ell_n(\theta) \text{ peak to } \hat{\theta}_n",
        )
        self.add(title_row, sub_tex)

        ax = Axes(
            x_range=[-1.15, 1.35, 0.5],
            y_range=[-3.6, 0.92, 1.0],
            x_length=9.2,
            y_length=4.9,
            axis_config={"color": MUTED_HEX, "stroke_width": 1.6, "include_tip": True},
        ).shift(DOWN * 0.56)

        x_lbl = MathTex(r"\theta", font_size=26, color=INK_HEX).next_to(ax.x_axis, RIGHT, buff=0.14)
        y_lbl = MathTex(r"\ell_n(\theta) - \ell_n(\hat{\theta}_n)", font_size=22, color=INK_HEX).next_to(ax.y_axis, UP, buff=0.12)

        skew_tr = ValueTracker(0.0)     # 0 = quadratic, 1 = exponential family
        curv_tr = ValueTracker(2.6)     # curvature at theta0 = 0
        thetahat_tr = ValueTracker(0.72)  # sample MLE peak (theta0 = 0 fixed as null / true center)

        a_exp = 0.65

        def eval_pop(t: float) -> float:
            # Population expected log-likelihood always peaks at true parameter theta_0 = 0
            c = curv_tr.get_value()
            w = skew_tr.get_value()
            q = -0.5 * c * t * t
            mu = max(0.24, 1.0 + a_exp * t)
            sk = (c / (a_exp * a_exp)) * (-math.log(mu) - 1.0 / mu + 1.0)
            return (1.0 - w) * q + w * sk

        def eval_ll(t: float) -> float:
            c = curv_tr.get_value()
            w = skew_tr.get_value()
            th = thetahat_tr.get_value()
            q = -0.5 * c * (t - th) * (t - th)
            mu = max(0.24, 1.0 + a_exp * t)
            mu_hat = max(0.24, 1.0 + a_exp * th)
            sk = (c / (a_exp * a_exp)) * (math.log(mu_hat / mu) - (mu_hat / mu) + 1.0)
            return (1.0 - w) * q + w * sk

        def eval_score(t: float) -> float:
            c = curv_tr.get_value()
            w = skew_tr.get_value()
            th = thetahat_tr.get_value()
            dq = -c * (t - th)
            mu = max(0.24, 1.0 + a_exp * t)
            mu_hat = max(0.24, 1.0 + a_exp * th)
            dsk = (c / a_exp) * (mu_hat - mu) / (mu * mu)
            return (1.0 - w) * dq + w * dsk

        def eval_wald_parabola(t: float) -> float:
            c = curv_tr.get_value()
            w = skew_tr.get_value()
            th = thetahat_tr.get_value()
            mu_hat = max(0.24, 1.0 + a_exp * th)
            i_hat = (1.0 - w) * c + w * (c / (mu_hat * mu_hat))
            return -0.5 * i_hat * (t - th) * (t - th)

        def valid_x_bounds(fn):
            lo = -1.04
            while lo < -0.05 and fn(lo) < -3.45:
                lo += 0.02
            hi = 1.25
            while hi > 0.05 and fn(hi) < -3.45:
                hi -= 0.02
            return [lo, hi]

        pop_curve = always_redraw(
            lambda: DashedLine(ax.c2p(0, 0), ax.c2p(0, 0), color=TEAL_HEX)
            if False
            else ax.plot(
                eval_pop,
                x_range=valid_x_bounds(eval_pop),
                color=TEAL_HEX,
                stroke_width=2.1,
            ).set_stroke(opacity=0.45)
        )
        ll_curve = always_redraw(
            lambda: ax.plot(
                eval_ll,
                x_range=valid_x_bounds(eval_ll),
                color=INDIGO_LIGHT_HEX,
                stroke_width=3.6,
            )
        )
        wald_parab = always_redraw(
            lambda: DashedLine(ax.c2p(0, 0), ax.c2p(0, 0), color=AMBER_HEX)
            if skew_tr.get_value() < 0.02
            else ax.plot(
                eval_wald_parabola,
                x_range=valid_x_bounds(eval_wald_parabola),
                color=AMBER_HEX,
                stroke_width=2.1,
            ).set_stroke(opacity=0.72)
        )

        # True / null parameter theta_0 is fixed at 0 (maximum of population curve!)
        dot_pop = Dot(ax.c2p(0.0, 0.0), color=TEAL_HEX, radius=0.075)
        lbl_pop0 = pill_tex(r"\theta_0\text{ (True)}", ax.c2p(-0.22, 0.24), TEAL_HEX, font_size=20)

        dot_hat = always_redraw(lambda: Dot(ax.c2p(thetahat_tr.get_value(), 0.0), color=INDIGO_LIGHT_HEX, radius=0.085))
        lbl_hat = always_redraw(
            lambda: pill_tex(
                r"\hat{\theta}_n",
                ax.c2p(thetahat_tr.get_value() + (0.14 if thetahat_tr.get_value() >= 0 else -0.14), 0.24),
                INDIGO_LIGHT_HEX,
                font_size=23,
            )
        )

        self.play(FadeIn(ax), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.5)
        self.play(Create(pop_curve), FadeIn(dot_pop), FadeIn(lbl_pop0), run_time=0.85, rate_func=smooth)
        self.play(Create(ll_curve), FadeIn(dot_hat), FadeIn(lbl_hat), run_time=0.95, rate_func=smooth)

        # Evaluation on sample curve at theta_0 = 0
        dot_0_on_ll = always_redraw(lambda: Dot(ax.c2p(0.0, max(-3.45, eval_ll(0.0))), color=CYAN_HEX, radius=0.085))

        # 1. Wald: horizontal distance |thetaHat - theta_0|
        wald_bar = always_redraw(
            lambda: Line(
                ax.c2p(0.0, 0.44),
                ax.c2p(thetahat_tr.get_value(), 0.44),
                color=AMBER_HEX,
                stroke_width=3.2,
            )
        )
        wald_tick_l = always_redraw(lambda: Line(ax.c2p(0.0, 0.36), ax.c2p(0.0, 0.52), color=AMBER_HEX, stroke_width=2.4))
        wald_tick_r = always_redraw(lambda: Line(ax.c2p(thetahat_tr.get_value(), 0.36), ax.c2p(thetahat_tr.get_value(), 0.52), color=AMBER_HEX, stroke_width=2.4))
        wald_lbl = always_redraw(
            lambda: pill_tex(
                r"\text{Wald: } \hat{\mathcal{I}}_n(\hat{\theta}_n-\theta_0)^2",
                ax.c2p(thetahat_tr.get_value() * 0.5 + (0.28 if thetahat_tr.get_value() >= 0 else -0.28), 0.72),
                AMBER_HEX,
                font_size=19,
            )
        )

        # 2. LR: vertical drop at theta_0 from peak level (0) down to sample curve ell_n(theta_0)
        lr_line = always_redraw(
            lambda: Line(
                ax.c2p(0.0, 0.0),
                ax.c2p(0.0, max(-3.45, eval_ll(0.0))),
                color=INDIGO_LIGHT_HEX,
                stroke_width=3.6,
            )
        )
        lr_lbl = always_redraw(
            lambda: pill_tex(
                r"\tfrac{1}{2}\mathrm{LR} = \ell_n(\hat{\theta}_n)-\ell_n(\theta_0)",
                ax.c2p(-0.56 if thetahat_tr.get_value() >= 0 else 0.56, max(-3.0, eval_ll(0.0) * 0.52)),
                INDIGO_LIGHT_HEX,
                font_size=19,
            )
        )

        # 3. LM: tangent slope at theta_0 = 0
        def make_tangent():
            y0 = eval_ll(0.0)
            s0 = eval_score(0.0)
            dt = 0.25
            p1 = ax.c2p(-dt, max(-3.45, min(0.55, y0 - s0 * dt)))
            p2 = ax.c2p(dt, max(-3.45, min(0.55, y0 + s0 * dt)))
            return Line(p1, p2, color=CYAN_HEX, stroke_width=3.2)

        lm_tangent = always_redraw(make_tangent)
        lm_lbl = always_redraw(
            lambda: pill_tex(
                r"\text{LM: } S_n(\theta_0) = \ell_n'(\theta_0)",
                ax.c2p(0.45 if thetahat_tr.get_value() >= 0 else -0.45, max(-3.2, eval_ll(0.0) - 0.34)),
                CYAN_HEX,
                font_size=19,
            )
        )

        self.play(FadeIn(dot_0_on_ll), Create(wald_bar), FadeIn(wald_tick_l), FadeIn(wald_tick_r), FadeIn(wald_lbl), run_time=0.75)
        self.play(Create(lr_line), FadeIn(lr_lbl), run_time=0.65)
        self.play(Create(lm_tangent), FadeIn(lm_lbl), run_time=0.65)
        self.wait(0.55)

        # Under H0 true: as n grows, sample peak thetaHat_n -> theta_0 while curvature sharpens
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{Under } H_0\text{ true: as } n \uparrow,\; \hat{\theta}_n \xrightarrow{p} \theta_0 \text{ (population peak) while curvature } n\mathcal{I}_1 \uparrow",
            font_size=19,
            color=TEAL_HEX,
        )
        self.play(
            thetahat_tr.animate.set_value(0.32),
            curv_tr.animate.set_value(5.0),
            run_time=1.8,
            rate_func=smooth,
        )
        self.wait(0.45)

        # Return to moderate n & morph to skewed likelihood -> divergence!
        self.add(wald_parab)
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{Finite-sample skewness (}\hat{\theta}_n > \theta_0\text{): Wald parabola is flatter at } \hat{\theta}_n \Rightarrow W < \mathrm{LR} < \mathrm{LM}",
            font_size=19,
            color=AMBER_HEX,
        )
        self.play(
            thetahat_tr.animate.set_value(0.66),
            curv_tr.animate.set_value(2.8),
            skew_tr.animate.set_value(1.0),
            run_time=1.6,
            rate_func=smooth,
        )
        self.wait(0.6)

        # Negative sample fluctuation (thetaHat_n < theta_0): peak sits on steep left side -> W > LR > LM
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{Negative fluctuation (}\hat{\theta}_n < \theta_0\text{): peak curvature } \hat{\mathcal{I}}_n \text{ is sharper } \Rightarrow W > \mathrm{LR} > \mathrm{LM}",
            font_size=19,
            color=CYAN_HEX,
        )
        self.play(thetahat_tr.animate.set_value(-0.46), run_time=2.0, rate_func=smooth)
        self.wait(1.3)


class MLEProjection3DScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_HEX
        self.set_camera_orientation(phi=59 * DEGREES, theta=-40 * DEGREES, zoom=1.18)

        title_row, sub_tex = make_hud(
            "Constrained MLE (3D):",
            r"\tilde{\theta}_n = \arg\max_{R(\theta)=r} \ell_n(\theta)",
            r"\text{True } \theta_0 \text{ lies on } R(\theta)=r\text{; finite-sample noise shifts peak } \hat{\theta}_n \text{ off the wall}",
        )
        self.add_fixed_in_frame_mobjects(title_row, sub_tex)

        r_max = 1.45
        floor = Polygon(
            pt3(-r_max, -r_max, 0.0),
            pt3(r_max, -r_max, 0.0),
            pt3(r_max, r_max, 0.0),
            pt3(-r_max, r_max, 0.0),
            fill_color="#111827",
            fill_opacity=0.06,
            stroke_color=GRID_HEX,
            stroke_width=1.4,
        )
        grid = VGroup()
        for g in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            col = "#3f4c63" if abs(g) < 1e-6 else GRID_HEX
            grid.add(Line(pt3(-r_max, g, 0.0), pt3(r_max, g, 0.0), color=col, stroke_width=1.0))
            grid.add(Line(pt3(g, -r_max, 0.0), pt3(g, r_max, 0.0), color=col, stroke_width=1.0))

        lbl_ax1 = pill_tex(r"\theta_1", pt3(r_max + 0.18, r_max * 0.78, 0.03), MUTED_HEX, font_size=21, use_bg=True)
        lbl_ax2 = pill_tex(r"\theta_2", pt3(-r_max * 0.85, -r_max - 0.18, 0.03), MUTED_HEX, font_size=21, use_bg=True)

        # Constraint R(theta) = t1 + t2 = 0; true parameter theta_0 on constraint near center
        r_const = 0.0
        t_true = np.array([0.30, -0.30])      # 0.30 - 0.30 = 0
        delta_sample = np.array([-0.40, 1.30])  # at s=1.0: t_hat = (-0.10, +1.00), gap = +0.90

        # Anisotropic Fisher info (strong asymmetry i11 > i22, rho = 0.62) so I-projection differs sharply from Euclidean
        i11, i22 = 2.40, 0.85
        i12 = 0.62 * math.sqrt(i11 * i22)
        det_i = i11 * i22 - i12 * i12
        info = np.array([[i11, i12], [i12, i22]])
        inv_i = np.array([[i22, -i12], [-i12, i11]]) / det_i
        eigvals, eigvecs = np.linalg.eigh(info)
        # Matrix A_base such that 0.5 * k_base * d' info d = ||A_base^{-1} d||^2 -> dome cap z = z_peak * (1 - u^2)
        z_peak = 1.62
        k_base = 1.80
        inv_sqrt_h = eigvecs @ np.diag(1.0 / np.sqrt(0.5 * k_base * eigvals / z_peak)) @ eigvecs.T

        G = np.array([1.0, 1.0])
        g_inv_g = float(G @ inv_i @ G)

        scale_s = ValueTracker(1.0)  # s ~ n^{-1/2}: shrinks 1.0 -> 0.45 while curvature H_n = s^{-2} I grows as n

        def state_for(s: float):
            t_hat = t_true + s * delta_sample
            gap = float(G @ t_hat) - r_const
            t_tilde = t_hat - (inv_i @ G) * (gap / g_inv_g)
            t_euc = t_hat - 0.5 * gap * G
            d_til = t_tilde - t_hat
            # Vertical drop = 0.5 * k_base * s^{-2} * (d_til' info d_til) = O_p(1) invariant as n grows!
            half_lr = 0.5 * (k_base / (s * s)) * float(d_til @ info @ d_til)
            z_til = max(0.10, z_peak - half_lr)
            return t_hat, t_tilde, t_euc, z_til, half_lr

        # Parametric concave elliptic paraboloid dome mesh (n_rad rings x n_ang sectors)
        n_rad, n_ang = 8, 24
        u_vals = np.linspace(0.0, 1.0, n_rad + 1)
        ang_vals = np.linspace(0.0, 2.0 * PI, n_ang + 1)

        def build_dome_pts(t_hat: np.ndarray, s: float):
            pts = []
            for u in u_vals:
                row = []
                z_v = z_peak * (1.0 - u * u)
                for th in ang_vals:
                    offset = s * u * (inv_sqrt_h @ np.array([math.cos(th), math.sin(th)]))
                    p2 = t_hat + offset
                    row.append(pt3(float(p2[0]), float(p2[1]), z_v))
                pts.append(row)
            return pts

        t_hat0, t_til0, t_euc0, z_til0, _hlr0 = state_for(1.0)
        pts0 = build_dome_pts(t_hat0, 1.0)
        dome = VGroup()
        for ir in range(n_rad):
            op_stroke = 0.45 + 0.35 * (1.0 - ir / n_rad)
            for ia in range(n_ang):
                dome.add(
                    Polygon(
                        pts0[ir][ia],
                        pts0[ir + 1][ia],
                        pts0[ir + 1][ia + 1],
                        pts0[ir][ia + 1],
                        fill_color=INDIGO_HEX,
                        fill_opacity=0.17,
                        stroke_color=INDIGO_LIGHT_HEX,
                        stroke_opacity=op_stroke,
                        stroke_width=0.95,
                    )
                )

        # Constraint line on floor: t1 + t2 = 0 -> t2 = -t1
        xs_c = np.linspace(-1.30, 1.30, 48)
        floor_c_pts = [pt3(float(x), float(r_const - x), 0.02) for x in xs_c]
        floor_c_line = Line(floor_c_pts[0], floor_c_pts[-1], color=AMBER_HEX, stroke_width=3.0)

        # True parameter theta_0 fixed on constraint wall
        p_true_fl = pt3(float(t_true[0]), float(t_true[1]), 0.03)
        dot_true_fl = Dot3D(p_true_fl, radius=0.085, color=TEAL_HEX, resolution=(8, 8))
        p_true_lbl = pt3(0.76, -1.16, 0.08)
        pin_true = Line(p_true_fl, p_true_lbl + np.array([-0.14, 0.22, -0.02]), color=TEAL_HEX, stroke_width=2.1)
        lbl_true = pill_tex(r"\theta_0\text{ (True)}", p_true_lbl, TEAL_HEX, font_size=21, use_bg=True)

        # Unconstrained sample peak hat{theta}_n
        p_hat_top0 = pt3(float(t_hat0[0]), float(t_hat0[1]), z_peak)
        p_hat_fl0 = pt3(float(t_hat0[0]), float(t_hat0[1]), 0.025)
        dot_hat_top = Dot3D(p_hat_top0, radius=0.075, color=INDIGO_LIGHT_HEX, resolution=(8, 8))
        dot_hat_fl = Dot3D(p_hat_fl0, radius=0.068, color=INDIGO_LIGHT_HEX, resolution=(8, 8))
        drop_hat = make_dashed_3d(p_hat_top0, p_hat_fl0, color=INDIGO_LIGHT_HEX, num_dashes=7)
        lbl_hat = pill_tex(r"\hat{\theta}_n", p_hat_top0 + np.array([0.22, 0.18, 0.28]), INDIGO_LIGHT_HEX, font_size=24, use_bg=True)

        # Constraint ridge arch on the paraboloid (only inside the dome cap where z >= 0)
        def compute_ridge(t_hat: np.ndarray, s: float):
            pts_f, pts_top = [], []
            for x in np.linspace(-1.32, 1.32, 70):
                y = r_const - x
                d = np.array([x - t_hat[0], y - t_hat[1]])
                z_v = z_peak - 0.5 * (k_base / (s * s)) * float(d @ info @ d)
                if z_v >= 0.0:
                    pts_f.append(pt3(float(x), float(y), 0.02))
                    pts_top.append(pt3(float(x), float(y), float(z_v)))
            return pts_f, pts_top

        rf0, rt0 = compute_ridge(t_hat0, 1.0)
        ridge_curve = VMobject(color=AMBER_HEX, stroke_width=4.0)
        ridge_curve.set_points_as_corners(rt0)
        wall_poly = Polygon(*rf0, *reversed(rt0), fill_color=AMBER_HEX, fill_opacity=0.14, stroke_opacity=0.0)

        p_til_top0 = pt3(float(t_til0[0]), float(t_til0[1]), z_til0)
        p_til_fl0 = pt3(float(t_til0[0]), float(t_til0[1]), 0.025)
        p_til_at_hat0 = pt3(float(t_til0[0]), float(t_til0[1]), z_peak)

        dot_til_top = Dot3D(p_til_top0, radius=0.075, color=AMBER_HEX, resolution=(8, 8))
        dot_til_fl = Dot3D(p_til_fl0, radius=0.072, color=AMBER_HEX, resolution=(8, 8))
        drop_til = make_dashed_3d(p_til_top0, p_til_fl0, color=AMBER_HEX, num_dashes=6)
        lr_bridge = make_dashed_3d(p_hat_top0, p_til_at_hat0, color=INDIGO_LIGHT_HEX, num_dashes=7)
        lr_drop = Line(p_til_at_hat0, p_til_top0, color=INDIGO_LIGHT_HEX, stroke_width=3.8)
        lbl_til = pill_tex(r"\tilde{\theta}_n", p_til_top0 + np.array([-0.34, -0.22, 0.04]), AMBER_HEX, font_size=24, use_bg=True)
        lbl_lr3d = pill_tex(r"\tfrac{1}{2}\mathrm{LR}", (p_til_at_hat0 + p_til_top0) * 0.5 + np.array([-0.38, 0.22, 0.16]), INDIGO_LIGHT_HEX, font_size=22, use_bg=True)

        # Horizontal level-set ellipse at z = z_til AND its vertical projection onto the floor (tangent at tilde_n!)
        def build_level_ellipses(t_hat: np.ndarray, t_tilde: np.ndarray, z_level: float):
            d_til = t_tilde - t_hat
            q_val = max(0.015, float(d_til @ info @ d_til))
            pts_surf, pts_floor = [], []
            for k in range(57):
                ang = (k / 56.0) * 2.0 * PI
                u = np.array([math.cos(ang), math.sin(ang)])
                rad = math.sqrt(q_val / float(u @ info @ u))
                p = t_hat + rad * u
                pts_surf.append(pt3(float(p[0]), float(p[1]), float(z_level)))
                pts_floor.append(pt3(float(p[0]), float(p[1]), 0.025))
            return pts_surf, pts_floor

        es0, ef0 = build_level_ellipses(t_hat0, t_til0, z_til0)
        surf_ellipse = VMobject(color=CYAN_HEX, stroke_width=2.6)
        surf_ellipse.set_points_as_corners(es0)
        floor_ellipse = VMobject(color=INDIGO_LIGHT_HEX, stroke_width=2.6)
        floor_ellipse.set_points_as_corners(ef0)

        p_euc_fl0 = pt3(float(t_euc0[0]), float(t_euc0[1]), 0.025)
        proj_euc = make_dashed_3d(p_hat_fl0, p_euc_fl0, color=MUTED_HEX, num_dashes=6, stroke_width=2.2)
        proj_info = Line(p_hat_fl0, p_til_fl0, color=CYAN_HEX, stroke_width=3.2)
        dot_euc = Dot3D(p_euc_fl0, radius=0.055, color=MUTED_HEX, resolution=(8, 8))
        lbl_euc = pill_tex(r"\hat{\theta}_{\text{Euc}}", p_euc_fl0 + np.array([-0.40, 0.34, 0.04]), MUTED_HEX, font_size=19, use_bg=True)

        # Score vector along G = (1, 1) (Euclidean 90-deg orthogonal to constraint tangent (1, -1))
        score_u = np.array([1.0, 1.0]) / math.sqrt(2.0)
        tan_u = np.array([1.0, -1.0]) / math.sqrt(2.0)
        p_score_end0 = pt3(float(t_til0[0] + score_u[0] * 0.58), float(t_til0[1] + score_u[1] * 0.58), 0.025)
        score_arrow = make_arrow_3d(p_til_fl0, p_score_end0, color=CYAN_HEX, stroke_width=4.0, tip_len=0.18, tip_rad=0.058)
        right_sq = make_right_angle_floor(
            p_til_fl0,
            pt3(float(tan_u[0]), float(tan_u[1]), 0.0) - pt3(0.0, 0.0, 0.0),
            pt3(float(score_u[0]), float(score_u[1]), 0.0) - pt3(0.0, 0.0, 0.0),
            size=0.16,
            color=CYAN_HEX,
        )
        p_score_lbl = pt3(0.92, 0.52, 0.12)
        pin_score = make_dashed_3d(p_score_end0, p_score_lbl + np.array([-0.16, -0.06, 0.0]), color=CYAN_HEX, num_dashes=4, stroke_width=1.8)
        lbl_score = pill_tex(r"\nabla\ell_n(\tilde{\theta}_n) \perp \ker\nabla R", p_score_lbl, CYAN_HEX, font_size=19, use_bg=True)

        # Phase 1: Floor, constraint wall, theta_0, and concave quadratic paraboloid peaking at hat{theta}_n
        self.add_fixed_orientation_mobjects(lbl_ax1, lbl_ax2, lbl_true)
        self.play(FadeIn(floor), FadeIn(grid), FadeIn(lbl_ax1), FadeIn(lbl_ax2), Create(floor_c_line), FadeIn(dot_true_fl), Create(pin_true), FadeIn(lbl_true), run_time=0.8)
        self.play(Create(dome), run_time=1.0, rate_func=smooth)
        self.add_fixed_orientation_mobjects(lbl_hat)
        self.play(FadeIn(dot_hat_top), FadeIn(dot_hat_fl), Create(drop_hat), FadeIn(lbl_hat), run_time=0.55)

        # Phase 2: Constraint slice -> ridge vertex tilde{theta}_n & vertical drop 1/2 LR + tangent level ellipse
        sub_tex = swap_sub_3d(
            self,
            title_row,
            sub_tex,
            r"\text{Wall } R(\theta)=r \text{ cuts parabolic ridge at } \tilde{\theta}_n\text{; vertical drop is exact } \tfrac{1}{2}\mathrm{LR} = \ell_n(\hat{\theta}_n)-\ell_n(\tilde{\theta}_n)",
            font_size=18,
            color=AMBER_HEX,
        )
        self.play(FadeIn(wall_poly), Create(ridge_curve), run_time=0.85, rate_func=smooth)
        self.add_fixed_orientation_mobjects(lbl_til, lbl_lr3d)
        self.play(
            FadeIn(dot_til_top),
            FadeIn(dot_til_fl),
            Create(drop_til),
            FadeIn(lbl_til),
            Create(lr_bridge),
            Create(lr_drop),
            FadeIn(lbl_lr3d),
            Create(surf_ellipse),
            run_time=0.85,
        )

        # Phase 3: Drop level-set ellipse onto floor -> I-projection vs Euclidean + Lagrange 90° score
        sub_tex = swap_sub_3d(
            self,
            title_row,
            sub_tex,
            r"\text{Level ellipse touches wall at } \tilde{\theta}_n \text{ (}\mathcal{I}\text{-projection } \neq \text{ Euclidean), with } \nabla\ell_n(\tilde{\theta}_n) \perp \text{wall}",
            font_size=18,
            color=CYAN_HEX,
        )
        self.add_fixed_orientation_mobjects(lbl_euc, lbl_score)
        self.play(
            Create(floor_ellipse),
            Create(proj_euc),
            FadeIn(dot_euc),
            FadeIn(lbl_euc),
            Create(proj_info),
            FadeIn(score_arrow),
            Create(right_sq),
            Create(pin_score),
            FadeIn(lbl_score),
            run_time=1.0,
        )
        self.wait(0.65)

        # Phase 4: Honest n ↑ scaling (s ~ n^{-1/2}, curvature nI ~ s^{-2}):
        # Horizontal estimators converge (hat{theta}_n, tilde{theta}_n -> theta_0) while vertical LR = O_p(1) stays constant height!
        sub_tex = swap_sub_3d(
            self,
            title_row,
            sub_tex,
            r"\text{As } n \uparrow\; (\text{curvature } n\mathcal{I} \uparrow):\; \hat{\theta}_n, \tilde{\theta}_n \xrightarrow{p} \theta_0 \text{ while } \mathrm{LR}_n = O_p(1) \text{ keeps finite drop height}",
            font_size=17.5,
            color=TEAL_HEX,
        )

        def sync_scene(s: float) -> None:
            t_hat, t_tilde, t_euc, z_til, _ = state_for(s)
            pts_d = build_dome_pts(t_hat, s)
            idx_q = 0
            for ir in range(n_rad):
                for ia in range(n_ang):
                    dome[idx_q].set_points_as_corners([
                        pts_d[ir][ia], pts_d[ir + 1][ia], pts_d[ir + 1][ia + 1], pts_d[ir][ia + 1], pts_d[ir][ia]
                    ])
                    idx_q += 1

            rf, rt = compute_ridge(t_hat, s)
            ridge_curve.set_points_as_corners(rt)
            wall_poly.set_points_as_corners([*rf, *reversed(rt), rf[0]])

            p_ht = pt3(float(t_hat[0]), float(t_hat[1]), z_peak)
            p_hf = pt3(float(t_hat[0]), float(t_hat[1]), 0.025)
            p_tt = pt3(float(t_tilde[0]), float(t_tilde[1]), z_til)
            p_tf = pt3(float(t_tilde[0]), float(t_tilde[1]), 0.025)
            p_tah = pt3(float(t_tilde[0]), float(t_tilde[1]), z_peak)
            p_ef = pt3(float(t_euc[0]), float(t_euc[1]), 0.025)

            dot_hat_top.move_to(p_ht)
            dot_hat_fl.move_to(p_hf)
            drop_hat.become(make_dashed_3d(p_ht, p_hf, color=INDIGO_LIGHT_HEX, num_dashes=7))
            lbl_hat.move_to(p_ht + np.array([0.22, 0.18, 0.28]))

            dot_til_top.move_to(p_tt)
            dot_til_fl.move_to(p_tf)
            drop_til.become(make_dashed_3d(p_tt, p_tf, color=AMBER_HEX, num_dashes=6))
            lr_bridge.become(make_dashed_3d(p_ht, p_tah, color=INDIGO_LIGHT_HEX, num_dashes=7))
            lr_drop.put_start_and_end_on(p_tah, p_tt)
            lbl_til.move_to(p_tt + np.array([-0.34, -0.22, 0.04]))
            lbl_lr3d.move_to((p_tah + p_tt) * 0.5 + np.array([-0.38, 0.22, 0.16]))

            es, ef = build_level_ellipses(t_hat, t_tilde, z_til)
            surf_ellipse.set_points_as_corners(es)
            floor_ellipse.set_points_as_corners(ef)

            proj_euc.become(make_dashed_3d(p_hf, p_ef, color=MUTED_HEX, num_dashes=6, stroke_width=2.2))
            proj_info.put_start_and_end_on(p_hf, p_tf)
            dot_euc.move_to(p_ef)
            lbl_euc.move_to(p_ef + np.array([-0.40, 0.34, 0.04]))

            s_len = 0.58 * s
            p_se = pt3(float(t_tilde[0] + score_u[0] * s_len), float(t_tilde[1] + score_u[1] * s_len), 0.025)
            score_arrow.become(make_arrow_3d(p_tf, p_se, color=CYAN_HEX, stroke_width=4.0, tip_len=0.15, tip_rad=0.05))
            right_sq.become(
                make_right_angle_floor(
                    p_tf,
                    pt3(float(tan_u[0]), float(tan_u[1]), 0.0) - pt3(0.0, 0.0, 0.0),
                    pt3(float(score_u[0]), float(score_u[1]), 0.0) - pt3(0.0, 0.0, 0.0),
                    size=0.14 * max(0.65, s),
                    color=CYAN_HEX,
                )
            )
            pin_score.become(make_dashed_3d(p_se, p_score_lbl + np.array([-0.16, -0.06, 0.0]), color=CYAN_HEX, num_dashes=4, stroke_width=1.8))

        dome.add_updater(lambda _m: sync_scene(scale_s.get_value()))
        self.begin_ambient_camera_rotation(rate=0.11)
        self.play(
            scale_s.animate.set_value(0.45),
            run_time=2.3,
            rate_func=smooth,
        )
        dome.clear_updaters()
        self.wait(1.3)
        self.stop_ambient_camera_rotation()
        self.wait(0.4)


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
            sys.executable,
            "-m",
            "manim",
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
