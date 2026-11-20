"""
Manim animations for Econometrics Visual Supplement 04 (MLE & Wald/LR/LM):
  - MLETrinityScene       -> assets/media/mle-trinity-geometry.mp4
  - MLEProjection3DScene  -> assets/media/mle-2d-projection.mp4

Usage:
  .venv\\Scripts\\python.exe visuals\\render_mle_tests.py [--quality low|high] [--scene all|trinity|proj3d]
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
    LEFT,
    PI,
    RIGHT,
    UL,
    UR,
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

XY_SCALE = 1.24
Z_FLOOR = -1.02
Z_SCALE = 1.05


def pt3(t1: float, t2: float, z_val: float) -> np.ndarray:
    return np.array([t1 * XY_SCALE, t2 * XY_SCALE, Z_FLOOR + z_val * Z_SCALE], dtype=float)


def pill_tex(tex_str: str, pos: np.ndarray, color: str, font_size: int = 24, use_bg: bool = True) -> MathTex:
    lbl = MathTex(tex_str, font_size=font_size, color=color).move_to(pos)
    if use_bg:
        lbl.add_background_rectangle(color=BG_HEX, opacity=0.84, buff=0.06)
    return lbl


def make_hud(title_str: str, math_str: str, sub_str: str):
    title = Text(title_str, font="Segoe UI", font_size=22, weight="BOLD", color=INK_HEX)
    title_math = MathTex(math_str, font_size=23, color=INDIGO_LIGHT_HEX)
    title_row = VGroup(title, title_math).arrange(RIGHT, buff=0.16)
    title_row.to_corner(UL, buff=0.40)

    sub = MathTex(sub_str, font_size=19, color=MUTED_HEX)
    sub.next_to(title_row, DOWN, buff=0.13, aligned_edge=LEFT)
    return title_row, sub


def swap_sub(
    scene: Scene,
    title_row: VGroup,
    old_sub: MathTex,
    new_tex: str,
    font_size: int = 19,
    color: str = MUTED_HEX,
) -> MathTex:
    scene.play(FadeOut(old_sub, shift=UP * 0.08, run_time=0.22, rate_func=smooth))
    new_sub = MathTex(new_tex, font_size=font_size, color=color)
    new_sub.next_to(title_row, DOWN, buff=0.13, aligned_edge=LEFT)
    scene.play(FadeIn(new_sub, shift=UP * 0.08, run_time=0.24, rate_func=smooth))
    return new_sub


def swap_sub_3d(
    scene: ThreeDScene,
    title_row: VGroup,
    old_sub: MathTex,
    new_tex: str,
    font_size: int = 19,
    color: str = MUTED_HEX,
) -> MathTex:
    scene.play(FadeOut(old_sub, shift=UP * 0.08, run_time=0.22, rate_func=smooth))
    scene.remove_fixed_in_frame_mobjects(old_sub)
    new_sub = MathTex(new_tex, font_size=font_size, color=color)
    new_sub.next_to(title_row, DOWN, buff=0.13, aligned_edge=LEFT)
    scene.add_fixed_in_frame_mobjects(new_sub)
    scene.play(FadeIn(new_sub, shift=UP * 0.08, run_time=0.24, rate_func=smooth))
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


def xy_arrow_tip(start_2d: np.ndarray, end_2d: np.ndarray, z_val: float, color: str, size: float = 0.085) -> Polygon:
    diff = end_2d - start_2d
    norm = max(float(np.linalg.norm(diff)), 1e-6)
    u = diff / norm
    perp = np.array([-u[1], u[0]])
    p_tip = pt3(float(end_2d[0]), float(end_2d[1]), z_val)
    b1 = end_2d - size * u + 0.48 * size * perp
    b2 = end_2d - size * u - 0.48 * size * perp
    return Polygon(
        p_tip,
        pt3(float(b1[0]), float(b1[1]), z_val),
        pt3(float(b2[0]), float(b2[1]), z_val),
        fill_color=color,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=1.0,
    )


class MLETrinityScene(Scene):
    def construct(self):
        self.camera.background_color = BG_HEX

        title_row, sub_tex = make_hud(
            "Likelihood Trinity:",
            r"W_n \approx \mathrm{LR}_n \approx \mathrm{LM}_n \xrightarrow{d} \chi^2_r",
            r"\text{Population curve } \mathbb{E}[\ell(\theta)] \text{ peaks at } \theta_0;\; \text{sample noise shifts } \ell_n(\theta) \text{ peak to } \hat{\theta}_n",
        )
        self.add(title_row, sub_tex)

        ax = Axes(
            x_range=[-1.15, 1.35, 0.5],
            y_range=[-2.50, 0.88, 0.5],
            x_length=9.3,
            y_length=4.95,
            axis_config={"color": MUTED_HEX, "stroke_width": 1.6, "include_tip": True},
        ).shift(DOWN * 0.54)

        x_lbl = MathTex(r"\theta", font_size=26, color=INK_HEX).next_to(ax.x_axis, RIGHT, buff=0.14)
        y_lbl = MathTex(r"\ell_n(\theta) - \ell_n(\hat{\theta}_n)", font_size=21, color=INK_HEX).next_to(ax.y_axis, UP, buff=0.12)

        th_init = 0.68
        curv_init = 4.6
        a_exp = 0.65

        skew_tr = ValueTracker(0.0)       # 0 = quadratic, 1 = exponential-family skew
        curv_tr = ValueTracker(curv_init) # J_n curvature around peak / theta_0
        thetahat_tr = ValueTracker(th_init)
        n_scale_tr = ValueTracker(1.0)    # 1 -> 0.45 during n^ growth phase

        def eval_pop(t: float) -> float:
            c = curv_tr.get_value()
            w = skew_tr.get_value()
            q = -0.5 * c * t * t
            mu = max(0.22, 1.0 + a_exp * t)
            sk = (c / (a_exp * a_exp)) * (-math.log(mu) - 1.0 / mu + 1.0)
            return (1.0 - w) * q + w * sk

        def eval_ll(t: float) -> float:
            c = curv_tr.get_value()
            w = skew_tr.get_value()
            th = thetahat_tr.get_value()
            q = -0.5 * c * (t - th) * (t - th)
            mu = max(0.22, 1.0 + a_exp * t)
            mu_hat = max(0.22, 1.0 + a_exp * th)
            sk = (c / (a_exp * a_exp)) * (math.log(mu_hat / mu) - (mu_hat / mu) + 1.0)
            return (1.0 - w) * q + w * sk

        def eval_score(t: float) -> float:
            c = curv_tr.get_value()
            w = skew_tr.get_value()
            th = thetahat_tr.get_value()
            dq = -c * (t - th)
            mu = max(0.22, 1.0 + a_exp * t)
            mu_hat = max(0.22, 1.0 + a_exp * th)
            dsk = (c / a_exp) * (mu_hat - mu) / (mu * mu)
            return (1.0 - w) * dq + w * dsk

        def eval_wald_parabola(t: float) -> float:
            c = curv_tr.get_value()
            w = skew_tr.get_value()
            th = thetahat_tr.get_value()
            mu_hat = max(0.22, 1.0 + a_exp * th)
            i_hat = (1.0 - w) * c + w * (c / (mu_hat * mu_hat))
            return -0.5 * i_hat * (t - th) * (t - th)

        def valid_x_bounds(fn: Callable[[float], float]):
            lo = -1.06
            while lo < -0.04 and fn(lo) < -2.38:
                lo += 0.015
            hi = 1.26
            while hi > 0.04 and fn(hi) < -2.38:
                hi -= 0.015
            return [lo, hi]

        pop_curve = always_redraw(
            lambda: ax.plot(
                eval_pop,
                x_range=valid_x_bounds(eval_pop),
                color=TEAL_HEX,
                stroke_width=2.1,
            ).set_stroke(opacity=0.48)
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
                stroke_width=2.2,
            ).set_stroke(opacity=0.80)
        )

        dot_pop = Dot(ax.c2p(0.0, 0.0), color=TEAL_HEX, radius=0.075)
        lbl_pop0 = pill_tex(r"\theta_0\text{ (True)}", ax.c2p(-0.26, 0.24), TEAL_HEX, font_size=20)

        dot_hat = always_redraw(lambda: Dot(ax.c2p(thetahat_tr.get_value(), 0.0), color=INDIGO_LIGHT_HEX, radius=0.085))
        lbl_hat = always_redraw(
            lambda: pill_tex(
                r"\hat{\theta}_n",
                ax.c2p(thetahat_tr.get_value() + (0.13 if thetahat_tr.get_value() >= 0 else -0.13), 0.24),
                INDIGO_LIGHT_HEX,
                font_size=23,
            )
        )

        # Compact live ordering badge at UR corner (pre-cached inequality MathTex + fast monospace Text readout)
        badge_titles = {
            "eq": MathTex(r"W_n = \mathrm{LR}_n = \mathrm{LM}_n", font_size=21, color=TEAL_HEX),
            "lt": MathTex(r"W_n < \mathrm{LR}_n < \mathrm{LM}_n", font_size=21, color=AMBER_HEX),
            "gt": MathTex(r"W_n > \mathrm{LR}_n > \mathrm{LM}_n", font_size=21, color=CYAN_HEX),
        }

        def make_order_badge():
            w = skew_tr.get_value()
            y_lr = -eval_ll(0.0)
            y_w = -eval_wald_parabola(0.0)
            s0 = eval_score(0.0)
            c = curv_tr.get_value()
            y_lm = 0.5 * (s0 * s0) / max(c, 1e-4)
            if w < 0.05:
                row1 = badge_titles["eq"].copy()
            elif y_w < y_lr - 0.02:
                row1 = badge_titles["lt"].copy()
            else:
                row1 = badge_titles["gt"].copy()
            row2 = Text(
                f"½W={y_w:.2f} · ½LR={y_lr:.2f} · ½LM={y_lm:.2f}",
                font="Consolas",
                font_size=14,
                color=MUTED_HEX,
            )
            grp = VGroup(row1, row2).arrange(DOWN, buff=0.09, aligned_edge=RIGHT)
            grp.to_corner(UR, buff=0.40)
            grp.add_background_rectangle(color="#111827", opacity=0.86, buff=0.12)
            return grp

        order_badge = always_redraw(make_order_badge)

        self.play(FadeIn(ax), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.48)
        self.play(Create(pop_curve), FadeIn(dot_pop), FadeIn(lbl_pop0), run_time=0.82, rate_func=smooth)
        self.play(Create(ll_curve), FadeIn(dot_hat), FadeIn(lbl_hat), FadeIn(order_badge), run_time=0.90, rate_func=smooth)

        # Sample evaluations at theta_0 = 0
        dot_0_on_ll = always_redraw(lambda: Dot(ax.c2p(0.0, max(-2.38, eval_ll(0.0))), color=CYAN_HEX, radius=0.085))
        dot_w_on_parab = always_redraw(
            lambda: Dot(ax.c2p(0.0, max(-2.38, eval_wald_parabola(0.0))), color=AMBER_HEX, radius=0.075)
            if skew_tr.get_value() >= 0.04
            else Dot(ax.c2p(0.0, 0.0), radius=0.001, fill_opacity=0.0)
        )

        # 1. Wald horizontal bracket
        wald_bar = always_redraw(
            lambda: Line(ax.c2p(0.0, 0.44), ax.c2p(thetahat_tr.get_value(), 0.44), color=AMBER_HEX, stroke_width=3.2)
        )
        wald_tick_l = always_redraw(lambda: Line(ax.c2p(0.0, 0.36), ax.c2p(0.0, 0.52), color=AMBER_HEX, stroke_width=2.4))
        wald_tick_r = always_redraw(lambda: Line(ax.c2p(thetahat_tr.get_value(), 0.36), ax.c2p(thetahat_tr.get_value(), 0.52), color=AMBER_HEX, stroke_width=2.4))
        wald_lbl = always_redraw(
            lambda: pill_tex(
                r"\text{Wald: } \hat{J}_n(\hat{\theta}_n-\theta_0)^2",
                ax.c2p(thetahat_tr.get_value() * 0.5 + (0.28 if thetahat_tr.get_value() >= 0 else -0.28), 0.72),
                AMBER_HEX,
                font_size=19,
            )
        )

        # 2. LR vertical drop at theta_0 = 0
        lr_line = always_redraw(
            lambda: Line(ax.c2p(0.0, 0.0), ax.c2p(0.0, max(-2.38, eval_ll(0.0))), color=INDIGO_LIGHT_HEX, stroke_width=3.8)
        )
        # Keep 1/2 LR pill clean of x-axis ticks and cleanly separated from 1/2 W
        lr_lbl = always_redraw(
            lambda: pill_tex(
                r"\tfrac{1}{2}\mathrm{LR}_n",
                ax.c2p(
                    -0.24 if thetahat_tr.get_value() >= 0 else 0.25,
                    min(-0.56, max(-2.05, eval_ll(0.0) * (0.90 if skew_tr.get_value() >= 0.15 and thetahat_tr.get_value() >= 0 else 0.74))),
                ),
                INDIGO_LIGHT_HEX,
                font_size=21,
            )
        )

        # Amber 1/2 W callout when skew turns on (placed on opposite height band from 1/2 LR)
        w_drop_lbl = always_redraw(
            lambda: pill_tex(
                r"\tfrac{1}{2}W_n",
                ax.c2p(
                    -0.22 if thetahat_tr.get_value() >= 0 else -0.22,
                    max(-2.12, eval_wald_parabola(0.0) + (0.12 if thetahat_tr.get_value() >= 0 else -0.02)),
                ),
                AMBER_HEX,
                font_size=20,
            )
            if skew_tr.get_value() >= 0.15
            else Dot(ax.c2p(0.0, 0.0), radius=0.001, fill_opacity=0.0)
        )

        # 3. LM tangent slope at theta_0 = 0
        def make_tangent():
            y0 = eval_ll(0.0)
            s0 = eval_score(0.0)
            dt = 0.24
            p1 = ax.c2p(-dt, max(-2.38, min(0.55, y0 - s0 * dt)))
            p2 = ax.c2p(dt, max(-2.38, min(0.55, y0 + s0 * dt)))
            return Line(p1, p2, color=CYAN_HEX, stroke_width=3.2)

        lm_tangent = always_redraw(make_tangent)
        lm_lbl = always_redraw(
            lambda: pill_tex(
                r"\text{LM: } S_n(\theta_0) = \ell_n'(\theta_0)",
                ax.c2p(0.45 if thetahat_tr.get_value() >= 0 else -0.56, max(-2.22, eval_ll(0.0) - (0.36 if thetahat_tr.get_value() >= 0 else 0.76))),
                CYAN_HEX,
                font_size=19,
            )
        )

        self.play(FadeIn(dot_0_on_ll), Create(wald_bar), FadeIn(wald_tick_l), FadeIn(wald_tick_r), FadeIn(wald_lbl), run_time=0.72)
        self.play(Create(lr_line), FadeIn(lr_lbl), run_time=0.62)
        self.play(Create(lm_tangent), FadeIn(lm_lbl), run_time=0.62)
        self.wait(0.55)

        # Under H0 true: as n grows, sample peak thetaHat_n -> theta_0 at O_p(n^{-1/2})
        # while curvature n J_1 grows as O(n), holding 1/2 LR = O_p(1) invariant!
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{As } n\uparrow \text{ under } H_0:\; \hat{\theta}_n-\theta_0 = O_p(n^{-1/2}) \text{ while } J_n = nJ_1\uparrow,\; \tfrac{1}{2}\mathrm{LR}_n = O_p(1)\text{ stays finite}",
            font_size=18,
            color=TEAL_HEX,
        )

        def sync_n_growth(_mob):
            s = n_scale_tr.get_value()
            thetahat_tr.set_value(th_init * s)
            curv_tr.set_value(curv_init / (s * s))

        ax.add_updater(sync_n_growth)
        self.play(n_scale_tr.animate.set_value(0.45), run_time=2.0, rate_func=smooth)
        ax.remove_updater(sync_n_growth)
        self.wait(0.55)

        # Moderate finite sample + exponential-family skew -> Wald < LR < LM
        self.add(wald_parab, dot_w_on_parab, w_drop_lbl)
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{Finite-sample skewness (}\hat{\theta}_n > \theta_0\text{): quadratic Wald parabola flattens at } \hat{\theta}_n \Rightarrow W_n < \mathrm{LR}_n < \mathrm{LM}_n",
            font_size=18,
            color=AMBER_HEX,
        )
        self.play(
            thetahat_tr.animate.set_value(0.64),
            curv_tr.animate.set_value(5.4),
            skew_tr.animate.set_value(1.0),
            run_time=1.6,
            rate_func=smooth,
        )
        self.wait(0.75)

        # Negative sample fluctuation -> steeper peak curvature J_hat -> W > LR > LM
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{Negative fluctuation (}\hat{\theta}_n < \theta_0\text{): peak curvature } \hat{J}_n \text{ sharpens } \Rightarrow W_n > \mathrm{LR}_n > \mathrm{LM}_n",
            font_size=18,
            color=CYAN_HEX,
        )
        self.play(thetahat_tr.animate.set_value(-0.44), run_time=2.0, rate_func=smooth)
        self.wait(1.3)


class MLEProjection3DScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_HEX
        self.set_camera_orientation(phi=61 * DEGREES, theta=-28 * DEGREES, zoom=1.30)

        title_row, sub_tex = make_hud(
            "Constrained MLE (3D):",
            r"\tilde{\theta}_n = \arg\max_{g(\theta)=0}\ell_n(\theta)",
            r"\text{Constraint } g(\theta)=\theta_1+\theta_2=0 \text{ (}G=\nabla g\text{)};\; \text{under } H_0,\; \theta_0 \text{ lies on } g(\theta)=0",
        )
        self.add_fixed_in_frame_mobjects(title_row, sub_tex)

        r_max = 1.50
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

        # Constraint g(theta) = theta_1 + theta_2 = 0 (G = [1, 1]) and true parameter theta_0
        t_true = np.array([0.70, -0.70])
        delta_sample = np.array([-0.35, 0.95])  # initial unconstrained t_hat = (0.35, 0.25)

        # Positive-definite population information matrix J (Ch 05 §5.20 notation)
        j11, j22, j12 = 2.4, 1.3, 1.05
        info_J = np.array([[j11, j12], [j12, j22]], dtype=float)
        inv_J = np.linalg.inv(info_J)
        L_chol = np.linalg.cholesky(info_J)  # J = L L^T -> d = r_max * (L^T)^{-1} [cos a, sin a]
        inv_LT = np.linalg.inv(L_chol.T)

        G_vec = np.array([1.0, 1.0])
        g_invJ_g = float(G_vec @ inv_J @ G_vec)

        z_peak = 1.62
        curv_0 = 2.467  # yields Delta z = 1/2 LR = 0.56 -> z_til = 1.06

        def state_for(s: float):
            t_hat = t_true + s * delta_sample
            gap = float(G_vec @ t_hat)
            t_tilde = t_hat - (inv_J @ G_vec) * (gap / g_invJ_g)
            t_euc = t_hat - 0.5 * gap * G_vec
            c_mult = curv_0 / (s * s)
            return t_hat, t_tilde, t_euc, c_mult

        def ll_height(t1: float, t2: float, t_hat: np.ndarray, c_mult: float) -> float:
            d = np.array([t1 - t_hat[0], t2 - t_hat[1]])
            return max(0.0, z_peak - 0.5 * c_mult * float(d @ info_J @ d))

        floor_c_line = Line(pt3(-r_max, r_max, 0.0), pt3(r_max, -r_max, 0.0), color=AMBER_HEX, stroke_width=2.8)

        # True parameter theta_0 on constraint line (open foreground, pill offset off the line)
        p_true_fl = pt3(float(t_true[0]), float(t_true[1]), 0.04)
        dot_true_fl = Dot3D(p_true_fl, radius=0.088, color=TEAL_HEX, resolution=(8, 8))
        p_true_lbl = pt3(1.22, -1.26, 0.14)
        pin_true = Line(p_true_fl, p_true_lbl + np.array([-0.16, 0.14, -0.04]), color=TEAL_HEX, stroke_width=2.1)
        lbl_true = pill_tex(r"\theta_0\text{ (True)}", p_true_lbl, TEAL_HEX, font_size=20)

        t_hat0, t_til0, t_euc0, c0 = state_for(1.0)
        z_hat0 = ll_height(t_hat0[0], t_hat0[1], t_hat0, c0)
        z_til0 = ll_height(t_til0[0], t_til0[1], t_hat0, c0)

        # Polar J-elliptic paraboloid dome mesh (clean exact floor edge at z = 0)
        n_rad, n_ang = 8, 24

        def make_dome_quads() -> VGroup:
            q_grp = VGroup()
            for ir in range(n_rad):
                for ia in range(n_ang):
                    q = Polygon(
                        ORIGIN := pt3(0.0, 0.0, 0.0),
                        ORIGIN,
                        ORIGIN,
                        ORIGIN,
                        fill_color=INDIGO_HEX,
                        fill_opacity=0.22,
                        stroke_color=INDIGO_LIGHT_HEX,
                        stroke_opacity=0.58,
                        stroke_width=1.05,
                    )
                    q_grp.add(q)
            return q_grp

        dome_mesh = make_dome_quads()

        def update_dome_quads(t_hat: np.ndarray, c_mult: float) -> None:
            r_J = math.sqrt(2.0 * z_peak / c_mult)
            idx_q = 0
            angs = [2.0 * PI * ia / n_ang for ia in range(n_ang + 1)]
            rings = []
            for ir in range(n_rad + 1):
                u_r = ir / n_rad
                z_r = z_peak * (1.0 - u_r * u_r)
                ring_pts = []
                for ang in angs:
                    unit_circle = np.array([math.cos(ang), math.sin(ang)])
                    p_2d = t_hat + (r_J * u_r) * (inv_LT @ unit_circle)
                    ring_pts.append(pt3(float(p_2d[0]), float(p_2d[1]), z_r))
                rings.append(ring_pts)

            for ir in range(n_rad):
                u_mid = (ir + 0.5) / n_rad
                fade = 1.0 - 0.45 * (u_mid ** 2)
                for ia in range(n_ang):
                    q = dome_mesh[idx_q]
                    q.set_points_as_corners([
                        rings[ir][ia],
                        rings[ir + 1][ia],
                        rings[ir + 1][ia + 1],
                        rings[ir][ia + 1],
                        rings[ir][ia],
                    ])
                    q.set_fill(opacity=0.24 * fade)
                    q.set_stroke(opacity=0.62 * fade)
                    idx_q += 1

        update_dome_quads(t_hat0, c0)

        p_hat_top0 = pt3(float(t_hat0[0]), float(t_hat0[1]), z_hat0)
        p_hat_fl0 = pt3(float(t_hat0[0]), float(t_hat0[1]), 0.0)
        dot_hat_top = Dot3D(p_hat_top0, radius=0.075, color=INDIGO_LIGHT_HEX, resolution=(8, 8))
        dot_hat_fl = Dot3D(p_hat_fl0, radius=0.064, color=INDIGO_LIGHT_HEX, resolution=(8, 8))
        drop_hat = make_dashed_3d(p_hat_top0, p_hat_fl0, color=INDIGO_LIGHT_HEX, num_dashes=8)
        lbl_hat = pill_tex(r"\hat{\theta}_n", p_hat_top0 + np.array([0.16, 0.24, 0.30]), INDIGO_LIGHT_HEX, font_size=23)

        # Exact active constraint ridge along g(theta)=0 -> (u, -u) where z(u, -u) >= 0
        n_ridge = 36

        def ridge_bounds(t_hat: np.ndarray, c_mult: float):
            # Minimize/evaluate q(u) = (d_0 + u v)' J (d_0 + u v) for v = (1, -1)
            v = np.array([1.0, -1.0])
            d0 = -t_hat
            A = float(v @ info_J @ v)
            B = 2.0 * float(d0 @ info_J @ v)
            C = float(d0 @ info_J @ d0) - (2.0 * z_peak / c_mult)
            disc = max(0.0, B * B - 4.0 * A * C)
            u_lo = (-B - math.sqrt(disc)) / (2.0 * A)
            u_hi = (-B + math.sqrt(disc)) / (2.0 * A)
            u_lo = max(-r_max, u_lo)
            u_hi = min(r_max, u_hi)
            us = np.linspace(u_lo, u_hi, n_ridge + 1)
            f_pts = [pt3(float(u), float(-u), 0.0) for u in us]
            r_pts = [pt3(float(u), float(-u), ll_height(float(u), float(-u), t_hat, c_mult)) for u in us]
            return f_pts, r_pts

        f_pts0, r_pts0 = ridge_bounds(t_hat0, c0)
        ridge_curve = VMobject(color=AMBER_HEX, stroke_width=4.0)
        ridge_curve.set_points_as_corners(r_pts0)
        wall_quads = VGroup()
        for idx in range(n_ridge):
            wall_quads.add(
                Polygon(
                    f_pts0[idx],
                    f_pts0[idx + 1],
                    r_pts0[idx + 1],
                    r_pts0[idx],
                    fill_color=AMBER_HEX,
                    fill_opacity=0.20,
                    stroke_opacity=0.0,
                )
            )

        p_til_top0 = pt3(float(t_til0[0]), float(t_til0[1]), z_til0)
        p_til_fl0 = pt3(float(t_til0[0]), float(t_til0[1]), 0.0)
        p_til_at_hat0 = pt3(float(t_til0[0]), float(t_til0[1]), z_hat0)

        dot_til_top = Dot3D(p_til_top0, radius=0.075, color=AMBER_HEX, resolution=(8, 8))
        dot_til_fl = Dot3D(p_til_fl0, radius=0.068, color=AMBER_HEX, resolution=(8, 8))
        drop_til = make_dashed_3d(p_til_top0, p_til_fl0, color=AMBER_HEX, num_dashes=6)
        lr_bridge = make_dashed_3d(p_hat_top0, p_til_at_hat0, color=INDIGO_LIGHT_HEX, num_dashes=6)
        lr_drop = Line(p_til_at_hat0, p_til_top0, color=INDIGO_LIGHT_HEX, stroke_width=3.8)

        lbl_til = pill_tex(r"\tilde{\theta}_n", p_til_top0 + np.array([0.24, -0.32, 0.14]), AMBER_HEX, font_size=23)
        lbl_lr3d = pill_tex(
            r"\tfrac{1}{2}\mathrm{LR}_n",
            (p_til_at_hat0 + p_til_top0) * 0.5 + np.array([-0.14, -0.44, 0.06]),
            INDIGO_LIGHT_HEX,
            font_size=20,
        )

        # Level-set J-ellipse at z = z_til (upper) and z = 0.02 (floor tangent to g=0)
        def build_level_ellipse_pts(t_hat: np.ndarray, t_tilde: np.ndarray, z_height: float) -> list[np.ndarray]:
            d_til = t_tilde - t_hat
            r_lvl = max(0.04, math.sqrt(float(d_til @ info_J @ d_til)))
            pts = []
            for k in range(57):
                ang = (k / 56.0) * 2.0 * PI
                u_c = np.array([math.cos(ang), math.sin(ang)])
                p = t_hat + r_lvl * (inv_LT @ u_c)
                pts.append(pt3(float(p[0]), float(p[1]), z_height))
            return pts

        upper_ellipse = VMobject(color=CYAN_HEX, stroke_width=2.6)
        upper_ellipse.set_points_as_corners(build_level_ellipse_pts(t_hat0, t_til0, z_til0))

        floor_ellipse = VMobject(color=INDIGO_LIGHT_HEX, stroke_width=2.4)
        floor_ellipse.set_points_as_corners(build_level_ellipse_pts(t_hat0, t_til0, 0.02))

        p_euc_fl0 = pt3(float(t_euc0[0]), float(t_euc0[1]), 0.02)
        p_hat_fl2 = pt3(float(t_hat0[0]), float(t_hat0[1]), 0.02)
        p_til_fl2 = pt3(float(t_til0[0]), float(t_til0[1]), 0.02)

        proj_euc = make_dashed_3d(p_hat_fl2, p_euc_fl0, color=MUTED_HEX, num_dashes=5, stroke_width=2.0)
        proj_info = Line(p_hat_fl2, p_til_fl2, color=CYAN_HEX, stroke_width=3.0)
        dot_euc = Dot3D(p_euc_fl0, radius=0.054, color=MUTED_HEX, resolution=(8, 8))
        p_euc_lbl0 = p_euc_fl0 + np.array([-0.92, -0.58, 0.14])
        pin_euc = make_dashed_3d(p_euc_fl0, p_euc_lbl0 + np.array([0.24, 0.16, -0.04]), color=MUTED_HEX, num_dashes=5, stroke_width=1.6)
        lbl_euc = pill_tex(r"\hat{\theta}_{\mathrm{Euc}}", p_euc_lbl0, MUTED_HEX, font_size=18)

        # Floor score normal arrow S_n(tilde{theta}_n) _|_ ker G + 90-deg square marker
        g_unit = G_vec / math.sqrt(2.0)
        t_tang = np.array([-1.0, 1.0]) / math.sqrt(2.0)
        s_len0 = 0.44
        p_score_end2d0 = t_til0 + s_len0 * g_unit
        score_shaft = Line(
            p_til_fl2,
            pt3(float(p_score_end2d0[0]), float(p_score_end2d0[1]), 0.02),
            color=CYAN_HEX,
            stroke_width=3.6,
        )
        score_tip = xy_arrow_tip(t_til0, p_score_end2d0, 0.02, CYAN_HEX, size=0.09)

        def build_right_angle_pts(base_2d: np.ndarray, sq: float = 0.11):
            p1 = base_2d + sq * g_unit
            p2 = base_2d + sq * g_unit + sq * t_tang
            p3 = base_2d + sq * t_tang
            return [
                pt3(float(p1[0]), float(p1[1]), 0.02),
                pt3(float(p2[0]), float(p2[1]), 0.02),
                pt3(float(p3[0]), float(p3[1]), 0.02),
            ]

        score_perp_box = VMobject(color=CYAN_HEX, stroke_width=2.0)
        score_perp_box.set_points_as_corners(build_right_angle_pts(t_til0))

        p_score_lbl0 = pt3(float(p_score_end2d0[0]), float(p_score_end2d0[1]), 0.02) + np.array([0.96, -0.14, 0.08])
        pin_score = make_dashed_3d(
            pt3(float(p_score_end2d0[0]), float(p_score_end2d0[1]), 0.02),
            p_score_lbl0 + np.array([-0.38, 0.04, -0.02]),
            color=CYAN_HEX,
            num_dashes=4,
            stroke_width=1.8,
        )
        lbl_score = pill_tex(r"S_n(\tilde{\theta}_n)\perp\ker G", p_score_lbl0, CYAN_HEX, font_size=18)

        # Phase 1: Floor, constraint line g=0, and true theta_0 (register lbl_true only now!)
        self.add_fixed_orientation_mobjects(lbl_true)
        self.play(FadeIn(floor), FadeIn(grid), Create(floor_c_line), FadeIn(dot_true_fl), Create(pin_true), FadeIn(lbl_true), run_time=0.85)

        # Phase 2: Elliptic paraboloid dome peaking at hat{theta}_n (register lbl_hat only now!)
        sub_tex = swap_sub_3d(
            self,
            title_row,
            sub_tex,
            r"\text{Sample log-likelihood dome peaks at } \hat{\theta}_n \text{ with curvature } J_n = nJ",
            font_size=19,
            color=INDIGO_LIGHT_HEX,
        )
        self.add_fixed_orientation_mobjects(lbl_hat)
        self.play(Create(dome_mesh), run_time=1.05, rate_func=smooth)
        self.play(FadeIn(dot_hat_top), FadeIn(dot_hat_fl), Create(drop_hat), FadeIn(lbl_hat), run_time=0.60)

        # Phase 3: Constraint slice g=0, ridge peak tilde{theta}_n, level-set ellipse, 1/2 LR
        sub_tex = swap_sub_3d(
            self,
            title_row,
            sub_tex,
            r"\text{Slice } g(\theta)=0 \text{ peaks at } \tilde{\theta}_n;\; \text{level-set ellipse kisses ridge giving } \tfrac{1}{2}\mathrm{LR}_n",
            font_size=18,
            color=AMBER_HEX,
        )
        self.add_fixed_orientation_mobjects(lbl_til, lbl_lr3d)
        self.play(FadeIn(wall_quads), Create(ridge_curve), run_time=0.85, rate_func=smooth)
        self.play(
            Create(upper_ellipse),
            FadeIn(dot_til_top),
            FadeIn(dot_til_fl),
            Create(drop_til),
            FadeIn(lbl_til),
            Create(lr_bridge),
            Create(lr_drop),
            FadeIn(lbl_lr3d),
            run_time=0.85,
        )

        # Phase 4: Floor J-projection vs Euclidean + score normal S_n(tilde{theta}_n) _|_ ker G
        sub_tex = swap_sub_3d(
            self,
            title_row,
            sub_tex,
            r"\text{Floor: } J\text{-projection } \tilde{\theta}_n \neq \hat{\theta}_{\mathrm{Euc}} \text{ and score } S_n(\tilde{\theta}_n) \perp \ker G",
            font_size=18,
            color=CYAN_HEX,
        )
        self.add_fixed_orientation_mobjects(lbl_euc, lbl_score)
        self.play(
            Create(floor_ellipse),
            Create(proj_euc),
            FadeIn(dot_euc),
            Create(pin_euc),
            FadeIn(lbl_euc),
            Create(proj_info),
            Create(score_shaft),
            FadeIn(score_tip),
            Create(score_perp_box),
            Create(pin_score),
            FadeIn(lbl_score),
            run_time=1.0,
        )
        self.wait(0.60)

        # Phase 5: Under H0 true, as n grows: both hat{theta}_n, tilde{theta}_n -> theta_0
        # while curvature scales as O(n) -> dome narrows onto theta_0 & 1/2 LR = O_p(1) stays finite!
        sub_tex = swap_sub_3d(
            self,
            title_row,
            sub_tex,
            r"\text{As } n\uparrow \text{ under } H_0:\; \hat{\theta}_n,\tilde{\theta}_n \xrightarrow{p} \theta_0 \text{ while } \tfrac{1}{2}\mathrm{LR}_n = O_p(1) \text{ stays finite}",
            font_size=18,
            color=TEAL_HEX,
        )

        scale_n = ValueTracker(1.0)

        def sync_scene_to_scale(s: float) -> None:
            t_hat, t_tilde, t_euc, c_mult = state_for(s)
            z_hat = ll_height(t_hat[0], t_hat[1], t_hat, c_mult)
            z_til = ll_height(t_tilde[0], t_tilde[1], t_hat, c_mult)

            update_dome_quads(t_hat, c_mult)

            f_pts, r_pts = ridge_bounds(t_hat, c_mult)
            ridge_curve.set_points_as_corners(r_pts)
            for idx in range(n_ridge):
                wall_quads[idx].set_points_as_corners([
                    f_pts[idx], f_pts[idx + 1], r_pts[idx + 1], r_pts[idx], f_pts[idx]
                ])

            p_ht = pt3(float(t_hat[0]), float(t_hat[1]), z_hat)
            p_hf = pt3(float(t_hat[0]), float(t_hat[1]), 0.0)
            p_tt = pt3(float(t_tilde[0]), float(t_tilde[1]), z_til)
            p_tf = pt3(float(t_tilde[0]), float(t_tilde[1]), 0.0)
            p_tah = pt3(float(t_tilde[0]), float(t_tilde[1]), z_hat)
            p_hf2 = pt3(float(t_hat[0]), float(t_hat[1]), 0.02)
            p_tf2 = pt3(float(t_tilde[0]), float(t_tilde[1]), 0.02)
            p_ef2 = pt3(float(t_euc[0]), float(t_euc[1]), 0.02)

            dot_hat_top.move_to(p_ht)
            dot_hat_fl.move_to(p_hf)
            drop_hat.become(make_dashed_3d(p_ht, p_hf, color=INDIGO_LIGHT_HEX, num_dashes=8))
            lbl_hat.move_to(p_ht + np.array([0.16, 0.24, 0.30]))

            dot_til_top.move_to(p_tt)
            dot_til_fl.move_to(p_tf)
            drop_til.become(make_dashed_3d(p_tt, p_tf, color=AMBER_HEX, num_dashes=6))
            lr_bridge.become(make_dashed_3d(p_ht, p_tah, color=INDIGO_LIGHT_HEX, num_dashes=6))
            lr_drop.put_start_and_end_on(p_tah, p_tt)

            lbl_til.move_to(p_tt + np.array([0.24, -0.32, 0.14]))
            lbl_lr3d.move_to((p_tah + p_tt) * 0.5 + np.array([-0.14, -0.44, 0.06]))

            upper_ellipse.set_points_as_corners(build_level_ellipse_pts(t_hat, t_tilde, z_til))
            floor_ellipse.set_points_as_corners(build_level_ellipse_pts(t_hat, t_tilde, 0.02))
            proj_euc.become(make_dashed_3d(p_hf2, p_ef2, color=MUTED_HEX, num_dashes=5, stroke_width=2.0))
            proj_info.put_start_and_end_on(p_hf2, p_tf2)
            dot_euc.move_to(p_ef2)
            p_el = p_ef2 + np.array([-0.92, -0.58, 0.14])
            lbl_euc.move_to(p_el)
            pin_euc.become(make_dashed_3d(p_ef2, p_el + np.array([0.24, 0.16, -0.04]), color=MUTED_HEX, num_dashes=5, stroke_width=1.6))

            s_len = s_len0 * (0.72 + 0.28 * s)
            p_se2d = t_tilde + s_len * g_unit
            p_se = pt3(float(p_se2d[0]), float(p_se2d[1]), 0.02)
            p_sl = p_se + np.array([0.96, -0.14, 0.08])
            score_shaft.put_start_and_end_on(p_tf2, p_se)
            score_tip.become(xy_arrow_tip(t_tilde, p_se2d, 0.02, CYAN_HEX, size=0.09))
            score_perp_box.set_points_as_corners(build_right_angle_pts(t_tilde))
            lbl_score.move_to(p_sl)
            pin_score.become(make_dashed_3d(p_se, p_sl + np.array([-0.38, 0.04, -0.02]), color=CYAN_HEX, num_dashes=4, stroke_width=1.8))

        dome_mesh.add_updater(lambda _m: sync_scene_to_scale(scale_n.get_value()))
        self.begin_ambient_camera_rotation(rate=0.14)
        self.play(scale_n.animate.set_value(0.42), run_time=2.3, rate_func=smooth)
        dome_mesh.clear_updaters()
        self.wait(1.3)
        self.stop_ambient_camera_rotation()
        self.wait(0.45)


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
