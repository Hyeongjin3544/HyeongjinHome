"""
Manim animations for ML Visual Supplement 01 (Gradient Descent Geometry):
  - GDStepSizeScene      -> assets/media/gd-step-size-geometry.mp4
  - GDConditioningScene  -> assets/media/gd-condition-momentum.mp4

Usage:
  .venv\\Scripts\\python.exe visuals\\render_gd.py [--quality low|high] [--scene all|steps|cond]
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

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UP,
    Axes,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    MathTex,
    Polygon,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    ValueTracker,
    always_redraw,
    smooth,
)

BG_HEX = "#0c0f14"
PANEL_HEX = "#111622"
GRID_HEX = "#2b3547"
CYAN_HEX = "#38bdf8"
TEAL_HEX = "#2dd4bf"
INDIGO_HEX = "#6366f1"
INDIGO_LIGHT_HEX = "#818cf8"
AMBER_HEX = "#fbbf24"
ROSE_HEX = "#fb7185"
INK_HEX = "#f4f4f5"
MUTED_HEX = "#a1a1aa"


def pill_tex(
    tex_str: str,
    pos: np.ndarray,
    color: str,
    font_size: int = 22,
    use_bg: bool = True,
) -> MathTex:
    lbl = MathTex(tex_str, font_size=font_size, color=color).move_to(pos)
    if use_bg:
        lbl.add_background_rectangle(color=BG_HEX, opacity=0.88, buff=0.06)
    return lbl


def pill_text(
    text_str: str,
    pos: np.ndarray,
    color: str,
    font_size: int = 16,
    weight: str = "BOLD",
    use_bg: bool = True,
) -> Text:
    lbl = Text(text_str, font="Segoe UI", font_size=font_size, weight=weight, color=color).move_to(pos)
    if use_bg:
        lbl.add_background_rectangle(color=BG_HEX, opacity=0.88, buff=0.06)
    return lbl


def make_hud(title_str: str, math_str: str, sub_str: str):
    title = Text(title_str, font="Segoe UI", font_size=22, weight="BOLD", color=INK_HEX)
    title_math = MathTex(math_str, font_size=23, color=INDIGO_LIGHT_HEX)
    title_row = VGroup(title, title_math).arrange(RIGHT, buff=0.16)
    title_row.to_corner(UL, buff=0.38)

    sub = MathTex(sub_str, font_size=18, color=MUTED_HEX)
    sub.next_to(title_row, DOWN, buff=0.12, aligned_edge=LEFT)
    return title_row, sub


def swap_sub(
    scene: Scene,
    title_row: VGroup,
    old_sub: MathTex,
    new_tex: str,
    font_size: int = 18,
    color: str = MUTED_HEX,
) -> MathTex:
    scene.play(FadeOut(old_sub, shift=UP * 0.08, run_time=0.22, rate_func=smooth))
    new_sub = MathTex(new_tex, font_size=font_size, color=color)
    new_sub.next_to(title_row, DOWN, buff=0.12, aligned_edge=LEFT)
    scene.play(FadeIn(new_sub, shift=UP * 0.08, run_time=0.24, rate_func=smooth))
    return new_sub


def make_arrow_2d(
    p0: np.ndarray,
    p1: np.ndarray,
    color: str,
    stroke_width: float = 3.2,
    tip_size: float = 0.14,
) -> VGroup:
    diff = p1 - p0
    norm = float(np.linalg.norm(diff[:2]))
    if norm < 1e-5:
        return VGroup()
    u = diff[:2] / norm
    perp = np.array([-u[1], u[0]])
    shaft_end = p1 - np.array([u[0], u[1], 0.0]) * (tip_size * 0.72)
    shaft = Line(p0, shaft_end, color=color, stroke_width=stroke_width)
    tip = Polygon(
        p1,
        p1 + np.array([(-tip_size * u[0] + 0.46 * tip_size * perp[0]), (-tip_size * u[1] + 0.46 * tip_size * perp[1]), 0.0]),
        p1 + np.array([(-tip_size * u[0] - 0.46 * tip_size * perp[0]), (-tip_size * u[1] - 0.46 * tip_size * perp[1]), 0.0]),
        fill_color=color,
        fill_opacity=1.0,
        stroke_color=color,
        stroke_width=1.0,
    )
    return VGroup(shaft, tip)


def clip_segment_to_box(
    p0: np.ndarray,
    p1: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Liang-Barsky 2D segment clipping so diverging trajectories stay neatly inside axes."""
    x0, y0 = float(p0[0]), float(p0[1])
    dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])
    t0, t1 = 0.0, 1.0
    checks = [
        (-dx, x0 - xlim[0]),
        (dx, xlim[1] - x0),
        (-dy, y0 - ylim[0]),
        (dy, ylim[1] - y0),
    ]
    for p, q in checks:
        if abs(p) < 1e-12:
            if q < 0:
                return False, p0, p1
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return False, p0, p1
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False, p0, p1
                t1 = min(t1, r)
    if t0 > t1:
        return False, p0, p1
    c0 = np.array([x0 + t0 * dx, y0 + t0 * dy])
    c1 = np.array([x0 + t1 * dx, y0 + t1 * dy])
    return True, c0, c1


def ellipse_contour_mobject(
    ax: Axes,
    center: np.ndarray,
    mat: np.ndarray,
    level: float,
    color: str,
    stroke_width: float = 1.6,
    stroke_opacity: float = 0.65,
    n_pts: int = 96,
) -> DashedLine | Line | VGroup:
    """Return closed polyline on Axes for (x-c)^T M (x-c) / 2 = level."""
    from manim import VMobject

    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, 1e-5)
    angles = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=True)
    # (u1^2 * eig1 + u2^2 * eig2)/2 = level => u_i = sqrt(2*level/eig_i) * trig
    pts_data = []
    r1 = math.sqrt(2.0 * level / eigvals[0])
    r2 = math.sqrt(2.0 * level / eigvals[1])
    for ang in angles:
        u = np.array([r1 * math.cos(ang), r2 * math.sin(ang)])
        w = center + eigvecs @ u
        pts_data.append(ax.c2p(float(w[0]), float(w[1])))
    curve = VMobject(color=color, stroke_width=stroke_width, stroke_opacity=stroke_opacity)
    curve.set_points_as_corners(pts_data)
    return curve


class GDStepSizeScene(Scene):
    """Clip 01: Fundamentals (-grad f, L-smoothness bound, eta stability, 2D contour orthogonality)."""

    def construct(self):
        self.camera.background_color = BG_HEX

        # ==================== PHASE 1: 1D TANGENT, L-BOUND & STEP RULE ====================
        title_row, sub_tex = make_hud(
            "Gradient Descent Basics:",
            r"\theta_{k+1} = \theta_k - \eta \nabla f(\theta_k)",
            r"\text{Move opposite the slope } f'(\theta_k) \text{ to minimize the } L\text{-smoothness quadratic upper bound}",
        )
        self.add(title_row, sub_tex)

        ax1 = Axes(
            x_range=[-3.1, 3.1, 1.0],
            y_range=[-0.35, 4.45, 1.0],
            x_length=9.2,
            y_length=4.75,
            axis_config={"color": MUTED_HEX, "stroke_width": 1.6, "include_tip": True},
        ).shift(DOWN * 0.56)

        ax1_xlbl = MathTex(r"\theta", font_size=24, color=INK_HEX).next_to(ax1.x_axis, RIGHT, buff=0.14)
        ax1_ylbl = MathTex(r"f(\theta)", font_size=22, color=INK_HEX).next_to(ax1.y_axis, UP, buff=0.10)

        # Pseudo-Huber + mild quadratic: global convexity 0.24 <= f''(t) <= L = 1.34
        L_bound = 1.34

        def f1(t: float) -> float:
            return 2.2 * (math.sqrt(1.0 + 0.5 * t * t) - 1.0) + 0.12 * t * t

        def df1(t: float) -> float:
            return (1.1 * t) / math.sqrt(1.0 + 0.5 * t * t) + 0.24 * t

        f1_curve = ax1.plot(f1, x_range=[-2.95, 2.95], color=INDIGO_LIGHT_HEX, stroke_width=3.4)
        f1_lbl = pill_tex(r"f(\theta)", ax1.c2p(-2.35, f1(-2.35) + 0.42), INDIGO_LIGHT_HEX, font_size=22)

        dot_star = Dot(ax1.c2p(0.0, 0.0), radius=0.075, color=TEAL_HEX)
        lbl_star = pill_tex(r"\theta^*", ax1.c2p(0.0, -0.30), TEAL_HEX, font_size=22)

        self.play(
            Create(ax1),
            FadeIn(ax1_xlbl),
            FadeIn(ax1_ylbl),
            Create(f1_curve),
            FadeIn(f1_lbl),
            FadeIn(dot_star),
            FadeIn(lbl_star),
            run_time=1.0,
        )

        th0 = 2.05
        y0 = f1(th0)
        g0 = df1(th0)
        th1 = th0 - g0 / L_bound  # optimal surrogate step: eta = 1/L
        y1 = f1(th1)

        def q_surr(t: float) -> float:
            return y0 + g0 * (t - th0) + 0.5 * L_bound * (t - th0) ** 2

        # Tangent line around th0 (stay cleanly clear of th1 / x-axis labels)
        t_lo, t_hi = 1.15, 2.70
        tan_line = Line(
            ax1.c2p(t_lo, y0 + g0 * (t_lo - th0)),
            ax1.c2p(t_hi, y0 + g0 * (t_hi - th0)),
            color=AMBER_HEX,
            stroke_width=2.6,
        )
        dot_0 = Dot(ax1.c2p(th0, y0), radius=0.075, color=CYAN_HEX)
        dot_0_x = Dot(ax1.c2p(th0, 0.0), radius=0.055, color=CYAN_HEX)
        drop_0 = DashedLine(ax1.c2p(th0, y0), ax1.c2p(th0, 0.0), color=CYAN_HEX, stroke_width=1.8, dash_length=0.08)
        lbl_0 = pill_tex(r"\theta_k", ax1.c2p(th0 + 0.08, -0.30), CYAN_HEX, font_size=21)
        lbl_slope = pill_tex(r"\text{slope } f'(\theta_k) > 0", ax1.c2p(th0 + 0.36, y0 + 0.85), AMBER_HEX, font_size=19)

        self.play(
            Create(tan_line),
            Create(drop_0),
            FadeIn(dot_0),
            FadeIn(dot_0_x),
            FadeIn(lbl_0),
            FadeIn(lbl_slope),
            run_time=0.95,
        )

        # Quadratic L-smoothness surrogate majorizer q_L(theta) >= f(theta)
        q_curve = ax1.plot(
            q_surr,
            x_range=[-0.85, 2.55],
            color=TEAL_HEX,
            stroke_width=2.4,
        )
        q_min_y = q_surr(th1)
        lbl_q = pill_tex(
            r"q_L(\theta) = f(\theta_k) + f'(\theta_k)(\theta-\theta_k) + \frac{L}{2}(\theta-\theta_k)^2 \ge f(\theta)",
            ax1.c2p(-1.42, 3.88),
            TEAL_HEX,
            font_size=17,
        )
        dot_qmin = Dot(ax1.c2p(th1, q_min_y), radius=0.06, color=TEAL_HEX)
        drop_q1 = DashedLine(ax1.c2p(th1, q_min_y), ax1.c2p(th1, y1), color=TEAL_HEX, stroke_width=1.8, dash_length=0.07)

        self.play(Create(q_curve), FadeIn(lbl_q), run_time=1.0)

        # Step arrow along x-axis from th0 -> th1
        step_arr = make_arrow_2d(ax1.c2p(th0, 0.10), ax1.c2p(th1, 0.10), CYAN_HEX, stroke_width=3.6, tip_size=0.14)
        lbl_step = pill_tex(r"-\frac{1}{L} f'(\theta_k)", ax1.c2p(0.5 * (th0 + th1), 0.44), CYAN_HEX, font_size=20)
        dot_1 = Dot(ax1.c2p(th1, y1), radius=0.075, color=TEAL_HEX)
        dot_1_x = Dot(ax1.c2p(th1, 0.0), radius=0.055, color=TEAL_HEX)
        drop_1 = DashedLine(ax1.c2p(th1, y1), ax1.c2p(th1, 0.0), color=TEAL_HEX, stroke_width=1.8, dash_length=0.08)
        lbl_1 = pill_tex(r"\theta_{k+1}", ax1.c2p(th1 - 0.04, -0.30), TEAL_HEX, font_size=21)

        self.play(
            FadeIn(dot_qmin),
            Create(drop_q1),
            Create(step_arr),
            FadeIn(lbl_step),
            Create(drop_1),
            FadeIn(dot_1),
            FadeIn(dot_1_x),
            FadeIn(lbl_1),
            run_time=1.05,
        )
        self.wait(1.0)

        # Clear Phase 1
        phase1_group = VGroup(
            ax1, ax1_xlbl, ax1_ylbl, f1_curve, f1_lbl, dot_star, lbl_star,
            tan_line, dot_0, dot_0_x, drop_0, lbl_0, lbl_slope,
            q_curve, lbl_q, dot_qmin, drop_q1,
            step_arr, lbl_step, dot_1, dot_1_x, drop_1, lbl_1,
        )
        self.play(FadeOut(phase1_group), run_time=0.45)

        # ==================== PHASE 2: 2x2 STEP-SIZE REGIMES ON QUADRATIC ====================
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{On } f(\theta)=\tfrac{L}{2}\theta^2:\; \theta_k = (1-\eta L)^k \theta_0 \;\Longrightarrow\; \text{stable iff } 0 < \eta < 2/L",
            font_size=18,
            color=CYAN_HEX,
        )

        # 4 panels in clean 2x2 grid below HUD
        regimes = [
            (0.22, TEAL_HEX, r"\eta = 0.22/L \text{ (Slow crawl)}", 7, LEFT * 3.1 + UP * 0.85),
            (1.00, CYAN_HEX, r"\eta = 1/L \text{ (Exact 1-step)}", 1, RIGHT * 3.1 + UP * 0.85),
            (1.78, AMBER_HEX, r"\eta = 1.78/L \text{ (Damped oscillation)}", 6, LEFT * 3.1 + DOWN * 1.95),
            (2.16, ROSE_HEX, r"\eta = 2.16/L > 2/L \text{ (Diverges!)}", 5, RIGHT * 3.1 + DOWN * 1.95),
        ]

        panel_mobs = VGroup()
        panel_traj = []
        L_q = 1.0
        th_start = 1.65
        xlim = (-3.1, 3.1)
        ylim = (-0.25, 4.6)

        for eta_val, col_hex, title_tex, n_steps, center_pos in regimes:
            box = RoundedRectangle(
                corner_radius=0.14,
                width=5.75,
                height=2.45,
                fill_color=PANEL_HEX,
                fill_opacity=0.88,
                stroke_color=GRID_HEX,
                stroke_width=1.4,
            ).move_to(center_pos)

            ax_m = Axes(
                x_range=[-3.1, 3.1, 1.0],
                y_range=[-0.25, 4.6, 2.0],
                x_length=5.1,
                y_length=1.78,
                axis_config={"color": MUTED_HEX, "stroke_width": 1.2, "include_tip": False},
            ).move_to(center_pos + DOWN * 0.20)

            parab = ax_m.plot(lambda t: 0.5 * L_q * t * t, x_range=[-2.95, 2.95], color=INDIGO_LIGHT_HEX, stroke_width=2.4)
            badge = MathTex(title_tex, font_size=18, color=col_hex).move_to(box.get_top() + DOWN * 0.24)
            opt_dot = Dot(ax_m.c2p(0.0, 0.0), radius=0.045, color=INK_HEX)

            panel_mobs.add(box, ax_m, parab, opt_dot, badge)

            # Build exact gradient descent iterates
            pts = [th_start]
            t_cur = th_start
            for _ in range(n_steps):
                t_cur = t_cur - eta_val * (L_q * t_cur)
                pts.append(t_cur)
            panel_traj.append((ax_m, col_hex, pts, eta_val))

        self.play(FadeIn(panel_mobs), run_time=0.85)

        # Place initial dots on all 4 mini panels
        init_dots = VGroup()
        for ax_m, col_hex, pts, _ in panel_traj:
            d = Dot(ax_m.c2p(pts[0], 0.5 * L_q * pts[0] ** 2), radius=0.06, color=col_hex)
            init_dots.add(d)
        self.play(FadeIn(init_dots), run_time=0.4)

        # Animate step-by-step cobweb jumps simultaneously across all 4 panels
        max_steps = max(len(item[2]) - 1 for item in panel_traj)
        step_mobs = VGroup(init_dots)
        diverge_marked = False
        for k in range(max_steps):
            anims = []
            for ax_m, col_hex, pts, eta_val in panel_traj:
                if k >= len(pts) - 1:
                    continue
                t0_k = pts[k]
                t1_k = pts[k + 1]
                p0_data = np.array([t0_k, 0.5 * L_q * t0_k * t0_k])
                p1_data = np.array([t1_k, 0.5 * L_q * t1_k * t1_k])
                ok, c0, c1 = clip_segment_to_box(p0_data, p1_data, xlim, ylim)
                if not ok:
                    continue
                seg = Line(
                    ax_m.c2p(float(c0[0]), float(c0[1])),
                    ax_m.c2p(float(c1[0]), float(c1[1])),
                    color=col_hex,
                    stroke_width=2.3,
                )
                step_mobs.add(seg)
                anims.append(Create(seg))
                if xlim[0] <= t1_k <= xlim[1] and ylim[0] <= p1_data[1] <= ylim[1]:
                    nd = Dot(ax_m.c2p(t1_k, float(p1_data[1])), radius=0.055, color=col_hex)
                    step_mobs.add(nd)
                    anims.append(FadeIn(nd))
                elif eta_val > 2.0 and not diverge_marked:
                    diverge_marked = True
                    bang = pill_text("Escapes!", ax_m.c2p(min(2.15, max(-2.15, c1[0])), 3.7), ROSE_HEX, font_size=14)
                    step_mobs.add(bang)
                    anims.append(FadeIn(bang))
            if anims:
                self.play(*anims, run_time=0.45, rate_func=smooth)
        self.wait(1.35)

        self.play(FadeOut(panel_mobs), FadeOut(step_mobs), run_time=0.45)

        # ==================== PHASE 3: 2D CONTOUR ORTHOGONALITY (-grad f perp level set) ====================
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{In } \mathbb{R}^d:\; \nabla f(\theta_k) \perp \text{contour tangent},\;\text{so } -\nabla f(\theta_k) \text{ is local steepest descent}",
            font_size=18,
            color=TEAL_HEX,
        )

        ax2 = Axes(
            x_range=[-3.4, 3.4, 1.0],
            y_range=[-2.3, 2.3, 1.0],
            x_length=6.9,
            y_length=4.7,
            axis_config={"color": MUTED_HEX, "stroke_width": 1.4, "include_tip": True},
        ).shift(DOWN * 0.52 + LEFT * 1.85)

        th1_lbl = MathTex(r"\theta_1", font_size=22, color=INK_HEX).next_to(ax2.x_axis, RIGHT, buff=0.10)
        th2_lbl = MathTex(r"\theta_2", font_size=22, color=INK_HEX).next_to(ax2.y_axis, UP, buff=0.08)

        # Tilted ellipse around (-0.4, -0.2)
        c2 = np.array([-0.35, -0.20])
        ang2 = 26.0 * math.pi / 180.0
        rot2 = np.array([[math.cos(ang2), -math.sin(ang2)], [math.sin(ang2), math.cos(ang2)]])
        hess2 = rot2 @ np.diag([0.48, 1.55]) @ rot2.T

        levels = [0.22, 0.58, 1.15, 1.95, 2.95]
        contours = VGroup(*[
            ellipse_contour_mobject(ax2, c2, hess2, lv, INDIGO_LIGHT_HEX, stroke_width=1.6, stroke_opacity=0.62)
            for lv in levels
        ])
        dot_opt2 = Dot(ax2.c2p(float(c2[0]), float(c2[1])), radius=0.07, color=TEAL_HEX)
        lbl_opt2 = pill_tex(r"\theta^*", ax2.c2p(float(c2[0] - 0.34), float(c2[1] - 0.28)), TEAL_HEX, font_size=21)

        # Right-side explanation card
        info_card = RoundedRectangle(
            corner_radius=0.14,
            width=4.15,
            height=4.4,
            fill_color=PANEL_HEX,
            fill_opacity=0.92,
            stroke_color=GRID_HEX,
            stroke_width=1.5,
        ).move_to(RIGHT * 4.55 + DOWN * 0.48)

        card_title = pill_text("Why perpendicular?", info_card.get_top() + DOWN * 0.42, INK_HEX, font_size=17, use_bg=False)
        eq_dir = MathTex(r"df = \nabla f(\theta)^\top v = 0", font_size=20, color=AMBER_HEX).next_to(card_title, DOWN, buff=0.26)
        eq_cs = MathTex(
            r"\min_{\|u\|=1} \nabla f(\theta)^\top u = -\|\nabla f(\theta)\|",
            font_size=18,
            color=CYAN_HEX,
        ).next_to(eq_dir, DOWN, buff=0.28)
        eq_sol = MathTex(
            r"\implies u^* = -\frac{\nabla f(\theta)}{\|\nabla f(\theta)\|}",
            font_size=20,
            color=TEAL_HEX,
        ).next_to(eq_cs, DOWN, buff=0.28)
        eq_desc = MathTex(
            r"\text{Step } -\eta\nabla f \text{ crosses contours}\\",
            r"\text{at } 90^\circ \text{ locally (not globally straight to } \theta^*)",
            font_size=16,
            color=MUTED_HEX,
        ).next_to(eq_sol, DOWN, buff=0.34)

        self.play(
            Create(ax2),
            FadeIn(th1_lbl),
            FadeIn(th2_lbl),
            Create(contours),
            FadeIn(dot_opt2),
            FadeIn(lbl_opt2),
            FadeIn(info_card),
            FadeIn(card_title),
            FadeIn(eq_dir),
            run_time=1.0,
        )

        # Point theta_0 right on level f = 1.95
        dir_pick = np.array([1.75, 1.35])
        quad_val = 0.5 * float(dir_pick @ hess2 @ dir_pick)
        th_pt = c2 + dir_pick * math.sqrt(1.95 / quad_val)
        grad_pt = hess2 @ (th_pt - c2)
        g_norm = float(np.linalg.norm(grad_pt))
        u_pos = grad_pt / g_norm
        u_tan = np.array([-u_pos[1], u_pos[0]])

        p_th = ax2.c2p(float(th_pt[0]), float(th_pt[1]))
        dot_th = Dot(p_th, radius=0.075, color=INK_HEX)
        lbl_th = pill_tex(r"\theta_0", ax2.c2p(float(th_pt[0] - 0.32), float(th_pt[1] + 0.30)), INK_HEX, font_size=21)

        tan_seg = DashedLine(
            ax2.c2p(float(th_pt[0] - 1.15 * u_tan[0]), float(th_pt[1] - 1.15 * u_tan[1])),
            ax2.c2p(float(th_pt[0] + 1.15 * u_tan[0]), float(th_pt[1] + 1.15 * u_tan[1])),
            color=AMBER_HEX,
            stroke_width=2.5,
            dash_length=0.10,
        )
        lbl_tan = pill_tex(r"\text{contour tangent } v", ax2.c2p(float(th_pt[0] - 1.05 * u_tan[0] + 0.55), float(th_pt[1] - 1.05 * u_tan[1] - 0.16)), AMBER_HEX, font_size=17)

        # +grad and -grad vectors
        arr_up = make_arrow_2d(p_th, ax2.c2p(float(th_pt[0] + 0.85 * u_pos[0]), float(th_pt[1] + 0.85 * u_pos[1])), ROSE_HEX, stroke_width=3.2, tip_size=0.13)
        lbl_up = pill_tex(r"+\nabla f", ax2.c2p(float(th_pt[0] + 1.08 * u_pos[0]), float(th_pt[1] + 1.02 * u_pos[1])), ROSE_HEX, font_size=19)

        arr_dn = make_arrow_2d(p_th, ax2.c2p(float(th_pt[0] - 0.95 * u_pos[0]), float(th_pt[1] - 0.95 * u_pos[1])), CYAN_HEX, stroke_width=3.6, tip_size=0.14)
        lbl_dn = pill_tex(r"-\nabla f", ax2.c2p(float(th_pt[0] - 1.08 * u_pos[0] + 0.42), float(th_pt[1] - 0.92 * u_pos[1] - 0.10)), CYAN_HEX, font_size=20)

        # Right-angle square marker between -grad and tangent
        sq_d = 0.20
        p_r1 = ax2.c2p(float(th_pt[0] + sq_d * u_tan[0]), float(th_pt[1] + sq_d * u_tan[1]))
        p_r2 = ax2.c2p(float(th_pt[0] + sq_d * u_tan[0] - sq_d * u_pos[0]), float(th_pt[1] + sq_d * u_tan[1] - sq_d * u_pos[1]))
        p_r3 = ax2.c2p(float(th_pt[0] - sq_d * u_pos[0]), float(th_pt[1] - sq_d * u_pos[1]))
        from manim import VMobject
        perp_corner = VMobject(color=AMBER_HEX, stroke_width=2.0)
        perp_corner.set_points_as_corners([p_r1, p_r2, p_r3])

        self.play(
            FadeIn(dot_th),
            FadeIn(lbl_th),
            Create(tan_seg),
            FadeIn(lbl_tan),
            Create(arr_up),
            FadeIn(lbl_up),
            Create(perp_corner),
            run_time=0.95,
        )
        self.play(
            FadeIn(eq_cs),
            FadeIn(eq_sol),
            Create(arr_dn),
            FadeIn(lbl_dn),
            run_time=0.95,
        )

        # Animate actual GD trajectory from th_pt down to c2
        self.play(FadeIn(eq_desc), FadeOut(arr_up), FadeOut(lbl_up), run_time=0.55)
        cur = th_pt.copy()
        eta2 = 0.55
        for step_idx in range(8):
            nxt = cur - eta2 * (hess2 @ (cur - c2))
            seg_gd = Line(ax2.c2p(float(cur[0]), float(cur[1])), ax2.c2p(float(nxt[0]), float(nxt[1])), color=TEAL_HEX, stroke_width=3.0)
            dot_gd = Dot(ax2.c2p(float(nxt[0]), float(nxt[1])), radius=0.055, color=TEAL_HEX)
            self.play(Create(seg_gd), FadeIn(dot_gd), run_time=0.28, rate_func=smooth)
            cur = nxt
        self.wait(1.35)


class GDConditioningScene(Scene):
    """Clip 02: Condition number kappa = L/mu zigzag, Ridge conditioning, Momentum & Newton race."""

    def construct(self):
        self.camera.background_color = BG_HEX

        # ==================== PHASE 1: ISOTROPIC (kappa=1) VS ILL-CONDITIONED (kappa=12) ====================
        title_row, sub_tex = make_hud(
            "Condition Number & Acceleration:",
            r"\kappa = \frac{L}{\mu}",
            r"\text{Large } \kappa \text{ creates narrow elliptical valleys where } -\nabla f \text{ zigzags across the canyon}",
        )
        self.add(title_row, sub_tex)

        c_left = LEFT * 3.15 + DOWN * 0.52
        c_right = RIGHT * 3.15 + DOWN * 0.52

        box_l = RoundedRectangle(corner_radius=0.14, width=5.95, height=5.15, fill_color=PANEL_HEX, fill_opacity=0.88, stroke_color=GRID_HEX, stroke_width=1.4).move_to(c_left)
        box_r = RoundedRectangle(corner_radius=0.14, width=5.95, height=5.15, fill_color=PANEL_HEX, fill_opacity=0.88, stroke_color=GRID_HEX, stroke_width=1.4).move_to(c_right)

        ax_l = Axes(
            x_range=[-2.6, 2.6, 1.0],
            y_range=[-2.3, 2.3, 1.0],
            x_length=4.7,
            y_length=4.15,
            axis_config={"color": MUTED_HEX, "stroke_width": 1.2, "include_tip": False},
        ).move_to(c_left + DOWN * 0.16)
        ax_r = Axes(
            x_range=[-2.6, 2.6, 1.0],
            y_range=[-2.3, 2.3, 1.0],
            x_length=4.7,
            y_length=4.15,
            axis_config={"color": MUTED_HEX, "stroke_width": 1.2, "include_tip": False},
        ).move_to(c_right + DOWN * 0.16)

        hdr_l = MathTex(r"\text{Isotropic: } \kappa = 1 \text{ (Circular)}", font_size=20, color=TEAL_HEX).move_to(box_l.get_top() + DOWN * 0.30)
        hdr_r = MathTex(r"\text{Ill-conditioned: } \kappa = 12 \text{ (Ravine)}", font_size=20, color=ROSE_HEX).move_to(box_r.get_top() + DOWN * 0.30)

        H_iso = np.diag([1.0, 1.0])
        ang_rav = -24.0 * math.pi / 180.0
        R_rav = np.array([[math.cos(ang_rav), -math.sin(ang_rav)], [math.sin(ang_rav), math.cos(ang_rav)]])
        H_rav = R_rav @ np.diag([0.125, 1.50]) @ R_rav.T  # kappa = 12

        cont_l = VGroup(*[
            ellipse_contour_mobject(ax_l, np.zeros(2), H_iso, lv, TEAL_HEX, stroke_width=1.4, stroke_opacity=0.48)
            for lv in [0.18, 0.55, 1.15, 2.0]
        ])
        cont_r = VGroup(*[
            ellipse_contour_mobject(ax_r, np.zeros(2), H_rav, lv, INDIGO_LIGHT_HEX, stroke_width=1.4, stroke_opacity=0.55)
            for lv in [0.03, 0.09, 0.18, 0.30]
        ])

        star_l = Dot(ax_l.c2p(0.0, 0.0), radius=0.06, color=INK_HEX)
        star_r = Dot(ax_r.c2p(0.0, 0.0), radius=0.06, color=INK_HEX)

        self.play(
            FadeIn(box_l), FadeIn(box_r),
            Create(ax_l), Create(ax_r),
            FadeIn(hdr_l), FadeIn(hdr_r),
            Create(cont_l), Create(cont_r),
            FadeIn(star_l), FadeIn(star_r),
            run_time=0.95,
        )

        x_l = np.array([1.85, 1.15])
        x_r = R_rav @ np.array([-1.68, 0.70])
        dl0 = Dot(ax_l.c2p(float(x_l[0]), float(x_l[1])), radius=0.065, color=CYAN_HEX)
        dr0 = Dot(ax_r.c2p(float(x_r[0]), float(x_r[1])), radius=0.065, color=ROSE_HEX)
        self.play(FadeIn(dl0), FadeIn(dr0), run_time=0.35)

        p1_traj = VGroup(dl0, dr0)
        eta_iso = 0.58
        eta_rav = 1.24  # < 2/1.50 = 1.333 => stable but oscillates wildly (-0.86) in steep axis!
        for k in range(13):
            anims = []
            if k < 6:
                xn_l = x_l - eta_iso * (H_iso @ x_l)
                seg_l = Line(ax_l.c2p(float(x_l[0]), float(x_l[1])), ax_l.c2p(float(xn_l[0]), float(xn_l[1])), color=CYAN_HEX, stroke_width=3.0)
                pt_l = Dot(ax_l.c2p(float(xn_l[0]), float(xn_l[1])), radius=0.05, color=CYAN_HEX)
                p1_traj.add(seg_l, pt_l)
                anims.extend([Create(seg_l), FadeIn(pt_l)])
                x_l = xn_l

            xn_r = x_r - eta_rav * (H_rav @ x_r)
            seg_r = Line(ax_r.c2p(float(x_r[0]), float(x_r[1])), ax_r.c2p(float(xn_r[0]), float(xn_r[1])), color=ROSE_HEX, stroke_width=2.6)
            pt_r = Dot(ax_r.c2p(float(xn_r[0]), float(xn_r[1])), radius=0.045, color=ROSE_HEX)
            p1_traj.add(seg_r, pt_r)
            anims.extend([Create(seg_r), FadeIn(pt_r)])
            x_r = xn_r
            self.play(*anims, run_time=0.22, rate_func=smooth)

        lbl_direct = pill_tex(r"\text{Straight to } \theta^*", ax_l.c2p(0.85, -0.75), CYAN_HEX, font_size=18)
        lbl_zig = pill_tex(r"\text{Steep-wall zigzag } (1-\mu/L)^k", ax_r.c2p(-0.35, -1.65), ROSE_HEX, font_size=17)
        self.play(FadeIn(lbl_direct), FadeIn(lbl_zig), run_time=0.45)
        self.wait(1.0)

        p1_all = VGroup(box_l, box_r, ax_l, ax_r, hdr_l, hdr_r, cont_l, cont_r, star_l, star_r, p1_traj, lbl_direct, lbl_zig)
        self.play(FadeOut(p1_all), run_time=0.45)

        # ==================== PHASE 2: RIDGE CONDITIONING + WARNING 5.14 ====================
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{Ridge penalty } +\tfrac{\lambda}{2}\|\theta\|^2 \text{ drops } \kappa(\lambda)=\frac{d_{\max}+\lambda}{d_{\min}+\lambda}\text{, but shifts the optimum!}",
            font_size=18,
            color=AMBER_HEX,
        )

        ax_rdg = Axes(
            x_range=[-2.3, 2.8, 1.0],
            y_range=[-2.0, 2.0, 1.0],
            x_length=6.2,
            y_length=4.75,
            axis_config={"color": MUTED_HEX, "stroke_width": 1.4, "include_tip": True},
        ).shift(LEFT * 2.35 + DOWN * 0.48)

        theta_ols = np.array([1.18, -0.38])
        dot_origin = Dot(ax_rdg.c2p(0.0, 0.0), radius=0.055, color=MUTED_HEX)
        lbl_origin = pill_tex(r"0", ax_rdg.c2p(-0.20, -0.22), MUTED_HEX, font_size=18)

        lam_tr = ValueTracker(0.0)

        def ridge_sol(lam: float) -> np.ndarray:
            return np.linalg.solve(H_rav + lam * np.eye(2), H_rav @ theta_ols)

        ridge_contours = always_redraw(lambda: VGroup(*[
            ellipse_contour_mobject(
                ax_rdg,
                ridge_sol(lam_tr.get_value()),
                H_rav + lam_tr.get_value() * np.eye(2),
                lv,
                AMBER_HEX if lam_tr.get_value() > 0.05 else INDIGO_LIGHT_HEX,
                stroke_width=1.6,
                stroke_opacity=0.60,
            )
            for lv in [0.025, 0.075, 0.15, 0.24]
        ]))

        dot_ols = Dot(ax_rdg.c2p(float(theta_ols[0]), float(theta_ols[1])), radius=0.075, color=INDIGO_LIGHT_HEX)
        lbl_ols = pill_tex(r"\hat{\theta}^{\mathrm{OLS}}", ax_rdg.c2p(float(theta_ols[0] + 0.35), float(theta_ols[1] - 0.30)), INDIGO_LIGHT_HEX, font_size=19)
        dot_rdg = always_redraw(lambda: Dot(
            ax_rdg.c2p(float(ridge_sol(lam_tr.get_value())[0]), float(ridge_sol(lam_tr.get_value())[1])),
            radius=0.075,
            color=AMBER_HEX,
        ))

        info_rdg = RoundedRectangle(
            corner_radius=0.14,
            width=4.4,
            height=4.5,
            fill_color=PANEL_HEX,
            fill_opacity=0.92,
            stroke_color=GRID_HEX,
            stroke_width=1.5,
        ).move_to(RIGHT * 4.35 + DOWN * 0.45)

        rdg_title = pill_text("Ridge Preconditioning (Ch 05)", info_rdg.get_top() + DOWN * 0.42, AMBER_HEX, font_size=16, use_bg=False)
        kappa_readout = always_redraw(lambda: MathTex(
            rf"\lambda = {lam_tr.get_value():.2f},\;\; \kappa(\lambda) = {(1.50 + lam_tr.get_value()) / (0.125 + lam_tr.get_value()):.1f}",
            font_size=20,
            color=CYAN_HEX,
        ).next_to(rdg_title, DOWN, buff=0.34))
        rdg_note = MathTex(
            r"H_\lambda = X^\top X + \lambda I\\[6pt]"
            r"\text{Contours round out } (\kappa \downarrow)\\[4pt]"
            r"\text{and GD takes fewer bounces.}",
            font_size=17,
            color=INK_HEX,
        ).next_to(rdg_title, DOWN, buff=1.12)
        rdg_warn = pill_tex(
            r"\text{Warning: }\hat{\theta}_\lambda^{\mathrm{Ridge}} \ne \hat{\theta}^{\mathrm{OLS}}\text{ (bias!)}",
            info_rdg.get_bottom() + UP * 0.62,
            ROSE_HEX,
            font_size=17,
        )

        # Static comparison trajectory OLS (k=0) vs Ridge (k=0.55)
        start_rdg = np.array([-1.10, 1.35])

        def trace_gd_ridge(lam: float, eta_step: float, n_it: int) -> list[np.ndarray]:
            c_target = ridge_sol(lam)
            h_eff = H_rav + lam * np.eye(2)
            pts = [start_rdg.copy()]
            cur = start_rdg.copy()
            for _ in range(n_it):
                cur = cur - eta_step * (h_eff @ (cur - c_target))
                pts.append(cur.copy())
            return pts

        pts_ols_gd = trace_gd_ridge(0.0, 1.20, 10)
        from manim import VMobject
        path_ols_gd = VMobject(color=ROSE_HEX, stroke_width=2.3, stroke_opacity=0.65)
        path_ols_gd.set_points_as_corners([ax_rdg.c2p(float(p[0]), float(p[1])) for p in pts_ols_gd])

        self.play(
            Create(ax_rdg),
            FadeIn(dot_origin), FadeIn(lbl_origin),
            FadeIn(ridge_contours),
            FadeIn(dot_ols), FadeIn(lbl_ols),
            FadeIn(dot_rdg),
            FadeIn(info_rdg), FadeIn(rdg_title),
            FadeIn(kappa_readout), FadeIn(rdg_note), FadeIn(rdg_warn),
            run_time=0.95,
        )
        self.play(Create(path_ols_gd), run_time=0.75)

        # Smoothly increase lambda 0 -> 0.55 so ellipse rounds & target slides toward 0
        self.play(lam_tr.animate.set_value(0.55), run_time=1.8, rate_func=smooth)

        pts_rdg_gd = trace_gd_ridge(0.55, 0.85, 8)
        path_rdg_gd = VMobject(color=TEAL_HEX, stroke_width=3.2)
        path_rdg_gd.set_points_as_corners([ax_rdg.c2p(float(p[0]), float(p[1])) for p in pts_rdg_gd])
        lbl_rdg_star = pill_tex(
            r"\hat{\theta}_\lambda^{\mathrm{Ridge}}",
            ax_rdg.c2p(float(ridge_sol(0.55)[0] - 0.15), float(ridge_sol(0.55)[1] - 0.45)),
            AMBER_HEX,
            font_size=19,
        )
        self.play(Create(path_rdg_gd), FadeIn(lbl_rdg_star), run_time=0.85)
        self.wait(1.0)

        # Freeze dynamic updaters cleanly before fading out
        ridge_contours.clear_updaters()
        dot_rdg.clear_updaters()
        kappa_readout.clear_updaters()
        self.play(
            FadeOut(VGroup(
                ax_rdg, dot_origin, lbl_origin, ridge_contours, dot_ols, lbl_ols,
                dot_rdg, info_rdg, rdg_title, kappa_readout, rdg_note, rdg_warn,
                path_ols_gd, path_rdg_gd, lbl_rdg_star,
            )),
            run_time=0.45,
        )

        # ==================== PHASE 3: 3-WAY CANYON RACE + LOG LOSS CURVE ====================
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{In the ravine: Momentum cancels } \perp \text{ zigzags; Newton } (-H^{-1}\nabla f) \text{ whitens curvature in 1 step}",
            font_size=17,
            color=TEAL_HEX,
        )

        ax_race = Axes(
            x_range=[-2.7, 2.7, 1.0],
            y_range=[-2.1, 2.1, 1.0],
            x_length=6.8,
            y_length=5.0,
            axis_config={"color": MUTED_HEX, "stroke_width": 1.3, "include_tip": True},
        ).shift(LEFT * 2.55 + DOWN * 0.48)

        ax_log = Axes(
            x_range=[0, 16, 4],
            y_range=[-4.4, 0.35, 1.0],
            x_length=4.5,
            y_length=4.4,
            axis_config={"color": MUTED_HEX, "stroke_width": 1.3, "include_tip": True},
        ).shift(RIGHT * 4.0 + DOWN * 0.54)
        ax_log.x_axis.set_y(ax_log.c2p(0.0, -4.4)[1])

        log_xlbl = MathTex(r"k \text{ (iterations)}", font_size=18, color=INK_HEX).next_to(ax_log.x_axis, DOWN, buff=0.14)
        log_ylbl = MathTex(r"\log_{10}(f(\theta_k) - f^*)", font_size=18, color=INK_HEX).next_to(ax_log.y_axis, UP, buff=0.10)

        # Narrow ravine kappa = 15
        ang_3 = 24.0 * math.pi / 180.0
        R3 = np.array([[math.cos(ang_3), -math.sin(ang_3)], [math.sin(ang_3), math.cos(ang_3)]])
        H3 = R3 @ np.diag([0.10, 1.50]) @ R3.T

        race_cont = VGroup(*[
            ellipse_contour_mobject(ax_race, np.zeros(2), H3, lv, INDIGO_LIGHT_HEX, stroke_width=1.4, stroke_opacity=0.52)
            for lv in [0.025, 0.075, 0.16, 0.29]
        ])
        race_star = Dot(ax_race.c2p(0.0, 0.0), radius=0.07, color=INK_HEX)

        start_3 = R3 @ np.array([1.95, 0.72])
        dot_s3 = Dot(ax_race.c2p(float(start_3[0]), float(start_3[1])), radius=0.07, color=INK_HEX)
        lbl_s3 = pill_tex(r"\theta_0", ax_race.c2p(float(start_3[0] + 0.16), float(start_3[1] + 0.24)), INK_HEX, font_size=20)

        self.play(
            Create(ax_race), Create(race_cont), FadeIn(race_star), FadeIn(dot_s3), FadeIn(lbl_s3),
            Create(ax_log), FadeIn(log_xlbl), FadeIn(log_ylbl),
            run_time=0.95,
        )

        # Compute exact trajectories for n_iter = 16
        n_iter = 16

        # 1. Vanilla GD (eta = 1.24)
        gd_pts = [start_3.copy()]
        cur = start_3.copy()
        for _ in range(n_iter):
            cur = cur - 1.24 * (H3 @ cur)
            gd_pts.append(cur.copy())

        # 2. Momentum (Heavy ball: beta = 0.62, eta = 0.68)
        mom_pts = [start_3.copy()]
        cur_m = start_3.copy()
        vel = np.zeros(2)
        for _ in range(n_iter):
            vel = 0.62 * vel + (H3 @ cur_m)
            cur_m = cur_m - 0.68 * vel
            mom_pts.append(cur_m.copy())

        # 3. Newton-Raphson (exact -H^{-1} grad f in 1 step on quadratic)
        newt_pts = [start_3.copy()] + [np.zeros(2) for _ in range(n_iter)]

        def log_loss(pt: np.ndarray) -> float:
            val = 0.5 * float(pt @ H3 @ pt)
            return max(-4.35, math.log10(max(val, 1e-5)))

        # Legend stacked neatly in empty upper-right of log plot
        leg_gd = pill_tex(r"\text{Vanilla GD}", ax_log.c2p(12.0, -0.05), ROSE_HEX, font_size=17)
        leg_mom = pill_tex(r"\text{Momentum } (\beta=0.62)", ax_log.c2p(11.6, -0.58), AMBER_HEX, font_size=17)
        leg_newt = pill_tex(r"\text{Newton } (-H^{-1}\nabla f)", ax_log.c2p(11.6, -1.12), TEAL_HEX, font_size=17)
        leg_mom.align_to(ax_log, RIGHT)
        leg_gd.align_to(leg_mom, RIGHT)
        leg_newt.align_to(leg_mom, RIGHT)
        self.play(FadeIn(leg_gd), FadeIn(leg_mom), FadeIn(leg_newt), run_time=0.45)

        dot_gd_h = Dot(ax_race.c2p(float(gd_pts[0][0]), float(gd_pts[0][1])), radius=0.06, color=ROSE_HEX)
        dot_mom_h = Dot(ax_race.c2p(float(mom_pts[0][0]), float(mom_pts[0][1])), radius=0.06, color=AMBER_HEX)
        dot_newt_h = Dot(ax_race.c2p(float(newt_pts[0][0]), float(newt_pts[0][1])), radius=0.065, color=TEAL_HEX)
        self.add(dot_gd_h, dot_mom_h, dot_newt_h)

        # Step k=1: Newton jumps straight to optimum + others start
        newt_arr = make_arrow_2d(
            ax_race.c2p(float(start_3[0]), float(start_3[1])),
            ax_race.c2p(0.0, 0.0),
            TEAL_HEX,
            stroke_width=3.2,
            tip_size=0.14,
        )
        newt_log_seg = Line(
            ax_log.c2p(0, log_loss(newt_pts[0])),
            ax_log.c2p(1, log_loss(newt_pts[1])),
            color=TEAL_HEX,
            stroke_width=3.0,
        )

        for k in range(n_iter):
            anims = []
            # GD segment + log
            g_seg = Line(ax_race.c2p(float(gd_pts[k][0]), float(gd_pts[k][1])), ax_race.c2p(float(gd_pts[k + 1][0]), float(gd_pts[k + 1][1])), color=ROSE_HEX, stroke_width=2.5)
            gl_seg = Line(ax_log.c2p(k, log_loss(gd_pts[k])), ax_log.c2p(k + 1, log_loss(gd_pts[k + 1])), color=ROSE_HEX, stroke_width=2.6)
            anims.extend([Create(g_seg), Create(gl_seg), dot_gd_h.animate.move_to(ax_race.c2p(float(gd_pts[k + 1][0]), float(gd_pts[k + 1][1])))])

            # Momentum segment + log
            m_seg = Line(ax_race.c2p(float(mom_pts[k][0]), float(mom_pts[k][1])), ax_race.c2p(float(mom_pts[k + 1][0]), float(mom_pts[k + 1][1])), color=AMBER_HEX, stroke_width=2.9)
            ml_seg = Line(ax_log.c2p(k, log_loss(mom_pts[k])), ax_log.c2p(k + 1, log_loss(mom_pts[k + 1])), color=AMBER_HEX, stroke_width=2.8)
            anims.extend([Create(m_seg), Create(ml_seg), dot_mom_h.animate.move_to(ax_race.c2p(float(mom_pts[k + 1][0]), float(mom_pts[k + 1][1])))])

            # Newton jumps at k == 0
            if k == 0:
                anims.extend([Create(newt_arr), Create(newt_log_seg), dot_newt_h.animate.move_to(ax_race.c2p(0.0, 0.0))])
            elif k >= 1:
                nl_flat = Line(ax_log.c2p(k, -4.35), ax_log.c2p(k + 1, -4.35), color=TEAL_HEX, stroke_width=2.6)
                anims.append(Create(nl_flat))

            self.play(*anims, run_time=0.24, rate_func=smooth)

        lbl_newton_done = pill_tex(r"-H^{-1}\nabla f(\theta_0) \text{ (1 step!)}", ax_race.c2p(0.85, -0.35), TEAL_HEX, font_size=17)
        self.play(FadeIn(lbl_newton_done), run_time=0.45)
        self.wait(1.4)


def render_all(quality: str, which: str = "all") -> None:
    root_dir = Path(__file__).resolve().parent.parent
    out_dir = root_dir / "assets" / "media"
    candidate_media = root_dir / "visuals" / "media"
    media_dir = (
        candidate_media
        if str(candidate_media).isascii()
        else Path(tempfile.gettempdir()) / "manim_gd"
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
        ("steps", "GDStepSizeScene", "gd-step-size-geometry.mp4"),
        ("cond", "GDConditioningScene", "gd-condition-momentum.mp4"),
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
    parser = argparse.ArgumentParser(description="Render Gradient Descent Manim videos")
    parser.add_argument("--quality", choices=["low", "high"], default="high")
    parser.add_argument("--scene", choices=["all", "steps", "cond"], default="all")
    args = parser.parse_args()
    render_all(args.quality, args.scene)
