"""
Manim animations for Econometrics Visual Supplement 03 (GMM):
  - GMMCoreIdeaScene       -> assets/media/gmm-core-idea.mp4
  - GMMOptimalWeightScene  -> assets/media/gmm-optimal-weight.mp4

Usage:
  .venv\\Scripts\\python.exe visuals\\render_gmm.py [--quality low|high]
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
    Axes,
    Circle,
    Create,
    DashedLine,
    Dot,
    DOWN,
    Ellipse,
    FadeIn,
    FadeOut,
    LEFT,
    Line,
    MathTex,
    ORIGIN,
    RIGHT,
    Scene,
    Text,
    UL,
    UP,
    ValueTracker,
    VGroup,
    always_redraw,
    smooth,
)

BG_HEX = "#0c0f14"
GRID_HEX = "#273040"
CYAN_HEX = "#38bdf8"
TEAL_HEX = "#2dd4bf"
INDIGO_HEX = "#6366f1"
INDIGO_LIGHT_HEX = "#818cf8"
AMBER_HEX = "#fbbf24"
ROSE_HEX = "#fb7185"
INK_HEX = "#f4f4f5"
MUTED_HEX = "#a1a1aa"


def make_hud(title_str: str, math_str: str, sub_str: str):
    title = Text(title_str, font="Segoe UI", font_size=22, weight="BOLD", color=INK_HEX)
    title_math = MathTex(math_str, font_size=24, color=INDIGO_LIGHT_HEX)
    title_row = VGroup(title, title_math).arrange(RIGHT, buff=0.18)
    title_row.to_corner(UL, buff=0.45)

    sub = MathTex(sub_str, font_size=21, color=MUTED_HEX)
    sub.next_to(title_row, DOWN, buff=0.16, aligned_edge=LEFT)
    return title_row, sub


def swap_sub(scene: Scene, title_row: VGroup, old_sub: MathTex, new_tex: str, font_size: int = 21, color: str = MUTED_HEX) -> MathTex:
    scene.play(FadeOut(old_sub, shift=UP * 0.08, run_time=0.22, rate_func=smooth))
    new_sub = MathTex(new_tex, font_size=font_size, color=color)
    new_sub.next_to(title_row, DOWN, buff=0.16, aligned_edge=LEFT)
    scene.play(FadeIn(new_sub, shift=UP * 0.08, run_time=0.26, rate_func=smooth))
    return new_sub


class GMMCoreIdeaScene(Scene):
    def construct(self):
        self.camera.background_color = BG_HEX

        title_row, sub_tex = make_hud(
            "What Is GMM?",
            r"\hat{\theta}_{\mathrm{GMM}} = \arg\min_\theta \; \bar{g}_n(\theta)' W \bar{g}_n(\theta)",
            r"\text{Method of Moments (}r=k=1\text{): Solve } \bar{g}_1(\theta) = 0 \text{ directly}",
        )
        self.add(title_row, sub_tex)

        # ------------------------------------------------------------------
        # ACT 1: 1D Moment Curves g1(theta) & g2(theta) -> Overidentification conflict
        # ------------------------------------------------------------------
        ax1 = Axes(
            x_range=[0.8, 4.6, 1.0],
            y_range=[-1.6, 1.6, 0.8],
            x_length=8.8,
            y_length=4.5,
            axis_config={"color": MUTED_HEX, "stroke_width": 1.6, "include_tip": True},
        ).shift(DOWN * 0.55)

        x_lbl = MathTex(r"\theta", font_size=26, color=INK_HEX).next_to(ax1.x_axis, RIGHT, buff=0.15)
        y_lbl = MathTex(r"\bar{g}_n(\theta)", font_size=24, color=INK_HEX).next_to(ax1.y_axis, UP, buff=0.12)
        zero_lbl = MathTex("0", font_size=20, color=MUTED_HEX).next_to(ax1.c2p(0.8, 0), LEFT, buff=0.12)

        # Moment 1: crosses 0 at theta = 2.15; Moment 2: crosses 0 at theta = 3.45
        g1_fn = lambda t: 0.72 * (t - 2.15)
        g2_fn = lambda t: 0.58 * (t - 3.45)

        curve1 = ax1.plot(g1_fn, x_range=[1.0, 4.2], color=CYAN_HEX, stroke_width=3.5)
        curve2 = ax1.plot(g2_fn, x_range=[1.1, 4.4], color=TEAL_HEX, stroke_width=3.5)

        lbl_c1 = MathTex(r"\bar{g}_1(\theta)", font_size=24, color=CYAN_HEX).next_to(ax1.c2p(3.85, g1_fn(3.85)), RIGHT, buff=0.12)
        lbl_c2 = MathTex(r"\bar{g}_2(\theta)", font_size=24, color=TEAL_HEX).next_to(ax1.c2p(4.25, g2_fn(4.25)), RIGHT, buff=0.12)

        dot_t1 = Dot(ax1.c2p(2.15, 0), color=CYAN_HEX, radius=0.085)
        lbl_t1 = MathTex(r"\hat{\theta}_{(1)}", font_size=24, color=CYAN_HEX).next_to(dot_t1, UP, buff=0.14)

        self.play(FadeIn(ax1), FadeIn(x_lbl), FadeIn(y_lbl), FadeIn(zero_lbl), run_time=0.6)
        self.play(Create(curve1), FadeIn(lbl_c1), run_time=1.0, rate_func=smooth)
        self.play(FadeIn(dot_t1, scale=1.3), FadeIn(lbl_t1), run_time=0.5)
        self.wait(0.35)

        # Add 2nd moment -> overdetermined conflict!
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{Overidentified (}r=2 > k=1\text{): Two moments give } \hat{\theta}_{(1)} \neq \hat{\theta}_{(2)} \text{ — no joint root!}",
            font_size=21,
            color=MUTED_HEX,
        )

        dot_t2 = Dot(ax1.c2p(3.45, 0), color=TEAL_HEX, radius=0.085)
        lbl_t2 = MathTex(r"\hat{\theta}_{(2)}", font_size=24, color=TEAL_HEX).next_to(dot_t2, DOWN, buff=0.16)
        gap_line = DashedLine(ax1.c2p(2.15, 0), ax1.c2p(2.15, g2_fn(2.15)), color=ROSE_HEX, stroke_width=2.4)
        gap_dot = Dot(ax1.c2p(2.15, g2_fn(2.15)), color=ROSE_HEX, radius=0.07)
        gap_lbl = MathTex(r"\bar{g}_2(\hat{\theta}_{(1)}) \neq 0", font_size=22, color=ROSE_HEX).next_to(gap_dot, DOWN, buff=0.12)

        self.play(Create(curve2), FadeIn(lbl_c2), run_time=1.0, rate_func=smooth)
        self.play(FadeIn(dot_t2, scale=1.3), FadeIn(lbl_t2), run_time=0.45)
        self.play(Create(gap_line), FadeIn(gap_dot), FadeIn(gap_lbl), run_time=0.65)
        self.wait(0.55)

        # ------------------------------------------------------------------
        # ACT 2: Lift into 2D Moment Space (g1, g2) -> Closest point to (0,0)
        # ------------------------------------------------------------------
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{GMM Idea: Map } \theta \mapsto \bar{g}_n(\theta) \in \mathbb{R}^2 \text{ and pick } \hat{\theta}_{\mathrm{GMM}} \text{ closest to } (0,0)",
            font_size=21,
            color=INK_HEX,
        )

        act1_group = VGroup(ax1, x_lbl, y_lbl, zero_lbl, curve1, curve2, lbl_c1, lbl_c2, dot_t1, lbl_t1, dot_t2, lbl_t2, gap_line, gap_dot, gap_lbl)
        self.play(FadeOut(act1_group), run_time=0.5)

        ax2 = Axes(
            x_range=[-1.6, 1.8, 0.8],
            y_range=[-1.6, 1.6, 0.8],
            x_length=6.0,
            y_length=5.0,
            axis_config={"color": GRID_HEX, "stroke_width": 1.6, "include_tip": True},
        ).shift(DOWN * 0.55 + LEFT * 0.4)

        g1_ax_lbl = MathTex(r"\bar{g}_1(\theta)", font_size=23, color=CYAN_HEX).next_to(ax2.x_axis, RIGHT, buff=0.14)
        g2_ax_lbl = MathTex(r"\bar{g}_2(\theta)", font_size=23, color=TEAL_HEX).next_to(ax2.y_axis, UP, buff=0.12)

        origin_pt = ax2.c2p(0, 0)
        origin_dot = Dot(origin_pt, color=AMBER_HEX, radius=0.09)
        origin_lbl = MathTex(r"\mathbf{0} = (0,0)", font_size=22, color=AMBER_HEX).next_to(origin_dot, UP + LEFT, buff=0.18).shift(LEFT * 0.18)

        # Moment curve in (g1, g2) space: passes through (0, g2(t1)) and (g1(t2), 0), missing (0,0)
        # Curve parameter t in [0.9, 4.3]: g1 = 0.72*(t - 2.15), g2 = 0.58*(t - 3.45)
        moment_locus = ax2.plot_parametric_curve(
            lambda t: np.array([g1_fn(t), g2_fn(t), 0.0]),
            t_range=[0.9, 4.3],
            color=INDIGO_LIGHT_HEX,
            stroke_width=3.6,
        )
        locus_lbl = MathTex(r"\{\bar{g}_n(\theta) : \theta \in \Theta\}", font_size=22, color=INDIGO_LIGHT_HEX).next_to(
            ax2.c2p(g1_fn(4.15), g2_fn(4.15)), RIGHT, buff=0.12
        )

        # Intercepts t1 (g1=0) and t2 (g2=0)
        pt_m1 = Dot(ax2.c2p(0, g2_fn(2.15)), color=CYAN_HEX, radius=0.075)
        lbl_m1 = MathTex(r"\hat{\theta}_{(1)} \, (\bar{g}_1=0)", font_size=20, color=CYAN_HEX).next_to(pt_m1, LEFT, buff=0.14)
        pt_m2 = Dot(ax2.c2p(g1_fn(3.45), 0), color=TEAL_HEX, radius=0.075)
        lbl_m2 = MathTex(r"\hat{\theta}_{(2)} \, (\bar{g}_2=0)", font_size=20, color=TEAL_HEX).next_to(pt_m2, UP, buff=0.14)

        self.play(FadeIn(ax2), FadeIn(g1_ax_lbl), FadeIn(g2_ax_lbl), FadeIn(origin_dot), FadeIn(origin_lbl), run_time=0.6)
        self.play(Create(moment_locus), FadeIn(locus_lbl), run_time=1.3, rate_func=smooth)
        self.play(FadeIn(pt_m1), FadeIn(lbl_m1), FadeIn(pt_m2), FadeIn(lbl_m2), run_time=0.6)

        # Compute exact Euclidean closest point on line g1(t), g2(t):
        # g(t) = a + b*(t - 2.15), where a = [0, g2(2.15)], b = [0.72, 0.58]
        a_v = np.array([0.0, g2_fn(2.15)])
        b_v = np.array([0.72, 0.58])
        t_gmm = 2.15 - float(a_v @ b_v) / float(b_v @ b_v)
        g_gmm = np.array([g1_fn(t_gmm), g2_fn(t_gmm)])
        pt_gmm_px = ax2.c2p(g_gmm[0], g_gmm[1])
        r_px = float(np.linalg.norm(pt_gmm_px - origin_pt))

        # Expanding distance contour circle until tangent to moment curve!
        r_trk = ValueTracker(0.05)
        dist_circle = always_redraw(
            lambda: Circle(radius=max(0.02, r_trk.get_value()), color=AMBER_HEX, stroke_width=2.2, stroke_opacity=0.85).move_to(origin_pt)
        )
        self.add(dist_circle)
        self.play(r_trk.animate.set_value(r_px), run_time=1.4, rate_func=smooth)

        gmm_dot = Dot(pt_gmm_px, color=AMBER_HEX, radius=0.095)
        resid_vec = Line(origin_pt, pt_gmm_px, color=ROSE_HEX, stroke_width=2.8)
        gmm_lbl = MathTex(r"\bar{g}_n(\hat{\theta}_{\mathrm{GMM}})", font_size=22, color=AMBER_HEX).next_to(gmm_dot, DR := DOWN + RIGHT, buff=0.14)

        # Orthogonal projection note
        note_box = MathTex(
            r"G' W \bar{g}_n(\hat{\theta}_{\mathrm{GMM}}) = 0",
            font_size=24,
            color=ROSE_HEX,
        ).to_corner(DR := DOWN + RIGHT, buff=0.55)

        self.play(FadeIn(gmm_dot, scale=1.3), Create(resid_vec), FadeIn(gmm_lbl), FadeIn(note_box), run_time=0.8)
        self.wait(1.6)


class GMMOptimalWeightScene(Scene):
    def construct(self):
        self.camera.background_color = BG_HEX

        title_row, sub_tex = make_hud(
            "Optimal Weighting Matrix:",
            r"W^* = S^{-1} \;\; (\mathrm{Avar}(\hat{\theta}) = (G'S^{-1}G)^{-1})",
            r"\text{Naive } W = I \text{ treats precise } \bar{g}_1 \text{ and noisy } \bar{g}_2 \text{ equally (circle distance)}",
        )
        self.add(title_row, sub_tex)

        ax = Axes(
            x_range=[-1.6, 1.8, 0.8],
            y_range=[-1.6, 1.6, 0.8],
            x_length=6.2,
            y_length=5.1,
            axis_config={"color": GRID_HEX, "stroke_width": 1.6, "include_tip": True},
        ).shift(DOWN * 0.55 + LEFT * 0.55)

        g1_lbl = MathTex(r"\bar{g}_1 \text{ (precise, }\sigma_1=0.3\text{)}", font_size=20, color=CYAN_HEX).next_to(ax.x_axis, RIGHT, buff=0.12)
        g2_lbl = MathTex(r"\bar{g}_2 \text{ (noisy, }\sigma_2=1.1\text{)}", font_size=20, color=TEAL_HEX).next_to(ax.y_axis, UP, buff=0.10)

        origin_pt = ax.c2p(0, 0)
        origin_dot = Dot(origin_pt, color=AMBER_HEX, radius=0.085)

        # Moment line: g1(t) = 0.72*(t - 2.2), g2(t) = 0.60*(t - 3.5)
        g1_fn = lambda t: 0.72 * (t - 2.20)
        g2_fn = lambda t: 0.60 * (t - 3.50)
        moment_locus = ax.plot_parametric_curve(
            lambda t: np.array([g1_fn(t), g2_fn(t), 0.0]),
            t_range=[0.9, 4.4],
            color=INDIGO_LIGHT_HEX,
            stroke_width=3.5,
        )

        # Sample uncertainty cloud around (0,0): narrow horizontally, tall vertically (sigma_2 >> sigma_1)
        s_cloud = Ellipse(
            width=0.75,
            height=2.60,
            color=TEAL_HEX,
            fill_color=TEAL_HEX,
            fill_opacity=0.08,
            stroke_width=1.3,
            stroke_opacity=0.45,
        ).move_to(origin_pt)
        cloud_lbl = MathTex(r"\text{Covariance } S", font_size=20, color=TEAL_HEX).next_to(s_cloud, UP, buff=0.08).shift(RIGHT * 1.15)

        # Compute tangent points for W = diag(1, w2) as w2 morphs from 1.0 -> (0.3/1.1)^2 = 0.074
        a_v = np.array([0.0, g2_fn(2.20)])
        b_v = np.array([0.72, 0.60])

        def solve_gmm(w2: float):
            W = np.diag([1.0, w2])
            t_star = 2.20 - float(a_v @ W @ b_v) / float(b_v @ W @ b_v)
            g_star = np.array([g1_fn(t_star), g2_fn(t_star)])
            q_min = float(g_star @ W @ g_star)
            return t_star, g_star, q_min

        # Show W = I tangent first
        _, g_ident, q_ident = solve_gmm(1.0)
        p_ident = ax.c2p(g_ident[0], g_ident[1])
        dot_ident = Dot(p_ident, color=ROSE_HEX, radius=0.08)
        lbl_ident = MathTex(r"\hat{\theta}_{I} \;\; (W=I)", font_size=21, color=ROSE_HEX).next_to(dot_ident, RIGHT, buff=0.14)

        # Convert math-unit radius to screen pixels
        px_per_x = float(np.linalg.norm(ax.c2p(1, 0) - origin_pt))
        px_per_y = float(np.linalg.norm(ax.c2p(0, 1) - origin_pt))

        w2_trk = ValueTracker(1.0)
        metric_ellipse = always_redraw(
            lambda: (
                lambda w2: (
                    lambda _t, _g, q: Ellipse(
                        width=2.0 * np.sqrt(q) * px_per_x,
                        height=2.0 * np.sqrt(q / w2) * px_per_y,
                        color=AMBER_HEX,
                        stroke_width=2.4,
                    ).move_to(origin_pt)
                )(*solve_gmm(w2))
            )(w2_trk.get_value())
        )
        moving_dot = always_redraw(
            lambda: (
                lambda _t, g, _q: Dot(ax.c2p(g[0], g[1]), color=AMBER_HEX, radius=0.095)
            )(*solve_gmm(w2_trk.get_value()))
        )

        self.play(FadeIn(ax), FadeIn(g1_lbl), FadeIn(g2_lbl), FadeIn(origin_dot), Create(moment_locus), run_time=0.9)
        self.play(FadeIn(s_cloud), FadeIn(cloud_lbl), run_time=0.6)
        self.play(FadeIn(metric_ellipse), FadeIn(dot_ident), FadeIn(lbl_ident), run_time=0.7)
        self.wait(0.45)

        # Morph W from I -> S^{-1} (w2 = (0.3/1.1)^2 ≈ 0.08)
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{Optimal } W^* = S^{-1}: \text{ Stretch ellipse along noisy } \bar{g}_2 \;\Rightarrow\; \text{pull } \hat{\theta}_{S^{-1}} \text{ toward precise } \bar{g}_1",
            font_size=20,
            color=INK_HEX,
        )

        self.add(moving_dot)
        self.play(
            dot_ident.animate.set_opacity(0.45),
            lbl_ident.animate.set_opacity(0.45),
            w2_trk.animate.set_value(0.08),
            run_time=2.4,
            rate_func=smooth,
        )

        _, g_opt, _ = solve_gmm(0.08)
        lbl_opt = MathTex(r"\hat{\theta}_{S^{-1}} \;\; (W^*=S^{-1})", font_size=21, color=AMBER_HEX).next_to(
            ax.c2p(g_opt[0], g_opt[1]), LEFT, buff=0.26
        )
        j_badge = MathTex(
            r"J = n\,\bar{g}_n(\hat{\theta})' S^{-1} \bar{g}_n(\hat{\theta}) \xrightarrow{d} \chi^2_{r-k}",
            font_size=22,
            color=INDIGO_LIGHT_HEX,
        ).to_corner(DOWN + RIGHT, buff=0.48)

        self.play(FadeIn(lbl_opt), FadeIn(j_badge), run_time=0.7)
        self.wait(1.8)


def render_all(quality: str, which: str = "all") -> None:
    root_dir = Path(__file__).resolve().parent.parent
    out_dir = root_dir / "assets" / "media"
    candidate_media = root_dir / "visuals" / "media"
    media_dir = (
        candidate_media
        if str(candidate_media).isascii()
        else Path(tempfile.gettempdir()) / "manim_gmm"
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
        ("idea", "GMMCoreIdeaScene", "gmm-core-idea.mp4"),
        ("weight", "GMMOptimalWeightScene", "gmm-optimal-weight.mp4"),
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
    parser = argparse.ArgumentParser(description="Render GMM Manim videos")
    parser.add_argument("--quality", choices=["low", "high"], default="high")
    parser.add_argument("--scene", choices=["all", "idea", "weight"], default="all")
    args = parser.parse_args()
    render_all(args.quality, args.scene)
