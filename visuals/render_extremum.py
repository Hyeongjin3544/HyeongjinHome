"""
Manim animations for Econometrics Visual Supplement 00 (Extremum Estimation):
  - ExtremumUllnScene        -> assets/media/extremum-ulln-consistency.mp4
  - ExtremumAsymptoticScene  -> assets/media/extremum-asymptotic-normality.mp4

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

import numpy as np
from manim import (
    Axes,
    Create,
    DashedLine,
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    LEFT,
    Line,
    MathTex,
    Polygon,
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


class ExtremumUllnScene(Scene):
    def construct(self):
        self.camera.background_color = BG_HEX

        title_row, sub_tex = make_hud(
            "Extremum Estimator:",
            r"\hat{\theta}_n = \arg\min_{\theta \in \Theta} \hat{Q}_n(\theta)",
            r"\text{Identification: Population objective } Q(\theta) \text{ has a unique global minimum at } \theta_0",
        )
        self.add(title_row, sub_tex)

        ax = Axes(
            x_range=[-0.2, 4.2, 1.0],
            y_range=[-0.2, 3.3, 1.0],
            x_length=9.4,
            y_length=4.7,
            axis_config={"color": GRID_HEX, "stroke_width": 1.6, "include_tip": True},
        ).shift(DOWN * 0.55)

        x_lbl = MathTex(r"\theta \in \Theta", font_size=24, color=INK_HEX).next_to(ax.x_axis, RIGHT, buff=0.14)
        y_lbl = MathTex(r"Q(\theta)", font_size=24, color=INK_HEX).next_to(ax.y_axis, UP, buff=0.12)

        theta0 = 2.10
        # Population objective Q(theta) with unique minimum at theta0 = 2.10
        q_pop = lambda t: 0.45 + 0.40 * (t - theta0) ** 2 + 0.05 * np.sin(2.0 * (t - theta0))
        pop_curve = ax.plot(q_pop, x_range=[0.35, 3.85], color=INDIGO_LIGHT_HEX, stroke_width=3.6)
        pop_lbl = MathTex(r"Q(\theta)", font_size=24, color=INDIGO_LIGHT_HEX).next_to(ax.c2p(3.85, q_pop(3.85)), UP, buff=0.12)

        dot_t0 = Dot(ax.c2p(theta0, 0), color=AMBER_HEX, radius=0.085)
        dot_q0 = Dot(ax.c2p(theta0, q_pop(theta0)), color=AMBER_HEX, radius=0.08)
        drop_t0 = DashedLine(ax.c2p(theta0, q_pop(theta0)), ax.c2p(theta0, 0), color=AMBER_HEX, stroke_width=2.0)
        lbl_t0 = MathTex(r"\theta_0", font_size=26, color=AMBER_HEX).next_to(dot_t0, DOWN, buff=0.14)

        self.play(FadeIn(ax), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.55)
        self.play(Create(pop_curve), FadeIn(pop_lbl), run_time=1.0, rate_func=smooth)
        self.play(Create(drop_t0), FadeIn(dot_q0), FadeIn(dot_t0), FadeIn(lbl_t0), run_time=0.6)
        self.wait(0.35)

        # ULLN: Sample objective Q_hat_n(theta) shrinks into epsilon-tube around Q(theta) as n grows
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{ULLN: } \sup_{\theta \in \Theta} |\hat{Q}_n(\theta) - Q(\theta)| \xrightarrow{p} 0 \;\Rightarrow\; \hat{\theta}_n \xrightarrow{p} \theta_0",
            font_size=21,
            color=INK_HEX,
        )

        # Sample perturbation: linear tilt + smooth oscillatory sample noise scaled by 1/sqrt(n)
        def q_samp(t, amp: float):
            return q_pop(t) + amp * (-0.68 * (t - theta0) + 0.30 * np.cos(2.6 * t - 0.5) + 0.20 * np.sin(4.1 * t))

        def find_min_t(amp: float) -> float:
            ts = np.linspace(0.45, 3.75, 240)
            vals = [q_samp(float(tt), amp) for tt in ts]
            return float(ts[int(np.argmin(vals))])

        # Epsilon band around Q(theta)
        ts_band = np.linspace(0.35, 3.85, 60)
        amp_trk = ValueTracker(0.72)  # 0.72 -> small n, 0.05 -> large n

        band_poly = always_redraw(
            lambda: (
                lambda eps: Polygon(
                    *[ax.c2p(float(tt), q_pop(float(tt)) + eps) for tt in ts_band],
                    *[ax.c2p(float(tt), q_pop(float(tt)) - eps) for tt in reversed(ts_band)],
                    fill_color=TEAL_HEX,
                    fill_opacity=0.12,
                    stroke_color=TEAL_HEX,
                    stroke_opacity=0.30,
                    stroke_width=1.0,
                )
            )(0.16 + 0.60 * amp_trk.get_value())
        )
        samp_curve = always_redraw(
            lambda: ax.plot(lambda t: q_samp(t, amp_trk.get_value()), x_range=[0.35, 3.85], color=CYAN_HEX, stroke_width=3.2)
        )
        dot_qn = always_redraw(
            lambda: (
                lambda th: Dot(ax.c2p(th, q_samp(th, amp_trk.get_value())), color=CYAN_HEX, radius=0.085)
            )(find_min_t(amp_trk.get_value()))
        )
        drop_tn = always_redraw(
            lambda: (
                lambda th: DashedLine(ax.c2p(th, q_samp(th, amp_trk.get_value())), ax.c2p(th, 0), color=CYAN_HEX, stroke_width=2.0)
            )(find_min_t(amp_trk.get_value()))
        )
        dot_tn = always_redraw(
            lambda: (
                lambda th: Dot(ax.c2p(th, 0), color=CYAN_HEX, radius=0.085)
            )(find_min_t(amp_trk.get_value()))
        )
        lbl_tn = always_redraw(
            lambda: (
                lambda th: MathTex(r"\hat{\theta}_n", font_size=24, color=CYAN_HEX).next_to(
                    ax.c2p(th, q_samp(th, amp_trk.get_value())), UP, buff=0.22
                )
            )(find_min_t(amp_trk.get_value()))
        )

        # Sample size badge
        n_badge = always_redraw(
            lambda: (
                lambda amp: MathTex(
                    rf"n = {int(round(16 / max(amp * amp, 0.035)))}",
                    font_size=26,
                    color=TEAL_HEX,
                ).to_corner(DOWN + RIGHT, buff=0.55)
            )(amp_trk.get_value())
        )
        samp_lbl = MathTex(r"\hat{Q}_n(\theta)", font_size=24, color=CYAN_HEX).next_to(ax.c2p(3.85, q_samp(3.85, 0.72)), RIGHT, buff=0.12)

        self.play(FadeIn(band_poly), FadeIn(samp_curve), FadeIn(samp_lbl), FadeIn(drop_tn), FadeIn(dot_qn), FadeIn(dot_tn), FadeIn(lbl_tn), FadeIn(n_badge), run_time=0.8)
        self.wait(0.45)

        # Smoothly grow n -> ULLN convergence!
        self.play(
            FadeOut(samp_lbl, run_time=0.4),
            amp_trk.animate.set_value(0.05),
            run_time=3.6,
            rate_func=smooth,
        )
        self.wait(1.2)


class ExtremumAsymptoticScene(Scene):
    def construct(self):
        self.camera.background_color = BG_HEX

        title_row, sub_tex = make_hud(
            "Asymptotic Normality:",
            r"\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}(0, \, H^{-1}\Omega H^{-1})",
            r"\text{Taylor expansion of FOC: } 0 = \nabla\hat{Q}_n(\hat{\theta}_n) \approx \nabla\hat{Q}_n(\theta_0) + H\,(\hat{\theta}_n - \theta_0)",
        )
        self.add(title_row, sub_tex)

        # Left axes: Steep curvature H (precise); Right axes: Flat curvature H (noisy!)
        ax_l = Axes(
            x_range=[-1.6, 1.6, 0.8],
            y_range=[-2.2, 2.2, 1.0],
            x_length=4.8,
            y_length=4.4,
            axis_config={"color": GRID_HEX, "stroke_width": 1.5, "include_tip": True},
        ).shift(LEFT * 3.1 + DOWN * 0.55)

        ax_r = Axes(
            x_range=[-1.6, 1.6, 0.8],
            y_range=[-2.2, 2.2, 1.0],
            x_length=4.8,
            y_length=4.4,
            axis_config={"color": GRID_HEX, "stroke_width": 1.5, "include_tip": True},
        ).shift(RIGHT * 3.1 + DOWN * 0.55)

        lbl_l = MathTex(r"\text{Sharp Curvature } (|H| \text{ large})", font_size=22, color=TEAL_HEX).next_to(ax_l, UP, buff=0.18)
        lbl_r = MathTex(r"\text{Flat Curvature } (|H| \text{ small})", font_size=22, color=ROSE_HEX).next_to(ax_r, UP, buff=0.18)

        # Derivative curves (FOC lines with slopes H_steep = 2.4, H_flat = 0.65), clamped strictly inside axes y=[-2.0, 2.0]
        h_steep, h_flat = 2.4, 0.65
        score_err = ValueTracker(0.0)

        def clamped_range(s: float, slope: float):
            d_lo = max(-1.40, (-2.00 - s) / slope)
            d_hi = min(1.40, (2.00 - s) / slope)
            return [d_lo, d_hi]

        foc_l = always_redraw(
            lambda: ax_l.plot(
                lambda d: score_err.get_value() + h_steep * d,
                x_range=clamped_range(score_err.get_value(), h_steep),
                color=CYAN_HEX,
                stroke_width=3.2,
            )
        )
        foc_r = always_redraw(
            lambda: ax_r.plot(
                lambda d: score_err.get_value() + h_flat * d,
                x_range=clamped_range(score_err.get_value(), h_flat),
                color=CYAN_HEX,
                stroke_width=3.2,
            )
        )

        # Vertical score dot at theta_0 (d=0) and resulting root d_hat = -score / H
        dot_sc_l = always_redraw(lambda: Dot(ax_l.c2p(0, score_err.get_value()), color=AMBER_HEX, radius=0.075))
        dot_sc_r = always_redraw(lambda: Dot(ax_r.c2p(0, score_err.get_value()), color=AMBER_HEX, radius=0.075))

        dot_rt_l = always_redraw(lambda: Dot(ax_l.c2p(-score_err.get_value() / h_steep, 0), color=TEAL_HEX, radius=0.085))
        dot_rt_r = always_redraw(lambda: Dot(ax_r.c2p(np.clip(-score_err.get_value() / h_flat, -1.5, 1.5), 0), color=ROSE_HEX, radius=0.085))

        t0_l = MathTex(r"\theta_0", font_size=20, color=MUTED_HEX).next_to(ax_l.c2p(0, 0), DOWN, buff=0.14)
        t0_r = MathTex(r"\theta_0", font_size=20, color=MUTED_HEX).next_to(ax_r.c2p(0, 0), DOWN, buff=0.14)
        y_l = MathTex(r"\nabla \hat{Q}_n", font_size=20, color=CYAN_HEX).next_to(ax_l.y_axis, LEFT, buff=0.10).shift(UP * 1.4)
        y_r = MathTex(r"\nabla \hat{Q}_n", font_size=20, color=CYAN_HEX).next_to(ax_r.y_axis, LEFT, buff=0.10).shift(UP * 1.4)

        self.play(
            FadeIn(ax_l), FadeIn(ax_r), FadeIn(lbl_l), FadeIn(lbl_r),
            FadeIn(t0_l), FadeIn(t0_r), FadeIn(y_l), FadeIn(y_r),
            FadeIn(foc_l), FadeIn(foc_r),
            FadeIn(dot_sc_l), FadeIn(dot_sc_r),
            FadeIn(dot_rt_l), FadeIn(dot_rt_r),
            run_time=0.8,
        )

        # Oscillate score fluctuation +/- 0.85 to contrast horizontal spread (-score / H)
        sub_tex = swap_sub(
            self,
            title_row,
            sub_tex,
            r"\text{Identical score noise } \nabla\hat{Q}_n(\theta_0) \text{ produces tight } \hat{\theta}_n \text{ under steep } H \text{ vs wide } \hat{\theta}_n \text{ under flat } H",
            font_size=20,
            color=INK_HEX,
        )

        self.play(score_err.animate.set_value(0.82), run_time=1.35, rate_func=smooth)
        self.wait(0.35)
        self.play(score_err.animate.set_value(-0.82), run_time=1.8, rate_func=smooth)
        self.wait(0.35)
        self.play(score_err.animate.set_value(0.0), run_time=1.1, rate_func=smooth)

        # Highlight resulting spread bands on x-axis
        band_l = Line(ax_l.c2p(-0.82 / h_steep, 0), ax_l.c2p(0.82 / h_steep, 0), color=TEAL_HEX, stroke_width=7.0)
        band_r = Line(ax_r.c2p(-0.82 / h_flat, 0), ax_r.c2p(0.82 / h_flat, 0), color=ROSE_HEX, stroke_width=7.0)
        var_l = MathTex(r"\mathrm{SE} \propto |H|^{-1} \sqrt{\Omega} \text{ (small)}", font_size=21, color=TEAL_HEX).next_to(ax_l, DOWN, buff=0.22)
        var_r = MathTex(r"\mathrm{SE} \propto |H|^{-1} \sqrt{\Omega} \text{ (large)}", font_size=21, color=ROSE_HEX).next_to(ax_r, DOWN, buff=0.22)

        self.play(Create(band_l), Create(band_r), FadeIn(var_l), FadeIn(var_r), run_time=0.8)
        self.wait(1.6)


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
        ("ulln", "ExtremumUllnScene", "extremum-ulln-consistency.mp4"),
        ("asym", "ExtremumAsymptoticScene", "extremum-asymptotic-normality.mp4"),
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
    parser = argparse.ArgumentParser(description="Render Extremum Estimation Manim videos")
    parser.add_argument("--quality", choices=["low", "high"], default="high")
    parser.add_argument("--scene", choices=["all", "ulln", "asym"], default="all")
    args = parser.parse_args()
    render_all(args.quality, args.scene)
