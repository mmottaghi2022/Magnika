#!/usr/bin/env python3
"""
MAGNIKA Stage H
Synthetic PWM-current excitation of the 2-D grain model.

This is an exploratory, dimensionless, phenomenological dynamic model.
It is NOT full LLG and is NOT calibrated to a specific ferrite or motor.

Place this file beside stageb.py (the corrected Stage B/C model).

Synthetic input:
    Balanced three-phase fundamental currents plus zero-mean bipolar
    triangular ripple, with phase-shifted ripple carriers.

Dynamic law:
    d(theta)/dt = -(1/tau) * dE/d(theta)

Forward Euler integration is used with a timestep resolving each PWM period.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import stageb as base
except ImportError as exc:
    raise SystemExit(
        "Could not import stageb.py. Put stageH.py beside the corrected "
        "Stage B/C file and name that file stageb.py."
    ) from exc


# ============================================================
# 1. SYNTHETIC PWM CURRENT GENERATOR
# ============================================================

def triangle_wave(phase: np.ndarray) -> np.ndarray:
    """Generate a unit bipolar triangular waveform in [-1, 1]."""

    cycles = np.mod(phase / (2.0 * np.pi), 1.0)

    return 4.0 * np.abs(cycles - 0.5) - 1.0


def synthesize_currents(
    t: np.ndarray,
    fe: float,
    fsw: float,
    ripple_fraction: float,
    current_peak: float
):
    """
    Generate three-phase currents.

    The current consists of:
        fundamental sinusoidal current
        +
        synthetic bipolar triangular ripple

    IMPORTANT:
    This is a controlled ripple surrogate, not a prediction of
    the current ripple produced by a real inverter and motor winding.
    """

    fundamental_phase = 2.0 * np.pi * fe * t

    phase_offsets = np.array([
        0.0,
        -2.0 * np.pi / 3.0,
        2.0 * np.pi / 3.0
    ])

    carrier_offsets = np.array([
        0.0,
        -2.0 * np.pi / 3.0,
        2.0 * np.pi / 3.0
    ])

    # Three-phase fundamental currents.
    fundamental = current_peak * np.sin(
        fundamental_phase[:, None]
        + phase_offsets[None, :]
    )

    # Three-phase synthetic ripple.
    carrier_phase = (
        2.0 * np.pi * fsw * t[:, None]
        + carrier_offsets[None, :]
    )

    ripple = (
        ripple_fraction
        * current_peak
        * triangle_wave(carrier_phase)
    )

    currents = fundamental + ripple

    ia, ib, ic = currents.T

    # Clarke transformation: abc -> alpha-beta.
    i_alpha = (2.0 / 3.0) * (
        ia - 0.5 * ib - 0.5 * ic
    )

    i_beta = (
        (2.0 / 3.0)
        * (np.sqrt(3.0) / 2.0)
        * (ib - ic)
    )

    return currents, i_alpha, i_beta


# ============================================================
# 2. DYNAMIC MAGNETIC RELAXATION
# ============================================================

def dynamic_step(
    model,
    theta,
    hx,
    hy,
    eta,
    dt,
    tau
):
    """
    One explicit Euler step of phenomenological overdamped
    angular relaxation.

    d(theta)/dt = -(1/tau) * dE/d(theta)

    This is NOT the Landau-Lifshitz-Gilbert equation.
    """

    _, gradient, _ = model.energy_gradient(
        theta,
        hx,
        hy,
        eta
    )

    theta_new = theta - (dt / tau) * gradient

    # Wrap angles into [-pi, pi].
    theta_new = (
        (theta_new + np.pi) % (2.0 * np.pi)
    ) - np.pi

    return theta_new


# ============================================================
# 3. GRAIN COHERENCE
# ============================================================

def grain_coherence(theta, grain_map):
    """
    Calculate the average local magnetic coherence within grains.

    Each grain's coherence is the magnitude of its mean
    in-plane magnetization vector.
    """

    grid = theta.reshape(grain_map.shape)

    values = []

    for gid in np.unique(grain_map):

        local = grid[grain_map == gid]

        mx = np.mean(np.cos(local))
        my = np.mean(np.sin(local))

        coherence = np.hypot(mx, my)

        values.append(coherence)

    return float(np.mean(values))


# ============================================================
# 4. RUN ONE EXPERIMENT
# ============================================================

def run_case(
    args,
    fsw: float,
    ripple_fraction: float,
    seed: int
):

    print(
        f"Generating grain structure: "
        f"{args.nx} x {args.ny}, "
        f"{args.n_grains} grains"
    )

    grain_map, grain_angles = base.generate_grains(
        args.nx,
        args.ny,
        args.n_grains,
        seed
    )

    model = base.MicroMagneticModel(
        grain_map,
        grain_angles,
        K=base.K_ANIS,
        J=base.J_EXCHANGE
    )

    # --------------------------------------------------------
    # Time discretization
    # --------------------------------------------------------

    dt = 1.0 / (
        args.samples_per_pwm * fsw
    )

    duration = (
        args.cycles / args.fundamental_hz
    )

    n_steps = int(
        np.ceil(duration / dt)
    )

    t = np.arange(
        n_steps + 1,
        dtype=float
    ) * dt

    # --------------------------------------------------------
    # Generate synthetic current waveforms
    # --------------------------------------------------------

    currents, i_alpha, i_beta = synthesize_currents(
        t,
        args.fundamental_hz,
        fsw,
        ripple_fraction,
        args.current_peak
    )

    # --------------------------------------------------------
    # Map current to dimensionless magnetic field
    # --------------------------------------------------------

    # The fundamental current peak maps to the Stage-F
    # reference field. This is a comparison convention,
    # not a physical SI calibration.

    field_scale = (
        args.reference_field / args.current_peak
    )

    hx_series = field_scale * i_alpha
    hy_series = field_scale * i_beta

    # --------------------------------------------------------
    # Initial magnetic state
    # --------------------------------------------------------

    theta = np.zeros(
        model.n,
        dtype=float
    )

    rows = []

    # --------------------------------------------------------
    # Time-domain simulation
    # --------------------------------------------------------

    for k in range(n_steps + 1):

        if k > 0:

            theta = dynamic_step(
                model,
                theta,
                hx_series[k],
                hy_series[k],
                args.eta_boundary,
                dt,
                args.relaxation_time
            )

        mx, my = model.magnetization(theta)

        coherence = grain_coherence(
            theta,
            grain_map
        )

        rows.append({

            "time_s": t[k],

            "ia": currents[k, 0],
            "ib": currents[k, 1],
            "ic": currents[k, 2],

            "i_alpha": i_alpha[k],
            "i_beta": i_beta[k],

            "hx": hx_series[k],
            "hy": hy_series[k],

            "Mx": mx,
            "My": my,

            "M_magnitude": float(
                np.hypot(mx, my)
            ),

            "grain_coherence": coherence,

        })

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Magnetic loop-work proxy
    # --------------------------------------------------------

    dmx = np.diff(
        df["Mx"].to_numpy()
    )

    dmy = np.diff(
        df["My"].to_numpy()
    )

    hx_mid = 0.5 * (
        hx_series[:-1] + hx_series[1:]
    )

    hy_mid = 0.5 * (
        hy_series[:-1] + hy_series[1:]
    )

    loop_work = float(
        np.sum(
            hx_mid * dmx
            + hy_mid * dmy
        )
    )

    # --------------------------------------------------------
    # Current ripple diagnostic
    # --------------------------------------------------------

    alpha = df["i_alpha"].to_numpy()
    beta = df["i_beta"].to_numpy()

    current_magnitude = np.hypot(
        alpha,
        beta
    )

    ripple_rms = float(
        np.std(
            current_magnitude
            - np.mean(current_magnitude)
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "seed": seed,

        "fundamental_hz": args.fundamental_hz,

        "pwm_hz": fsw,

        "ripple_fraction_peak": ripple_fraction,

        "eta_boundary": args.eta_boundary,

        "relaxation_time_s_model_parameter":
            args.relaxation_time,

        "dt_s": dt,

        "samples_per_pwm_period":
            args.samples_per_pwm,

        "time_steps": n_steps,

        "Mx_mean": float(
            df["Mx"].mean()
        ),

        "My_mean": float(
            df["My"].mean()
        ),

        "M_magnitude_mean": float(
            df["M_magnitude"].mean()
        ),

        "M_magnitude_std": float(
            df["M_magnitude"].std()
        ),

        "grain_coherence_mean": float(
            df["grain_coherence"].mean()
        ),

        "normalized_loop_work_proxy": loop_work,

        "alpha_beta_current_magnitude_std":
            ripple_rms,

    }

    return df, summary


# ============================================================
# 5. PLOTS
# ============================================================

def make_plots(
    all_data,
    summaries,
    outdir: Path
):

    # Representative case:
    # highest PWM frequency and highest ripple.
    key = sorted(all_data.keys())[-1]

    df = all_data[key]

    # --------------------------------------------------------
    # Representative time trace
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True
    )

    axes[0].plot(
        df.time_s * 1000,
        df.ia,
        label="ia"
    )

    axes[0].plot(
        df.time_s * 1000,
        df.ib,
        label="ib",
        alpha=0.8
    )

    axes[0].plot(
        df.time_s * 1000,
        df.ic,
        label="ic",
        alpha=0.8
    )

    axes[0].set_ylabel(
        "Normalized current"
    )

    axes[0].legend(ncol=3)

    axes[1].plot(
        df.time_s * 1000,
        df.hx,
        label="Hx"
    )

    axes[1].plot(
        df.time_s * 1000,
        df.hy,
        label="Hy"
    )

    axes[1].set_ylabel(
        "Dimensionless field"
    )

    axes[1].legend()

    axes[2].plot(
        df.time_s * 1000,
        df.Mx,
        label="Mx"
    )

    axes[2].plot(
        df.time_s * 1000,
        df.My,
        label="My"
    )

    axes[2].plot(
        df.time_s * 1000,
        df.M_magnitude,
        label="|M|"
    )

    axes[2].set_ylabel(
        "Magnetization"
    )

    axes[2].set_xlabel(
        "Time (ms)"
    )

    axes[2].legend(ncol=3)

    fig.suptitle(
        f"Stage H representative trace: {key}"
    )

    fig.tight_layout()

    fig.savefig(
        outdir / "stageH_representative_trace.png",
        dpi=180
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Summary plots
    # --------------------------------------------------------

    sdf = pd.DataFrame(summaries)

    plot_metrics = [

        (
            "M_magnitude_std",
            "Std. dev. of |M|",
            "stageH_magnitude_fluctuation.png"
        ),

        (
            "grain_coherence_mean",
            "Mean grain coherence",
            "stageH_grain_coherence.png"
        ),

        (
            "normalized_loop_work_proxy",
            "Normalized loop-work proxy",
            "stageH_loop_work_proxy.png"
        ),

    ]

    for metric, ylabel, filename in plot_metrics:

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        for ripple in sorted(
            sdf.ripple_fraction_peak.unique()
        ):

            sub = sdf[
                sdf.ripple_fraction_peak == ripple
            ].sort_values("pwm_hz")

            ax.plot(
                sub.pwm_hz / 1000,
                sub[metric],
                marker="o",
                label=f"ripple {100*ripple:g}%"
            )

        ax.set_xlabel(
            "PWM switching frequency (kHz)"
        )

        ax.set_ylabel(ylabel)

        ax.legend()

        ax.grid(
            True,
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            outdir / filename,
            dpi=180
        )

        plt.close(fig)


# ============================================================
# 6. COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description=__doc__
    )

    p.add_argument(
        "--fundamental-hz",
        type=float,
        default=50.0
    )

    p.add_argument(
        "--current-peak",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--reference-field",
        type=float,
        default=0.45
    )

    p.add_argument(
        "--pwm-frequencies",
        type=float,
        nargs="+",
        default=[
            4000,
            8000,
            12000,
            16000
        ]
    )

    p.add_argument(
        "--ripple-fractions",
        type=float,
        nargs="+",
        default=[
            0.0,
            0.01,
            0.03,
            0.05,
            0.10
        ]
    )

    p.add_argument(
        "--eta-boundary",
        type=float,
        default=0.20
    )

    p.add_argument(
        "--relaxation-time",
        type=float,
        default=1.0e-3,
        help=(
            "Phenomenological relaxation time in seconds; "
            "not material-calibrated."
        )
    )

    p.add_argument(
        "--cycles",
        type=float,
        default=2.0
    )

    p.add_argument(
        "--samples-per-pwm",
        type=int,
        default=20
    )

    p.add_argument(
        "--nx",
        type=int,
        default=40
    )

    p.add_argument(
        "--ny",
        type=int,
        default=40
    )

    p.add_argument(
        "--n-grains",
        type=int,
        default=100
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("stageH_results")
    )

    return p.parse_args()


# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def main():

    args = parse_args()

    if (
        args.fundamental_hz <= 0
        or args.current_peak <= 0
    ):
        raise SystemExit(
            "Fundamental frequency and current peak must be positive."
        )

    if (
        args.relaxation_time <= 0
        or args.samples_per_pwm < 8
    ):
        raise SystemExit(
            "Relaxation time must be positive and "
            "samples-per-pwm must be >= 8."
        )

    if any(
        f <= 0
        for f in args.pwm_frequencies
    ):
        raise SystemExit(
            "All PWM frequencies must be positive."
        )

    if any(
        r < 0 or r > 0.5
        for r in args.ripple_fractions
    ):
        raise SystemExit(
            "Ripple fractions must be between 0 and 0.5."
        )

    args.outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_data = {}
    summaries = []

    total = (
        len(args.pwm_frequencies)
        * len(args.ripple_fractions)
    )

    case = 0

    for fsw in args.pwm_frequencies:

        for ripple in args.ripple_fractions:

            case += 1

            print(
                f"[{case}/{total}] "
                f"PWM={fsw:g} Hz, "
                f"ripple={100*ripple:g}%",
                flush=True
            )

            df, summary = run_case(
                args,
                fsw,
                ripple,
                args.seed
            )

            label = (
                f"pwm_{int(fsw)}Hz_"
                f"ripple_{ripple:.3f}"
            )

            df.to_csv(
                args.outdir / f"{label}.csv",
                index=False
            )

            all_data[(fsw, ripple)] = df

            summaries.append(summary)

    # --------------------------------------------------------
    # Save summary CSV
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        args.outdir / "stageH_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "stage": "MAGNIKA Stage H",

        "model": (
            "2-D in-plane phenomenological "
            "overdamped relaxation"
        ),

        "input": (
            "Synthetic three-phase fundamental plus "
            "bipolar triangular current ripple"
        ),

        "physical_calibration": False,

        "parameters": (
            vars(args)
            | {"outdir": str(args.outdir)}
        ),

        "cases": len(summaries),

    }

    (
        args.outdir / "stageH_run_summary.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    make_plots(
        all_data,
        summaries,
        args.outdir
    )

    print("\nStage H completed.")

    print(
        summary_df.to_string(index=False)
    )

    print(
        f"\nOutputs: {args.outdir.resolve()}"
    )


if __name__ == "__main__":
    main()