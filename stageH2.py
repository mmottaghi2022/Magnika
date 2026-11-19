#!/usr/bin/env python3
"""
MAGNIKA Stage H2
Dynamic B*-H response, phase lag, cycle convergence, and timestep sensitivity.

This extends Stage H's 2-D in-plane phenomenological model.
B* = H + M is dimensionless (B* corresponds to B/mu_0 in SI).
The relaxation law is phenomenological overdamped angular relaxation,
not LLG, and tau is not calibrated to a real ferrite.

Place stageH2.py beside stageb.py.
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
        "Cannot import stageb.py. Put stageH2.py beside the verified Stage B/C file."
    ) from exc


def triangle_wave(phase):
    cycles = np.mod(phase / (2.0 * np.pi), 1.0)
    return 4.0 * np.abs(cycles - 0.5) - 1.0


def synthesize_currents(t, fe, fsw, ripple_fraction, current_peak):
    fundamental_phase = 2.0 * np.pi * fe * t
    phase_offsets = np.array([0.0, -2.0*np.pi/3.0, 2.0*np.pi/3.0])
    carrier_offsets = np.array([0.0, -2.0*np.pi/3.0, 2.0*np.pi/3.0])

    fundamental = current_peak * np.sin(
        fundamental_phase[:, None] + phase_offsets[None, :]
    )
    carrier_phase = (
        2.0*np.pi*fsw*t[:, None] + carrier_offsets[None, :]
    )
    ripple = ripple_fraction * current_peak * triangle_wave(carrier_phase)
    currents = fundamental + ripple

    ia, ib, ic = currents.T
    i_alpha = (2.0/3.0) * (ia - 0.5*ib - 0.5*ic)
    i_beta = (2.0/3.0) * (np.sqrt(3.0)/2.0) * (ib - ic)
    return currents, i_alpha, i_beta


def advance(model, theta, hx, hy, eta, dt, tau):
    _, grad, _ = model.energy_gradient(theta, hx, hy, eta)
    theta = theta - (dt / tau) * grad
    return (theta + np.pi) % (2.0*np.pi) - np.pi


def lockin_phasor(z, t, frequency):
    """Complex fundamental coefficient; normalization cancels in ratios."""
    phase = np.exp(-2j * np.pi * frequency * t)
    return np.mean(np.asarray(z, dtype=complex) * phase)


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x*x)))


def run_case(args, fsw, ripple, samples_per_pwm):
    grain_map, grain_angles = base.generate_grains(
        args.nx, args.ny, args.n_grains, args.seed
    )
    model = base.MicroMagneticModel(
        grain_map, grain_angles, K=base.K_ANIS, J=base.J_EXCHANGE
    )

    dt = 1.0 / (samples_per_pwm * fsw)
    samples_per_cycle_float = (1.0 / args.fundamental_hz) / dt
    samples_per_cycle = int(round(samples_per_cycle_float))
    if not np.isclose(samples_per_cycle_float, samples_per_cycle, rtol=0, atol=1e-9):
        raise SystemExit(
            "The selected PWM/fundamental frequencies do not give an integer "
            "number of samples per electrical cycle. Choose compatible values."
        )

    n_steps = args.cycles * samples_per_cycle
    t = np.arange(n_steps + 1, dtype=float) * dt
    currents, i_alpha, i_beta = synthesize_currents(
        t, args.fundamental_hz, fsw, ripple, args.current_peak
    )

    field_scale = args.reference_field / args.current_peak
    hx = field_scale * i_alpha
    hy = field_scale * i_beta

    theta = np.zeros(model.n, dtype=float)
    rows = []
    for k in range(n_steps + 1):
        if k:
            theta = advance(
                model, theta, hx[k], hy[k],
                args.eta_boundary, dt, args.relaxation_time
            )
        mx, my = model.magnetization(theta)
        bx = hx[k] + mx
        by = hy[k] + my
        rows.append({
            "time_s": t[k],
            "cycle": min(k // samples_per_cycle + 1, args.cycles),
            "ia": currents[k, 0], "ib": currents[k, 1], "ic": currents[k, 2],
            "Hx": hx[k], "Hy": hy[k],
            "Mx": mx, "My": my,
            "Bx_star": bx, "By_star": by,
            "H_magnitude": np.hypot(hx[k], hy[k]),
            "B_star_magnitude": np.hypot(bx, by),
        })

    df = pd.DataFrame(rows)
    cycle_rows = []
    cycle_traces = []

    for c in range(args.cycles):
        start = c * samples_per_cycle
        stop = (c + 1) * samples_per_cycle
        # Include the endpoint for integration and trajectory comparison.
        sl = slice(start, stop + 1)
        tc = t[sl]
        hxc, hyc = hx[sl], hy[sl]
        mxc = df["Mx"].to_numpy()[sl]
        myc = df["My"].to_numpy()[sl]
        bxc = hxc + mxc
        byc = hyc + myc

        h_phasor = lockin_phasor(hxc + 1j*hyc, tc, args.fundamental_hz)
        b_phasor = lockin_phasor(bxc + 1j*byc, tc, args.fundamental_hz)
        ratio = b_phasor / h_phasor if abs(h_phasor) > 1e-14 else np.nan + 1j*np.nan
        phase_lag = -np.degrees(np.angle(ratio)) if np.isfinite(ratio.real) else np.nan
        gain = abs(ratio) if np.isfinite(ratio.real) else np.nan

        # Vector magnetic-work proxy for this cycle: integral H dot dM.
        dmx, dmy = np.diff(mxc), np.diff(myc)
        hxmid = 0.5*(hxc[:-1] + hxc[1:])
        hymid = 0.5*(hyc[:-1] + hyc[1:])
        work_proxy = float(np.sum(hxmid*dmx + hymid*dmy))

        if c == 0:
            cycle_delta = np.nan
        else:
            prev_start = (c - 1) * samples_per_cycle
            prev = slice(prev_start, prev_start + samples_per_cycle + 1)
            prev_mx = df["Mx"].to_numpy()[prev]
            prev_my = df["My"].to_numpy()[prev]
            cycle_delta = rms(np.concatenate([mxc-prev_mx, myc-prev_my]))

        cycle_rows.append({
            "cycle": c + 1,
            "Bstar_over_H_fundamental_gain": gain,
            "Bstar_lag_vs_H_deg": phase_lag,
            "Bstar_lead_vs_H_deg": -phase_lag,
            "H_dot_dM_loop_work_proxy": work_proxy,
            "cycle_to_previous_M_trajectory_rms": cycle_delta,
            "Mx_start": float(mxc[0]), "My_start": float(myc[0]),
            "Mx_end": float(mxc[-1]), "My_end": float(myc[-1]),
        })
        cycle_traces.append((tc, hxc, mxc, bxc))

    cycles_df = pd.DataFrame(cycle_rows)
    summary = {
        "seed": args.seed,
        "fundamental_hz": args.fundamental_hz,
        "pwm_hz": fsw,
        "ripple_fraction_peak": ripple,
        "samples_per_pwm": samples_per_pwm,
        "dt_s": dt,
        "samples_per_electrical_cycle": samples_per_cycle,
        "cycles": args.cycles,
        "eta_boundary": args.eta_boundary,
        "relaxation_time_s_model_parameter": args.relaxation_time,
        "last_cycle_Bstar_over_H_gain": float(cycles_df.iloc[-1]["Bstar_over_H_fundamental_gain"]),
        "last_cycle_Bstar_lag_deg": float(cycles_df.iloc[-1]["Bstar_lag_vs_H_deg"]),
        "last_cycle_work_proxy": float(cycles_df.iloc[-1]["H_dot_dM_loop_work_proxy"]),
        "last_cycle_trajectory_delta_rms": float(cycles_df.iloc[-1]["cycle_to_previous_M_trajectory_rms"])
            if args.cycles > 1 else np.nan,
    }
    return df, cycles_df, summary, cycle_traces


def make_plots(all_runs, outdir):
    # Use the highest-resolution, highest-ripple run for the detailed trace.
    key = sorted(all_runs.keys())[-1]
    df, cycles_df, _, _ = all_runs[key]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df.time_s*1000, df.Hx, label="Hx")
    axes[0].plot(df.time_s*1000, df.Hy, label="Hy")
    axes[0].set_ylabel("Applied H (dimensionless)")
    axes[0].legend()
    axes[1].plot(df.time_s*1000, df.Bx_star, label="Bx* = Hx + Mx")
    axes[1].plot(df.time_s*1000, df.By_star, label="By* = Hy + My")
    axes[1].set_ylabel("B* (dimensionless)")
    axes[1].legend()
    axes[2].plot(df.time_s*1000, df.Mx, label="Mx")
    axes[2].plot(df.time_s*1000, df.My, label="My")
    axes[2].set_ylabel("Magnetization")
    axes[2].set_xlabel("Time (ms)")
    axes[2].legend()
    fig.suptitle(f"H2 time response: {key}")
    fig.tight_layout()
    fig.savefig(outdir / "stageH2_time_response.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df.Hx, df.Bx_star, label="Bx* vs Hx")
    ax.set_xlabel("Hx (dimensionless)")
    ax.set_ylabel("Bx* (dimensionless)")
    ax.set_title("Component B*-H trajectory (all cycles)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "stageH2_component_BH_loop.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].plot(cycles_df.cycle, cycles_df.Bstar_lag_vs_H_deg, marker="o")
    axes[0].set_ylabel("B* fundamental lag vs H (deg)")
    axes[0].grid(True, alpha=0.25)
    axes[1].semilogy(
        cycles_df.cycle,
        np.maximum(cycles_df.cycle_to_previous_M_trajectory_rms.fillna(np.nan), 1e-16),
        marker="o"
    )
    axes[1].set_ylabel("M trajectory RMS change vs prior cycle")
    axes[1].set_xlabel("Electrical cycle")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "stageH2_cycle_convergence.png", dpi=180)
    plt.close(fig)

    summary = pd.DataFrame([v[2] for v in all_runs.values()])
    fig, ax = plt.subplots(figsize=(9, 5))
    for spp in sorted(summary.samples_per_pwm.unique()):
        sub = summary[summary.samples_per_pwm == spp]
        labels = [
            f"{int(r.pwm_hz/1000)}kHz / {100*r.ripple_fraction_peak:g}%"
            for r in sub.itertuples()
        ]
        ax.plot(labels, sub.last_cycle_Bstar_lag_deg, marker="o", label=f"{spp} samples/PWM")
    ax.set_ylabel("Last-cycle B* fundamental lag vs H (deg)")
    ax.set_xlabel("PWM / current-ripple case")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "stageH2_timestep_comparison.png", dpi=180)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fundamental-hz", type=float, default=50.0)
    p.add_argument("--current-peak", type=float, default=1.0)
    p.add_argument("--reference-field", type=float, default=0.45)
    p.add_argument("--pwm-frequencies", type=float, nargs="+", default=[4000.0])
    p.add_argument("--ripple-fractions", type=float, nargs="+", default=[0.0, 0.05])
    p.add_argument("--samples-per-pwm", type=int, nargs="+", default=[20, 40])
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--eta-boundary", type=float, default=0.20)
    p.add_argument("--relaxation-time", type=float, default=1.0e-3)
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--n-grains", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("stageH2_results"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.fundamental_hz <= 0 or args.current_peak <= 0 or args.relaxation_time <= 0:
        raise SystemExit("fundamental frequency, current peak and tau must be positive")
    if args.cycles < 2:
        raise SystemExit("Use at least 2 cycles to assess cycle-to-cycle convergence.")
    if any(f <= 0 for f in args.pwm_frequencies):
        raise SystemExit("PWM frequencies must be positive.")
    if any(r < 0 or r > 0.5 for r in args.ripple_fractions):
        raise SystemExit("Ripple fractions must be between 0 and 0.5.")
    if any(s < 8 for s in args.samples_per_pwm):
        raise SystemExit("Use at least 8 samples per PWM period.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_runs = {}
    summaries = []
    total = len(args.pwm_frequencies)*len(args.ripple_fractions)*len(args.samples_per_pwm)
    n = 0

    for fsw in args.pwm_frequencies:
        for ripple in args.ripple_fractions:
            for spp in args.samples_per_pwm:
                n += 1
                print(
                    f"[{n}/{total}] PWM={fsw:g} Hz, ripple={100*ripple:g}%, "
                    f"samples/PWM={spp}, cycles={args.cycles}",
                    flush=True
                )
                df, cdf, summary, traces = run_case(args, fsw, ripple, spp)
                label = f"pwm_{int(fsw)}Hz_ripple_{ripple:.3f}_spp_{spp}"
                df.to_csv(args.outdir / f"{label}_trace.csv", index=False)
                cdf.to_csv(args.outdir / f"{label}_cycles.csv", index=False)
                all_runs[(fsw, ripple, spp)] = (df, cdf, summary, traces)
                summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.outdir / "stageH2_summary.csv", index=False)
    (args.outdir / "stageH2_metadata.json").write_text(
        json.dumps({
            "stage": "MAGNIKA H2",
            "definition": "Bstar = H + M; Bstar is dimensionless and corresponds to B/mu0",
            "model": "2-D in-plane phenomenological overdamped angular relaxation",
            "full_LLG": False,
            "material_calibrated": False,
            "synthetic_current_ripple": True,
            "parameters": {**vars(args), "outdir": str(args.outdir)},
            "cases": len(summaries),
        }, indent=2, default=str),
        encoding="utf-8"
    )
    make_plots(all_runs, args.outdir)
    print("\nMAGNIKA Stage H2 completed.")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
