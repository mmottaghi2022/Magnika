#!/usr/bin/env python3
"""
MAGNIKA Stage H2.1
Direct fundamental phase relationship between normalized magnetization M and
applied field H. This removes the physically invalid B* = H + M combination
used as a diagnostic in H2.

IMPORTANT:
- H is still dimensionless and mapped to the Stage-F reference field.
- M is normalized magnetization.
- Therefore M/H gain is a model diagnostic, NOT physical permeability.
- The M-versus-H phase is a property of this phenomenological model, not
  measured ferrite behavior.
- Relaxation is overdamped and phenomenological; this is not LLG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import stageb as base


def triangle_wave(phase):
    cycles = np.mod(phase / (2.0 * np.pi), 1.0)
    return 4.0 * np.abs(cycles - 0.5) - 1.0


def synthesize_currents(t, fe, fsw, ripple_fraction, current_peak):
    fundamental_phase = 2.0 * np.pi * fe * t
    offsets = np.array([0.0, -2.0*np.pi/3.0, 2.0*np.pi/3.0])
    fundamental = current_peak * np.sin(
        fundamental_phase[:, None] + offsets[None, :]
    )
    carrier_phase = 2.0*np.pi*fsw*t[:, None] + offsets[None, :]
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


def fundamental_phasor(z, t, frequency):
    """Trapezoidal Fourier coefficient over exactly one electrical cycle."""
    z = np.asarray(z, dtype=complex)
    kernel = z * np.exp(-2j * np.pi * frequency * t)
    duration = float(t[-1] - t[0])
    if duration <= 0:
        return np.nan + 1j*np.nan
    return (2.0 / duration) * np.trapezoid(kernel, t)


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
            "PWM/fundamental frequencies must produce an integer number "
            "of samples per electrical cycle."
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
    mx_values = np.empty(n_steps + 1)
    my_values = np.empty(n_steps + 1)

    for k in range(n_steps + 1):
        if k:
            theta = advance(
                model, theta, hx[k], hy[k],
                args.eta_boundary, dt, args.relaxation_time
            )
        mx_values[k], my_values[k] = model.magnetization(theta)

    h_complex = hx + 1j*hy
    m_complex = mx_values + 1j*my_values
    trace = pd.DataFrame({
        "time_s": t,
        "cycle": np.minimum(np.floor(np.arange(n_steps + 1) / samples_per_cycle).astype(int) + 1, args.cycles),
        "ia": currents[:, 0], "ib": currents[:, 1], "ic": currents[:, 2],
        "Hx": hx, "Hy": hy,
        "Mx": mx_values, "My": my_values,
        "H_magnitude": np.abs(h_complex),
        "M_magnitude": np.abs(m_complex),
    })

    cycle_rows = []
    for c in range(args.cycles):
        start = c * samples_per_cycle
        stop = (c + 1) * samples_per_cycle
        sl = slice(start, stop + 1)
        tc = t[sl]
        hc = h_complex[sl]
        mc = m_complex[sl]

        h1 = fundamental_phasor(hc, tc, args.fundamental_hz)
        m1 = fundamental_phasor(mc, tc, args.fundamental_hz)
        ratio = m1 / h1 if abs(h1) > 1e-14 else np.nan + 1j*np.nan
        phase_m_minus_h = float(np.degrees(np.angle(ratio))) if np.isfinite(ratio.real) else np.nan
        # With the rotating-vector convention used here, negative phase means M lags H.
        lag_m_vs_h = -phase_m_minus_h
        gain_m_over_h = float(abs(ratio)) if np.isfinite(ratio.real) else np.nan

        dmx = np.diff(mx_values[sl])
        dmy = np.diff(my_values[sl])
        hxm = 0.5*(hx[sl][:-1] + hx[sl][1:])
        hym = 0.5*(hy[sl][:-1] + hy[sl][1:])
        work_proxy = float(np.sum(hxm*dmx + hym*dmy))

        if c == 0:
            trajectory_delta = np.nan
        else:
            prev = slice(start - samples_per_cycle, stop - samples_per_cycle + 1)
            trajectory_delta = rms(np.concatenate([
                mx_values[sl] - mx_values[prev],
                my_values[sl] - my_values[prev],
            ]))

        cycle_rows.append({
            "cycle": c + 1,
            "H_fundamental_amplitude": float(abs(h1)),
            "M_fundamental_amplitude": float(abs(m1)),
            "M_over_H_fundamental_gain_model_only": gain_m_over_h,
            "M_phase_minus_H_phase_deg": phase_m_minus_h,
            "M_lag_vs_H_deg_model_only": lag_m_vs_h,
            "H_dot_dM_loop_work_proxy": work_proxy,
            "cycle_to_previous_M_trajectory_rms": trajectory_delta,
            "Mx_start": float(mx_values[start]),
            "My_start": float(my_values[start]),
            "Mx_end": float(mx_values[stop]),
            "My_end": float(my_values[stop]),
        })

    cycles_df = pd.DataFrame(cycle_rows)
    last = cycles_df.iloc[-1]
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
        "last_cycle_M_lag_vs_H_deg_model_only": float(last["M_lag_vs_H_deg_model_only"]),
        "last_cycle_M_over_H_gain_model_only": float(last["M_over_H_fundamental_gain_model_only"]),
        "last_cycle_work_proxy": float(last["H_dot_dM_loop_work_proxy"]),
        "last_cycle_trajectory_delta_rms": float(last["cycle_to_previous_M_trajectory_rms"])
            if args.cycles > 1 else np.nan,
    }
    return trace, cycles_df, summary


def make_plots(all_runs, outdir):
    key = sorted(all_runs.keys())[-1]
    trace, cycles_df, _ = all_runs[key]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(trace.time_s*1000, trace.Hx, label="Hx")
    axes[0].plot(trace.time_s*1000, trace.Hy, label="Hy")
    axes[0].set_ylabel("Applied H (dimensionless)")
    axes[0].legend()
    axes[1].plot(trace.time_s*1000, trace.Mx, label="Mx")
    axes[1].plot(trace.time_s*1000, trace.My, label="My")
    axes[1].set_ylabel("Normalized M")
    axes[1].legend()
    axes[2].plot(trace.time_s*1000, trace.H_magnitude, label="|H|")
    axes[2].plot(trace.time_s*1000, trace.M_magnitude, label="|M|")
    axes[2].set_ylabel("Magnitude")
    axes[2].set_xlabel("Time (ms)")
    axes[2].legend()
    fig.suptitle(f"H2.1 direct H/M response: {key}")
    fig.tight_layout()
    fig.savefig(outdir / "stageH21_time_response.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].plot(cycles_df.cycle, cycles_df.M_lag_vs_H_deg_model_only, marker="o")
    axes[0].set_ylabel("M lag vs H (degrees; model only)")
    axes[0].grid(True, alpha=0.25)
    axes[1].semilogy(
        cycles_df.cycle,
        np.maximum(cycles_df.cycle_to_previous_M_trajectory_rms, 1e-16),
        marker="o"
    )
    axes[1].set_ylabel("M trajectory RMS vs previous cycle")
    axes[1].set_xlabel("Electrical cycle")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "stageH21_cycle_convergence.png", dpi=180)
    plt.close(fig)

    summary = pd.DataFrame([v[2] for v in all_runs.values()])
    fig, ax = plt.subplots(figsize=(10, 5))
    for spp in sorted(summary.samples_per_pwm.unique()):
        sub = summary[summary.samples_per_pwm == spp]
        labels = [
            f"{int(r.pwm_hz/1000)}kHz / {100*r.ripple_fraction_peak:g}%"
            for r in sub.itertuples()
        ]
        ax.plot(labels, sub.last_cycle_M_lag_vs_H_deg_model_only,
                marker="o", label=f"{spp} samples/PWM")
    ax.set_ylabel("Last-cycle M lag vs H (degrees; model only)")
    ax.set_xlabel("PWM / ripple case")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "stageH21_ripple_timestep_comparison.png", dpi=180)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fundamental-hz", type=float, default=50.0)
    p.add_argument("--current-peak", type=float, default=1.0)
    p.add_argument("--reference-field", type=float, default=0.45)
    p.add_argument("--pwm-frequencies", type=float, nargs="+", default=[4000.0])
    p.add_argument("--ripple-fractions", type=float, nargs="+", default=[0.0, 0.03, 0.05, 0.10])
    p.add_argument("--samples-per-pwm", type=int, nargs="+", default=[20, 40])
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--eta-boundary", type=float, default=0.20)
    p.add_argument("--relaxation-time", type=float, default=1.0e-3)
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--n-grains", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("stageH21_results"))
    return p.parse_args()


def main():
    args = parse_args()
    if args.fundamental_hz <= 0 or args.current_peak <= 0 or args.relaxation_time <= 0:
        raise SystemExit("fundamental frequency, current peak and tau must be positive")
    if args.cycles < 2:
        raise SystemExit("Use at least 2 cycles to assess convergence.")
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
    count = 0

    for fsw in args.pwm_frequencies:
        for ripple in args.ripple_fractions:
            for spp in args.samples_per_pwm:
                count += 1
                print(
                    f"[{count}/{total}] PWM={fsw:g} Hz, ripple={100*ripple:g}%, "
                    f"samples/PWM={spp}, cycles={args.cycles}",
                    flush=True
                )
                trace, cycles_df, summary = run_case(args, fsw, ripple, spp)
                label = f"pwm_{int(fsw)}Hz_ripple_{ripple:.3f}_spp_{spp}"
                trace.to_csv(args.outdir / f"{label}_trace.csv", index=False)
                cycles_df.to_csv(args.outdir / f"{label}_cycles.csv", index=False)
                all_runs[(fsw, ripple, spp)] = (trace, cycles_df, summary)
                summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.outdir / "stageH21_summary.csv", index=False)
    metadata = {
        "stage": "MAGNIKA H2.1",
        "measurement": "direct fundamental phase relationship between normalized M and applied H",
        "Bstar_H_plus_M_removed": True,
        "H_scale": "dimensionless Stage-F reference-field mapping",
        "M_scale": "normalized magnetization",
        "M_over_H_gain_is_physical_permeability": False,
        "phase_is_measured_ferrite_property": False,
        "dynamic_model": "phenomenological overdamped angular relaxation; not LLG",
        "material_calibrated": False,
        "synthetic_current_ripple": True,
        "parameters": {**vars(args), "outdir": str(args.outdir)},
        "cases": len(summaries),
    }
    (args.outdir / "stageH21_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    make_plots(all_runs, args.outdir)
    print("\nMAGNIKA Stage H2.1 completed.")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
