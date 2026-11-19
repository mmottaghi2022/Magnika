#!/usr/bin/env python3
"""
MAGNIKA Stage H2.6: grain-level dynamics, relaxation-time sensitivity,
and full-microstate periodicity analysis.

This extends H2.5 without changing the Stage B energy model, Stage H2.3
synthetic current waveform, or Euler/RK4 integrators.

The model remains dimensionless and phenomenological. It is not a calibrated
ferrite model, an LLG solver, or a physical loss/thermal model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stageb as base
import stageH23 as h23


def circular_difference(a, b):
    d = np.asarray(a) - np.asarray(b)
    return np.arctan2(np.sin(d), np.cos(d))


def circular_std_from_resultant(r):
    return np.sqrt(np.maximum(0.0, -2.0 * np.log(np.clip(r, 1e-15, 1.0))))


def grain_observables(theta, grain_flat, counts, n_grains):
    """Per-grain mean direction, coherence and within-grain circular spread."""
    c = np.cos(theta)
    s = np.sin(theta)
    sum_c = np.bincount(grain_flat, weights=c, minlength=n_grains)
    sum_s = np.bincount(grain_flat, weights=s, minlength=n_grains)
    mx = sum_c / counts
    my = sum_s / counts
    resultant = np.clip(np.hypot(mx, my), 0.0, 1.0)
    angle = np.arctan2(my, mx)
    spread = circular_std_from_resultant(resultant)
    return mx, my, angle, resultant, spread


def grain_rows_at_time(theta, time_s, boundary_index, grain_flat, counts,
                       fe, fsw, tau_ms, n_grains):
    gmx, gmy, gang, gres, gspread = grain_observables(
        theta, grain_flat, counts, n_grains
    )
    return pd.DataFrame({
        "time_s": time_s,
        "electrical_cycle": int(round(time_s * fe)),
        "pwm_boundary_index": boundary_index,
        "fundamental_hz": fe,
        "pwm_hz": fsw,
        "relaxation_time_ms": tau_ms,
        "grain_id": np.arange(n_grains),
        "grain_Mx_normalized": gmx,
        "grain_My_normalized": gmy,
        "grain_mean_angle_rad": gang,
        "grain_coherence": gres,
        "within_grain_circular_std_rad": gspread,
        "grain_pixel_count": counts,
    })


def run_case(args, fe, fsw, spp, tau_ms, integrator, outdir):
    args.fundamental_hz = float(fe)
    args.relaxation_time = float(tau_ms) * 1e-3

    grain_map, grain_angles = base.generate_grains(
        args.nx, args.ny, args.n_grains, args.seed
    )
    model = base.MicroMagneticModel(
        grain_map, grain_angles, K=base.K_ANIS, J=base.J_EXCHANGE
    )
    grain_flat = np.asarray(grain_map, dtype=int).ravel()
    n_grains = int(args.n_grains)
    counts = np.bincount(grain_flat, minlength=n_grains).astype(float)
    if np.any(counts == 0):
        raise RuntimeError("At least one generated grain has no pixels.")

    dt = 1.0 / (spp * fsw)
    spc_float = (1.0 / fe) / dt
    spc = int(round(spc_float))
    if not np.isclose(spc_float, spc, rtol=0, atol=1e-8):
        raise SystemExit(
            f"{fe:g} Hz fundamental and {fsw:g} Hz PWM do not produce "
            "an integer samples-per-electrical-cycle count."
        )
    pwm_per_cycle_float = fsw / fe
    pwm_per_cycle = int(round(pwm_per_cycle_float))
    if not np.isclose(pwm_per_cycle_float, pwm_per_cycle, rtol=0, atol=1e-10):
        raise SystemExit("PWM periods per electrical cycle must be an integer.")

    n_steps = args.cycles * spc
    t = np.arange(n_steps + 1, dtype=float) * dt
    currents, i_alpha, i_beta = h23.synthesize_currents(
        t, fe, fsw, args.ripple_fraction, args.current_peak
    )
    field_scale = args.reference_field / args.current_peak
    hx, hy = field_scale * i_alpha, field_scale * i_beta

    theta = np.zeros(model.n, dtype=float)
    mx = np.empty(n_steps + 1)
    my = np.empty(n_steps + 1)
    energy = np.empty(n_steps + 1)
    grad2_mean = np.empty(n_steps + 1)

    mx[0], my[0] = model.magnetization(theta)
    _, energy[0], grad0 = h23.rhs(
        model, theta, t[0], args, fsw, args.ripple_fraction
    )
    grad2_mean[0] = float(np.mean(grad0 * grad0))

    # Save the complete microstate at every PWM boundary.
    theta_pwm = [theta.copy()]
    grain_frames = [
        grain_rows_at_time(theta, t[0], 0, grain_flat, counts,
                           fe, fsw, tau_ms, n_grains)
    ]
    stepper = h23.step_euler if integrator == "euler" else h23.step_rk4

    for k in range(n_steps):
        theta, energy[k], grad2_mean[k] = stepper(
            model, theta, t[k], dt, args, fsw, args.ripple_fraction
        )
        mx[k + 1], my[k + 1] = model.magnetization(theta)
        if (k + 1) % spp == 0:
            boundary_index = (k + 1) // spp
            theta_pwm.append(theta.copy())
            grain_frames.append(
                grain_rows_at_time(
                    theta, t[k + 1], boundary_index, grain_flat, counts,
                    fe, fsw, tau_ms, n_grains
                )
            )

    _, energy[-1], grad_last = h23.rhs(
        model, theta, t[-1], args, fsw, args.ripple_fraction
    )
    grad2_mean[-1] = float(np.mean(grad_last * grad_last))

    n_spins = int(model.n)
    dissipation_rate = n_spins * grad2_mean / args.relaxation_time
    dW = -n_spins * (
        0.5 * (mx[:-1] + mx[1:]) * np.diff(hx)
        + 0.5 * (my[:-1] + my[1:]) * np.diff(hy)
    )
    dD = 0.5 * (dissipation_rate[:-1] + dissipation_rate[1:]) * dt
    W_cum = np.r_[0.0, np.cumsum(dW)]
    D_cum = np.r_[0.0, np.cumsum(dD)]
    balance_residual = energy - energy[0] - W_cum + D_cum

    trace = pd.DataFrame({
        "time_s": t,
        "ia": currents[:, 0], "ib": currents[:, 1], "ic": currents[:, 2],
        "Hx_model_units": hx, "Hy_model_units": hy,
        "Mx_normalized": mx, "My_normalized": my,
        "energy_model_units": energy,
        "external_work_cumulative_model_units": W_cum,
        "relaxation_dissipation_cumulative_model_units": D_cum,
        "energy_balance_residual_model_units": balance_residual,
        "gradient_squared_sum_model_units": n_spins * grad2_mean,
    })

    cycle_rows = []
    grain_cycle_rows = []
    for c in range(args.cycles):
        start, stop = c * spc, (c + 1) * spc
        sl = slice(start, stop + 1)
        tc = t[sl]
        hvec = hx[sl] + 1j * hy[sl]
        mvec = mx[sl] + 1j * my[sl]
        h1 = h23.fundamental_phasor(hvec, tc, fe)
        m1 = h23.fundamental_phasor(mvec, tc, fe)
        ratio = m1 / h1 if abs(h1) > 1e-14 else np.nan + 1j * np.nan
        phase = float(np.degrees(np.angle(ratio))) if np.isfinite(ratio.real) else np.nan
        lag = -phase
        loop_proxy = float(np.sum(
            0.5 * (hx[sl][:-1] + hx[sl][1:]) * np.diff(mx[sl])
            + 0.5 * (hy[sl][:-1] + hy[sl][1:]) * np.diff(my[sl])
        ))
        dE = float(energy[stop] - energy[start])
        W = float(W_cum[stop] - W_cum[start])
        D = float(D_cum[stop] - D_cum[start])
        R = float(dE - W + D)
        scale = max(abs(dE), abs(W), abs(D), 1e-14)
        mdelta = np.nan if c == 0 else h23.rms(np.r_[
            mx[sl] - mx[start-spc:stop-spc+1],
            my[sl] - my[start-spc:stop-spc+1]
        ])

        cycle_rows.append({
            "cycle": c + 1, "integrator": integrator,
            "fundamental_hz": fe, "pwm_hz": fsw,
            "ripple_fraction_peak": args.ripple_fraction,
            "samples_per_pwm": spp, "dt_s": dt,
            "relaxation_time_ms": tau_ms,
            "M_lag_vs_H_deg_model_only": lag,
            "M_over_H_gain_model_only": float(abs(ratio)) if np.isfinite(ratio.real) else np.nan,
            "M_loop_work_proxy_model_only": loop_proxy,
            "M_trajectory_delta_rms": mdelta,
            "delta_energy_model_units": dE,
            "external_work_model_units": W,
            "relaxation_dissipation_model_units": D,
            "energy_balance_residual_model_units": R,
            "energy_balance_residual_relative": abs(R) / scale,
            "final_cumulative_energy_balance_residual_model_units":
                float(balance_residual[stop]),
        })

        # Aggregate grain-level statistics at each electrical-cycle boundary.
        pwm_idx = (c + 1) * pwm_per_cycle
        frame = grain_frames[pwm_idx]
        ang = frame["grain_mean_angle_rad"].to_numpy()
        coherence = frame["grain_coherence"].to_numpy()
        within = frame["within_grain_circular_std_rad"].to_numpy()
        intergrain_resultant = float(np.abs(np.mean(np.exp(1j * ang))))
        grain_cycle_rows.append({
            "cycle": c + 1, "time_s": (c + 1) / fe,
            "fundamental_hz": fe, "relaxation_time_ms": tau_ms,
            "global_Mx": mx[stop], "global_My": my[stop],
            "intergrain_mean_direction_coherence": intergrain_resultant,
            "intergrain_circular_std_rad":
                float(circular_std_from_resultant(intergrain_resultant)),
            "mean_grain_coherence": float(np.mean(coherence)),
            "min_grain_coherence": float(np.min(coherence)),
            "mean_within_grain_circular_std_rad": float(np.mean(within)),
            "max_within_grain_circular_std_rad": float(np.max(within)),
        })

    # Full-state recurrence: compare complete PWM-phase trajectories separated
    # by p electrical cycles, using the last up-to-five available comparisons.
    states = np.asarray(theta_pwm)
    periodicity_rows = []
    max_p = min(args.periodicity_max_period, args.cycles - 1)
    for pcycle in range(1, max_p + 1):
        pair_values = []
        pair_global = []
        first_c = max(pcycle, args.cycles - args.periodicity_tail_pairs)
        for current_cycle in range(first_c, args.cycles):
            prior_cycle = current_cycle - pcycle
            a0 = prior_cycle * pwm_per_cycle
            a1 = (prior_cycle + 1) * pwm_per_cycle
            b0 = current_cycle * pwm_per_cycle
            b1 = (current_cycle + 1) * pwm_per_cycle
            delta = circular_difference(states[b0:b1+1], states[a0:a1+1])
            pair_values.append(float(np.sqrt(np.mean(delta * delta))))
            # Global M trajectories at the same PWM-boundary phase samples.
            ia0, ia1 = prior_cycle * pwm_per_cycle, (prior_cycle + 1) * pwm_per_cycle
            ib0, ib1 = current_cycle * pwm_per_cycle, (current_cycle + 1) * pwm_per_cycle
            ma = np.column_stack((mx[ia0*spp:ia1*spp+1:spp],
                                  my[ia0*spp:ia1*spp+1:spp]))
            mb = np.column_stack((mx[ib0*spp:ib1*spp+1:spp],
                                  my[ib0*spp:ib1*spp+1:spp]))
            pair_global.append(float(np.sqrt(np.mean((mb - ma) ** 2))))
        periodicity_rows.append({
            "periodicity_lag_electrical_cycles": pcycle,
            "full_microstate_trajectory_rms_rad_mean_last_pairs":
                float(np.mean(pair_values)),
            "full_microstate_trajectory_rms_rad_max_last_pairs":
                float(np.max(pair_values)),
            "global_M_trajectory_rms_mean_last_pairs":
                float(np.mean(pair_global)),
            "number_of_pairs": len(pair_values),
            "tail_pairs_requested": args.periodicity_tail_pairs,
        })

    grain_evolution = pd.concat(grain_frames, ignore_index=True)
    grain_cycles = pd.DataFrame(grain_cycle_rows)
    cycles_df = pd.DataFrame(cycle_rows)
    periodicity_df = pd.DataFrame(periodicity_rows)
    last = cycles_df.iloc[-1]
    summary = {
        "integrator": integrator,
        "fundamental_hz": fe,
        "pwm_hz": fsw,
        "ripple_fraction_peak": args.ripple_fraction,
        "samples_per_pwm": spp,
        "dt_s": dt,
        "cycles": args.cycles,
        "eta_boundary": args.eta_boundary,
        "relaxation_time_ms": tau_ms,
        "last_cycle_M_lag_vs_H_deg_model_only": float(last.M_lag_vs_H_deg_model_only),
        "last_cycle_M_over_H_gain_model_only": float(last.M_over_H_gain_model_only),
        "last_cycle_M_loop_work_proxy_model_only": float(last.M_loop_work_proxy_model_only),
        "last_cycle_M_trajectory_delta_rms": float(last.M_trajectory_delta_rms),
        "last_cycle_delta_energy_model_units": float(last.delta_energy_model_units),
        "last_cycle_external_work_model_units": float(last.external_work_model_units),
        "last_cycle_relaxation_dissipation_model_units": float(last.relaxation_dissipation_model_units),
        "last_cycle_energy_balance_residual_model_units": float(last.energy_balance_residual_model_units),
        "last_cycle_energy_balance_residual_relative": float(last.energy_balance_residual_relative),
        "final_cumulative_energy_balance_residual_model_units": float(balance_residual[-1]),
        "final_intergrain_coherence": float(grain_cycle_rows[-1]["intergrain_mean_direction_coherence"]),
        "final_mean_within_grain_std_rad": float(grain_cycle_rows[-1]["mean_within_grain_circular_std_rad"]),
        "final_full_state_period1_rms_rad":
            float(periodicity_df.loc[periodicity_df.periodicity_lag_electrical_cycles == 1,
                                     "full_microstate_trajectory_rms_rad_mean_last_pairs"].iloc[0]),
    }
    tag = f"{int(fe)}Hz_tau{tau_ms:g}ms_{integrator}_spp{spp}"
    trace.to_csv(outdir / f"{tag}_trace.csv", index=False)
    cycles_df.to_csv(outdir / f"{tag}_energy_balance_cycles.csv", index=False)
    grain_evolution.to_csv(outdir / f"{tag}_grain_evolution.csv", index=False)
    grain_cycles.to_csv(outdir / f"{tag}_grain_cycle_summary.csv", index=False)
    periodicity_df.to_csv(outdir / f"{tag}_periodicity.csv", index=False)
    np.savez_compressed(
        outdir / f"{tag}_grain_states_pwm_boundaries.npz",
        theta=states, samples_per_pwm=spp, fundamental_hz=fe,
        pwm_hz=fsw, cycles=args.cycles, relaxation_time_ms=tau_ms,
        grain_map=grain_map,
    )
    return summary, cycles_df, grain_cycles, periodicity_df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fundamental-frequencies", type=float, nargs="+", default=[100, 200, 400])
    p.add_argument("--pwm-frequency", type=float, default=4000.0)
    p.add_argument("--ripple-fraction", type=float, default=0.0)
    p.add_argument("--samples-per-pwm", type=int, default=40)
    p.add_argument("--integrator", choices=["euler", "rk4"], default="rk4")
    p.add_argument("--cycles", type=int, default=50)
    p.add_argument("--relaxation-times-ms", type=float, nargs="+", default=[1.0])
    p.add_argument("--periodicity-max-period", type=int, default=15)
    p.add_argument("--periodicity-tail-pairs", type=int, default=5)
    p.add_argument("--current-peak", type=float, default=1.0)
    p.add_argument("--reference-field", type=float, default=0.45)
    p.add_argument("--eta-boundary", type=float, default=0.20)
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--n-grains", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("stageH26_results"))
    args = p.parse_args()

    if min(args.fundamental_frequencies) <= 0 or args.pwm_frequency <= 0:
        p.error("frequencies must be positive")
    if args.samples_per_pwm < 8 or args.cycles < 2:
        p.error("use at least 8 samples per PWM period and 2 cycles")
    if args.current_peak <= 0 or min(args.relaxation_times_ms) <= 0:
        p.error("current peak and relaxation times must be positive")
    if not 0 <= args.ripple_fraction <= 0.5:
        p.error("ripple fraction must be between 0 and 0.5")
    if args.periodicity_max_period < 1 or args.periodicity_tail_pairs < 1:
        p.error("periodicity limits must be positive")
    if len(set(args.relaxation_times_ms)) != len(args.relaxation_times_ms):
        p.error("relaxation times must be unique")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summaries, all_cycles, all_grain_cycles, all_periodicity = [], [], [], []
    total = len(args.fundamental_frequencies) * len(args.relaxation_times_ms)
    count = 0
    for fe in args.fundamental_frequencies:
        for tau_ms in args.relaxation_times_ms:
            count += 1
            print(
                f"[{count}/{total}] f={fe:g} Hz, PWM={args.pwm_frequency:g} Hz, "
                f"tau={tau_ms:g} ms, spp={args.samples_per_pwm}, "
                f"integrator={args.integrator}, cycles={args.cycles}",
                flush=True,
            )
            summary, cyc, gc, per = run_case(
                args, fe, args.pwm_frequency, args.samples_per_pwm,
                tau_ms, args.integrator, args.outdir
            )
            summaries.append(summary)
            all_cycles.append(cyc)
            all_grain_cycles.append(gc)
            per.insert(0, "fundamental_hz", fe)
            per.insert(1, "relaxation_time_ms", tau_ms)
            all_periodicity.append(per)

    summary_df = pd.DataFrame(summaries)
    cycles_df = pd.concat(all_cycles, ignore_index=True)
    grain_cycles_df = pd.concat(all_grain_cycles, ignore_index=True)
    periodicity_df = pd.concat(all_periodicity, ignore_index=True)
    summary_df.to_csv(args.outdir / "stageH26_summary.csv", index=False)
    cycles_df.to_csv(args.outdir / "stageH26_energy_balance_cycles.csv", index=False)
    grain_cycles_df.to_csv(args.outdir / "stageH26_grain_cycle_summary.csv", index=False)
    periodicity_df.to_csv(args.outdir / "stageH26_periodicity.csv", index=False)

    # Sensitivity relative to tau=1 ms, where that reference case was included.
    sensitivity_rows = []
    for fe in args.fundamental_frequencies:
        subset = summary_df[summary_df.fundamental_hz == fe]
        ref = subset[np.isclose(subset.relaxation_time_ms, 1.0)]
        if len(ref) == 1:
            ref = ref.iloc[0]
            for _, row in subset.iterrows():
                sensitivity_rows.append({
                    "fundamental_hz": fe,
                    "relaxation_time_ms": row.relaxation_time_ms,
                    "reference_tau_ms": 1.0,
                    "last_cycle_phase_lag_change_deg":
                        row.last_cycle_M_lag_vs_H_deg_model_only - ref.last_cycle_M_lag_vs_H_deg_model_only,
                    "last_cycle_gain_change":
                        row.last_cycle_M_over_H_gain_model_only - ref.last_cycle_M_over_H_gain_model_only,
                    "period1_full_state_rms_rad": row.final_full_state_period1_rms_rad,
                    "final_intergrain_coherence": row.final_intergrain_coherence,
                })
    if sensitivity_rows:
        pd.DataFrame(sensitivity_rows).to_csv(
            args.outdir / "stageH26_relaxation_sensitivity_vs_1ms.csv", index=False
        )

    # Plots: grain-level metrics and complete-state periodicity.
    for fe in args.fundamental_frequencies:
        fig, ax = plt.subplots(figsize=(9, 5))
        for tau in args.relaxation_times_ms:
            q = grain_cycles_df[
                (grain_cycles_df.fundamental_hz == fe)
                & np.isclose(grain_cycles_df.relaxation_time_ms, tau)
            ].sort_values("cycle")
            ax.plot(q.cycle, q.intergrain_mean_direction_coherence,
                    marker=".", label=f"tau={tau:g} ms")
        ax.set_xlabel("Electrical cycle")
        ax.set_ylabel("Intergrain mean-direction coherence")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"H2.6 intergrain coherence — {fe:g} Hz")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / f"H26_{int(fe)}Hz_intergrain_coherence.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for tau in args.relaxation_times_ms:
            q = periodicity_df[
                (periodicity_df.fundamental_hz == fe)
                & np.isclose(periodicity_df.relaxation_time_ms, tau)
            ].sort_values("periodicity_lag_electrical_cycles")
            ax.loglog(q.periodicity_lag_electrical_cycles,
                     np.maximum(q.full_microstate_trajectory_rms_rad_mean_last_pairs, 1e-16),
                     marker="o", label=f"tau={tau:g} ms")
        ax.set_xlabel("Recurrence lag (electrical cycles)")
        ax.set_ylabel("Full microstate trajectory RMS difference (rad)")
        ax.set_title(f"H2.6 full-state recurrence — {fe:g} Hz")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / f"H26_{int(fe)}Hz_full_state_recurrence.png", dpi=180)
        plt.close(fig)

    metadata = {
        "stage": "MAGNIKA H2.6",
        "purpose": [
            "grain-level dynamics",
            "relaxation-time sensitivity",
            "full-microstate periodicity and recurrence",
            "continued H2.5 dynamic energy-balance audit",
        ],
        "source_model": "Stage B via stageb.py",
        "waveform_and_integrators": "reused from stageH23.py",
        "grain_metrics": {
            "grain_mean_direction": "circular mean of pixel spin angles within each Voronoi grain",
            "grain_coherence": "resultant length of within-grain unit spin vectors",
            "within_grain_spread": "circular standard deviation derived from resultant length",
            "intergrain_spread": "circular standard deviation of grain mean directions",
        },
        "periodicity": {
            "method": "compare complete spin-angle trajectories sampled at PWM boundaries, phase-aligned by electrical cycle",
            "lags": "1..periodicity_max_period electrical cycles",
            "aggregation": "mean and maximum RMS across the last up-to-periodicity_tail_pairs cycle-pairs",
            "caveat": "finite-horizon recurrence diagnostic; not proof of an attractor or asymptotic periodicity",
        },
        "energy_balance": {
            "equation": "Delta E = W_external - D_relaxation + R_numerical",
            "units": "dimensionless model units; not joules, watts, heat, or measured material loss",
        },
        "physical_calibration": False,
        "llg_solver": False,
        "relaxation_time_is_measured_material_parameter": False,
        "phase_and_gain_are_measured_ferrite_properties": False,
        "parameters": {
            "fundamental_frequencies": args.fundamental_frequencies,
            "pwm_frequency": args.pwm_frequency,
            "ripple_fraction": args.ripple_fraction,
            "samples_per_pwm": args.samples_per_pwm,
            "integrator": args.integrator,
            "cycles": args.cycles,
            "relaxation_times_ms": args.relaxation_times_ms,
            "periodicity_max_period": args.periodicity_max_period,
            "periodicity_tail_pairs": args.periodicity_tail_pairs,
            "current_peak": args.current_peak,
            "reference_field": args.reference_field,
            "eta_boundary": args.eta_boundary,
            "nx": args.nx, "ny": args.ny, "n_grains": args.n_grains,
            "seed": args.seed,
        },
        "cases": len(summaries),
    }
    (args.outdir / "stageH26_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nMAGNIKA Stage H2.6 completed.")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
