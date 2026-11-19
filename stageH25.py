#!/usr/bin/env python3
"""MAGNIKA Stage H2.5: dynamic energy-balance and full-state convergence audit.

Uses the existing Stage H2.3 waveform, Stage B model, and integrators.
This remains a dimensionless phenomenological angular-relaxation study:
not a calibrated ferrite model, not an LLG solver, and not a loss/thermal model.

For E(theta,H) = E_internal(theta) - N M(theta).H and
d(theta)/dt = -(1/tau) grad_theta(E), the continuous balance is

    dE/dt = -N M.dH/dt - (1/tau) sum_i (dE/dtheta_i)^2.

The script evaluates external field work and relaxation dissipation by
trapezoidal quadrature, then reports the numerical balance residual.
It also compares complete grain-angle states at common PWM boundaries
against the finest requested timestep. That comparison is a convergence
diagnostic, not proof that the finest result is exact.
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
    """Signed wrapped angular difference in [-pi, pi)."""
    d = np.asarray(a) - np.asarray(b)
    return np.arctan2(np.sin(d), np.cos(d))


def run_case(args, fe, fsw, spp, integrator):
    args.fundamental_hz = float(fe)
    grain_map, grain_angles = base.generate_grains(
        args.nx, args.ny, args.n_grains, args.seed
    )
    model = base.MicroMagneticModel(
        grain_map, grain_angles, K=base.K_ANIS, J=base.J_EXCHANGE
    )

    dt = 1.0 / (spp * fsw)
    spc_float = (1.0 / fe) / dt
    spc = int(round(spc_float))
    if not np.isclose(spc_float, spc, rtol=0, atol=1e-8):
        raise SystemExit(
            f"{fe:g} Hz fundamental and {fsw:g} Hz PWM do not produce "
            "an integer samples-per-electrical-cycle count."
        )

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
    # Store mean(grad**2) at each time node; multiply by N for sum_i grad_i**2.
    grad2_mean = np.empty(n_steps + 1)

    mx[0], my[0] = model.magnetization(theta)
    _, energy[0], grad0 = h23.rhs(
        model, theta, t[0], args, fsw, args.ripple_fraction
    )
    grad2_mean[0] = float(np.mean(grad0 * grad0))

    # Full microstate at every PWM boundary: common comparison grid for all spp.
    theta_pwm = [theta.copy()]
    stepper = h23.step_euler if integrator == "euler" else h23.step_rk4

    for k in range(n_steps):
        theta, energy[k], grad2_mean[k] = stepper(
            model, theta, t[k], dt, args, fsw, args.ripple_fraction
        )
        mx[k + 1], my[k + 1] = model.magnetization(theta)
        if (k + 1) % spp == 0:
            theta_pwm.append(theta.copy())

    _, energy[-1], grad_last = h23.rhs(
        model, theta, t[-1], args, fsw, args.ripple_fraction
    )
    grad2_mean[-1] = float(np.mean(grad_last * grad_last))

    # Dimensionless external work: -N integral(Mx dHx + My dHy).
    # Dimensionless relaxation dissipation: integral(sum(grad_i^2)/tau dt).
    n_spins = int(model.n)
    dissipation_rate = n_spins * grad2_mean / args.relaxation_time
    dH_x = np.diff(hx)
    dH_y = np.diff(hy)
    dW = -n_spins * (
        0.5 * (mx[:-1] + mx[1:]) * dH_x
        + 0.5 * (my[:-1] + my[1:]) * dH_y
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

        dE_cycle = float(energy[stop] - energy[start])
        W_cycle = float(W_cum[stop] - W_cum[start])
        D_cycle = float(D_cum[stop] - D_cum[start])
        R_cycle = float(dE_cycle - W_cycle + D_cycle)
        scale = max(abs(dE_cycle), abs(W_cycle), abs(D_cycle), 1e-14)

        if c == 0:
            m_trajectory_delta = np.nan
        else:
            prev = slice(start - spc, stop - spc + 1)
            m_trajectory_delta = h23.rms(np.r_[
                mx[sl] - mx[prev], my[sl] - my[prev]
            ])

        cycle_rows.append({
            "cycle": c + 1,
            "integrator": integrator,
            "fundamental_hz": fe,
            "pwm_hz": fsw,
            "ripple_fraction_peak": args.ripple_fraction,
            "samples_per_pwm": spp,
            "dt_s": dt,
            "M_lag_vs_H_deg_model_only": lag,
            "M_over_H_gain_model_only": float(abs(ratio)) if np.isfinite(ratio.real) else np.nan,
            "M_loop_work_proxy_model_only": loop_proxy,
            "M_trajectory_delta_rms": m_trajectory_delta,
            "delta_energy_model_units": dE_cycle,
            "external_work_model_units": W_cycle,
            "relaxation_dissipation_model_units": D_cycle,
            "energy_balance_residual_model_units": R_cycle,
            "energy_balance_residual_relative": abs(R_cycle) / scale,
            "mean_gradient_squared_sum_model_units": float(
                np.mean(n_spins * grad2_mean[start:stop])
            ),
        })

    cycles_df = pd.DataFrame(cycle_rows)
    last = cycles_df.iloc[-1]
    summary = {
        "integrator": integrator,
        "fundamental_hz": fe,
        "pwm_hz": fsw,
        "ripple_fraction_peak": args.ripple_fraction,
        "samples_per_pwm": spp,
        "dt_s": dt,
        "samples_per_electrical_cycle": spc,
        "cycles": args.cycles,
        "eta_boundary": args.eta_boundary,
        "relaxation_time_s_model_parameter": args.relaxation_time,
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
    }

    return trace, cycles_df, summary, np.asarray(theta_pwm)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fundamental-frequencies", type=float, nargs="+", default=[100, 200, 400])
    p.add_argument("--pwm-frequency", type=float, default=4000.0)
    p.add_argument("--ripple-fraction", type=float, default=0.0)
    p.add_argument("--samples-per-pwm", type=int, nargs="+", default=[20, 40, 80, 160])
    p.add_argument("--integrators", nargs="+", choices=["euler", "rk4"], default=["rk4"])
    p.add_argument("--cycles", type=int, default=8)
    p.add_argument("--current-peak", type=float, default=1.0)
    p.add_argument("--reference-field", type=float, default=0.45)
    p.add_argument("--eta-boundary", type=float, default=0.20)
    p.add_argument("--relaxation-time", type=float, default=1e-3)
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--n-grains", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("stageH25_results"))
    args = p.parse_args()

    if min(args.fundamental_frequencies) <= 0 or args.pwm_frequency <= 0:
        p.error("frequencies must be positive")
    if args.relaxation_time <= 0 or args.current_peak <= 0:
        p.error("relaxation time and current peak must be positive")
    if args.cycles < 2 or min(args.samples_per_pwm) < 8:
        p.error("use at least 2 cycles and 8 samples per PWM period")
    if len(set(args.samples_per_pwm)) != len(args.samples_per_pwm):
        p.error("samples-per-pwm values must be unique")
    if not 0 <= args.ripple_fraction <= 0.5:
        p.error("ripple fraction must be between 0 and 0.5")

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    all_cycles = []
    convergence_rows = []
    total = (
        len(args.fundamental_frequencies)
        * len(args.samples_per_pwm)
        * len(args.integrators)
    )
    count = 0

    for fe in args.fundamental_frequencies:
        for integrator in args.integrators:
            case_states = {}
            for spp in args.samples_per_pwm:
                count += 1
                print(
                    f"[{count}/{total}] f={fe:g} Hz, PWM={args.pwm_frequency:g} Hz, "
                    f"integrator={integrator}, samples/PWM={spp}, cycles={args.cycles}",
                    flush=True,
                )
                trace, cycles_df, summary, states = run_case(
                    args, fe, args.pwm_frequency, spp, integrator
                )
                tag = f"{int(fe)}Hz_{integrator}_spp{spp}"
                trace.to_csv(args.outdir / f"{tag}_trace.csv", index=False)
                cycles_df.to_csv(args.outdir / f"{tag}_energy_balance_cycles.csv", index=False)
                np.savez_compressed(
                    args.outdir / f"{tag}_grain_states_pwm_boundaries.npz",
                    theta=states,
                    samples_per_pwm=spp,
                    fundamental_hz=fe,
                    pwm_hz=args.pwm_frequency,
                    cycles=args.cycles,
                )
                all_summaries.append(summary)
                all_cycles.append(cycles_df)
                case_states[spp] = states

            # Compare complete angular microstates at the same PWM-boundary times.
            reference_spp = max(args.samples_per_pwm)
            reference = case_states[reference_spp]
            for spp in args.samples_per_pwm:
                states = case_states[spp]
                if states.shape != reference.shape:
                    raise RuntimeError(
                        "PWM-boundary state arrays do not align; check frequency and timestep settings."
                    )
                delta = circular_difference(states, reference)
                convergence_rows.append({
                    "integrator": integrator,
                    "fundamental_hz": fe,
                    "pwm_hz": args.pwm_frequency,
                    "samples_per_pwm": spp,
                    "reference_samples_per_pwm": reference_spp,
                    "full_grain_state_rms_vs_finest_rad": float(np.sqrt(np.mean(delta * delta))),
                    "full_grain_state_max_abs_vs_finest_rad": float(np.max(np.abs(delta))),
                    "final_full_grain_state_rms_vs_finest_rad": float(np.sqrt(np.mean(delta[-1] * delta[-1]))),
                    "number_of_pwm_boundary_states": int(states.shape[0]),
                    "number_of_spin_angles": int(states.shape[1]),
                })

    summary_df = pd.DataFrame(all_summaries)
    cycles_all_df = pd.concat(all_cycles, ignore_index=True)
    convergence_df = pd.DataFrame(convergence_rows)
    summary_df.to_csv(args.outdir / "stageH25_summary.csv", index=False)
    cycles_all_df.to_csv(args.outdir / "stageH25_energy_balance_cycles.csv", index=False)
    convergence_df.to_csv(args.outdir / "stageH25_full_state_convergence.csv", index=False)

    # Plot energy accounting for each frequency, using the finest requested spp.
    for fe in args.fundamental_frequencies:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for integrator in args.integrators:
            q = cycles_all_df[
                (cycles_all_df.fundamental_hz == fe)
                & (cycles_all_df.integrator == integrator)
                & (cycles_all_df.samples_per_pwm == max(args.samples_per_pwm))
            ].sort_values("cycle")
            axs[0].plot(q.cycle, q.delta_energy_model_units, marker="o", label=f"{integrator}: ΔE")
            axs[0].plot(q.cycle, q.external_work_model_units - q.relaxation_dissipation_model_units,
                        marker="s", linestyle="--", label=f"{integrator}: Wext − D")
            axs[1].semilogy(
                q.cycle,
                np.maximum(q.energy_balance_residual_relative, 1e-18),
                marker="o", label=integrator
            )
        axs[0].set_ylabel("Per-cycle energy change (model units)")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(fontsize=8)
        axs[1].set_ylabel("Relative energy-balance residual")
        axs[1].set_xlabel("Electrical cycle")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        fig.suptitle(f"MAGNIKA H2.5 energy-balance audit — {fe:g} Hz")
        fig.tight_layout()
        fig.savefig(args.outdir / f"H25_{int(fe)}Hz_energy_balance.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for integrator in args.integrators:
            q = convergence_df[
                (convergence_df.fundamental_hz == fe)
                & (convergence_df.integrator == integrator)
            ].sort_values("samples_per_pwm")
            ax.loglog(
                q.samples_per_pwm,
                np.maximum(q.full_grain_state_rms_vs_finest_rad, 1e-18),
                marker="o", label=integrator
            )
        ax.set_xlabel("Samples per PWM period")
        ax.set_ylabel("Full grain-state RMS difference vs finest (rad)")
        ax.set_title(f"H2.5 full-state timestep comparison — {fe:g} Hz")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / f"H25_{int(fe)}Hz_full_state_convergence.png", dpi=180)
        plt.close(fig)

    metadata = {
        "stage": "MAGNIKA H2.5",
        "purpose": "dimensionless dynamic energy-balance audit and full grain-state timestep comparison",
        "source_model": "Stage B via stageb.py",
        "waveform_and_integrators": "reused from stageH23.py",
        "energy_balance": {
            "equation": "Delta E = W_external - D_relaxation + R_numerical",
            "external_work": "-N * integral(Mx dHx + My dHy), trapezoidal in M over field increments",
            "relaxation_dissipation": "integral(sum_i gradient_i^2 / relaxation_time) dt, trapezoidal",
            "residual": "Delta E - W_external + D_relaxation",
            "units": "dimensionless model units; not joules, watts, heat, or measured material loss",
        },
        "full_state_comparison": {
            "state": "all pixel/grain angular degrees of freedom",
            "sampling": "every PWM boundary",
            "reference": "finest samples_per_pwm requested for each frequency and integrator",
            "metric": "wrapped angular RMS and maximum absolute difference",
            "caveat": "comparison to the finest requested grid is a convergence diagnostic, not proof that the reference is exact",
        },
        "physical_calibration": False,
        "llg_solver": False,
        "H_and_M_same_SI_scale": False,
        "M_over_H_gain_is_physical_permeability": False,
        "phase_is_measured_ferrite_property": False,
        "loop_work_proxy_is_physical_loss": False,
        "parameters": {
            **vars(args),
            "outdir": str(args.outdir),
            "integrators": args.integrators,
            "samples_per_pwm": args.samples_per_pwm,
            "fundamental_frequencies": args.fundamental_frequencies,
        },
        "cases": len(all_summaries),
    }
    (args.outdir / "stageH25_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    print("\nMAGNIKA Stage H2.5 completed.")
    print(summary_df.to_string(index=False))
    print("\nFull-state timestep comparison:")
    print(convergence_df.to_string(index=False))
    print(f"\nOutputs: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
