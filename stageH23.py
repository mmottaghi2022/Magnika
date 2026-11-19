#!/usr/bin/env python3
"""MAGNIKA Stage H2.3: integrator and timestep validation.

Compares explicit Euler and classical RK4 for the existing phenomenological
angular-relaxation model. This is a numerical-method study, not a calibrated
ferrite model and not an LLG solver.

The synthetic three-phase current waveform and dimensionless Stage-F field
mapping are retained. H and normalized M are not in common SI units, so M/H
gain is reported only as a model diagnostic. The M-H phase is also model-only.
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


def triangle_wave(phase):
    cycles = np.mod(phase / (2.0 * np.pi), 1.0)
    return 4.0 * np.abs(cycles - 0.5) - 1.0


def synthesize_currents(t, fe, fsw, ripple_fraction, current_peak):
    t = np.asarray(t, dtype=float)
    fundamental_phase = 2.0 * np.pi * fe * t
    offsets = np.array([0.0, -2.0*np.pi/3.0, 2.0*np.pi/3.0])
    fundamental = current_peak * np.sin(fundamental_phase[..., None] + offsets)
    carrier_phase = 2.0*np.pi*fsw*t[..., None] + offsets
    ripple = ripple_fraction * current_peak * triangle_wave(carrier_phase)
    currents = fundamental + ripple
    ia, ib, ic = np.moveaxis(currents, -1, 0)
    i_alpha = (2.0/3.0) * (ia - 0.5*ib - 0.5*ic)
    i_beta = (2.0/3.0) * (np.sqrt(3.0)/2.0) * (ib - ic)
    return currents, i_alpha, i_beta


def field_at(t, args, fsw, ripple):
    _, ia, ib = synthesize_currents(
        np.asarray(t), args.fundamental_hz, fsw, ripple, args.current_peak
    )
    scale = args.reference_field / args.current_peak
    return float(scale * ia), float(scale * ib)


def rhs(model, theta, t, args, fsw, ripple):
    hx, hy = field_at(t, args, fsw, ripple)
    energy, grad, _ = model.energy_gradient(theta, hx, hy, args.eta_boundary)
    return -grad / args.relaxation_time, float(energy), grad


def wrap_angles(theta):
    return (theta + np.pi) % (2.0*np.pi) - np.pi


def step_euler(model, theta, t, dt, args, fsw, ripple):
    k1, energy, grad = rhs(model, theta, t, args, fsw, ripple)
    return wrap_angles(theta + dt*k1), energy, float(np.mean(grad*grad))


def step_rk4(model, theta, t, dt, args, fsw, ripple):
    k1, energy, grad1 = rhs(model, theta, t, args, fsw, ripple)
    k2, _, _ = rhs(model, theta + 0.5*dt*k1, t + 0.5*dt, args, fsw, ripple)
    k3, _, _ = rhs(model, theta + 0.5*dt*k2, t + 0.5*dt, args, fsw, ripple)
    k4, _, _ = rhs(model, theta + dt*k3, t + dt, args, fsw, ripple)
    new_theta = theta + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return wrap_angles(new_theta), energy, float(np.mean(grad1*grad1))


def fundamental_phasor(z, t, frequency):
    z = np.asarray(z, dtype=complex)
    duration = float(t[-1] - t[0])
    if duration <= 0:
        return np.nan + 1j*np.nan
    return (2.0/duration) * np.trapezoid(
        z*np.exp(-2j*np.pi*frequency*t), t
    )


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x*x)))


def run_case(args, fsw, spp, integrator):
    grain_map, grain_angles = base.generate_grains(
        args.nx, args.ny, args.n_grains, args.seed
    )
    model = base.MicroMagneticModel(
        grain_map, grain_angles, K=base.K_ANIS, J=base.J_EXCHANGE
    )

    dt = 1.0/(spp*fsw)
    spc_float = (1.0/args.fundamental_hz)/dt
    spc = int(round(spc_float))
    if not np.isclose(spc_float, spc, rtol=0, atol=1e-8):
        raise SystemExit("PWM/fundamental frequencies must yield an integer samples-per-cycle count.")
    n_steps = args.cycles*spc
    t = np.arange(n_steps+1, dtype=float)*dt
    currents, i_alpha, i_beta = synthesize_currents(
        t, args.fundamental_hz, fsw, args.ripple_fraction, args.current_peak
    )
    field_scale = args.reference_field/args.current_peak
    hx, hy = field_scale*i_alpha, field_scale*i_beta

    theta = np.zeros(model.n, dtype=float)
    mx = np.empty(n_steps+1); my = np.empty(n_steps+1)
    energy = np.empty(n_steps+1); grad2 = np.empty(n_steps)
    mx[0], my[0] = model.magnetization(theta)
    _, energy[0], _ = rhs(model, theta, t[0], args, fsw, args.ripple_fraction)

    stepper = step_euler if integrator == "euler" else step_rk4
    for k in range(n_steps):
        theta, energy[k], grad2[k] = stepper(
            model, theta, t[k], dt, args, fsw, args.ripple_fraction
        )
        mx[k+1], my[k+1] = model.magnetization(theta)
    _, energy[-1], _ = rhs(model, theta, t[-1], args, fsw, args.ripple_fraction)

    hvec = hx + 1j*hy
    mvec = mx + 1j*my
    cycle_rows = []
    for c in range(args.cycles):
        start, stop = c*spc, (c+1)*spc
        sl = slice(start, stop+1)
        tc, hc, mc = t[sl], hvec[sl], mvec[sl]
        h1 = fundamental_phasor(hc, tc, args.fundamental_hz)
        m1 = fundamental_phasor(mc, tc, args.fundamental_hz)
        ratio = m1/h1 if abs(h1) > 1e-14 else np.nan+1j*np.nan
        phase = float(np.degrees(np.angle(ratio))) if np.isfinite(ratio.real) else np.nan
        lag = -phase
        work = float(np.sum(
            0.5*(hx[sl][:-1]+hx[sl][1:])*np.diff(mx[sl]) +
            0.5*(hy[sl][:-1]+hy[sl][1:])*np.diff(my[sl])
        ))
        if c == 0:
            trajectory_delta = np.nan
        else:
            prev = slice(start-spc, stop-spc+1)
            trajectory_delta = rms(np.r_[mx[sl]-mx[prev], my[sl]-my[prev]])
        cycle_rows.append({
            "cycle": c+1,
            "H_fundamental_amplitude": float(abs(h1)),
            "M_fundamental_amplitude": float(abs(m1)),
            "M_over_H_gain_model_only": float(abs(ratio)) if np.isfinite(ratio.real) else np.nan,
            "M_lag_vs_H_deg_model_only": lag,
            "H_dot_dM_loop_work_proxy": work,
            "cycle_to_previous_M_trajectory_rms": trajectory_delta,
            "energy_start_model_units": float(energy[start]),
            "energy_end_model_units": float(energy[stop]),
            "energy_change_model_units": float(energy[stop]-energy[start]),
            "mean_gradient_squared_model_proxy": float(np.mean(grad2[start:stop])),
            "Mx_start": float(mx[start]), "My_start": float(my[start]),
            "Mx_end": float(mx[stop]), "My_end": float(my[stop]),
        })

    cycles_df = pd.DataFrame(cycle_rows)
    last = cycles_df.iloc[-1]
    summary = {
        "integrator": integrator, "fundamental_hz": args.fundamental_hz,
        "pwm_hz": fsw, "ripple_fraction_peak": args.ripple_fraction,
        "samples_per_pwm": spp, "dt_s": dt,
        "samples_per_electrical_cycle": spc, "cycles": args.cycles,
        "eta_boundary": args.eta_boundary,
        "relaxation_time_s_model_parameter": args.relaxation_time,
        "last_cycle_M_lag_vs_H_deg_model_only": float(last.M_lag_vs_H_deg_model_only),
        "last_cycle_M_over_H_gain_model_only": float(last.M_over_H_gain_model_only),
        "last_cycle_work_proxy": float(last.H_dot_dM_loop_work_proxy),
        "last_cycle_trajectory_delta_rms": float(last.cycle_to_previous_M_trajectory_rms),
        "last_cycle_energy_change_model_units": float(last.energy_change_model_units),
    }
    trace = pd.DataFrame({
        "time_s": t, "ia": currents[:,0], "ib": currents[:,1], "ic": currents[:,2],
        "Hx": hx, "Hy": hy, "Mx": mx, "My": my,
        "H_magnitude": np.abs(hvec), "M_magnitude": np.abs(mvec),
        "model_energy": energy,
    })
    return trace, cycles_df, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fundamental-frequencies", type=float, nargs="+", default=[100,200,400])
    p.add_argument("--pwm-frequency", type=float, default=4000.0)
    p.add_argument("--ripple-fraction", type=float, default=0.0)
    p.add_argument("--samples-per-pwm", type=int, nargs="+", default=[20,40,80,160])
    p.add_argument("--integrators", nargs="+", choices=["euler","rk4"], default=["euler","rk4"])
    p.add_argument("--cycles", type=int, default=8)
    p.add_argument("--current-peak", type=float, default=1.0)
    p.add_argument("--reference-field", type=float, default=0.45)
    p.add_argument("--eta-boundary", type=float, default=0.20)
    p.add_argument("--relaxation-time", type=float, default=1e-3)
    p.add_argument("--nx", type=int, default=40); p.add_argument("--ny", type=int, default=40)
    p.add_argument("--n-grains", type=int, default=100); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=Path, default=Path("stageH23_results"))
    args = p.parse_args()
    if min(args.fundamental_frequencies) <= 0 or args.pwm_frequency <= 0 or args.relaxation_time <= 0:
        p.error("frequencies and relaxation time must be positive")
    if args.cycles < 2 or min(args.samples_per_pwm) < 8:
        p.error("use at least 2 cycles and 8 samples per PWM period")
    if not 0 <= args.ripple_fraction <= 0.5:
        p.error("ripple fraction must be between 0 and 0.5")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summaries=[]; total=len(args.fundamental_frequencies)*len(args.samples_per_pwm)*len(args.integrators); count=0
    for fe in args.fundamental_frequencies:
        args.fundamental_hz = fe
        for integ in args.integrators:
            for spp in args.samples_per_pwm:
                count += 1
                print(f"[{count}/{total}] f={fe:g} Hz, PWM={args.pwm_frequency:g} Hz, integrator={integ}, samples/PWM={spp}, cycles={args.cycles}", flush=True)
                trace, cyc, summ = run_case(args,args.pwm_frequency,spp,integ)
                tag=f"{int(fe)}Hz_{integ}_spp{spp}"
                trace.to_csv(args.outdir/f"{tag}_trace.csv",index=False)
                cyc.to_csv(args.outdir/f"{tag}_cycles.csv",index=False)
                summaries.append(summ)
    sdf=pd.DataFrame(summaries)
    sdf.to_csv(args.outdir/"stageH23_summary.csv",index=False)
    (args.outdir/"stageH23_metadata.json").write_text(json.dumps({
        "stage":"MAGNIKA H2.3", "purpose":"numerical integrator and timestep validation",
        "integrators":"explicit Euler and classical RK4", "field_evaluation":"analytic synthetic waveform at RK4 substeps",
        "physical_calibration":False, "H_and_M_same_SI_scale":False,
        "M_over_H_gain_is_physical_permeability":False,
        "phase_is_measured_ferrite_property":False,
        "relaxation_model":"phenomenological overdamped angular relaxation; not LLG",
        "parameters":{**vars(args),"outdir":str(args.outdir)},
        "cases":len(summaries)
    },indent=2,default=str),encoding="utf-8")

    # Overview plots: phase and cycle convergence versus timestep, grouped by f/integrator.
    for fe in args.fundamental_frequencies:
        sub=sdf[sdf.fundamental_hz==fe]
        fig,axs=plt.subplots(2,1,figsize=(9,8),sharex=True)
        for integ in args.integrators:
            q=sub[sub.integrator==integ].sort_values("samples_per_pwm")
            axs[0].plot(q.samples_per_pwm,q.last_cycle_M_lag_vs_H_deg_model_only,marker="o",label=integ)
            axs[1].semilogy(q.samples_per_pwm,np.maximum(q.last_cycle_trajectory_delta_rms,1e-16),marker="o",label=integ)
        axs[0].set_ylabel("Last-cycle M lag vs H (deg; model only)"); axs[0].legend(); axs[0].grid(True,alpha=.3)
        axs[1].set_ylabel("Last-cycle trajectory RMS"); axs[1].set_xlabel("Samples per PWM period"); axs[1].legend(); axs[1].grid(True,alpha=.3)
        fig.suptitle(f"H2.3 timestep/integrator comparison — {fe:g} Hz")
        fig.tight_layout(); fig.savefig(args.outdir/f"H23_{int(fe)}Hz_convergence.png",dpi=180); plt.close(fig)
    print("\nMAGNIKA Stage H2.3 completed.")
    print(sdf.to_string(index=False))
    print(f"\nOutputs: {args.outdir.resolve()}")

if __name__ == "__main__":
    main()
