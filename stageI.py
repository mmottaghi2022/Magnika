#!/usr/bin/env python3
"""
MAGNIKA Stage I — controlled grain-population screening.

Uses the existing Stage B dimensionless 2-D energy model unchanged.
This is exploratory numerical screening, not a calibrated ferrite model.

Place this file beside stageb.py, then run:
    python stageI.py

Optional:
    python stageI.py --out stageI_results --grain-counts 16 25 50 100 \
        --seeds 17 42 73 --eta 0.20
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import stageb as base


NX, NY = 40, 40
DEFAULT_GRAIN_COUNTS = [16, 25, 50, 100]
DEFAULT_SEEDS = [17, 42, 73]
K_ANIS = 0.12
J_EXCHANGE = 1.0
H_MAX = 1.20
N_BRANCH = 17


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def circular_std_rad(angles):
    angles = np.asarray(angles, dtype=float)
    if angles.size == 0:
        return float("nan")
    c = float(np.mean(np.cos(angles)))
    s = float(np.mean(np.sin(angles)))
    r = float(np.clip(np.hypot(c, s), 1e-12, 1.0))
    return float(np.sqrt(-2.0 * np.log(r)))


def grain_diagnostics(theta, grain_map):
    """Grain-level alignment and within-grain angular dispersion."""
    theta2 = np.asarray(theta).reshape(grain_map.shape)
    grain_ids = np.unique(grain_map)
    mean_angles, resultants, within_stds, pixel_counts = [], [], [], []

    for gid in grain_ids:
        mask = grain_map == gid
        local = theta2[mask]
        c = float(np.mean(np.cos(local)))
        s = float(np.mean(np.sin(local)))
        mean_angle = float(np.arctan2(s, c))
        mean_angles.append(mean_angle)
        resultants.append(float(np.hypot(c, s)))
        pixel_counts.append(int(np.count_nonzero(mask)))

        wrapped = np.arctan2(
            np.sin(local - mean_angle), np.cos(local - mean_angle)
        )
        within_stds.append(float(np.sqrt(np.mean(wrapped ** 2))))

    return {
        "grain_direction_resultant": float(np.hypot(
            np.mean(np.cos(mean_angles)), np.mean(np.sin(mean_angles))
        )),
        "grain_mean_angle_std_deg": float(
            np.degrees(circular_std_rad(mean_angles))
        ),
        "mean_grain_resultant": float(np.mean(resultants)),
        "mean_within_grain_std_deg": float(np.degrees(np.mean(within_stds))),
        "max_within_grain_std_deg": float(np.degrees(np.max(within_stds))),
        "min_grain_pixels": int(np.min(pixel_counts)),
        "mean_grain_pixels": float(np.mean(pixel_counts)),
        "max_grain_pixels": int(np.max(pixel_counts)),
    }


def magnetization_at_zero(hvals, mvals):
    """Return magnetization at H=0 using linear interpolation."""
    hvals = np.asarray(hvals, dtype=float)
    mvals = np.asarray(mvals, dtype=float)

    if hvals.ndim != 1 or mvals.ndim != 1 or hvals.size != mvals.size:
        raise ValueError("hvals and mvals must be 1-D arrays of equal length")
    if hvals.size < 2:
        raise ValueError("At least two field samples are required")
    if not np.all(np.isfinite(hvals)) or not np.all(np.isfinite(mvals)):
        raise ValueError("Field and magnetization arrays must be finite")
    if np.min(hvals) > 0.0 or np.max(hvals) < 0.0:
        raise ValueError("The field sweep does not bracket H=0")

    order = np.argsort(hvals)
    return float(np.interp(0.0, hvals[order], mvals[order]))


def zero_crossing(hvals, mvals):
    for i in range(len(mvals) - 1):
        if mvals[i] == 0:
            return float(hvals[i])
        if mvals[i] * mvals[i + 1] < 0:
            return float(
                hvals[i] - mvals[i] *
                (hvals[i + 1] - hvals[i]) /
                (mvals[i + 1] - mvals[i])
            )
    return float("nan")


def run_case(grain_count, seed, eta, output_dir, save_map=False):
    grain_map, grain_angles = base.generate_grains(
        NX, NY, grain_count, seed
    )
    model = base.MicroMagneticModel(
        grain_map, grain_angles, K=K_ANIS, J=J_EXCHANGE
    )

    actual_grains = int(len(np.unique(grain_map)))
    pixel_counts = np.asarray([
        np.count_nonzero(grain_map == gid)
        for gid in np.unique(grain_map)
    ])
    boundary_fraction = float(np.mean(model.boundary))

    if save_map:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(grain_map, origin="lower", interpolation="nearest")
        ax.set_title(
            f"Stage I grain map: requested {grain_count}, "
            f"actual {actual_grains}, seed {seed}"
        )
        ax.set_xlabel("X cell")
        ax.set_ylabel("Y cell")
        fig.colorbar(im, ax=ax, label="Grain ID")
        fig.tight_layout()
        fig.savefig(output_dir / f"grain_map_{grain_count}.png", dpi=160)
        plt.close(fig)

    ascending = np.linspace(-H_MAX, H_MAX, N_BRANCH)
    descending = np.linspace(H_MAX, -H_MAX, N_BRANCH)[1:]
    fields = np.concatenate((ascending, descending))
    branches = ["ascending"] * len(ascending) + ["descending"] * len(descending)

    all_rows = []
    summaries = []

    # Separate continuation histories are maintained for X and Y.
    for axis in ("x", "y"):
        theta = np.full(model.n, np.pi, dtype=float)
        axis_rows = []

        for step, (h, branch) in enumerate(zip(fields, branches)):
            hx, hy = (float(h), 0.0) if axis == "x" else (0.0, float(h))
            theta, opt = model.relax(theta, hx, hy, eta)
            energy, gradient, parts = model.energy_gradient(theta, hx, hy, eta)
            mx, my = model.magnetization(theta)
            m_parallel = float(mx if axis == "x" else my)
            m_transverse = float(my if axis == "x" else mx)

            row = {
                "grain_count_requested": int(grain_count),
                "grain_count_actual": actual_grains,
                "seed": int(seed),
                "axis": axis,
                "branch": branch,
                "step": int(step),
                "H": float(h),
                "Mx": float(mx),
                "My": float(my),
                "M_parallel": m_parallel,
                "M_transverse": m_transverse,
                "E_exchange": float(parts["exchange"]),
                "E_anisotropy": float(parts["anisotropy"]),
                "E_zeeman": float(parts["zeeman"]),
                "E_total": float(energy),
                "max_abs_gradient": float(np.max(np.abs(gradient))),
                "optimizer_iterations": int(opt.nit),
                "optimizer_success": bool(opt.success),
                "boundary_edge_fraction": boundary_fraction,
                **grain_diagnostics(theta, grain_map),
            }
            axis_rows.append(row)
            all_rows.append(row)

        n_ascending = len(ascending)
        h_asc = np.asarray([r["H"] for r in axis_rows[:n_ascending]])
        m_asc = np.asarray([
            r["M_parallel"] for r in axis_rows[:n_ascending]
        ])
        h_desc = np.asarray([r["H"] for r in axis_rows[n_ascending - 1:]])
        m_desc = np.asarray([
            r["M_parallel"] for r in axis_rows[n_ascending - 1:]
        ])

        remanence_ascending = magnetization_at_zero(h_asc, m_asc)
        remanence_descending = magnetization_at_zero(h_desc, m_desc)

        near_zero = np.abs(h_asc) <= 0.30
        near_zero_slope = (
            float(np.polyfit(h_asc[near_zero], m_asc[near_zero], 1)[0])
            if np.count_nonzero(near_zero) >= 2 else float("nan")
        )

        loop_h = np.concatenate((h_asc, h_desc[1:]))
        loop_m = np.concatenate((m_asc, m_desc[1:]))
        loop_integral = float(abs(np.trapezoid(loop_m, loop_h)))

        summaries.append({
            "grain_count_requested": int(grain_count),
            "grain_count_actual": actual_grains,
            "seed": int(seed),
            "axis": axis,
            "eta_boundary": float(eta),
            "K_anisotropy": K_ANIS,
            "J_exchange": J_EXCHANGE,
            "H_max": H_MAX,
            "remanence_ascending_model": remanence_ascending,
            "remanence_descending_model": remanence_descending,
            "zero_crossing_ascending_model": zero_crossing(h_asc, m_asc),
            "zero_crossing_descending_model": zero_crossing(h_desc, m_desc),
            "near_zero_slope_model": near_zero_slope,
            "loop_integral_model": loop_integral,
            "M_at_positive_Hmax": float(m_asc[-1]),
            "M_at_negative_Hmax": float(m_asc[0]),
            "all_optimizer_steps_success": bool(
                all(r["optimizer_success"] for r in axis_rows)
            ),
            "max_gradient_over_sweep": float(
                max(r["max_abs_gradient"] for r in axis_rows)
            ),
            "boundary_edge_fraction": boundary_fraction,
            "min_grain_pixels": int(np.min(pixel_counts)),
            "mean_grain_pixels": float(np.mean(pixel_counts)),
            "max_grain_pixels": int(np.max(pixel_counts)),
        })

    return all_rows, summaries


def plot_curves(curve_rows, axis, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = sorted(set(r["grain_count_requested"] for r in curve_rows))
    for count in counts:
        subset = [
            r for r in curve_rows
            if r["axis"] == axis and r["grain_count_requested"] == count
        ]
        for seed in sorted(set(r["seed"] for r in subset)):
            srows = sorted(
                [r for r in subset if r["seed"] == seed],
                key=lambda r: r["step"]
            )
            ax.plot(
                [r["H"] for r in srows],
                [r["M_parallel"] for r in srows],
                alpha=0.7,
                linewidth=1.1,
                label=f"N={count}, seed={seed}"
            )
    ax.set_xlabel("Applied field H (dimensionless model units)")
    ax.set_ylabel("Normalized Mx" if axis == "x" else "Normalized My")
    ax.set_title(f"Stage I: {axis.upper()}-axis quasi-static M-H sweeps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"stageI_MH_{axis}.png", dpi=170)
    plt.close(fig)


def aggregate_summaries(case_rows):
    metrics = [
        "remanence_ascending_model", "remanence_descending_model",
        "zero_crossing_ascending_model", "zero_crossing_descending_model",
        "near_zero_slope_model", "loop_integral_model",
        "M_at_positive_Hmax", "M_at_negative_Hmax",
        "boundary_edge_fraction", "mean_grain_pixels",
        "min_grain_pixels", "max_gradient_over_sweep",
    ]
    groups = {}
    for row in case_rows:
        groups.setdefault(
            (row["grain_count_requested"], row["axis"]), []
        ).append(row)

    output = []
    for (count, axis), group in sorted(groups.items()):
        result = {
            "grain_count_requested": int(count),
            "axis": axis,
            "n_seeds": len(group),
        }
        for metric in metrics:
            values = np.asarray(
                [r[metric] for r in group if np.isfinite(r[metric])],
                dtype=float
            )
            result[f"{metric}_mean"] = (
                float(np.mean(values)) if values.size else float("nan")
            )
            result[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1
                else (0.0 if values.size == 1 else float("nan"))
            )
        result["all_cases_optimizer_success"] = bool(
            all(r["all_optimizer_steps_success"] for r in group)
        )
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="MAGNIKA Stage I controlled grain-population screening"
    )
    parser.add_argument("--out", default="stageI_results")
    parser.add_argument(
        "--grain-counts", nargs="+", type=int,
        default=DEFAULT_GRAIN_COUNTS
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--eta", type=float, default=0.20)
    args = parser.parse_args()

    if not 0.0 <= args.eta <= 1.0:
        parser.error("--eta must be between 0 and 1")
    if any(n < 2 or n >= NX * NY for n in args.grain_counts):
        parser.error("Each grain count must be >=2 and less than NX*NY")
    if not args.seeds:
        parser.error("At least one seed is required")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate the existing Stage B analytical gradient before screening.
    ref_map, ref_angles = base.generate_grains(
        NX, NY, args.grain_counts[0], args.seeds[0]
    )
    ref_model = base.MicroMagneticModel(
        ref_map, ref_angles, K=K_ANIS, J=J_EXCHANGE
    )
    gradient_error = base.verify_gradient(ref_model)

    all_curves, all_cases = [], []
    total = len(args.grain_counts) * len(args.seeds)
    case_number = 0

    for grain_count in args.grain_counts:
        for seed in args.seeds:
            case_number += 1
            print(
                f"[{case_number}/{total}] grains={grain_count}, "
                f"seed={seed}, eta={args.eta}"
            )
            curves, summaries = run_case(
                grain_count, seed, args.eta, output_dir,
                save_map=(seed == args.seeds[0])
            )
            all_curves.extend(curves)
            all_cases.extend(summaries)

    save_csv(output_dir / "stageI_magnetization_curves.csv", all_curves)
    save_csv(output_dir / "stageI_case_summary.csv", all_cases)
    save_csv(
        output_dir / "stageI_summary_by_grain_size.csv",
        aggregate_summaries(all_cases)
    )
    plot_curves(all_curves, "x", output_dir)
    plot_curves(all_curves, "y", output_dir)

    metadata = {
        "stage": "MAGNIKA Stage I",
        "purpose": (
            "Controlled screening of grain-population effects using "
            "the unchanged Stage B dimensionless energy model."
        ),
        "grid": {"nx": NX, "ny": NY, "cells": NX * NY},
        "grain_counts_requested": args.grain_counts,
        "seeds": args.seeds,
        "eta_boundary": args.eta,
        "K_anisotropy": K_ANIS,
        "J_exchange": J_EXCHANGE,
        "H_max": H_MAX,
        "field_points_per_branch": N_BRANCH,
        "axes": ["x", "y"],
        "gradient_verification_max_error": gradient_error,
        "limitations": [
            "2-D in-plane angle model",
            "dimensionless field, energy and magnetization",
            "no nonlocal magnetostatic/demagnetizing energy",
            "no 3-D domain structure",
            "no thermal activation or thermal variation",
            "no LLG dynamics",
            "no eddy-current or physical hysteresis-loss model",
            "no magnetoelastic coupling",
            "no SI material calibration",
            "grain count changes both characteristic grain size and boundary density",
            "finite 40x40 grid can under-resolve the smallest grains",
            "loop integral and zero-crossing are model-space diagnostics only",
            "no rotor geometry, d/q inductance, torque or motor efficiency is calculated"
        ],
        "interpretation_warning": (
            "Do not interpret model-space loop integral as W/kg or J/m^3, "
            "zero-crossing as physical coercivity, or slope as SI permeability."
        )
    }
    with (output_dir / "stageI_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\nSTAGE I COMPLETED")
    print(f"Results directory: {output_dir.resolve()}")
    print("Summary: stageI_summary_by_grain_size.csv")
    print("Detailed curves: stageI_magnetization_curves.csv")
    print("Case summary: stageI_case_summary.csv")
    print("Plots: stageI_MH_x.png, stageI_MH_y.png")


if __name__ == "__main__":
    main()
