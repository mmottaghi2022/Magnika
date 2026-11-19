#!/usr/bin/env python3
"""
MAGNIKA — Stage F
Field-strength × grain-boundary-coupling response study.

Runs 5 independent grain structures across:
    H = 0.25, 0.45, 0.70
    eta_boundary = 0.01 ... 1.00

Uses quasi-static rotating-field continuation (72 angular steps, including
both 0 and 360 degrees) and independently verifies the final gradient.

Expected base module: stageb.py
Expected API:
    generate_grains(nx, ny, n_grains, seed)
    MicroMagneticModel(grain_map, grain_angles, K=..., J=...)
    model.energy_gradient(theta, hx, hy, eta)
    model.magnetization(theta)
    grain_boundary_diagnostics(theta, grain_map, eta, J=...)

Run from the MAGNIKA project directory:
    python3 stageF.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import stageb as base


# ---------------------------------------------------------------------
# EXPERIMENT SETTINGS
# ---------------------------------------------------------------------

SEEDS = [17, 23, 42, 101, 2026]
FIELD_STRENGTHS = [0.25, 0.45, 0.70]
ETAS = [0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]

N_STEPS = 72
GRADIENT_TOL = 1.0e-5
MAX_OPTIMIZER_PASSES = 3

OPTIMIZER_OPTIONS = {
    "maxiter": 5000,
    "ftol": 1.0e-15,
    "gtol": 1.0e-9,
    "maxls": 100,
    "maxcor": 30,
}

OUTPUT_DIR = Path("stageF_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# ANGULAR DIAGNOSTICS
# ---------------------------------------------------------------------

def circular_difference(a, b):
    """Signed wrapped angular difference, in radians."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def grain_angular_metrics(theta, grain_map):
    """
    Returns:
      grain_angle_std_deg:
          Circular dispersion of the mean orientation of different grains.
      intragranular_dispersion_deg:
          Mean per-grain RMS angular deviation around that grain's
          circular mean orientation.
      grain_mean_angles:
          Circular mean orientation for each grain, in radians.
    """
    theta_grid = np.asarray(theta).reshape(grain_map.shape)
    grain_ids = np.unique(grain_map)

    grain_mean_angles = []
    intragranular_rms = []

    for grain_id in grain_ids:
        local = theta_grid[grain_map == grain_id]
        mean_x = np.mean(np.cos(local))
        mean_y = np.mean(np.sin(local))
        mean_angle = np.arctan2(mean_y, mean_x)

        residuals = circular_difference(local, mean_angle)
        intragranular_rms.append(np.sqrt(np.mean(residuals ** 2)))
        grain_mean_angles.append(mean_angle)

    grain_mean_angles = np.asarray(grain_mean_angles)

    # Circular standard deviation across grain mean orientations.
    resultant = np.sqrt(
        np.mean(np.cos(grain_mean_angles)) ** 2
        + np.mean(np.sin(grain_mean_angles)) ** 2
    )
    resultant = np.clip(resultant, 1.0e-15, 1.0)
    grain_std_rad = np.sqrt(max(0.0, -2.0 * np.log(resultant)))

    return (
        float(np.degrees(grain_std_rad)),
        float(np.degrees(np.mean(intragranular_rms))),
        grain_mean_angles,
    )


# ---------------------------------------------------------------------
# STRICT OPTIMIZATION
# ---------------------------------------------------------------------

def strict_relax(model, theta_initial, hx, hy, eta):
    """
    Minimize the energy and independently check max(abs(gradient)).

    L-BFGS-B may report success because of function-reduction termination
    even when the gradient is not sufficiently small. Therefore this
    function uses the gradient threshold as the numerical acceptance test.
    """
    theta = np.asarray(theta_initial, dtype=float).copy()
    last_result = None

    def objective(x):
        energy, gradient, _ = model.energy_gradient(x, hx, hy, eta)
        return energy, gradient

    for _ in range(MAX_OPTIMIZER_PASSES):
        result = minimize(
            objective,
            theta,
            method="L-BFGS-B",
            jac=True,
            options=OPTIMIZER_OPTIONS,
        )
        last_result = result
        theta = result.x

        _, gradient, _ = model.energy_gradient(theta, hx, hy, eta)
        gradient_inf = float(np.max(np.abs(gradient)))

        if gradient_inf <= GRADIENT_TOL:
            break

    _, gradient, _ = model.energy_gradient(theta, hx, hy, eta)
    gradient_inf = float(np.max(np.abs(gradient)))
    gradient_pass = gradient_inf <= GRADIENT_TOL

    return theta, last_result, gradient_inf, gradient_pass


# ---------------------------------------------------------------------
# ONE ROTATING-FIELD EXPERIMENT
# ---------------------------------------------------------------------

def run_one_case(seed, field_strength, eta):
    grain_map, grain_angles = base.generate_grains(
        base.NX, base.NY, base.N_GRAINS, seed
    )

    model = base.MicroMagneticModel(
        grain_map,
        grain_angles,
        K=base.K_ANIS,
        J=base.J_EXCHANGE,
    )

    theta = np.zeros(model.n, dtype=float)

    sample_rows = []
    optimizer_failures = []

    first_m = None
    first_grain_angles = None
    last_grain_angles = None

    angles_deg = np.linspace(0.0, 360.0, N_STEPS + 1)

    for step, angle_deg in enumerate(angles_deg):
        angle_rad = np.radians(angle_deg)
        hx = field_strength * np.cos(angle_rad)
        hy = field_strength * np.sin(angle_rad)

        theta, result, gradient_inf, gradient_pass = strict_relax(
            model, theta, hx, hy, eta
        )

        energy, gradient, components = model.energy_gradient(
            theta, hx, hy, eta
        )
        mx, my = model.magnetization(theta)

        grain_dispersion, intra_dispersion, grain_mean_angles = (
            grain_angular_metrics(theta, grain_map)
        )

        boundary = base.grain_boundary_diagnostics(
            theta, grain_map, eta, J=base.J_EXCHANGE
        )

        boundary_rms_deg = np.degrees(
            boundary["boundary_rms_angle_rad"]
        )
        boundary_mean_deg = np.degrees(
            boundary["boundary_mean_angle_rad"]
        )
        boundary_max_deg = np.degrees(
            boundary["boundary_max_angle_rad"]
        )

        if step == 0:
            first_m = (mx, my)
            first_grain_angles = grain_mean_angles.copy()

        if step == len(angles_deg) - 1:
            last_grain_angles = grain_mean_angles.copy()

        scipy_success = bool(result.success) if result is not None else False

        row = {
            "seed": seed,
            "field_strength": field_strength,
            "eta_boundary": eta,
            "step": step,
            "field_angle_deg": angle_deg,
            "Mx": mx,
            "My": my,
            "M_magnitude": float(np.hypot(mx, my)),
            "E_total": float(energy),
            "E_exchange": float(components["exchange"]),
            "E_anisotropy": float(components["anisotropy"]),
            "E_zeeman": float(components["zeeman"]),
            "E_exchange_intragranular": float(
                boundary["E_exchange_intragranular"]
            ),
            "E_exchange_boundary": float(
                boundary["E_exchange_boundary"]
            ),
            "grain_angle_std_deg": grain_dispersion,
            "boundary_mean_angle_deg": boundary_mean_deg,
            "boundary_rms_angle_deg": boundary_rms_deg,
            "boundary_max_angle_deg": boundary_max_deg,
            "intragranular_dispersion_deg": intra_dispersion,
            "scipy_success": scipy_success,
            "gradient_inf_norm": gradient_inf,
            "gradient_pass": gradient_pass,
        }
        sample_rows.append(row)

        if not gradient_pass:
            optimizer_failures.append({
                "seed": seed,
                "field_strength": field_strength,
                "eta_boundary": eta,
                "step": step,
                "field_angle_deg": angle_deg,
                "gradient_inf_norm": gradient_inf,
                "scipy_success": scipy_success,
                "message": str(result.message) if result is not None else
                           "No optimizer result",
            })

    macro_closure = float(np.hypot(
        last_mx_minus_first(first_m=first_m, last_m=(mx, my))[0],
        last_mx_minus_first(first_m=first_m, last_m=(mx, my))[1],
    ))

    grain_closure = circular_difference(
        last_grain_angles, first_grain_angles
    )
    grain_closure_rms_deg = float(
        np.degrees(np.sqrt(np.mean(grain_closure ** 2)))
    )
    grain_closure_max_deg = float(
        np.degrees(np.max(np.abs(grain_closure)))
    )

    closure_row = {
        "seed": seed,
        "field_strength": field_strength,
        "eta_boundary": eta,
        "macro_closure_error": macro_closure,
        "grain_closure_rms_deg": grain_closure_rms_deg,
        "grain_closure_max_deg": grain_closure_max_deg,
    }

    return sample_rows, closure_row, optimizer_failures


def last_mx_minus_first(first_m, last_m):
    return (
        last_m[0] - first_m[0],
        last_m[1] - first_m[1],
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("=" * 88)
    print("MAGNIKA — STAGE F: FIELD STRENGTH × BOUNDARY COUPLING")
    print("=" * 88)
    print(f"Seeds: {SEEDS}")
    print(f"Field strengths: {FIELD_STRENGTHS}")
    print(f"Boundary coefficients: {ETAS}")
    print(f"Gradient acceptance threshold: {GRADIENT_TOL:.1e}")
    print()

    all_samples = []
    all_closures = []
    all_failures = []

    total_cases = len(SEEDS) * len(FIELD_STRENGTHS) * len(ETAS)
    case_number = 0

    for seed in SEEDS:
        for field_strength in FIELD_STRENGTHS:
            for eta in ETAS:
                case_number += 1
                print(
                    f"[{case_number:03d}/{total_cases}] "
                    f"seed={seed:4d}, H={field_strength:.2f}, "
                    f"eta={eta:.3f}",
                    flush=True,
                )

                samples, closure, failures = run_one_case(
                    seed, field_strength, eta
                )
                all_samples.extend(samples)
                all_closures.append(closure)
                all_failures.extend(failures)

    samples_df = pd.DataFrame(all_samples)
    closures_df = pd.DataFrame(all_closures)
    failures_df = pd.DataFrame(all_failures)

    samples_path = OUTPUT_DIR / "stageF_samples.csv"
    closures_path = OUTPUT_DIR / "stageF_cycle_closure.csv"
    failures_path = OUTPUT_DIR / "stageF_optimizer_failures.csv"

    samples_df.to_csv(samples_path, index=False)
    closures_df.to_csv(closures_path, index=False)
    failures_df.to_csv(failures_path, index=False)

    # First average over the rotating-field cycle for each independent
    # seed, then calculate statistics across seeds.
    metrics = [
        "grain_angle_std_deg",
        "boundary_rms_angle_deg",
        "intragranular_dispersion_deg",
        "M_magnitude",
        "E_exchange_intragranular",
        "E_exchange_boundary",
        "gradient_inf_norm",
    ]

    per_seed = (
        samples_df
        .groupby(["field_strength", "eta_boundary", "seed"])[metrics]
        .mean()
        .reset_index()
    )
    per_seed_path = OUTPUT_DIR / "stageF_seed_statistics.csv"
    per_seed.to_csv(per_seed_path, index=False)

    summary = (
        per_seed
        .groupby(["field_strength", "eta_boundary"])[metrics]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns
    ]
    summary = summary.reset_index()

    closure_summary = (
        closures_df
        .groupby(["field_strength", "eta_boundary"])[
            [
                "macro_closure_error",
                "grain_closure_rms_deg",
                "grain_closure_max_deg",
            ]
        ]
        .mean()
        .reset_index()
    )

    summary = summary.merge(
        closure_summary,
        on=["field_strength", "eta_boundary"],
        how="left",
    )

    summary_path = OUTPUT_DIR / "stageF_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Compact terminal output: 27 rows, with variation across seeds.
    display_columns = [
        "field_strength",
        "eta_boundary",
        "grain_angle_std_deg_mean",
        "grain_angle_std_deg_std",
        "boundary_rms_angle_deg_mean",
        "intragranular_dispersion_deg_mean",
        "M_magnitude_mean",
        "macro_closure_error",
        "grain_closure_rms_deg",
    ]

    print("\n" + "=" * 112)
    print("STAGE F SUMMARY — CYCLE MEANS, THEN STATISTICS ACROSS FIVE SEEDS")
    print("=" * 112)
    print(
        summary[display_columns].to_string(
            index=False,
            formatters={
                "field_strength": "{:.2f}".format,
                "eta_boundary": "{:.3f}".format,
                "grain_angle_std_deg_mean": "{:.4f}".format,
                "grain_angle_std_deg_std": "{:.4f}".format,
                "boundary_rms_angle_deg_mean": "{:.4f}".format,
                "intragranular_dispersion_deg_mean": "{:.4f}".format,
                "M_magnitude_mean": "{:.6f}".format,
                "macro_closure_error": "{:.3e}".format,
                "grain_closure_rms_deg": "{:.6f}".format,
            },
        )
    )

    print("\n" + "=" * 88)
    print("NUMERICAL DIAGNOSTICS")
    print("=" * 88)
    print(f"Total field samples: {len(samples_df)}")
    print(f"Total independent cases: {total_cases}")
    print(f"Samples failing gradient threshold: {len(failures_df)}")
    print(
        "Maximum gradient infinity norm: "
        f"{samples_df['gradient_inf_norm'].max():.6e}"
    )
    print(
        "Maximum macro cycle closure: "
        f"{closures_df['macro_closure_error'].max():.6e}"
    )
    print(
        "Maximum grain cycle closure RMS: "
        f"{closures_df['grain_closure_rms_deg'].max():.6e} deg"
    )

    print("\nFiles written:")
    print(f"  {samples_path}")
    print(f"  {summary_path}")
    print(f"  {per_seed_path}")
    print(f"  {closures_path}")
    print(f"  {failures_path}")
    print("\nNo plots are generated.")
    print("=" * 88)


if __name__ == "__main__":
    main()
