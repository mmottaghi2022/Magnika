
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import stageb as base


# ============================================================
# STAGE E: MULTI-SEED BOUNDARY COUPLING EXPERIMENT
# ============================================================

SEEDS = [17, 23, 42, 101, 2026]

ETAS = [
    0.01, 0.025, 0.05, 0.1, 0.2,
    0.3, 0.5, 0.7, 1.0
]

H_ROT = 0.45
N_STEPS = 72

OUTPUT_DIR = "stageE_results"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# CIRCULAR STATISTICS
# ============================================================

def angular_difference(a, b):
    return np.arctan2(
        np.sin(a - b),
        np.cos(a - b)
    )


def intragranular_dispersion(theta, grain_map):

    theta_grid = np.asarray(theta).reshape(
        grain_map.shape
    )

    grain_rms_values = []

    for gid in np.unique(grain_map):

        mask = grain_map == gid
        local = theta_grid[mask]

        cx = np.mean(np.cos(local))
        sy = np.mean(np.sin(local))

        mean_angle = np.arctan2(sy, cx)

        deviations = angular_difference(
            local,
            mean_angle
        )

        rms = np.sqrt(np.mean(deviations**2))

        grain_rms_values.append(rms)

    return float(np.degrees(np.mean(grain_rms_values)))


# ============================================================
# RUN ONE SEED / ONE ETA
# ============================================================

def run_one_case(seed, eta):

    print(
        f"\nSeed={seed} | eta_boundary={eta}"
    )

    grain_map, grain_angles = base.generate_grains(
        base.NX,
        base.NY,
        base.N_GRAINS,
        seed
    )

    model = base.MicroMagneticModel(
        grain_map,
        grain_angles
    )

    grain_ids = np.unique(grain_map)

    # Same initial condition for every seed and eta.
    # Uniform magnetization along +x.
    theta = np.zeros(model.n, dtype=float)

    field_angles = np.linspace(
        0.0,
        360.0,
        N_STEPS + 1
    )

    summary_rows = []
    grain_rows = []

    for step, angle_deg in enumerate(field_angles):

        angle_rad = np.deg2rad(angle_deg)

        hx = H_ROT * np.cos(angle_rad)
        hy = H_ROT * np.sin(angle_rad)

        theta, optimizer = model.relax(
            theta,
            hx,
            hy,
            eta
        )

        energy, gradient, components = model.energy_gradient(
            theta,
            hx,
            hy,
            eta
        )

        mx, my = model.magnetization(theta)

        magnetization_magnitude = np.sqrt(
            mx**2 + my**2
        )

        magnetization_angle = np.rad2deg(
            np.arctan2(my, mx)
        )

        phase_lag = (
            magnetization_angle - angle_deg + 180.0
        ) % 360.0 - 180.0

        # Stage B grain-boundary diagnostics.
        diagnostics = base.grain_boundary_diagnostics(
            theta,
            grain_map,
            eta,
            J=model.J
        )

        # Intragranular angular dispersion.
        intra_dispersion = intragranular_dispersion(
            theta,
            grain_map
        )

        # Record individual grain orientations.
        theta_grid = theta.reshape(grain_map.shape)

        for gid in grain_ids:

            mask = grain_map == gid
            local = theta_grid[mask]

            cx = np.mean(np.cos(local))
            sy = np.mean(np.sin(local))

            grain_angle = np.arctan2(sy, cx)

            grain_resultant = np.sqrt(
                cx**2 + sy**2
            )

            grain_rows.append({

                "seed": seed,
                "eta_boundary": eta,
                "step": step,
                "field_angle_deg": angle_deg,
                "grain_id": int(gid),

                "grain_angle_deg": float(
                    np.degrees(grain_angle)
                ),

                "grain_resultant": float(
                    grain_resultant
                )
            })

        row = {

            "seed": seed,
            "eta_boundary": eta,

            "step": step,
            "field_angle_deg": angle_deg,

            "Hx": hx,
            "Hy": hy,

            "Mx": float(mx),
            "My": float(my),

            "M_magnitude": float(
                magnetization_magnitude
            ),

            "magnetization_angle_deg": float(
                magnetization_angle
            ),

            "phase_lag_deg": float(phase_lag),

            "E_total": float(energy),

            "E_exchange": float(
                components["exchange"]
            ),

            "E_anisotropy": float(
                components["anisotropy"]
            ),

            "E_zeeman": float(
                components["zeeman"]
            ),

            "E_exchange_intragranular": float(
                diagnostics["E_exchange_intragranular"]
            ),

            "E_exchange_boundary": float(
                diagnostics["E_exchange_boundary"]
            ),

            "grain_angle_std_deg": float(
                diagnostics["grain_angle_std_deg"]
            ),

            "boundary_mean_angle_deg": float(
                np.degrees(
                    diagnostics["boundary_mean_angle_rad"]
                )
            ),

            "boundary_rms_angle_deg": float(
                np.degrees(
                    diagnostics["boundary_rms_angle_rad"]
                )
            ),

            "boundary_max_angle_deg": float(
                np.degrees(
                    diagnostics["boundary_max_angle_rad"]
                )
            ),

            "intragranular_dispersion_deg": (
                intra_dispersion
            ),

            "grain_direction_resultant": float(
                diagnostics["grain_direction_resultant"]
            ),

            "optimizer_success": bool(
                optimizer.success
            ),

            "optimizer_iterations": int(
                optimizer.nit
            ),

            "max_abs_gradient": float(
                np.max(np.abs(gradient))
            )
        }

        summary_rows.append(row)

        if step % 12 == 0:

            print(
                f"  {angle_deg:6.1f} deg | "
                f"M=({mx:+.4f},{my:+.4f}) | "
                f"grain_std={diagnostics['grain_angle_std_deg']:.3f} deg | "
                f"boundary_rms={row['boundary_rms_angle_deg']:.3f} deg | "
                f"success={optimizer.success}"
            )

    # --------------------------------------------------------
    # Cycle closure
    # --------------------------------------------------------

    initial = summary_rows[0]
    final = summary_rows[-1]

    macro_closure = np.sqrt(
        (final["Mx"] - initial["Mx"])**2
        + (final["My"] - initial["My"])**2
    )

    grain_closure_values = []

    for gid in grain_ids:

        first = next(
            r for r in grain_rows
            if r["step"] == 0
            and r["grain_id"] == int(gid)
        )

        last = next(
            r for r in grain_rows
            if r["step"] == N_STEPS
            and r["grain_id"] == int(gid)
        )

        difference = angular_difference(
            np.deg2rad(last["grain_angle_deg"]),
            np.deg2rad(first["grain_angle_deg"])
        )

        grain_closure_values.append(
            np.degrees(difference)
        )

    grain_closure_rms = float(
        np.sqrt(
            np.mean(
                np.asarray(grain_closure_values)**2
            )
        )
    )

    grain_closure_max = float(
        np.max(np.abs(grain_closure_values))
    )

    closure_row = {

        "seed": seed,
        "eta_boundary": eta,

        "macro_closure_error": float(
            macro_closure
        ),

        "grain_closure_rms_deg": (
            grain_closure_rms
        ),

        "grain_closure_max_deg": (
            grain_closure_max
        )
    }

    print(
        f"  Macro closure: {macro_closure:.8e} | "
        f"Grain closure RMS: {grain_closure_rms:.6f} deg"
    )

    return summary_rows, grain_rows, closure_row


# ============================================================
# AGGREGATE RESULTS
# ============================================================

def aggregate_results(summary):

    metrics = [

        "grain_angle_std_deg",
        "boundary_mean_angle_deg",
        "boundary_rms_angle_deg",
        "boundary_max_angle_deg",
        "intragranular_dispersion_deg",

        "M_magnitude",
        "E_exchange_intragranular",
        "E_exchange_boundary",

        "max_abs_gradient"
    ]

    rows = []

    for eta in ETAS:

        subset = summary[
            summary["eta_boundary"] == eta
        ]

        for metric in metrics:

            # First average over the rotating-field cycle
            # separately for each seed.
            per_seed = subset.groupby("seed")[metric].mean()

            rows.append({

                "eta_boundary": eta,
                "metric": metric,

                "mean_across_seeds": float(
                    per_seed.mean()
                ),

                "std_across_seeds": float(
                    per_seed.std(ddof=1)
                ),

                "min_across_seeds": float(
                    per_seed.min()
                ),

                "max_across_seeds": float(
                    per_seed.max()
                )
            })

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================

def plot_metric(summary, metric, ylabel, filename):

    per_seed = (
        summary.groupby(
            ["seed", "eta_boundary"]
        )[metric]
        .mean()
        .reset_index()
    )

    stats = (
        per_seed.groupby("eta_boundary")[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for seed in SEEDS:

        data = per_seed[
            per_seed["seed"] == seed
        ]

        plt.plot(
            data["eta_boundary"],
            data[metric],
            marker="o",
            alpha=0.55,
            label=f"Seed {seed}"
        )

    plt.errorbar(
        stats["eta_boundary"],
        stats["mean"],
        yerr=stats["std"],
        marker="s",
        linewidth=2.5,
        capsize=4,
        label="Mean ± SD"
    )

    plt.xscale("log")

    plt.xlabel("Boundary coupling coefficient η")
    plt.ylabel(ylabel)

    plt.title(
        f"{ylabel} vs. Boundary Coupling"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURE_DIR, filename),
        dpi=200
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("MAGNIKA — STAGE E")
    print("MULTI-SEED BOUNDARY COUPLING REPRODUCIBILITY")
    print("=" * 65)

    print(f"Seeds: {SEEDS}")
    print(f"Boundary coefficients: {ETAS}")
    print(f"Field magnitude: {H_ROT}")
    print(f"Angular steps: {N_STEPS}")

    all_summary = []
    all_grains = []
    all_closures = []

    total_runs = len(SEEDS) * len(ETAS)
    completed = 0

    for seed in SEEDS:

        print("\n" + "=" * 55)
        print(f"GENERATING GRAIN STRUCTURE — SEED {seed}")
        print("=" * 55)

        for eta in ETAS:

            summary, grains, closure = run_one_case(
                seed,
                eta
            )

            all_summary.extend(summary)
            all_grains.extend(grains)
            all_closures.append(closure)

            completed += 1

            print(
                f"Progress: {completed}/{total_runs}"
            )

    # --------------------------------------------------------
    # Save raw data
    # --------------------------------------------------------

    summary_df = pd.DataFrame(all_summary)
    grain_df = pd.DataFrame(all_grains)
    closure_df = pd.DataFrame(all_closures)

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "stageE_summary.csv"),
        index=False
    )

    grain_df.to_csv(
        os.path.join(OUTPUT_DIR, "stageE_grain_angles.csv"),
        index=False
    )

    closure_df.to_csv(
        os.path.join(OUTPUT_DIR, "stageE_cycle_closure.csv"),
        index=False
    )

    # --------------------------------------------------------
    # Aggregate across seeds
    # --------------------------------------------------------

    aggregate_df = aggregate_results(summary_df)

    aggregate_df.to_csv(
        os.path.join(OUTPUT_DIR, "stageE_seed_statistics.csv"),
        index=False
    )

    # --------------------------------------------------------
    # Optimizer diagnostics
    # --------------------------------------------------------

    failures = summary_df[
        ~summary_df["optimizer_success"]
    ]

    failures.to_csv(
        os.path.join(OUTPUT_DIR, "stageE_optimizer_failures.csv"),
        index=False
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_metric(
        summary_df,
        "grain_angle_std_deg",
        "Grain angular dispersion (degrees)",
        "grain_dispersion_vs_eta.png"
    )

    plot_metric(
        summary_df,
        "boundary_rms_angle_deg",
        "Boundary RMS mismatch (degrees)",
        "boundary_rms_vs_eta.png"
    )

    plot_metric(
        summary_df,
        "boundary_mean_angle_deg",
        "Mean boundary mismatch (degrees)",
        "boundary_mean_vs_eta.png"
    )

    plot_metric(
        summary_df,
        "intragranular_dispersion_deg",
        "Intragranular dispersion (degrees)",
        "intragranular_dispersion_vs_eta.png"
    )

    plot_metric(
        summary_df,
        "E_exchange_boundary",
        "Boundary exchange energy",
        "boundary_exchange_vs_eta.png"
    )

    plot_metric(
        summary_df,
        "E_exchange_intragranular",
        "Intragranular exchange energy",
        "intragranular_exchange_vs_eta.png"
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    grain_stats = (
        summary_df.groupby(
            ["seed", "eta_boundary"]
        )["grain_angle_std_deg"]
        .mean()
        .reset_index()
    )

    final_stats = (
        grain_stats.groupby("eta_boundary")[
            "grain_angle_std_deg"
        ]
        .agg(["mean", "std", "min", "max"])
    )

    print("\n" + "=" * 65)
    print("STAGE E COMPLETED")
    print("=" * 65)

    print("\nMEAN GRAIN DISPERSION ACROSS SEEDS")
    print(final_stats.to_string())

    print("\nOptimizer failures:", len(failures))

    print("\nOUTPUT DIRECTORY:")
    print(os.path.abspath(OUTPUT_DIR))

    print("\nFILES:")
    print("  stageE_summary.csv")
    print("  stageE_grain_angles.csv")
    print("  stageE_cycle_closure.csv")
    print("  stageE_seed_statistics.csv")
    print("  stageE_optimizer_failures.csv")

    print("\nFIGURES:")
    print(os.path.abspath(FIGURE_DIR))


if __name__ == "__main__":
    main()
