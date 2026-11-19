
#!/usr/bin/env python3

"""
MAGNIKA — Stage D
Boundary Coupling Sensitivity Sweep

Uses the existing stageb.py model without changing its physics.

Fixed parameters:
    SEED       = 17
    NX, NY     = 40, 40
    N_GRAINS   = 100
    K_ANIS     = 0.12
    J_EXCHANGE = 1.0
    H_ROT      = 0.45
    n_steps    = 72
    MAX_ITER   = 250

Swept parameter:
    eta_boundary = [0.01, 0.025, 0.05, 0.1, 0.2,
                    0.3, 0.5, 0.7, 1.0]

Outputs:
    stageD_results/
        stageD_summary.csv
        stageD_grain_angles.csv
        stageD_cycle_closure.csv
        stageD_key_metrics.csv
        stageD_report.txt
        figures/
"""

from pathlib import Path
import csv
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import stageb as base


# ------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ------------------------------------------------------------

ETAS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

H_ROT = 0.45
N_STEPS = 72

OUTPUT_DIR = Path("stageD_results")


# ------------------------------------------------------------
# ANGLE UTILITIES
# ------------------------------------------------------------

def wrap_degrees(angle):
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles):
    angles = np.asarray(angles, dtype=float)
    c = np.mean(np.cos(np.radians(angles)))
    s = np.mean(np.sin(np.radians(angles)))
    return float(np.degrees(np.arctan2(s, c)))


def circular_std_deg(angles):
    angles = np.asarray(angles, dtype=float)
    c = np.mean(np.cos(np.radians(angles)))
    s = np.mean(np.sin(np.radians(angles)))

    resultant = np.sqrt(c*c + s*s)
    resultant = np.clip(resultant, 1e-15, 1.0)

    return float(np.degrees(np.sqrt(max(0.0, -2.0*np.log(resultant)))))


# ------------------------------------------------------------
# GRAIN STATISTICS
# ------------------------------------------------------------

def calculate_grain_statistics(theta, grain_map):
    """
    Calculate the circular mean orientation and resultant
    for every grain, using the same circular-mean definition
    as Stage C.
    """

    theta_grid = np.asarray(theta).reshape(grain_map.shape)

    grain_ids = np.unique(grain_map)

    grain_angles = []
    grain_resultants = []
    grain_records = []

    for grain_id in grain_ids:

        local = theta_grid[grain_map == grain_id]

        c = np.mean(np.cos(local))
        s = np.mean(np.sin(local))

        resultant = float(np.sqrt(c*c + s*s))
        angle = float(np.degrees(np.arctan2(s, c)))

        grain_angles.append(angle)
        grain_resultants.append(resultant)

        grain_records.append({
            "grain_id": int(grain_id),
            "grain_angle_deg": angle,
            "grain_resultant": resultant,
        })

    grain_angles = np.asarray(grain_angles)

    mean_angle = circular_mean_deg(grain_angles)
    dispersion = circular_std_deg(grain_angles)

    global_c = np.mean(np.cos(np.radians(grain_angles)))
    global_s = np.mean(np.sin(np.radians(grain_angles)))

    global_resultant = float(
        np.sqrt(global_c**2 + global_s**2)
    )

    return {
        "grain_records": grain_records,
        "grain_angles": grain_angles,
        "grain_mean_angle_deg": mean_angle,
        "grain_angle_std_deg": dispersion,
        "grain_direction_resultant": global_resultant,
        "mean_grain_resultant": float(np.mean(grain_resultants)),
    }


# ------------------------------------------------------------
# ENERGY DECOMPOSITION
# ------------------------------------------------------------

def calculate_energy_components(model, theta, hx, hy, eta):
    """
    Use the original model's energy function.

    Split the exchange energy into:
        - intragranular exchange
        - intergranular boundary exchange

    The sum must equal the model's original exchange energy.
    """

    total, gradient, components = model.energy_gradient(
        theta, hx, hy, eta
    )

    theta = np.asarray(theta)

    delta = theta[model.edge_i] - theta[model.edge_j]

    edge_energy = model.J * (1.0 - np.cos(delta))

    boundary_mask = np.asarray(model.boundary, dtype=bool)

    e_boundary = float(
        np.sum(edge_energy[boundary_mask]) * eta
    )

    e_intragranular = float(
        np.sum(edge_energy[~boundary_mask])
    )

    e_exchange = float(components["exchange"])

    split_error = abs(
        e_boundary + e_intragranular - e_exchange
    )

    return {
        "E_exchange": e_exchange,
        "E_intragranular": e_intragranular,
        "E_boundary": e_boundary,
        "E_anisotropy": float(components["anisotropy"]),
        "E_zeeman": float(components["zeeman"]),
        "E_total": float(total),
        "energy_split_error": split_error,
        "max_abs_gradient": float(
            np.max(np.abs(gradient))
        ),
    }


# ------------------------------------------------------------
# ONE COMPLETE ROTATING-FIELD RUN
# ------------------------------------------------------------

def run_one_eta(model, grain_map, theta_initial, eta):

    summary_records = []
    grain_records = []

    theta = theta_initial.copy()

    angles = np.linspace(0.0, 360.0, N_STEPS + 1)

    print(f"\nRunning eta_boundary = {eta:g}")

    for step, field_angle in enumerate(angles):

        angle_rad = np.radians(field_angle)

        hx = H_ROT * np.cos(angle_rad)
        hy = H_ROT * np.sin(angle_rad)

        # Relax using the existing Stage B model.
        theta, result = model.relax(
            theta, hx, hy, eta
        )

        theta = np.asarray(theta).copy()

        # Macroscopic magnetization.
        mx, my = model.magnetization(theta)

        mx = float(mx)
        my = float(my)

        magnetization_magnitude = float(
            np.sqrt(mx*mx + my*my)
        )

        magnetization_angle = float(
            np.degrees(np.arctan2(my, mx))
        )

        # Signed phase lag: magnetization angle minus field angle.
        phase_lag = float(
            wrap_degrees(magnetization_angle - field_angle)
        )

        # Grain-level measurements.
        grain_stats = calculate_grain_statistics(
            theta, grain_map
        )

        # Energy components and gradient.
        energy = calculate_energy_components(
            model, theta, hx, hy, eta
        )

        optimizer_success = bool(result.success)

        summary_records.append({
            "eta_boundary": eta,
            "step": step,
            "field_angle_deg": float(field_angle),
            "Hx": float(hx),
            "Hy": float(hy),
            "Mx": mx,
            "My": my,
            "M_magnitude": magnetization_magnitude,
            "magnetization_angle_deg": magnetization_angle,
            "phase_lag_deg": phase_lag,

            "grain_mean_angle_deg":
                grain_stats["grain_mean_angle_deg"],

            "grain_angle_std_deg":
                grain_stats["grain_angle_std_deg"],

            "grain_direction_resultant":
                grain_stats["grain_direction_resultant"],

            "mean_grain_resultant":
                grain_stats["mean_grain_resultant"],

            "optimizer_success": optimizer_success,
            "optimizer_iterations": int(
                getattr(result, "nit", 0)
            ),

            "optimizer_function_evaluations": int(
                getattr(result, "nfev", 0)
            ),

            "optimizer_message": str(result.message),

            **energy,
        })

        for record in grain_stats["grain_records"]:
            grain_records.append({
                "eta_boundary": eta,
                "step": step,
                "field_angle_deg": float(field_angle),
                **record,
            })

        if step % 12 == 0 or step == N_STEPS:
            print(
                f"  {field_angle:6.1f} deg | "
                f"M=({mx:+.6f},{my:+.6f}) | "
                f"grain_std={grain_stats['grain_angle_std_deg']:.4f} deg | "
                f"success={optimizer_success}"
            )

    return summary_records, grain_records


# ------------------------------------------------------------
# CYCLE CLOSURE
# ------------------------------------------------------------

def calculate_cycle_closure(summary, grains):

    macro_rows = []
    grain_rows = []

    for eta in ETAS:

        s = summary[
            summary["eta_boundary"] == eta
        ].sort_values("step")

        first = s.iloc[0]
        last = s.iloc[-1]

        macro_error = float(np.sqrt(
            (last["Mx"] - first["Mx"])**2
            + (last["My"] - first["My"])**2
        ))

        macro_rows.append({
            "eta_boundary": eta,
            "initial_Mx": first["Mx"],
            "initial_My": first["My"],
            "final_Mx": last["Mx"],
            "final_My": last["My"],
            "magnetization_cycle_closure_error": macro_error,
            "initial_energy": first["E_total"],
            "final_energy": last["E_total"],
            "energy_cycle_difference":
                last["E_total"] - first["E_total"],
        })

        g = grains[
            grains["eta_boundary"] == eta
        ]

        first_grains = g[g["step"] == g["step"].min()]
        last_grains = g[g["step"] == g["step"].max()]

        merged = first_grains.merge(
            last_grains,
            on="grain_id",
            suffixes=("_initial", "_final"),
        )

        delta = wrap_degrees(
            merged["grain_angle_deg_final"].to_numpy()
            - merged["grain_angle_deg_initial"].to_numpy()
        )

        grain_rows.append({
            "eta_boundary": eta,
            "matched_grains": len(merged),
            "grain_closure_mean_abs_deg":
                float(np.mean(np.abs(delta))),
            "grain_closure_rms_deg":
                float(np.sqrt(np.mean(delta**2))),
            "grain_closure_max_deg":
                float(np.max(np.abs(delta))),
        })

    return pd.DataFrame(macro_rows), pd.DataFrame(grain_rows)


# ------------------------------------------------------------
# SUMMARY METRICS
# ------------------------------------------------------------

def calculate_key_metrics(summary, grain_closure):

    rows = []

    for eta in ETAS:

        s = summary[
            summary["eta_boundary"] == eta
        ]

        closure = grain_closure[
            grain_closure["eta_boundary"] == eta
        ].iloc[0]

        rows.append({
            "eta_boundary": eta,

            "mean_grain_dispersion_deg":
                float(s["grain_angle_std_deg"].mean()),

            "max_grain_dispersion_deg":
                float(s["grain_angle_std_deg"].max()),

            "min_grain_dispersion_deg":
                float(s["grain_angle_std_deg"].min()),

            "mean_magnetization_magnitude":
                float(s["M_magnitude"].mean()),

            "min_magnetization_magnitude":
                float(s["M_magnitude"].min()),

            "max_abs_phase_lag_deg":
                float(s["phase_lag_deg"].abs().max()),

            "mean_abs_phase_lag_deg":
                float(s["phase_lag_deg"].abs().mean()),

            "mean_intragranular_exchange":
                float(s["E_intragranular"].mean()),

            "mean_boundary_exchange":
                float(s["E_boundary"].mean()),

            "mean_total_exchange":
                float(s["E_exchange"].mean()),

            "mean_anisotropy_energy":
                float(s["E_anisotropy"].mean()),

            "mean_zeeman_energy":
                float(s["E_zeeman"].mean()),

            "optimizer_failures":
                int((~s["optimizer_success"]).sum()),

            "max_abs_gradient":
                float(s["max_abs_gradient"].max()),

            "max_energy_split_error":
                float(s["energy_split_error"].max()),

            "grain_closure_rms_deg":
                float(closure["grain_closure_rms_deg"]),

            "grain_closure_max_deg":
                float(closure["grain_closure_max_deg"]),
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------

def make_plots(summary, metrics, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    etas = sorted(summary["eta_boundary"].unique())

    # 1. Mean grain dispersion against eta.
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        metrics["eta_boundary"],
        metrics["mean_grain_dispersion_deg"],
        marker="o",
        label="Mean over rotation",
    )

    ax.plot(
        metrics["eta_boundary"],
        metrics["max_grain_dispersion_deg"],
        marker="s",
        label="Maximum over rotation",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Boundary coupling coefficient (eta)")
    ax.set_ylabel("Grain angular dispersion (degrees)")
    ax.set_title("Stage D — Grain Dispersion vs Boundary Coupling")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "D1_grain_dispersion_vs_eta.png", dpi=220)
    plt.close(fig)

    # 2. Macroscopic magnetization deviation.
    reference = metrics[
        metrics["eta_boundary"] == 1.0
    ]["mean_magnetization_magnitude"].iloc[0]

    metrics = metrics.copy()

    metrics["magnetization_deviation_percent"] = (
        100.0
        * (
            metrics["mean_magnetization_magnitude"]
            - reference
        )
        / reference
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        metrics["eta_boundary"],
        metrics["magnetization_deviation_percent"],
        marker="o",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Boundary coupling coefficient (eta)")
    ax.set_ylabel("Mean magnetization deviation from eta=1 (%)")
    ax.set_title("Stage D — Macroscopic Magnetization Sensitivity")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "D2_magnetization_sensitivity.png", dpi=220)
    plt.close(fig)

    # 3. Grain dispersion over the field rotation.
    fig, ax = plt.subplots(figsize=(10, 6))

    for eta in etas:
        g = summary[
            summary["eta_boundary"] == eta
        ].sort_values("field_angle_deg")

        ax.plot(
            g["field_angle_deg"],
            g["grain_angle_std_deg"],
            label=f"eta={eta:g}",
        )

    ax.set_xlabel("Applied field angle (degrees)")
    ax.set_ylabel("Grain angular dispersion (degrees)")
    ax.set_title("Grain Dispersion During Rotation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "D3_dispersion_vs_angle.png", dpi=220)
    plt.close(fig)

    # 4. Macroscopic magnetization trajectories.
    fig, ax = plt.subplots(figsize=(7, 7))

    for eta in etas:
        g = summary[
            summary["eta_boundary"] == eta
        ].sort_values("field_angle_deg")

        ax.plot(g["Mx"], g["My"], label=f"eta={eta:g}")

    ax.set_xlabel("Mx")
    ax.set_ylabel("My")
    ax.set_title("Macroscopic Magnetization Trajectories")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "D4_magnetization_trajectories.png", dpi=220)
    plt.close(fig)

    # 5. Exchange energy split.
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(
        metrics["eta_boundary"],
        metrics["mean_intragranular_exchange"],
        marker="o",
        label="Intragranular",
    )

    axes[0].plot(
        metrics["eta_boundary"],
        metrics["mean_boundary_exchange"],
        marker="s",
        label="Intergranular boundary",
    )

    axes[0].set_ylabel("Mean exchange energy")
    axes[0].set_title("Exchange Energy Decomposition")
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        metrics["eta_boundary"],
        metrics["mean_boundary_exchange"],
        marker="o",
    )

    axes[1].set_xlabel("Boundary coupling coefficient (eta)")
    axes[1].set_ylabel("Mean boundary exchange energy")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "D5_exchange_energy.png", dpi=220)
    plt.close(fig)

    # 6. Phase lag over the rotation.
    fig, ax = plt.subplots(figsize=(10, 5))

    for eta in etas:
        g = summary[
            summary["eta_boundary"] == eta
        ].sort_values("field_angle_deg")

        ax.plot(
            g["field_angle_deg"],
            g["phase_lag_deg"],
            label=f"eta={eta:g}",
        )

    ax.set_xlabel("Applied field angle (degrees)")
    ax.set_ylabel("Magnetization minus field angle (degrees)")
    ax.set_title("Phase Lag During Rotation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "D6_phase_lag.png", dpi=220)
    plt.close(fig)


# ------------------------------------------------------------
# TEXT REPORT
# ------------------------------------------------------------

def write_report(summary, macro, grain_closure, metrics, output_dir):

    lines = [
        "MAGNIKA STAGE D — BOUNDARY COUPLING SENSITIVITY",
        "=" * 60,
        "",
        f"Summary records: {len(summary)}",
        f"Grain records: {summary.shape[0] * base.N_GRAINS}",
        f"Eta values: {list(ETAS)}",
        f"Field magnitude: {H_ROT}",
        f"Angular intervals: {N_STEPS}",
        "",
        "KEY METRICS",
        "-" * 60,
        metrics.to_string(index=False),
        "",
        "MACROSCOPIC CYCLE CLOSURE",
        "-" * 60,
        macro.to_string(index=False),
        "",
        "MICROSCOPIC GRAIN CLOSURE",
        "-" * 60,
        grain_closure.to_string(index=False),
        "",
        "INTERPRETATION NOTES",
        "-" * 60,
        "1. Lower eta represents weaker intergranular exchange coupling.",
        "2. Grain dispersion measures variation among grain mean orientations.",
        "3. Macroscopic magnetization is calculated from all lattice sites.",
        "4. Intragranular and boundary exchange are split using the",
        "   model's original nearest-neighbour exchange-edge definition.",
        "5. Cycle closure compares the initial 0-degree state with the",
        "   final 360-degree state.",
        "6. This monotonic rotation does not independently measure",
        "   angular hysteresis.",
        "7. The model remains dimensionless and exploratory.",
        "8. The grain map and easy-axis angles are held fixed across eta.",
        "",
    ]

    (output_dir / "stageD_report.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ------------------------------------------------------------
# MAIN EXPERIMENT
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
    )

    args = parser.parse_args()

    output_dir = args.output.expanduser().resolve()
    figures_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MAGNIKA — STAGE D")
    print("BOUNDARY COUPLING SENSITIVITY SWEEP")
    print("=" * 60)

    # Verify that Stage B constants match the experiment.
    expected = {
        "SEED": 17,
        "NX": 40,
        "NY": 40,
        "N_GRAINS": 100,
        "K_ANIS": 0.12,
        "J_EXCHANGE": 1.0,
        "MAX_ITER": 250,
    }

    for name, value in expected.items():
        actual = getattr(base, name)

        if actual != value:
            raise ValueError(
                f"Stage B configuration mismatch: "
                f"{name} is {actual}, expected {value}"
            )

    print("\nGenerating fixed grain structure...")

    grain_map, grain_angles = base.generate_grains(
        base.NX,
        base.NY,
        base.N_GRAINS,
        base.SEED,
    )

    # Construct the model once. It is reused for all eta values.
    model = base.MicroMagneticModel(
        grain_map,
        grain_angles,
        K=base.K_ANIS,
        J=base.J_EXCHANGE,
    )

    # Identical initial state for every eta experiment.
    theta_initial = np.zeros(grain_map.size, dtype=float)

    all_summary = []
    all_grains = []

    for eta in ETAS:

        # Reset the initial state for each independent eta run.
        summary_records, grain_records = run_one_eta(
            model,
            grain_map,
            theta_initial.copy(),
            eta,
        )

        all_summary.extend(summary_records)
        all_grains.extend(grain_records)

    summary = pd.DataFrame(all_summary)
    grains = pd.DataFrame(all_grains)

    summary.to_csv(
        output_dir / "stageD_summary.csv",
        index=False,
    )

    grains.to_csv(
        output_dir / "stageD_grain_angles.csv",
        index=False,
    )

    # Closure calculations.
    macro_closure, grain_closure = calculate_cycle_closure(
        summary,
        grains,
    )

    macro_closure.to_csv(
        output_dir / "stageD_cycle_closure.csv",
        index=False,
    )

    grain_closure.to_csv(
        output_dir / "stageD_grain_closure.csv",
        index=False,
    )

    metrics = calculate_key_metrics(
        summary,
        grain_closure,
    )

    metrics.to_csv(
        output_dir / "stageD_key_metrics.csv",
        index=False,
    )

    make_plots(summary, metrics, figures_dir)

    write_report(
        summary,
        macro_closure,
        grain_closure,
        metrics,
        output_dir,
    )

    print("\n" + "=" * 60)
    print("STAGE D COMPLETED")
    print("=" * 60)

    print("\nKEY METRICS:")
    print(metrics.to_string(index=False))

    print("\nOUTPUT DIRECTORY:")
    print(output_dir)

    print("\nIMPORTANT FILES:")
    for filename in [
        "stageD_summary.csv",
        "stageD_grain_angles.csv",
        "stageD_cycle_closure.csv",
        "stageD_grain_closure.csv",
        "stageD_key_metrics.csv",
        "stageD_report.txt",
    ]:
        print(" ", output_dir / filename)

    print("\nFIGURES:")
    print(" ", figures_dir)


if __name__ == "__main__":
    main()
