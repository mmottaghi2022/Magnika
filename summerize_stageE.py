
import os
import numpy as np
import pandas as pd

FOLDER = "stageE_results"

summary_file = os.path.join(FOLDER, "stageE_summary.csv")
closure_file = os.path.join(FOLDER, "stageE_cycle_closure.csv")

if not os.path.exists(summary_file):
    raise FileNotFoundError(summary_file)

df = pd.read_csv(summary_file)
closure = pd.read_csv(closure_file)

ETAS = sorted(df["eta_boundary"].unique())
SEEDS = sorted(df["seed"].unique())

# ============================================================
# 1. Average over the rotating-field cycle for each seed
# ============================================================

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

per_seed = (
    df.groupby(["seed", "eta_boundary"])[metrics]
    .mean()
    .reset_index()
)

# ============================================================
# 2. Statistics across independent grain structures
# ============================================================

print("\n" + "=" * 85)
print("MAGNIKA — STAGE E SUMMARY")
print("=" * 85)

print(f"Seeds: {len(SEEDS)}")
print(f"Boundary coefficients: {len(ETAS)}")
print(f"Total simulation samples: {len(df)}")

print("\n1. GRAIN DISPERSION")
print("-" * 85)

dispersion = per_seed.groupby("eta_boundary")[
    "grain_angle_std_deg"
].agg(["mean", "std", "min", "max"])

print(
    dispersion.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n2. BOUNDARY RMS ANGULAR MISMATCH")
print("-" * 85)

boundary = per_seed.groupby("eta_boundary")[
    "boundary_rms_angle_deg"
].agg(["mean", "std", "min", "max"])

print(
    boundary.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n3. INTRAGRANULAR DISPERSION")
print("-" * 85)

intra = per_seed.groupby("eta_boundary")[
    "intragranular_dispersion_deg"
].agg(["mean", "std"])

print(
    intra.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n4. EXCHANGE ENERGY PARTITION")
print("-" * 85)

energy = per_seed.groupby("eta_boundary")[
    [
        "E_exchange_intragranular",
        "E_exchange_boundary"
    ]
].mean()

print(
    energy.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n5. MACROSCOPIC MAGNETIZATION")
print("-" * 85)

mag = per_seed.groupby("eta_boundary")[
    "M_magnitude"
].agg(["mean", "std"])

print(
    mag.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)

# ============================================================
# 6. Cycle closure
# ============================================================

print("\n6. CYCLE CLOSURE")
print("-" * 85)

closure_metrics = [
    "macro_closure_error",
    "grain_closure_rms_deg",
    "grain_closure_max_deg"
]

print(
    closure.groupby("eta_boundary")[closure_metrics]
    .mean()
    .to_string(float_format=lambda x: f"{x:.8e}")
)

# ============================================================
# 7. Optimizer diagnostics
# ============================================================

print("\n7. OPTIMIZER DIAGNOSTICS")
print("-" * 85)

failures = df[~df["optimizer_success"]]

print(f"Failed optimizer samples: {len(failures)}")
print(f"Total optimizer samples: {len(df)}")

print(
    "Maximum gradient:",
    f"{df['max_abs_gradient'].max():.8e}"
)

if len(failures):
    print("\nFailure details:")
    print(
        failures[
            ["seed", "eta_boundary", "step",
             "field_angle_deg", "max_abs_gradient"]
        ].head(15).to_string(index=False)
    )

# ============================================================
# 8. Reproducibility assessment
# ============================================================

print("\n8. REPRODUCIBILITY")
print("-" * 85)

for metric, label in [
    ("grain_angle_std_deg", "Grain dispersion"),
    ("boundary_rms_angle_deg", "Boundary RMS mismatch")
]:

    stats = per_seed.groupby("eta_boundary")[metric].agg(
        ["mean", "std"]
    )

    relative_std = (
        100.0 * stats["std"] / stats["mean"].abs()
    )

    print(f"\n{label}:")
    for eta, value in relative_std.items():
        print(
            f"  eta={eta:6.3f} | "
            f"between-seed CV={value:6.2f}%"
        )

# ============================================================
# 9. Overall trend
# ============================================================

print("\n9. OVERALL TREND")
print("-" * 85)

first_eta = ETAS[0]
last_eta = ETAS[-1]

first = dispersion.loc[first_eta, "mean"]
last = dispersion.loc[last_eta, "mean"]

reduction = 100.0 * (first - last) / first

print(f"Lowest eta: {first_eta}")
print(f"Highest eta: {last_eta}")

print(f"Mean dispersion at lowest eta: {first:.4f} deg")
print(f"Mean dispersion at highest eta: {last:.4f} deg")

print(f"Dispersion reduction: {reduction:.2f}%")

# Check whether each seed shows the same direction.
seed_changes = []

for seed in SEEDS:

    values = per_seed[
        per_seed["seed"] == seed
    ].set_index("eta_boundary")["grain_angle_std_deg"]

    decreases = values.loc[last_eta] < values.loc[first_eta]

    seed_changes.append(decreases)

print(
    f"Seeds showing reduced dispersion: "
    f"{sum(seed_changes)}/{len(SEEDS)}"
)

print("\n" + "=" * 85)
print("END OF STAGE E SUMMARY")
print("=" * 85)
