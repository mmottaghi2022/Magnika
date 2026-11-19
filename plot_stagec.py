
#!/usr/bin/env python3

"""
STAGE C VISUAL ANALYSIS
Magnika micromagnetic simulation

Reads:
  rotating_field_summary.csv
  rotating_field_grain_angles.csv
  rotating_field_cycle_closure.csv

Creates:
  stageC_visual_report.png
  stageC_grain_analysis.png
  stageC_analysis.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE = Path(__file__).resolve().parent

DATA = BASE / "stageB_results" / "stageC_results"

SUMMARY_FILE = DATA / "rotating_field_summary.csv"
GRAIN_FILE = DATA / "rotating_field_grain_angles.csv"
CLOSURE_FILE = DATA / "rotating_field_cycle_closure.csv"

OUTPUT = DATA / "visual_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"Loaded {path.name}: {len(df)} rows")
    return df


summary = load_csv(SUMMARY_FILE)
grains = load_csv(GRAIN_FILE)

closure = None
if CLOSURE_FILE.exists():
    closure = load_csv(CLOSURE_FILE)
else:
    print("WARNING: Cycle closure CSV not found.")

# Check required columns
required_summary = [
    "eta_boundary",
    "field_angle_deg",
    "Mx",
    "My",
    "E_total",
    "grain_angle_std_deg"
]

required_grains = [
    "eta_boundary",
    "field_angle_deg",
    "grain_id",
    "grain_angle_deg"
]

for col in required_summary:
    if col not in summary.columns:
        raise ValueError(f"Missing summary column: {col}")

for col in required_grains:
    if col not in grains.columns:
        raise ValueError(f"Missing grain column: {col}")

# Convert numerical columns
for col in summary.columns:
    summary[col] = pd.to_numeric(summary[col], errors="coerce")

for col in grains.columns:
    grains[col] = pd.to_numeric(grains[col], errors="coerce")

# Report missing values
print("\nMissing values:")
print("Summary:", int(summary.isna().sum().sum()))
print("Grains:", int(grains.isna().sum().sum()))

# Remove exact duplicate rows
summary = summary.drop_duplicates()
grains = grains.drop_duplicates()

etas = sorted(summary["eta_boundary"].dropna().unique())

print("\nBoundary coefficients:", etas)

# --------------------------------------------------
# 3. ANGULAR UTILITIES
# --------------------------------------------------

def circular_difference(a, b):
    """Signed angular difference in degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


# --------------------------------------------------
# 4. FIGURE 1: MAIN DASHBOARD
# --------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(16, 17))
fig.suptitle(
    "MAGNIKA — STAGE C ROTATING FIELD ANALYSIS",
    fontsize=17,
    fontweight="bold"
)

# Plot 1: Mx and My
ax = axes[0, 0]

for eta in etas:
    d = summary[summary.eta_boundary == eta].sort_values(
        "field_angle_deg"
    )

    ax.plot(
        d.field_angle_deg, d.Mx,
        label=f"Mx, η={eta}"
    )
    ax.plot(
        d.field_angle_deg, d.My,
        linestyle="--",
        label=f"My, η={eta}"
    )

ax.set_title("1. Magnetization vs Rotation Angle")
ax.set_xlabel("Field angle (degrees)")
ax.set_ylabel("Normalized magnetization")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, ncol=2)

# Plot 2: Polar trajectory
ax = axes[0, 1]

for eta in etas:
    d = summary[summary.eta_boundary == eta]

    ax.plot(
        d.Mx,
        d.My,
        marker=".",
        markersize=3,
        label=f"η={eta}"
    )

ax.set_title("2. Polar Magnetization Trajectory")
ax.set_xlabel("Mx")
ax.set_ylabel("My")
ax.axis("equal")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Grain dispersion
ax = axes[1, 0]

for eta in etas:
    d = summary[summary.eta_boundary == eta].sort_values(
        "field_angle_deg"
    )

    ax.plot(
        d.field_angle_deg,
        d.grain_angle_std_deg,
        label=f"η={eta}"
    )

ax.set_title("3. Grain Angular Dispersion")
ax.set_xlabel("Field angle (degrees)")
ax.set_ylabel("Grain angular std. (degrees)")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Total energy
ax = axes[1, 1]

for eta in etas:
    d = summary[summary.eta_boundary == eta].sort_values(
        "field_angle_deg"
    )

    ax.plot(
        d.field_angle_deg,
        d.E_total,
        label=f"η={eta}"
    )

ax.set_title("4. Total Energy")
ax.set_xlabel("Field angle (degrees)")
ax.set_ylabel("Dimensionless energy")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 5: Energy components
ax = axes[2, 0]

energy_columns = [
    ("E_exchange", "Exchange"),
    ("E_anisotropy", "Anisotropy"),
    ("E_zeeman", "Zeeman")
]

available = [
    (c, label)
    for c, label in energy_columns
    if c in summary.columns
]

if available:
    d = summary[summary.eta_boundary == etas[0]].sort_values(
        "field_angle_deg"
    )

    for col, label in available:
        ax.plot(
            d.field_angle_deg,
            d[col],
            label=label
        )

    ax.set_title(f"5. Energy Components (η={etas[0]})")
    ax.set_xlabel("Field angle (degrees)")
    ax.set_ylabel("Dimensionless energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, "Energy components unavailable",
            ha="center", va="center")

# Plot 6: Macroscopic closure
ax = axes[2, 1]

closure_rows = []

for eta in etas:
    d = summary[summary.eta_boundary == eta].sort_values(
        "field_angle_deg"
    )

    first = d.iloc[0]
    last = d.iloc[-1]

    dx = last.Mx - first.Mx
    dy = last.My - first.My

    error = np.sqrt(dx**2 + dy**2)

    closure_rows.append({
        "eta": eta,
        "initial_Mx": first.Mx,
        "final_Mx": last.Mx,
        "initial_My": first.My,
        "final_My": last.My,
        "closure_error": error
    })

closure_df = pd.DataFrame(closure_rows)

x = np.arange(len(etas))
width = 0.35

ax.bar(
    x - width/2,
    closure_df.closure_error,
    width,
    label="Closure error"
)

ax.set_xticks(x)
ax.set_xticklabels([str(e) for e in etas])
ax.set_xlabel("Boundary coefficient η")
ax.set_ylabel("Magnetization closure error")
ax.set_title("6. Macroscopic Cycle Closure")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])

dashboard = OUTPUT / "stageC_visual_report.png"
plt.savefig(dashboard, dpi=180, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# 5. FIGURE 2: GRAIN ORIENTATION ANALYSIS
# --------------------------------------------------

target_angles = [0, 90, 180, 270, 360]

fig, axes = plt.subplots(
    len(etas), len(target_angles),
    figsize=(18, max(9, len(etas) * 3.2)),
    squeeze=False
)

fig.suptitle(
    "GRAIN ORIENTATION DISTRIBUTION DURING ROTATION",
    fontsize=16,
    fontweight="bold"
)

for i, eta in enumerate(etas):

    gd = grains[grains.eta_boundary == eta]

    for j, target in enumerate(target_angles):

        ax = axes[i, j]

        available_angles = gd.field_angle_deg.unique()

        if len(available_angles) == 0:
            continue

        actual = available_angles[
            np.argmin(np.abs(available_angles - target))
        ]

        d = gd[np.isclose(gd.field_angle_deg, actual)]

        angles = d.grain_angle_deg.dropna().values

        ax.hist(
            angles,
            bins=18,
            density=True
        )

        ax.set_title(
            f"η={eta}, Field={target}°"
        )

        ax.set_xlim(-180, 180)
        ax.set_xlabel("Grain direction (degrees)")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])

grain_image = OUTPUT / "stageC_grain_analysis.png"

plt.savefig(
    grain_image,
    dpi=180,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 6. MICROSCOPIC CYCLE CLOSURE
# --------------------------------------------------

micro_rows = []

for eta in etas:

    gd = grains[grains.eta_boundary == eta]

    first_angle = gd.field_angle_deg.min()
    last_angle = gd.field_angle_deg.max()

    first = gd[np.isclose(gd.field_angle_deg, first_angle)]
    last = gd[np.isclose(gd.field_angle_deg, last_angle)]

    merged = first.merge(
        last,
        on="grain_id",
        suffixes=("_initial", "_final")
    )

    differences = circular_difference(
        merged.grain_angle_deg_final.values,
        merged.grain_angle_deg_initial.values
    )

    micro_rows.append({
        "eta": eta,
        "matched_grains": len(merged),
        "grain_closure_rms_deg": float(
            np.sqrt(np.mean(differences**2))
        ) if len(differences) else np.nan,
        "grain_closure_max_deg": float(
            np.max(np.abs(differences))
        ) if len(differences) else np.nan
    })

micro_df = pd.DataFrame(micro_rows)

# --------------------------------------------------
# 7. ROTATIONAL HYSTERESIS CHECK
# --------------------------------------------------

# A single monotonic 0-to-360-degree rotation does not
# provide two independent states at each field angle.
# Therefore a full angular hysteresis loop cannot be
# established from this experiment alone.

# --------------------------------------------------
# 8. GENERATE TEXT REPORT
# --------------------------------------------------

report = []

report.append("MAGNIKA STAGE C ANALYSIS")
report.append("=" * 55)

report.append(
    f"\nSummary records: {len(summary)}"
)
report.append(
    f"Grain records: {len(grains)}"
)

report.append("\nMACROSCOPIC CYCLE CLOSURE")
report.append(closure_df.to_string(index=False))

report.append("\nMICROSCOPIC GRAIN CLOSURE")
report.append(micro_df.to_string(index=False))

report.append("\nBOUNDARY DISPERSION")

for eta in etas:

    d = summary[summary.eta_boundary == eta]

    initial = d.iloc[0].grain_angle_std_deg
    maximum = d.grain_angle_std_deg.max()
    minimum = d.grain_angle_std_deg.min()

    report.append(
        f"eta={eta}: initial={initial:.4f} deg, "
        f"minimum={minimum:.4f} deg, "
        f"maximum={maximum:.4f} deg"
    )

report.append("\nINTERPRETATION")

report.append(
    "1. Compare grain dispersion across boundary coefficients. "
    "A higher dispersion at lower eta indicates greater "
    "grain-level angular variation in this model."
)

report.append(
    "2. Compare Mx and My trajectories to assess sensitivity "
    "of macroscopic magnetization to boundary coupling."
)

report.append(
    "3. Macroscopic closure and microscopic grain closure "
    "are different measurements. Both must be considered."
)

report.append(
    "4. Small cycle closure error demonstrates return of the "
    "measured state, not absence of hysteresis."
)

report.append(
    "5. The present 0-to-360-degree monotonic rotation does "
    "not independently measure angular hysteresis. A reverse "
    "rotation or repeated-angle comparison is required."
)

report.append(
    "6. This is a dimensionless exploratory model without "
    "nonlocal magnetostatic interactions or material calibration."
)

report.append("\nCYCLE CLOSURE CSV")

if closure is not None:
    report.append(closure.to_string(index=False))
else:
    report.append("Not available; calculated from summary CSV.")

report_file = OUTPUT / "stageC_analysis.txt"

report_file.write_text(
    "\n".join(report),
    encoding="utf-8"
)

# --------------------------------------------------
# 9. SAVE NUMERICAL TABLES
# --------------------------------------------------

closure_df.to_csv(
    OUTPUT / "macroscopic_closure.csv",
    index=False
)

micro_df.to_csv(
    OUTPUT / "microscopic_closure.csv",
    index=False
)

# --------------------------------------------------
# 10. FINISH
# --------------------------------------------------

print("\n========================================")
print("STAGE C VISUAL ANALYSIS COMPLETED")
print("========================================")

print("Output directory:", OUTPUT.resolve())

print("\nGenerated files:")

for f in [
    dashboard,
    grain_image,
    report_file,
    OUTPUT / "macroscopic_closure.csv",
    OUTPUT / "microscopic_closure.csv"
]:
    print(f.name)

print("\nUpload stageC_visual_report.png here first.")
