#!/usr/bin/env python3
"""
STAGE B + STAGE C: 2D POLYCRYSTALLINE MICROMAGNETIC MODEL

Exploratory dimensionless model:
- 2D in-plane magnetization
- Voronoi grain structure
- Grain-specific in-plane projection of cubic anisotropy
- Nearest-neighbour exchange
- Reduced exchange across grain boundaries
- Zeeman interaction
- Quasi-static X/Y field sweeps
- Grain-boundary sensitivity experiment
- Quasi-static rotating-field experiment (Stage C)

Limitations:
No nonlocal magnetostatic energy, thermal noise, LLG dynamics,
eddy currents, magnetoelasticity, or SI material calibration.
The rotating-field experiment is quasi-static continuation, not a
time-domain simulation and does not calculate physical spin-rotation rates.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.spatial import cKDTree


# ---------------------------------------------------------
# 1. MODEL PARAMETERS
# ---------------------------------------------------------

SEED = 17

NX = 40
NY = 40
N_GRAINS = 100

K_ANIS = 0.12
J_EXCHANGE = 1.0

ETA_BOUNDARY = 0.20

H_MAX = 1.20
N_BRANCH = 17

MAX_ITER = 250

# Stage C parameters
STAGE_C_ETAS = (1.0, 0.5, 0.2, 0.05)
STAGE_C_FIELD = 0.45
STAGE_C_STEPS = 72


# ---------------------------------------------------------
# 2. GENERATE VORONOI GRAIN STRUCTURE
# ---------------------------------------------------------

def generate_grains(nx, ny, n_grains, seed):
    rng = np.random.default_rng(seed)

    grain_seeds = np.column_stack([
        rng.uniform(0, nx, n_grains),
        rng.uniform(0, ny, n_grains)
    ])

    yy, xx = np.mgrid[0:ny, 0:nx]

    points = np.column_stack([
        xx.ravel(),
        yy.ravel()
    ])

    tree = cKDTree(grain_seeds)
    _, labels = tree.query(points)

    grain_map = labels.reshape(ny, nx)

    # In-plane crystallographic orientation of each grain.
    grain_angles = rng.uniform(
        0.0,
        np.pi / 2,
        n_grains
    )

    return grain_map, grain_angles


# ---------------------------------------------------------
# 3. MICROMAGNETIC MODEL
# ---------------------------------------------------------

class MicroMagneticModel:

    def __init__(
        self,
        grain_map,
        grain_angles,
        K=K_ANIS,
        J=J_EXCHANGE
    ):
        self.grain_map = np.asarray(grain_map)
        self.ny, self.nx = self.grain_map.shape

        self.K = K
        self.J = J

        self.phi = grain_angles[self.grain_map].ravel()
        self.n = self.nx * self.ny

        self.build_exchange_edges()

    def build_exchange_edges(self):
        edge_i = []
        edge_j = []
        boundary = []

        for y in range(self.ny):
            for x in range(self.nx):
                i = y * self.nx + x

                # Right neighbour.
                if x + 1 < self.nx:
                    j = i + 1
                    edge_i.append(i)
                    edge_j.append(j)

                    boundary.append(
                        self.grain_map[y, x]
                        != self.grain_map[y, x + 1]
                    )

                # Upper neighbour.
                if y + 1 < self.ny:
                    j = i + self.nx
                    edge_i.append(i)
                    edge_j.append(j)

                    boundary.append(
                        self.grain_map[y, x]
                        != self.grain_map[y + 1, x]
                    )

        self.edge_i = np.asarray(edge_i, dtype=int)
        self.edge_j = np.asarray(edge_j, dtype=int)
        self.boundary = np.asarray(boundary, dtype=bool)

    # -----------------------------------------------------
    # 4. ENERGY FUNCTION AND ANALYTIC GRADIENT
    # -----------------------------------------------------

    def energy_gradient(self, theta, hx, hy, eta):
        theta = np.asarray(theta)

        # Exchange coupling.
        weights = self.J * np.where(
            self.boundary,
            eta,
            1.0
        )

        delta = (
            theta[self.edge_i]
            - theta[self.edge_j]
        )

        E_exchange = np.sum(
            weights * (1.0 - np.cos(delta))
        )

        gradient = np.zeros_like(theta)

        edge_gradient = weights * np.sin(delta)

        np.add.at(
            gradient,
            self.edge_i,
            edge_gradient
        )

        np.add.at(
            gradient,
            self.edge_j,
            -edge_gradient
        )

        # In-plane cubic anisotropy.
        angle_difference = theta - self.phi
        s = np.sin(2.0 * angle_difference)

        E_anisotropy = np.sum(
            (self.K / 4.0) * s**2
        )

        gradient += (
            self.K
            * s
            * np.cos(2.0 * angle_difference)
        )

        # Zeeman interaction.
        E_zeeman = -np.sum(
            hx * np.cos(theta)
            + hy * np.sin(theta)
        )

        gradient += (
            hx * np.sin(theta)
            - hy * np.cos(theta)
        )

        E_total = (
            E_exchange
            + E_anisotropy
            + E_zeeman
        )

        components = {
            "exchange": E_exchange,
            "anisotropy": E_anisotropy,
            "zeeman": E_zeeman,
            "total": E_total
        }

        return E_total, gradient, components

    # -----------------------------------------------------
    # 5. ENERGY MINIMIZATION
    # -----------------------------------------------------

    def relax(self, theta_initial, hx, hy, eta):

        def objective(theta):
            energy, gradient, _ = self.energy_gradient(
                theta,
                hx,
                hy,
                eta
            )
            return energy, gradient

        result = minimize(
            objective,
            theta_initial,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": MAX_ITER,
                "ftol": 1e-10,
                "gtol": 1e-7,
                "maxls": 30
            }
        )

        return result.x, result

    # -----------------------------------------------------
    # 6. MAGNETIZATION
    # -----------------------------------------------------

    def magnetization(self, theta):
        mx = np.mean(np.cos(theta))
        my = np.mean(np.sin(theta))
        return mx, my


# ---------------------------------------------------------
# 7. NUMERICAL GRADIENT VERIFICATION
# ---------------------------------------------------------

def verify_gradient(model):
    rng = np.random.default_rng(123)

    theta = rng.uniform(
        -np.pi,
        np.pi,
        model.n
    )

    hx = 0.31
    hy = -0.17
    eta = ETA_BOUNDARY

    _, analytic, _ = model.energy_gradient(
        theta,
        hx,
        hy,
        eta
    )

    indices = rng.choice(
        model.n,
        size=12,
        replace=False
    )

    epsilon = 1e-6
    errors = []

    for i in indices:
        plus = theta.copy()
        minus = theta.copy()

        plus[i] += epsilon
        minus[i] -= epsilon

        ep = model.energy_gradient(
            plus, hx, hy, eta
        )[0]

        em = model.energy_gradient(
            minus, hx, hy, eta
        )[0]

        numerical = (ep - em) / (2 * epsilon)
        errors.append(abs(numerical - analytic[i]))

    maximum_error = float(max(errors))

    print(
        "\nGradient verification:",
        f"maximum absolute error = {maximum_error:.3e}"
    )

    if maximum_error > 1e-5:
        raise RuntimeError("Gradient verification failed.")

    return maximum_error


# ---------------------------------------------------------
# 8. QUASI-STATIC FIELD SWEEP
# ---------------------------------------------------------

def run_sweep(model, axis, eta):
    increasing = np.linspace(
        -H_MAX,
        H_MAX,
        N_BRANCH
    )

    decreasing = np.linspace(
        H_MAX,
        -H_MAX,
        N_BRANCH
    )[1:]

    fields = np.concatenate([
        increasing,
        decreasing
    ])

    # Initial negative saturation.
    theta = np.full(model.n, np.pi)

    rows = []
    snapshots = {}
    failed_steps = []

    for step, h in enumerate(fields):
        if axis == "x":
            hx, hy = h, 0.0
        else:
            hx, hy = 0.0, h

        theta, result = model.relax(
            theta,
            hx,
            hy,
            eta
        )

        if not result.success:
            failed_steps.append({
                "step": step,
                "field": float(h),
                "message": str(result.message)
            })

        energy, _, parts = model.energy_gradient(
            theta,
            hx,
            hy,
            eta
        )

        mx, my = model.magnetization(theta)

        rows.append({
            "axis": axis,
            "field": float(h),
            "Mx": float(mx),
            "My": float(my),
            "E_exchange": float(parts["exchange"]),
            "E_anisotropy": float(parts["anisotropy"]),
            "E_zeeman": float(parts["zeeman"]),
            "E_total": float(energy),
            "iterations": int(result.nit),
            "optimizer_success": bool(result.success)
        })

        # Save selected spatial states.
        if step in [
            0,
            len(increasing) // 2,
            len(increasing) - 1,
            len(fields) - 1
        ]:
            snapshots[step] = theta.copy()

    return rows, snapshots, failed_steps


# ---------------------------------------------------------
# 9. GRAIN-BOUNDARY SENSITIVITY
# ---------------------------------------------------------

def grain_boundary_diagnostics(theta, grain_map, eta, J=1.0):
    """
    Measure intragranular and grain-boundary exchange,
    plus grain-level and boundary-level angular differences.
    """
    theta = np.asarray(theta).reshape(grain_map.shape)
    grain_map = np.asarray(grain_map)

    t1 = np.concatenate([
        theta[:, :-1].ravel(),
        theta[:-1, :].ravel()
    ])

    t2 = np.concatenate([
        theta[:, 1:].ravel(),
        theta[1:, :].ravel()
    ])

    g1 = np.concatenate([
        grain_map[:, :-1].ravel(),
        grain_map[:-1, :].ravel()
    ])

    g2 = np.concatenate([
        grain_map[:, 1:].ravel(),
        grain_map[1:, :].ravel()
    ])

    boundary = g1 != g2

    delta = np.arctan2(
        np.sin(t1 - t2),
        np.cos(t1 - t2)
    )

    mismatch = np.abs(delta)
    e_pair = J * (1.0 - np.cos(delta))

    e_intra = float(np.sum(e_pair[~boundary]))
    e_boundary = float(eta * np.sum(e_pair[boundary]))

    if np.any(boundary):
        boundary_mean = float(np.mean(mismatch[boundary]))
        boundary_rms = float(
            np.sqrt(np.mean(mismatch[boundary] ** 2))
        )
        boundary_max = float(np.max(mismatch[boundary]))
    else:
        boundary_mean = 0.0
        boundary_rms = 0.0
        boundary_max = 0.0

    grain_ids = np.unique(grain_map)
    grain_angles = []

    for gid in grain_ids:
        mask = grain_map == gid
        cx = np.mean(np.cos(theta[mask]))
        sy = np.mean(np.sin(theta[mask]))
        grain_angles.append(np.arctan2(sy, cx))

    grain_angles = np.asarray(grain_angles)

    mean_cos = np.mean(np.cos(grain_angles))
    mean_sin = np.mean(np.sin(grain_angles))

    resultant = np.sqrt(mean_cos**2 + mean_sin**2)
    resultant = np.clip(resultant, 1e-12, 1.0)

    grain_angle_std = float(
        np.sqrt(-2.0 * np.log(resultant))
    )

    return {
        "eta_boundary": float(eta),
        "E_exchange_intragranular": e_intra,
        "E_exchange_boundary": e_boundary,
        "E_exchange_total": e_intra + e_boundary,
        "boundary_mean_angle_rad": boundary_mean,
        "boundary_rms_angle_rad": boundary_rms,
        "boundary_max_angle_rad": boundary_max,
        "grain_angle_std_rad": grain_angle_std,
        "grain_angle_std_deg": float(np.degrees(grain_angle_std)),
        "grain_direction_resultant": float(resultant),
        "number_of_grains": int(len(grain_ids))
    }


def boundary_experiment(model, grain_map=None):
    results = []
    field = 0.45
    J = J_EXCHANGE

    if grain_map is None:
        grain_map = getattr(model, "grain_map", None)

    if grain_map is None:
        raise ValueError(
            "Grain map not found. Pass the original grain_map "
            "used to construct the model."
        )

    grain_map = np.asarray(grain_map)

    for eta in [1.0, 0.5, 0.2, 0.05]:
        theta0 = np.full(model.n, 0.15)

        theta, optimizer = model.relax(
            theta0,
            field,
            0.0,
            eta
        )

        energy, _, parts = model.energy_gradient(
            theta,
            field,
            0.0,
            eta
        )

        mx, my = model.magnetization(theta)

        row = {
            "eta_boundary": eta,
            "Mx": float(mx),
            "My": float(my),
            "E_exchange": float(parts["exchange"]),
            "E_anisotropy": float(parts["anisotropy"]),
            "E_zeeman": float(parts["zeeman"]),
            "E_total": float(energy),
            "optimizer_success": bool(optimizer.success)
        }

        row.update(
            grain_boundary_diagnostics(
                theta,
                grain_map,
                eta,
                J
            )
        )

        # Preserve the established Stage B column names.
        row["boundary_mean_angle_deg"] = float(
            np.degrees(row.pop("boundary_mean_angle_rad"))
        )
        row["boundary_rms_angle_deg"] = float(
            np.degrees(row.pop("boundary_rms_angle_rad"))
        )
        row["boundary_max_angle_deg"] = float(
            np.degrees(row.pop("boundary_max_angle_rad"))
        )
        row["grain_angle_std_deg"] = float(
            row["grain_angle_std_deg"]
        )

        results.append(row)

    return results


# ---------------------------------------------------------
# 10. STAGE C: QUASI-STATIC ROTATING-FIELD EXPERIMENT
# ---------------------------------------------------------

def rotating_field_experiment(
    model,
    grain_map,
    theta_initial,
    etas=STAGE_C_ETAS,
    H_ROT=STAGE_C_FIELD,
    n_steps=STAGE_C_STEPS,
    output_dir="stageC_results"
):
    """
    Quasi-static rotating-field experiment.

    The field magnitude is constant and its direction rotates from
    0 to 360 degrees. The previous relaxed state initializes the
    next field-angle step.

    This is not time-domain LLG dynamics.
    """

    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    grain_map = np.asarray(grain_map)
    if grain_map.shape != (model.ny, model.nx):
        raise ValueError("grain_map shape does not match model grid")

    theta_initial = np.asarray(theta_initial, dtype=float)
    if theta_initial.size != model.n:
        raise ValueError("theta_initial must contain model.n angles")

    grain_ids = np.unique(grain_map)
    summary_rows = []
    grain_rows = []
    closure_rows = []

    field_angles = np.linspace(0.0, 360.0, n_steps + 1)

    print("\n========================================")
    print("STAGE C: ROTATING FIELD EXPERIMENT")
    print("========================================")
    print(f"Field magnitude: {H_ROT}")
    print(f"Angular steps: {n_steps}")
    print(f"Number of grains: {len(grain_ids)}")
    print(f"Boundary coefficients: {etas}")

    for eta in etas:
        print(f"\nRunning eta_boundary = {eta}")

        theta = theta_initial.copy()
        eta_rows = []

        for step, angle_deg in enumerate(field_angles):
            angle_rad = np.deg2rad(angle_deg)

            hx = H_ROT * np.cos(angle_rad)
            hy = H_ROT * np.sin(angle_rad)

            theta_opt, result = model.relax(
                theta,
                hx,
                hy,
                eta
            )
            theta = theta_opt.copy()

            energy, gradient, components = model.energy_gradient(
                theta,
                hx,
                hy,
                eta
            )

            mx, my = model.magnetization(theta)
            magnetization_angle = np.degrees(np.arctan2(my, mx))

            phase_lag = (
                magnetization_angle - angle_deg + 180.0
            ) % 360.0 - 180.0

            grain_angles = []
            grain_resultants = []

            for gid in grain_ids:
                mask = grain_map == gid
                local_theta = theta.reshape(grain_map.shape)[mask]

                mean_cos = np.mean(np.cos(local_theta))
                mean_sin = np.mean(np.sin(local_theta))

                grain_angle = np.degrees(
                    np.arctan2(mean_sin, mean_cos)
                )

                resultant = np.sqrt(
                    mean_cos**2 + mean_sin**2
                )

                grain_angles.append(grain_angle)
                grain_resultants.append(resultant)

                grain_rows.append({
                    "eta_boundary": eta,
                    "step": step,
                    "field_angle_deg": float(angle_deg),
                    "grain_id": int(gid),
                    "grain_angle_deg": float(grain_angle),
                    "grain_resultant": float(resultant)
                })

            grain_angles_rad = np.deg2rad(grain_angles)

            global_grain_cos = np.mean(np.cos(grain_angles_rad))
            global_grain_sin = np.mean(np.sin(grain_angles_rad))

            grain_direction_resultant = np.sqrt(
                global_grain_cos**2 + global_grain_sin**2
            )
            grain_direction_resultant = np.clip(
                grain_direction_resultant,
                1e-12,
                1.0
            )

            grain_angle_std_deg = np.degrees(
                np.sqrt(-2.0 * np.log(grain_direction_resultant))
            )

            row = {
                "eta_boundary": float(eta),
                "step": int(step),
                "field_angle_deg": float(angle_deg),
                "Hx": float(hx),
                "Hy": float(hy),
                "Mx": float(mx),
                "My": float(my),
                "M_magnitude": float(np.sqrt(mx**2 + my**2)),
                "magnetization_angle_deg": float(magnetization_angle),
                "phase_lag_deg": float(phase_lag),
                "E_total": float(energy),
                "E_exchange": float(components["exchange"]),
                "E_anisotropy": float(components["anisotropy"]),
                "E_zeeman": float(components["zeeman"]),
                "max_abs_gradient": float(np.max(np.abs(gradient))),
                "optimizer_success": bool(result.success),
                "optimizer_iterations": int(result.nit),
                "grain_angle_std_deg": float(grain_angle_std_deg),
                "grain_direction_resultant": float(
                    grain_direction_resultant
                )
            }

            summary_rows.append(row)
            eta_rows.append(row)

            if step % 12 == 0:
                print(
                    f"  {angle_deg:6.1f} deg | "
                    f"M=({mx:+.4f}, {my:+.4f}) | "
                    f"E={energy:+.4f} | "
                    f"grain_std={grain_angle_std_deg:.3f} deg | "
                    f"success={result.success}"
                )

        # Compare the first and final magnetization vectors for this eta.
        first = eta_rows[0]
        last = eta_rows[-1]

        closure_error = float(np.hypot(
            last["Mx"] - first["Mx"],
            last["My"] - first["My"]
        ))

        closure_rows.append({
            "eta_boundary": float(eta),
            "initial_Mx": first["Mx"],
            "initial_My": first["My"],
            "final_Mx": last["Mx"],
            "final_My": last["My"],
            "magnetization_cycle_closure_error": closure_error,
            "initial_energy": first["E_total"],
            "final_energy": last["E_total"],
            "energy_cycle_difference": (
                last["E_total"] - first["E_total"]
            )
        })

        print(
            f"  Cycle closure error for eta={eta}: "
            f"{closure_error:.8f}"
        )

    save_csv(
        output_path / "rotating_field_summary.csv",
        summary_rows
    )

    save_csv(
        output_path / "rotating_field_grain_angles.csv",
        grain_rows
    )

    save_csv(
        output_path / "rotating_field_cycle_closure.csv",
        closure_rows
    )

    # Plot macroscopic magnetization components against field angle.
    plt.figure(figsize=(9, 6))

    for eta in etas:
        rows = [
            r for r in summary_rows
            if r["eta_boundary"] == eta
        ]
        angles = [r["field_angle_deg"] for r in rows]

        plt.plot(
            angles,
            [r["Mx"] for r in rows],
            label=f"Mx, eta={eta}"
        )

        plt.plot(
            angles,
            [r["My"] for r in rows],
            "--",
            label=f"My, eta={eta}"
        )

    plt.xlabel("Rotating field angle (degrees)")
    plt.ylabel("Normalized magnetization")
    plt.title("Stage C: Magnetization during rotating-field sweep")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        output_path / "rotating_field_magnetization.png",
        dpi=160
    )
    plt.close()

    # Plot grain-orientation dispersion against field angle.
    plt.figure(figsize=(9, 5))

    for eta in etas:
        rows = [
            r for r in summary_rows
            if r["eta_boundary"] == eta
        ]

        plt.plot(
            [r["field_angle_deg"] for r in rows],
            [r["grain_angle_std_deg"] for r in rows],
            label=f"eta={eta}"
        )

    plt.xlabel("Rotating field angle (degrees)")
    plt.ylabel("Grain orientation dispersion (degrees)")
    plt.title("Stage C: Grain orientation dispersion")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path / "rotating_field_grain_dispersion.png",
        dpi=160
    )
    plt.close()

    # Plot energy components against field angle.
    plt.figure(figsize=(9, 6))

    for eta in etas:
        rows = [
            r for r in summary_rows
            if r["eta_boundary"] == eta
        ]

        plt.plot(
            [r["field_angle_deg"] for r in rows],
            [r["E_exchange"] for r in rows],
            label=f"Exchange, eta={eta}"
        )

    plt.xlabel("Rotating field angle (degrees)")
    plt.ylabel("Exchange energy")
    plt.title("Stage C: Exchange energy during rotation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path / "rotating_field_exchange_energy.png",
        dpi=160
    )
    plt.close()

    print("\n========================================")
    print("STAGE C COMPLETED")
    print("========================================")
    print("Summary:", output_path / "rotating_field_summary.csv")
    print("Grain orientations:", output_path / "rotating_field_grain_angles.csv")
    print("Cycle closure:", output_path / "rotating_field_cycle_closure.csv")

    return summary_rows, grain_rows, closure_rows


# ---------------------------------------------------------
# 11. SAVE RESULTS
# ---------------------------------------------------------

def save_csv(filename, rows):
    if not rows:
        return

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------
# 12. MAIN PROGRAM
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        default="stageB_results"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=ETA_BOUNDARY
    )

    parser.add_argument(
        "--skip-stage-c",
        action="store_true",
        help="Run Stage B only; skip the rotating-field experiment."
    )

    args = parser.parse_args()

    if not 0.0 <= args.eta <= 1.0:
        raise ValueError("eta must be between 0 and 1")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    print("Generating grain structure...")

    grain_map, grain_angles = generate_grains(
        NX,
        NY,
        N_GRAINS,
        args.seed
    )

    model = MicroMagneticModel(
        grain_map,
        grain_angles
    )

    print("Grains:", len(np.unique(grain_map)))
    print("Cells:", model.n)
    print("Exchange edges:", len(model.edge_i))
    print("Boundary edges:", int(np.sum(model.boundary)))

    gradient_error = verify_gradient(model)

    print("\nRunning X-direction sweep...")

    x_rows, x_snapshots, x_failures = run_sweep(
        model,
        "x",
        args.eta
    )

    print("Running Y-direction sweep...")

    y_rows, y_snapshots, y_failures = run_sweep(
        model,
        "y",
        args.eta
    )

    save_csv(output / "x_sweep.csv", x_rows)
    save_csv(output / "y_sweep.csv", y_rows)

    # Stage B boundary experiment.
    print("\nRunning boundary sensitivity experiment...")

    boundary_results = boundary_experiment(
        model,
        grain_map
    )

    save_csv(
        output / "boundary_sensitivity.csv",
        boundary_results
    )

    # Grain map.
    plt.figure(figsize=(7, 6))
    plt.imshow(
        grain_map,
        origin="lower",
        interpolation="nearest"
    )
    plt.title("Stage B: Voronoi grain structure")
    plt.xlabel("X cell")
    plt.ylabel("Y cell")
    plt.colorbar(label="Grain ID")
    plt.tight_layout()
    plt.savefig(output / "grain_map.png", dpi=160)
    plt.close()

    # Magnetization loops.
    plt.figure(figsize=(8, 6))

    plt.plot(
        [r["field"] for r in x_rows],
        [r["Mx"] for r in x_rows],
        "o-",
        label="X sweep: Mx"
    )

    plt.plot(
        [r["field"] for r in y_rows],
        [r["My"] for r in y_rows],
        "s-",
        label="Y sweep: My"
    )

    plt.xlabel("Applied field (dimensionless)")
    plt.ylabel("Normalized magnetization")
    plt.title("Stage B: directional magnetization")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "magnetization_loops.png", dpi=160)
    plt.close()

    # Energy decomposition.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, rows, name in [
        (axes[0], x_rows, "X direction"),
        (axes[1], y_rows, "Y direction")
    ]:
        fields = [r["field"] for r in rows]

        for key, label in [
            ("E_exchange", "Exchange"),
            ("E_anisotropy", "Anisotropy"),
            ("E_zeeman", "Zeeman")
        ]:
            ax.plot(
                fields,
                [r[key] for r in rows],
                label=label
            )

        ax.set_title(name)
        ax.set_xlabel("Applied field")
        ax.set_ylabel("Dimensionless energy")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output / "energy_decomposition.png", dpi=160)
    plt.close()

    # Boundary sensitivity plot.
    plt.figure(figsize=(7, 5))

    plt.plot(
        [r["eta_boundary"] for r in boundary_results],
        [r["Mx"] for r in boundary_results],
        "o-"
    )

    plt.xlabel("Boundary exchange coefficient eta")
    plt.ylabel("Mx")
    plt.title("Grain-boundary sensitivity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "boundary_sensitivity.png", dpi=160)
    plt.close()

    # Save spatial magnetization maps.
    for axis, snapshots in [
        ("x", x_snapshots),
        ("y", y_snapshots)
    ]:
        for step, theta in snapshots.items():
            angle = theta.reshape(NY, NX)

            plt.figure(figsize=(6, 5))
            plt.imshow(
                angle,
                origin="lower",
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi
            )

            plt.title(f"{axis.upper()} sweep, step {step}")
            plt.xlabel("X cell")
            plt.ylabel("Y cell")
            plt.colorbar(label="Magnetization angle (rad)")
            plt.tight_layout()

            plt.savefig(
                output / f"magnetization_{axis}_{step:02d}.png",
                dpi=140
            )
            plt.close()

    # Stage C rotating-field experiment.
    if not args.skip_stage_c:
        print("\nRunning Stage C rotating-field experiment...")

        theta_initial = np.zeros(model.n)

        rotating_field_experiment(
            model=model,
            grain_map=grain_map,
            theta_initial=theta_initial,
            etas=STAGE_C_ETAS,
            H_ROT=STAGE_C_FIELD,
            n_steps=STAGE_C_STEPS,
            output_dir=output / "stageC_results"
        )

    # Run summary.
    summary = {
        "grid": [NY, NX],
        "realized_grains": int(len(np.unique(grain_map))),
        "boundary_edge_fraction": float(np.mean(model.boundary)),
        "gradient_max_error": gradient_error,
        "x_optimizer_nonconverged_steps": len(x_failures),
        "y_optimizer_nonconverged_steps": len(y_failures),
        "x_failures": x_failures,
        "y_failures": y_failures,
        "parameters": {
            "K_anisotropy": K_ANIS,
            "J_exchange": J_EXCHANGE,
            "eta_boundary": args.eta,
            "H_max": H_MAX,
            "stage_c_field": STAGE_C_FIELD,
            "stage_c_steps": STAGE_C_STEPS,
            "stage_c_etas": list(STAGE_C_ETAS)
        },
        "stage_c_enabled": not args.skip_stage_c,
        "warning": (
            "Dimensionless exploratory model. "
            "Not calibrated to electrical steel. "
            "No nonlocal magnetostatic energy or dynamic LLG."
        )
    }

    with open(output / "run_summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    np.save(output / "grain_map.npy", grain_map)
    np.save(output / "grain_angles.npy", grain_angles)

    print("\n===================================")
    print("STAGE B + STAGE C SIMULATION FINISHED")
    print("===================================")
    print("Results folder:", output.resolve())
    print("Gradient error:", gradient_error)
    print("X optimizer warnings:", len(x_failures))
    print("Y optimizer warnings:", len(y_failures))


if __name__ == "__main__":
    main()
