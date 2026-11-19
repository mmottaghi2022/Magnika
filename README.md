# MAGNIKA

**A materials-aware electromagnetic design platform for PM-free synchronous reluctance motors.**

MAGNIKA is an exploratory numerical research project that asks a deliberately unusual
question: *what if the magnetic material itself were an optimization variable, rather
than a fixed input to the electromagnetic design?*

Conventional motor design starts from an established electrical-steel datasheet, then
optimizes topology. MAGNIKA works the other way around — it starts from
**composition → microstructure → magnetic behaviour**, and aims to carry that chain
all the way to rotor/stator geometry and inverter control, ultimately targeting a
competitive permanent-magnet-free synchronous reluctance machine (SynRM).

> **Status:** early-stage numerical research. The current model is a *dimensionless,
> two-dimensional, phenomenological* spin-lattice model. It is **not** a calibrated
> electrical-steel model, **not** an LLG solver, and **not** a finite-element motor
> solver. No engineering predictions about real machines should be drawn from it yet.

---

## The research chain

```
Composition → Microstructure → Magnetic Properties → Electromagnetic FEM
            → Motor Topology → VFD → Optimized Machine
```

The project charter is organised around a "digital foundry" analogy, in which each
stage supplies *validated* information to the next. The repository currently covers
the **microstructure → magnetic behaviour** link only.

## The model

A 2-D in-plane lattice of magnetic orientations (an XY spin model):

| Element | Description |
|---|---|
| **Lattice** | 40 × 40 sites (1600 cells) |
| **Grain structure** | Voronoi grains; grain-dependent anisotropy axes |
| **Exchange** | Nearest-neighbour coupling, `J = 1.0` |
| **Grain boundaries** | Reduced exchange across boundaries, scaled by `η ∈ [0, 1]` |
| **Anisotropy** | In-plane projection of cubic anisotropy, `K = 0.12` |
| **Zeeman** | Applied in-plane field `H` |
| **Equilibrium** | Analytic energy gradient + numerical minimization |
| **Dynamics** (Stage H+) | Overdamped angular relaxation `dθ/dt = -(1/τ) ∂E/∂θ`, Euler and RK4 |

Diagnostics distinguish three things that a single macroscopic magnetization curve
hides: **grain-to-grain dispersion**, **boundary angular mismatch**, and
**intragranular dispersion**.

The full mathematics — energy functional, analytic gradient, equilibrium solver and
diagnostic definitions — is in
[`documentation/Part 2: Scientific and Mathematical Specification`](documentation/Part%202:%20Scientific%20and%20Mathematical%20Specification).

## Stages completed

| Stage | Study | Script | Result directory |
|---|---|---|---|
| **B** | Numerical foundation: grain map, energy decomposition, quasi-static X/Y sweeps, analytic-gradient validation | `stageb.py` | `stageB_results/` |
| **C** | Quasi-static rotating-field experiment (72 angular steps) across `η` | `stageb.py` | `stageB_results/stageC_results/` |
| **D** | Boundary-coupling sweep, `η = 0.01 … 1.00` (9 values) | `staged.py` | `stageD_results/` |
| **E** | Statistical reproducibility over 5 independent grain seeds × 9 `η` values | `stagee.py` | `stageE_results/` |
| **F** | Field-strength × boundary-coupling interaction (5 seeds × 3 fields × 9 `η`) | `stageF.py` | `stageF_results/` |
| **G** | Grain size and microstructure dependence (design in `part3`) | — | `stageG_results/` |
| **H** | Synthetic three-phase PWM excitation, B\*–H response, phase lag | `stageH.py`, `stageH2.py` | `stageH_results/`, `stageH2_results/` |
| **H2.1** | Fundamental M–H phase relationship (replaces the invalid `B* = H + M` diagnostic) | `stageH21.py` | `stageH21_results/` |
| **H2.3** | Integrator validation: explicit Euler vs classical RK4, timestep study | `stageH23.py` | `stageH23_results/` |
| **H2.5** | Dynamic energy balance and full-state convergence audit | `stageH25.py` | `stageH25_results/` |
| **H2.6** | Grain-level dynamics, relaxation-time sensitivity, microstate periodicity | `stageH26.py` | `stageH26_C_results/` |
| **I** | Controlled grain-population screening (16/25/50/100 grains × 3 seeds) | `stageI.py` | `stageI_results/` |

### Headline findings (Stages B–F)

- Increasing grain-boundary coupling `η` **reduces** grain-to-grain angular dispersion
  (≈5.24° → ≈2.72°, a ≈48% reduction), and this held across all five seeds (≈47% mean).
- Increasing applied field strength also reduces dispersion; the *relative* sensitivity
  to `η` shrinks at higher field.
- Boundary mismatch **falls** while **intragranular** dispersion **rises** with stronger
  coupling — angular distortion is *redistributed* between boundaries and grain
  interiors, not simply eliminated.
- Macroscopic magnetization stays close to unity throughout, which is precisely why
  macroscopic curves alone are insufficient to describe the internal magnetic state.
- Numerical health: zero failed gradient checks (max gradient ∞-norm ≈1.96 × 10⁻⁶),
  maximum cycle-closure error ≈4.12 × 10⁻⁸.

## Repository layout

```
.
├── stageb.py                 # Core model: geometry, energy, gradient, solver (Stages B + C)
├── staged.py                 # Stage D — boundary-coupling sweep
├── stagee.py                 # Stage E — multi-seed reproducibility
├── stageF.py                 # Stage F — field × coupling response
├── stageH.py                 # Stage H  — synthetic PWM excitation
├── stageH2.py                # Stage H2 — B*–H response, phase lag
├── stageH21.py               # Stage H2.1 — fundamental M–H phase
├── stageH23.py               # Stage H2.3 — Euler vs RK4, timestep validation
├── stageH25.py               # Stage H2.5 — energy balance, state convergence
├── stageH26.py               # Stage H2.6 — grain-level dynamics, periodicity
├── stageI.py                 # Stage I — grain-population screening
├── plot_stagec.py            # Stage C visual report
├── analyze_periodicity.py    # Periodicity analysis over H2.4 traces
├── summerize_stageE.py       # Post-processing for Stage E outputs
├── documentation/
│   ├── master_project_charter
│   ├── Part 2: Scientific and Mathematical Specification
│   └── part3: stage3 experimental design
└── LICENSE
```

All stage scripts are standalone and expect to be run **from the repository root**,
where `stageb.py` acts as a shared importable module.

> **Not tracked:** every `*_results/` directory. Run outputs total ~550 MB and are
> fully reproducible from the scripts and fixed seeds (see `.gitignore`).

## Requirements

Python 3.12 (tested with 3.12.6) and:

```
numpy>=2.0
pandas>=2.0
matplotlib>=3.8
scipy>=1.11
```

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib scipy

# Stage B + C (writes stageB_results/)
python3 stageb.py

# Stage B only
python3 stageb.py --skip-stage-c --seed 17 --eta 0.20

# Subsequent stages (each writes its own *_results/ directory)
python3 staged.py
python3 stagee.py
python3 stageF.py
python3 stageI.py --grain-counts 16 25 50 100 --seeds 17 42 73 --eta 0.20
```

Baseline parameters are fixed in `stageb.py`: `SEED = 17`, `NX = NY = 40`,
`N_GRAINS = 100`, `K_ANIS = 0.12`, `J_EXCHANGE = 1.0`, `ETA_BOUNDARY = 0.20`.
Every run writes a `*_metadata.json` recording the exact configuration used, and the
scripts cache nothing between runs, so a fresh clone reproduces the published figures.

## Reproducibility principles

1. Every result carries the metadata needed to regenerate it (seed, grid, parameters).
2. Analytic gradients are verified against finite differences before any optimization.
3. Convergence is demonstrated by timestep and state-comparison studies, not assumed.
4. Reported quantities are stated as *model diagnostics*, never as physical material
   properties, until calibration against measurement exists.

## Known limitations

- 2-D in-plane angles only — no 3-D domain structure.
- Dimensionless field, energy and magnetization; no SI calibration.
- No nonlocal magnetostatic / demagnetizing energy, no thermal activation.
- Overdamped phenomenological relaxation — **not** LLG; no eddy currents.
- No magnetoelastic coupling, no physical hysteresis-loss or thermal model.
- Changing grain count changes both characteristic grain size *and* boundary density.
- `τ`, ripple and PWM parameters are synthetic and uncalibrated to any real ferrite.

## Roadmap

- **Phase I** — microstructure and magnetic physics: grain size, texture, homogenization.
- **Phase II** — metallurgical composition, processing and experimental validation.
- **Phase III** — FEM integration (e.g. ANSYS Maxwell) and rotor/stator topology optimization.
- **Phase IV** — motor + VFD co-design (MTPA, flux weakening, switching effects).
- **Phase V** — prototype and industrialization.

See the [project charter](documentation/master_project_charter) for the full roadmap.

## License

MIT — see [LICENSE](LICENSE).
