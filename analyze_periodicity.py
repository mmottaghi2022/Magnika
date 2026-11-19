import csv
import numpy as np
from pathlib import Path

frequencies = [100, 200, 400]
samples_per_cycle = {
    100: 1600,
    200: 800,
    400: 400,
}

for f in frequencies:
    path = Path(
        f"stageH24_results/{f}Hz_rk4_spp40_trace.csv"
    )

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    mx = np.array([float(r["Mx"]) for r in rows])
    my = np.array([float(r["My"]) for r in rows])

    n = samples_per_cycle[f]

    print(f"\n{'='*65}")
    print(f"FREQUENCY: {f} Hz")
    print(f"Trace rows: {len(rows)}")
    print(f"Samples per cycle: {n}")
    print("Period | Last comparison RMS | Mean of last 5 comparisons")

    for period in [1, 2, 3, 4, 5, 10]:
        shift = period * n

        if len(mx) <= shift:
            continue

        dx = mx[shift:] - mx[:-shift]
        dy = my[shift:] - my[:-shift]

        # RMS of the two-component magnetization difference
        squared = dx**2 + dy**2

        # RMS for each complete electrical-cycle comparison
        cycle_values = []
        for start in range(0, len(squared) - n + 1, n):
            segment = squared[start:start+n]
            cycle_values.append(np.sqrt(np.mean(segment)))

        if not cycle_values:
            continue

        last = cycle_values[-1]
        recent = np.mean(cycle_values[-5:])

        print(
            f"{period:>6} | "
            f"{last:>20.8f} | "
            f"{recent:>25.8f}"
        )
