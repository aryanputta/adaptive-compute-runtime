#!/usr/bin/env python3
"""
Offline result analysis for the adaptive compute runtime benchmark.
Reads benchmark_results.csv and produces:
  - per-path timing breakdown bar chart
  - transfer overhead pie charts per workload
  - wall time comparison across paths
"""

import sys
import csv
import pathlib
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

CSV_PATH = pathlib.Path("benchmark_results.csv")


def load_csv(path: pathlib.Path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "workload_id":           row["workload_id"],
                "path":                  row["path"],
                "wall_ms":               float(row["wall_ms"]),
                "kernel_ms":             float(row["kernel_ms"]),
                "h2d_ms":                float(row["h2d_ms"]),
                "d2h_ms":                float(row["d2h_ms"]),
                "memory_throughput_GBs": float(row["memory_throughput_GBs"]),
                "correct":               int(row["correct"]),
            })
    return rows


def print_summary(rows):
    paths = defaultdict(list)
    for r in rows:
        paths[r["path"]].append(r["wall_ms"])

    print(f"\n{'Path':<20} {'Count':>6} {'Avg Wall (ms)':>14} {'Min (ms)':>10} {'Max (ms)':>10}")
    print("-" * 64)
    for path, times in sorted(paths.items()):
        print(f"{path:<20} {len(times):>6} {sum(times)/len(times):>14.3f} "
              f"{min(times):>10.3f} {max(times):>10.3f}")


def plot(rows):
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed — skipping plots. pip install matplotlib numpy")
        return

    paths = sorted(set(r["path"] for r in rows))
    avg_wall   = {p: [] for p in paths}
    avg_kernel = {p: [] for p in paths}
    avg_h2d    = {p: [] for p in paths}
    avg_d2h    = {p: [] for p in paths}

    for r in rows:
        p = r["path"]
        avg_wall[p].append(r["wall_ms"])
        avg_kernel[p].append(r["kernel_ms"])
        avg_h2d[p].append(r["h2d_ms"])
        avg_d2h[p].append(r["d2h_ms"])

    x    = np.arange(len(paths))
    wall = [np.mean(avg_wall[p])   for p in paths]
    kern = [np.mean(avg_kernel[p]) for p in paths]
    h2d  = [np.mean(avg_h2d[p])   for p in paths]
    d2h  = [np.mean(avg_d2h[p])   for p in paths]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Adaptive Compute Runtime — Benchmark Results", fontsize=13)

    # Left: stacked bar (kernel + transfer)
    ax = axes[0]
    ax.bar(x, kern, label="Kernel",    color="#4c78a8")
    ax.bar(x, h2d,  bottom=kern,       label="H2D Transfer", color="#f28e2b")
    d2h_bottom = [kern[i] + h2d[i] for i in range(len(paths))]
    ax.bar(x, d2h,  bottom=d2h_bottom, label="D2H Transfer", color="#e15759")
    ax.set_xticks(x)
    ax.set_xticklabels(paths, rotation=20, ha="right")
    ax.set_ylabel("Average time (ms)")
    ax.set_title("Time Breakdown per Execution Path")
    ax.legend()

    # Right: wall time comparison
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))
    ax2.bar(x, wall, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(paths, rotation=20, ha="right")
    ax2.set_ylabel("Average wall time (ms)")
    ax2.set_title("End-to-End Wall Time per Path")

    plt.tight_layout()
    out = pathlib.Path("benchmark_plot.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}. Run the benchmark first.")
        sys.exit(1)

    rows = load_csv(CSV_PATH)
    print(f"Loaded {len(rows)} result rows from {CSV_PATH}")
    print_summary(rows)
    plot(rows)
