#!/usr/bin/env python3
"""
Clean convergence plots for RealNVP + KL vs (1+1)-EA.

Creates:
  - TSP-20
  - QAP Nug20
  - PFSP Ta001

Plot design:
  x-axis: objective evaluations (log scale)
  y-axis: actual best-so-far objective value
  thick line: median over seeds
  shaded band: 25th-75th percentile
  dashed line: optimum / best-known reference

Run from the repository root:

    python plotting/clean_anytime_plots.py

Optional custom data directory:

    python plotting/clean_anytime_plots.py \
        --data-dir results/permutation_optimization/anytime
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------

PROBLEMS = [
    {
        "name": "TSP-20",
        "slug": "tsp20",
        "title": "TSP-20: convergence",
        "ylabel": "Best-so-far tour length",
        "reference": 3.513668846,
        "reference_label": "Global optimum",
        "ea_trace": "tsp20_one_plus_one_ea_trace.csv",
        "realnvp_trace": "tsp20_realnvp_kl_trace.csv",
        "output": "tsp20_clean_convergence.png",
    },
    {
        "name": "QAP Nug20",
        "slug": "qap20",
        "title": "QAP Nug20: convergence",
        "ylabel": "Best-so-far cost",
        "reference": 2570.0,
        "reference_label": "Global optimum",
        "ea_trace": "qap20_one_plus_one_ea_trace.csv",
        "realnvp_trace": "qap20_realnvp_kl_trace.csv",
        "output": "qap_nug20_clean_convergence.png",
    },
    {
        "name": "PFSP Ta001",
        "slug": "pfsp_ta001",
        "title": "PFSP Ta001: convergence",
        "ylabel": "Best-so-far makespan",
        "reference": 1278.0,
        "reference_label": "Best-known",
        "ea_trace": "pfsp_ta001_one_plus_one_ea_trace.csv",
        "realnvp_trace": "pfsp_ta001_realnvp_baseline_kl_trace.csv",
        "output": "pfsp_ta001_clean_convergence.png",
    },
]


# ---------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------

def normalise(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(df, preferred_names, contains=None):
    """
    Find a column robustly, allowing slightly different CSV headers.
    """
    normalised = {normalise(c): c for c in df.columns}

    for name in preferred_names:
        key = normalise(name)
        if key in normalised:
            return normalised[key]

    if contains is not None:
        for key, column in normalised.items():
            if normalise(contains) in key:
                return column

    raise KeyError(f"none of {preferred_names} in columns {list(df.columns)}")


# Load a trace CSV into {seed: (evaluations, best_so_far)}

def load_trace(path):
    df = pd.read_csv(path)

    seed_column = find_column(df, ["seed"])
    evals_column = find_column(df, ["evaluations", "evals"], contains="eval")
    best_column = find_column(df, ["best_so_far", "best"], contains="bestso")

    runs = {}

    for seed, rows in df.groupby(seed_column):
        rows = rows.sort_values(evals_column, kind="stable")
        evals = rows[evals_column].to_numpy()
        best = np.minimum.accumulate(rows[best_column].to_numpy(dtype=float))
        runs[int(seed)] = (evals, best)

    return runs


# Best-so-far of one run at the given evaluation counts (NaN before its first logged point)

def best_at(run, grid):
    evals, best = run
    index = np.searchsorted(evals, grid, side="right") - 1
    values = best[np.clip(index, 0, None)]
    return np.where(index >= 0, values, np.nan)


# Median and 25th-75th percentile over seeds, on a log-spaced grid up to the run's own budget

def summarise(runs, grid):
    end = max(evals[-1] for evals, _ in runs.values())
    grid = np.append(grid[grid < end], end)

    values = np.array([best_at(run, grid) for run in runs.values()])

    # RealNVP logs after each full batch, so skip points before every seed has one
    logged = ~np.isnan(values).any(axis=0)
    grid = grid[logged]
    values = values[:, logged]

    low, median, high = np.percentile(values, [25, 50, 75], axis=0)
    return grid, low, median, high


# Plotting

METHODS = [
    ("(1+1)-EA", "ea_trace", "#2a78d6"),
    ("RealNVP + KL", "realnvp_trace", "#eb6834"),
]

TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"


def evals_label(x, _):
    if x >= 1e6:
        return f"{x / 1e6:g}M"
    if x >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def plot_problem(problem, data_dir):
    runs = {
        label: load_trace(data_dir / problem[key])
        for label, key, _ in METHODS
    }

    end = max(evals[-1] for method_runs in runs.values() for evals, _ in method_runs.values())

    # Both lines start at the first evaluation count where every run of every method has a value.
    # RealNVP logs after each full batch, so this is its first batch (1024 evaluations).
    start = max(evals[0] for method_runs in runs.values() for evals, _ in method_runs.values())

    # The axis starts at the power of ten just below (1k), so the ticks are round numbers;
    # on a log axis the gap between 1,000 and 1,024 is not visible.
    axis_start = 10 ** np.floor(np.log10(start))
    grid = np.unique(np.geomspace(axis_start, end, 1000).astype(int))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for label, _, color in METHODS:
        x, low, median, high = summarise(runs[label], grid)

        ax.fill_between(x, low, high, step="post", color=color, alpha=0.2, linewidth=0)
        ax.step(x, median, where="post", color=color, linewidth=2.5, label=label)

        # A run that stops before the others (RealNVP TSP-20 at 2.2M) ends in a dot
        if x[-1] < end:
            ax.plot(x[-1], median[-1], "o", color=color, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)

        print(f"    {label:<13} median at end ({x[-1]:,} evals): {median[-1]:g}")

    ax.axhline(problem["reference"], color=TEXT_MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate(
        f"{problem['reference_label']} {problem['reference']:g}",
        (axis_start, problem["reference"]),
        xytext=(4, -4),
        textcoords="offset points",
        color=TEXT_MUTED,
        fontsize=9,
        va="top",
    )

    # Room below the reference line for its label, which the EA line would cover above it
    bottom, top = ax.get_ylim()
    ax.set_ylim(min(bottom, problem["reference"] - 0.1 * (top - problem["reference"])), top)

    ax.set_xscale("log")
    # Ticks at powers of ten, plus the end of the budget
    ax.set_xlim(axis_start, end * 1.03)
    decades = [10 ** k for k in range(int(np.log10(axis_start)), int(np.log10(end)) + 1)]
    ax.set_xticks(decades + [end])
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M" if x == end else evals_label(x, _))
    )

    ax.set_title(problem["title"], color=TEXT, loc="left", fontsize=14, pad=12)
    ax.set_xlabel("Objective evaluations (log scale)", color=TEXT, fontsize=11)
    ax.set_ylabel(problem["ylabel"], color=TEXT, fontsize=11)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelcolor=TEXT, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    legend = ax.legend(loc="upper right", frameon=False, fontsize=11)
    for text in legend.get_texts():
        text.set_color(TEXT)

    fig.text(
        0.01, 0.01,
        "Line: median over 10 seeds. Band: 25th-75th percentile. "
        f"RealNVP's first point is at {start:,} evaluations, after its first batch.",
        color=TEXT_MUTED,
        fontsize=8.5,
        va="bottom",
    )
    fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.17)

    return fig


# The repository root is the first parent folder that contains results/

def repository_root():
    for folder in Path(__file__).resolve().parents:
        if (folder / "results" / "permutation_optimization").is_dir():
            return folder
    return Path.cwd()


def main():
    default_dir = repository_root() / "results" / "permutation_optimization" / "anytime"

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data-dir", type=Path, default=default_dir)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where to save the figures (default: the data directory)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for problem in PROBLEMS:
        print(problem["name"])

        fig = plot_problem(problem, args.data_dir)

        png_path = output_dir / problem["output"]
        fig.savefig(png_path, dpi=200, facecolor=SURFACE)
        fig.savefig(png_path.with_suffix(".pdf"), facecolor=SURFACE)
        plt.close(fig)

        print(f"    saved {png_path} (+ .pdf)")


if __name__ == "__main__":
    main()
