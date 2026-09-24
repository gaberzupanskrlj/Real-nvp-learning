from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results/benchmark_100seeds")
OUT = ROOT / "figures"

MAX_EVALUATIONS = 4_096_000
GRID_POINTS = 1200

PROBLEMS = {
    "onemax": "OneMax",
    "leading_ones": "LeadingOnes",
    "concatenated_trap": "ConcatenatedTrap",
    "nk_landscapes": "NKLandscapes",
    "ising_torus": "IsingTorus",
}

ALGORITHMS = {
    "realnvp": "RealNVP",
    "one_plus_one_ea": "(1+1)-EA",
}


def load_seed_curves(folder):
    curves = []

    for seed_dir in sorted(folder.glob("seed_*")):
        path = seed_dir / "convergence.csv"

        if not path.exists():
            continue

        df = pd.read_csv(path)

        if not {"evaluations", "best_so_far"}.issubset(df.columns):
            raise ValueError(f"Missing required columns in {path}")

        df = (
            df[["evaluations", "best_so_far"]]
            .dropna()
            .sort_values("evaluations")
            .drop_duplicates("evaluations", keep="last")
        )

        if len(df):
            curves.append(df)

    return curves


def step_values(df, grid):
    x = df["evaluations"].to_numpy(dtype=float)
    y = df["best_so_far"].to_numpy(dtype=float)

    idx = np.searchsorted(x, grid, side="right") - 1
    values = np.full(grid.shape, np.nan, dtype=float)

    valid = idx >= 0
    values[valid] = y[idx[valid]]

    if len(y):
        values[grid > x[-1]] = y[-1]

    return values


def common_grid(curve_groups):
    first_points = []

    for curves in curve_groups:
        for df in curves:
            first_points.append(float(df["evaluations"].iloc[0]))

    if not first_points:
        raise RuntimeError("No convergence curves found.")

    start = max(first_points)

    return np.linspace(
        start,
        MAX_EVALUATIONS,
        GRID_POINTS,
    )


def aggregate(curves, grid):
    matrix = np.vstack(
        [step_values(df, grid) for df in curves]
    )

    mean = np.nanmean(matrix, axis=0)

    if matrix.shape[0] > 1:
        std = np.nanstd(matrix, axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)

    return mean, std, matrix.shape[0]


def plot_problem(problem_key, problem_name):
    folders = {
        key: ROOT / key / problem_key
        for key in ALGORITHMS
    }

    curves = {
        key: load_seed_curves(folder)
        for key, folder in folders.items()
    }

    missing = [
        ALGORITHMS[key]
        for key, value in curves.items()
        if not value
    ]

    if missing:
        print(
            f"Skipping {problem_name}: missing "
            + ", ".join(missing)
        )
        return

    grid = common_grid(list(curves.values()))

    stats = {
        key: aggregate(value, grid)
        for key, value in curves.items()
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for key, label in ALGORITHMS.items():
        mean, std, n = stats[key]

        line, = ax.plot(
            grid,
            mean,
            linewidth=2.2,
            label=f"{label} mean (n={n})",
        )

        ax.fill_between(
            grid,
            mean - std,
            mean + std,
            alpha=0.18,
            color=line.get_color(),
            linewidth=0,
            label=f"{label} ±1 std",
        )

    ax.set_title(f"{problem_name}: convergence comparison")
    ax.set_xlabel("Objective function evaluations")
    ax.set_ylabel("Best objective found so far")

    ax.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(6, 6),
    )
    if problem_key == "onemax":
        ax.set_xlim(0, 200_000)
    if problem_key == "concatenated_trap":
        ax.set_ylim(10, 20.5)
        ax.axhline(
            20,
            linestyle="--",
            linewidth=1.5,
            label="Global optimum = 20",
        )
    if problem_key == "ising_torus":
        ax.axhline(
        200,
        linestyle="--",
        linewidth=1.5,
        label="Global optimum = 200",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)

    png_path = OUT / f"{problem_key}_convergence.png"
    svg_path = OUT / f"{problem_key}_convergence.svg"

    fig.savefig(
        png_path,
        dpi=250,
        bbox_inches="tight",
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {svg_path}")


def main():
    for key, name in PROBLEMS.items():
        plot_problem(key, name)


if __name__ == "__main__":
    main()
