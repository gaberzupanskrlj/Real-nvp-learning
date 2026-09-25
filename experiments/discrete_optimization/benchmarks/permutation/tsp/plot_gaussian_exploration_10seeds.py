import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Settings

RESULTS_CSV = "tsp_gaussian_exploration_10seeds.csv"

REFERENCE_OPTIMUM = 3.513668846

OUTPUT_DIR = Path(
    "tsp_gaussian_exploration_plots"
)

EXPLORATION_LEVELS = [
    0.0,
    0.10,
    0.25,
]


# Load results

def load_results(filename):
    rows = []

    with open(
        filename,
        "r",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            rows.append(
                {
                    "epsilon": float(
                        row[
                            "exploration_fraction_requested"
                        ]
                    ),
                    "seed": int(
                        row["seed"]
                    ),
                    "test_mean": float(
                        row["test_mean_length"]
                    ),
                    "test_best": float(
                        row["test_best_length"]
                    ),
                    "best_ever": float(
                        row["best_length_ever"]
                    ),
                    "unique_flow": float(
                        row["final_unique_flow"]
                    ),
                    "grad_norm": float(
                        row["final_grad_norm"]
                    ),
                    "target_hit": (
                        row["target_hit"]
                        .strip()
                        .lower()
                        == "true"
                    ),
                }
            )

    return rows


def values_for(
    rows,
    epsilon,
    key,
):
    return np.array(
        [
            row[key]
            for row in rows
            if np.isclose(
                row["epsilon"],
                epsilon,
            )
        ],
        dtype=float,
    )


def mean_std(values):
    return (
        np.mean(values),
        np.std(
            values,
            ddof=1,
        ),
    )


def setup_axis(
    ax,
    ylabel,
):
    ax.set_xticks(
        range(
            len(EXPLORATION_LEVELS)
        )
    )

    ax.set_xticklabels(
        [
            "0.00",
            "0.10",
            "0.25",
        ]
    )

    ax.set_xlabel(
        "Gaussian exploration fraction ε"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )


# Test mean

def plot_test_mean(rows):
    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    means = []
    stds = []

    for x, epsilon in enumerate(
        EXPLORATION_LEVELS
    ):
        values = values_for(
            rows,
            epsilon,
            "test_mean",
        )

        mean_value, std_value = (
            mean_std(values)
        )

        means.append(mean_value)
        stds.append(std_value)

        offsets = np.linspace(
            -0.12,
            0.12,
            len(values),
        )

        ax.scatter(
            x + offsets,
            values,
            alpha=0.65,
            s=35,
        )

    ax.errorbar(
        range(
            len(EXPLORATION_LEVELS)
        ),
        means,
        yerr=stds,
        fmt="o-",
        linewidth=2,
        markersize=7,
        capsize=5,
        label="Mean ± std",
    )

    ax.axhline(
        REFERENCE_OPTIMUM,
        linestyle="--",
        linewidth=1.5,
        label="Reference optimum",
    )

    setup_axis(
        ax,
        "Final test mean tour length",
    )

    ax.set_title(
        "TSP-20: final generator quality"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "test_mean_vs_exploration.png",
        dpi=250,
    )

    plt.close(fig)


# Test best

def plot_test_best(rows):
    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    means = []
    stds = []

    for x, epsilon in enumerate(
        EXPLORATION_LEVELS
    ):
        values = values_for(
            rows,
            epsilon,
            "test_best",
        )

        mean_value, std_value = (
            mean_std(values)
        )

        means.append(mean_value)
        stds.append(std_value)

        offsets = np.linspace(
            -0.12,
            0.12,
            len(values),
        )

        ax.scatter(
            x + offsets,
            values,
            alpha=0.65,
            s=35,
        )

    ax.errorbar(
        range(
            len(EXPLORATION_LEVELS)
        ),
        means,
        yerr=stds,
        fmt="o-",
        linewidth=2,
        markersize=7,
        capsize=5,
        label="Mean ± std",
    )

    ax.axhline(
        REFERENCE_OPTIMUM,
        linestyle="--",
        linewidth=1.5,
        label="Reference optimum",
    )

    setup_axis(
        ax,
        "Best tour in final test",
    )

    ax.set_title(
        "TSP-20: final best solution"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "test_best_vs_exploration.png",
        dpi=250,
    )

    plt.close(fig)


# Flow diversity

def plot_flow_diversity(rows):
    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    means = []
    stds = []

    for x, epsilon in enumerate(
        EXPLORATION_LEVELS
    ):
        values = values_for(
            rows,
            epsilon,
            "unique_flow",
        )

        mean_value, std_value = (
            mean_std(values)
        )

        means.append(mean_value)
        stds.append(std_value)

        offsets = np.linspace(
            -0.12,
            0.12,
            len(values),
        )

        ax.scatter(
            x + offsets,
            values,
            alpha=0.65,
            s=35,
        )

    ax.errorbar(
        range(
            len(EXPLORATION_LEVELS)
        ),
        means,
        yerr=stds,
        fmt="o-",
        linewidth=2,
        markersize=7,
        capsize=5,
        label="Mean ± std",
    )

    setup_axis(
        ax,
        "Final flow unique fraction",
    )

    ax.set_ylim(
        bottom=0.0
    )

    ax.set_title(
        "TSP-20: mode collapse vs exploration"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "flow_diversity_vs_exploration.png",
        dpi=250,
    )

    plt.close(fig)


# Quality-diversity trade-off

def plot_quality_diversity(rows):
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for epsilon in EXPLORATION_LEVELS:
        subset = [
            row
            for row in rows
            if np.isclose(
                row["epsilon"],
                epsilon,
            )
        ]

        x = np.array(
            [
                row["unique_flow"]
                for row in subset
            ]
        )

        y = np.array(
            [
                row["test_mean"]
                for row in subset
            ]
        )

        ax.scatter(
            x,
            y,
            s=55,
            alpha=0.7,
            label=f"ε = {epsilon:.2f}",
        )

        ax.scatter(
            np.mean(x),
            np.mean(y),
            marker="X",
            s=140,
        )

    ax.axhline(
        REFERENCE_OPTIMUM,
        linestyle="--",
        linewidth=1.5,
        label="Reference optimum",
    )

    ax.set_xlabel(
        "Final flow unique fraction"
    )

    ax.set_ylabel(
        "Final test mean tour length"
    )

    ax.set_title(
        "TSP-20: quality-diversity trade-off"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "quality_diversity_tradeoff.png",
        dpi=250,
    )

    plt.close(fig)


# Target-hit rate

def plot_target_hits(rows):
    hit_rates = []

    for epsilon in EXPLORATION_LEVELS:
        subset = [
            row
            for row in rows
            if np.isclose(
                row["epsilon"],
                epsilon,
            )
        ]

        hits = sum(
            row["target_hit"]
            for row in subset
        )

        hit_rates.append(
            100.0
            * hits
            / len(subset)
        )

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    bars = ax.bar(
        [
            "0.00",
            "0.10",
            "0.25",
        ],
        hit_rates,
    )

    ax.set_xlabel(
        "Gaussian exploration fraction ε"
    )

    ax.set_ylabel(
        "Runs finding reference optimum (%)"
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_title(
        "TSP-20: optimum discovery rate"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        hit_rates,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "target_hit_rate.png",
        dpi=250,
    )

    plt.close(fig)


# Console summary

def print_summary(rows):
    print()
    print(
        "Gaussian exploration summary"
    )
    print()

    for epsilon in EXPLORATION_LEVELS:
        subset = [
            row
            for row in rows
            if np.isclose(
                row["epsilon"],
                epsilon,
            )
        ]

        test_mean = np.array(
            [
                row["test_mean"]
                for row in subset
            ]
        )

        test_best = np.array(
            [
                row["test_best"]
                for row in subset
            ]
        )

        unique_flow = np.array(
            [
                row["unique_flow"]
                for row in subset
            ]
        )

        hits = sum(
            row["target_hit"]
            for row in subset
        )

        print(
            f"ε={epsilon:.2f}"
        )

        print(
            f"  test mean:   "
            f"{test_mean.mean():.6f} "
            f"± "
            f"{test_mean.std(ddof=1):.6f}"
        )

        print(
            f"  test best:   "
            f"{test_best.mean():.6f} "
            f"± "
            f"{test_best.std(ddof=1):.6f}"
        )

        print(
            f"  unique flow: "
            f"{unique_flow.mean():.4f} "
            f"± "
            f"{unique_flow.std(ddof=1):.4f}"
        )

        print(
            f"  optimum hit: "
            f"{hits}/{len(subset)}"
        )

        print()


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_results(
        RESULTS_CSV
    )

    expected_runs = (
        len(EXPLORATION_LEVELS)
        * 10
    )

    if len(rows) != expected_runs:
        print(
            f"Warning: found {len(rows)} runs, "
            f"expected {expected_runs}."
        )

    plot_test_mean(rows)
    plot_test_best(rows)
    plot_flow_diversity(rows)
    plot_quality_diversity(rows)
    plot_target_hits(rows)

    print_summary(rows)

    print(
        f"Plots saved to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()