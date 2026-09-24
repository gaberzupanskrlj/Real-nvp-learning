from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "results" / "permutation_optimization" / "tsp20"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

REFERENCE_OPTIMUM = 3.513668846
METHODS = ["realnvp_baseline_v1", "cosine_elite", "annealed_kl"]
LABELS = {
    "realnvp_baseline_v1": "Baseline",
    "cosine_elite": "Cosine + elite",
    "annealed_kl": "Annealed KL",
}

def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

def plot_metric(summary_file, mean_key, std_key, ylabel, title, filename):
    rows = {row["method"]: row for row in read_rows(summary_file)}
    labels = [LABELS[m] for m in METHODS]
    means = [float(rows[m][mean_key]) for m in METHODS]
    stds = [float(rows[m][std_key]) for m in METHODS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(labels, means, yerr=stds, fmt="o", capsize=5)
    ax.axhline(REFERENCE_OPTIMUM, linestyle="--", label="Reference optimum")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=200)
    plt.close(fig)

plot_metric(
    RESULTS / "summary.csv",
    "mean_best",
    "std_best",
    "Best-ever tour length",
    "TSP-20 search quality across 10 seeds",
    "search_quality_10seeds.png",
)

plot_metric(
    RESULTS / "generator_summary.csv",
    "mean_test_mean",
    "std_test_mean",
    "Final test mean tour length",
    "TSP-20 final generator quality across 10 seeds",
    "final_generator_quality_10seeds.png",
)

print(f"Saved figures to {FIGURES}")
