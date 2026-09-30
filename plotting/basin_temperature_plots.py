import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# Settings
DATA_DIR = Path(__file__).resolve().parents[1] / "experiments" / "continuous_optimization"
DELTA_F = 0.2
TEMPERATURES = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
EPOCHS = 2000
# Right basin is retained if model p_right >= RETAIN_FRACTION * target p_right (fixed before results)
RETAIN_FRACTION = 0.5

SUMMARY_PATH = DATA_DIR / f"basin2d_delta{DELTA_F}_temperature_summary.csv"
FIGURE_PATH = DATA_DIR / f"basin2d_delta{DELTA_F}_temperature_p_right.png"

REALNVP_COLOR = "#eb6834"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"


# Exact Boltzmann target on the 50x50 grid (same as basin_2d_trace.py)
def boltzmann_target(temperature):
    step = 2.0 / 49
    grid = -1.0 + torch.arange(50, dtype=torch.float64) * step
    width = torch.full((50,), step, dtype=torch.float64)
    width[0] = width[-1] = step / 2

    X, Y = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    cell = (width[:, None] * width[None, :]).flatten()

    q1 = (points[:, 0] + 0.5) ** 2 + (points[:, 1] + 0.5) ** 2
    q2 = (points[:, 0] - 0.5) ** 2 + (points[:, 1] + 0.5) ** 2 + DELTA_F
    objective = torch.minimum(q1, q2)
    right = q2 < q1

    mass = torch.softmax(torch.log(cell) - objective / temperature, dim=0)

    return mass[right].sum().item(), (mass * objective).sum().item()


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


# Per-seed p_right traces for completed seeds only (a seed is complete when epoch EPOCHS is logged)
def load_traces(temperature):
    path = DATA_DIR / f"basin2d_delta{DELTA_F}_T{temperature}_trace.csv"
    if not path.exists():
        return {}

    traces = {}
    for row in read_csv(path):
        # A running job can leave a half-written last line
        if any(value in (None, "") for value in row.values()):
            continue
        traces.setdefault(int(row["seed"]), []).append(
            (int(row["epoch"]), float(row["p_right"]), float(row["mean_f"]))
        )

    return {
        seed: np.array(rows)
        for seed, rows in traces.items()
        if rows[-1][0] == EPOCHS
    }


# Final values: 100k fresh samples if the run finished, else the fixed eval batch at the last epoch
def load_final(temperature, traces):
    path = DATA_DIR / f"basin2d_delta{DELTA_F}_T{temperature}_final.csv"
    if path.exists():
        rows = read_csv(path)
        return {int(r["seed"]): (float(r["p_right"]), float(r["mean_f"])) for r in rows}, "final_100k"

    return {seed: (trace[-1, 1], trace[-1, 2]) for seed, trace in traces.items()}, "trace_last_epoch"


# Summary table
summary = []
all_traces = {}

for temperature in TEMPERATURES:
    target_p_right, target_mean_f = boltzmann_target(temperature)
    traces = load_traces(temperature)
    all_traces[temperature] = traces

    if not traces:
        continue

    final, source = load_final(temperature, traces)
    seeds = sorted(final)
    p_right = np.array([final[s][0] for s in seeds])
    mean_f = np.array([final[s][1] for s in seeds])
    min_p_right = np.array([traces[s][:, 1].min() for s in seeds])
    min_epoch = np.array([traces[s][traces[s][:, 1].argmin(), 0] for s in seeds])

    summary.append({
        "temperature": temperature,
        "seeds": len(seeds),
        "source": source,
        "target_p_right": target_p_right,
        "median_p_right": np.median(p_right),
        "min_p_right": p_right.min(),
        "max_p_right": p_right.max(),
        "median_ratio": np.median(p_right / target_p_right),
        "retained": int((p_right >= RETAIN_FRACTION * target_p_right).sum()),
        "median_training_min_ratio": np.median(min_p_right / target_p_right),
        "median_training_min_epoch": np.median(min_epoch),
        "target_mean_f": target_mean_f,
        "mean_mean_f": mean_f.mean(),
    })

if summary:
    with open(SUMMARY_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

print(f"Retained: model p_right >= {RETAIN_FRACTION} * target p_right")
print("T     | seeds | target p_R | median p_R (min-max)      | ratio  | retained | training min ratio (epoch) | mean f / target")

for row in summary:
    print(
        f"{row['temperature']:<5} | {row['seeds']:5d} | {row['target_p_right']:.4f}     | "
        f"{row['median_p_right']:.4f} ({row['min_p_right']:.4f}-{row['max_p_right']:.4f}) | "
        f"{row['median_ratio']:.3f}  | {row['retained']:2d}/{row['seeds']:<2d}    | "
        f"{row['median_training_min_ratio']:.3f} ({row['median_training_min_epoch']:.0f})"
        f"{'':13s}| {row['mean_mean_f']:.4f} / {row['target_mean_f']:.4f}"
    )

missing = [t for t in TEMPERATURES if not all_traces[t]]
if missing:
    print(f"\nNo completed seeds yet for T = {missing}")


# Figure: p_right over training, one panel per T
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
fig.patch.set_facecolor(SURFACE)

for ax, temperature in zip(axes.flat, TEMPERATURES):
    target_p_right, _ = boltzmann_target(temperature)
    traces = all_traces[temperature]
    ax.set_facecolor(SURFACE)

    if traces:
        stacked = np.stack([trace[:, 1] for trace in traces.values()])
        epochs = next(iter(traces.values()))[:, 0]
        low, median, high = np.percentile(stacked, [25, 50, 75], axis=0)

        ax.fill_between(epochs, low, high, color=REALNVP_COLOR, alpha=0.2, linewidth=0)
        ax.plot(epochs, median, color=REALNVP_COLOR, linewidth=2, label="RealNVP")

    ax.axhline(target_p_right, color=TEXT_MUTED, linestyle="--", linewidth=1.2, label="Boltzmann target")

    # y range around the target so small targets stay visible (epoch 0 starts near 0.5)
    ax.set_ylim(0, min(0.6, 3 * target_p_right))
    ax.set_xlim(0, EPOCHS)
    ax.set_title(f"T = {temperature}  ({len(traces)} seed{'' if len(traces) == 1 else 's'})", color=TEXT, loc="left", fontsize=11)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelcolor=TEXT, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)

for ax in axes[1]:
    ax.set_xlabel("Epoch", color=TEXT, fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel("Mass in right (local) basin", color=TEXT, fontsize=10)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=10)
fig.suptitle(
    f"Right-basin mass vs Boltzmann target, Δf = {DELTA_F} (median, 25–75 % over seeds)",
    color=TEXT, x=0.01, ha="left", fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIGURE_PATH, dpi=200, facecolor=SURFACE)
fig.savefig(FIGURE_PATH.with_suffix(".pdf"), facecolor=SURFACE)

print(f"\nSaved {SUMMARY_PATH.name} and {FIGURE_PATH.name} (+ .pdf) in {DATA_DIR}")
