import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Settings
DATA_DIR = Path(__file__).resolve().parents[1] / "experiments" / "continuous_optimization"
DELTA_F = 0.2
TEMPERATURES = [0.2, 0.1, 0.05]
EPOCHS = 2000
MLE_EPOCHS = 5000
OCCUPIED_THRESHOLD = 0.01
# Common reverse KL range so the three temperatures are comparable
KL_YMAX = 0.06

SUMMARY_PATH = DATA_DIR / f"basin2d_mle_delta{DELTA_F}_summary.csv"
FIGURE_PATH = DATA_DIR / f"basin2d_mle_delta{DELTA_F}_p_right_kl.png"

# Categorical slots 1-3 (validated all-pairs); color follows the phase
SCRATCH_COLOR = "#eb6834"
MLE_COLOR = "#2a78d6"
WARM_COLOR = "#1baf7a"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

PHASES = [
    ("scratch", "REINFORCE from scratch", SCRATCH_COLOR),
    ("warm", "REINFORCE from MLE fit", WARM_COLOR),
]


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


# Per-seed traces; a seed counts only when its warm phase reached EPOCHS
def load_traces(temperature):
    path = DATA_DIR / f"basin2d_mle_delta{DELTA_F}_T{temperature}_trace.csv"
    if not path.exists():
        return {}

    traces = {}
    for row in read_csv(path):
        # A running job can leave a half-written last line
        if any(value in (None, "") for value in row.values()):
            continue
        key = (int(row["seed"]), row["phase"])
        traces.setdefault(key, []).append(
            (int(row["epoch"]), float(row["p_right"]), float(row["reverse_kl"]),
             float(row["kl_allocation"]), float(row["kl_shape"]), int(row["occupied_basins"]))
        )

    seeds = sorted(
        seed for seed, phase in traces
        if phase == "warm" and traces[(seed, phase)][-1][0] == EPOCHS
    )

    return {seed: {phase: np.array(traces[(seed, phase)]) for phase in ["scratch", "mle", "warm"]} for seed in seeds}


# Final values: 100k fresh samples if the run finished, else the fixed eval batch at the last epoch
def load_final(temperature, traces):
    path = DATA_DIR / f"basin2d_mle_delta{DELTA_F}_T{temperature}_final.csv"
    columns = ["p_right", "reverse_kl", "kl_allocation", "kl_shape", "occupied_basins"]

    if path.exists():
        rows = read_csv(path)
        final = {(int(r["seed"]), r["phase"]): [float(r[c]) for c in columns] for r in rows}
        return final, float(rows[0]["target_p_right"]), float(rows[0]["drop_right_kl"]), "final_100k"

    final = {
        (seed, phase): list(trace[-1, 1:])
        for seed, phases in traces.items()
        for phase, trace in phases.items()
    }
    return final, None, None, "trace_last_epoch"


# Exact target p_right and drop cost (same as build_target in basin_2d_mle_fit.py)
def target_values(temperature):
    step = 2.0 / 49
    grid = -1.0 + np.arange(50) * step
    low = np.clip(grid - step / 2, -1.0, None)
    width = np.clip(grid + step / 2, None, 1.0) - low

    X, Y = np.meshgrid(grid, grid, indexing="ij")
    cell = (width[:, None] * width[None, :]).flatten()
    q1 = (X.flatten() + 0.5) ** 2 + (Y.flatten() + 0.5) ** 2
    q2 = (X.flatten() - 0.5) ** 2 + (Y.flatten() + 0.5) ** 2 + DELTA_F
    objective = np.minimum(q1, q2)

    log_weight = np.log(cell) - objective / temperature
    mass = np.exp(log_weight - log_weight.max())
    mass /= mass.sum()
    p_right = mass[q2 < q1].sum()

    return p_right, -np.log(1.0 - p_right)


# Summary table
summary = []
all_traces = {}

for temperature in TEMPERATURES:
    traces = load_traces(temperature)
    all_traces[temperature] = traces

    if not traces:
        continue

    final, target_p_right, drop_kl, source = load_final(temperature, traces)
    if target_p_right is None:
        target_p_right, drop_kl = target_values(temperature)

    seeds = sorted(traces)
    kl = {}

    for phase in ["scratch", "mle", "warm"]:
        values = np.array([final[(seed, phase)] for seed in seeds])
        kl[phase] = values[:, 1]

        summary.append({
            "temperature": temperature,
            "phase": phase,
            "seeds": len(seeds),
            "source": source,
            "target_p_right": target_p_right,
            "drop_right_kl": drop_kl,
            "both_basins": int((values[:, 4] == 2).sum()),
            "median_p_right": np.median(values[:, 0]),
            "min_p_right": values[:, 0].min(),
            "max_p_right": values[:, 0].max(),
            "median_reverse_kl": np.median(values[:, 1]),
            "min_reverse_kl": values[:, 1].min(),
            "max_reverse_kl": values[:, 1].max(),
            "median_kl_allocation": np.median(values[:, 2]),
            "median_kl_shape": np.median(values[:, 3]),
            "below_drop_cost": int((values[:, 1] < drop_kl).sum()),
        })

    # Paired: is the MLE / warm solution better than scratch under the training loss?
    summary[-3]["mle_better_than_scratch"] = ""
    summary[-2]["mle_better_than_scratch"] = int((kl["mle"] < kl["scratch"]).sum())
    summary[-1]["mle_better_than_scratch"] = int((kl["warm"] < kl["scratch"]).sum())

if summary:
    with open(SUMMARY_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

print("T     | phase   | seeds | both | median p_R (min-max)      | median KL rev (min-max)   | alloc  | shape  | < drop | < scratch")

for row in summary:
    print(
        f"{row['temperature']:<5} | {row['phase']:7s} | {row['seeds']:5d} | {row['both_basins']:2d}   | "
        f"{row['median_p_right']:.4f} ({row['min_p_right']:.4f}-{row['max_p_right']:.4f}) | "
        f"{row['median_reverse_kl']:.4f} ({row['min_reverse_kl']:.4f}-{row['max_reverse_kl']:.4f}) | "
        f"{row['median_kl_allocation']:.4f} | {row['median_kl_shape']:.4f} | "
        f"{row['below_drop_cost']:2d}     | {row['mle_better_than_scratch']}"
    )
    if row["phase"] == "warm":
        print(
            f"      target p_R = {row['target_p_right']:.4f}, drop cost = {row['drop_right_kl']:.4f}, "
            f"source = {row['source']}"
        )

missing = [t for t in TEMPERATURES if not all_traces[t]]
if missing:
    print(f"\nNo completed seeds yet for T = {missing}")


# Figure: top row p_right, bottom row reverse KL; one column per T
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
fig.patch.set_facecolor(SURFACE)

for column, temperature in enumerate(TEMPERATURES):
    target_p_right, drop_kl = target_values(temperature)
    traces = all_traces[temperature]
    ax_p, ax_kl = axes[0, column], axes[1, column]

    if traces:
        seeds = sorted(traces)

        for phase, label, color in PHASES:
            epochs = traces[seeds[0]][phase][:, 0]

            for ax, index in [(ax_p, 1), (ax_kl, 2)]:
                stacked = np.stack([traces[seed][phase][:, index] for seed in seeds])
                low, median, high = np.percentile(stacked, [25, 50, 75], axis=0)
                ax.fill_between(epochs, low, high, color=color, alpha=0.2, linewidth=0)
                ax.plot(epochs, median, color=color, linewidth=2, label=label)

        # MLE fit is a reference level (its own 5000-epoch run), shown as its final median
        mle_kl = np.median([traces[seed]["mle"][-1, 2] for seed in seeds])
        ax_kl.axhline(mle_kl, color=MLE_COLOR, linestyle=":", linewidth=2, label="MLE fit (end)")

    ax_p.axhline(target_p_right, color=TEXT_MUTED, linestyle="--", linewidth=1.2, label="Boltzmann target")
    ax_kl.axhline(drop_kl, color=TEXT_MUTED, linestyle="-.", linewidth=1.2, label="Cost of dropping right basin")

    # y range around the target so small targets stay visible (epoch 0 starts near 0.5)
    ax_p.set_ylim(0, min(0.6, 3 * target_p_right))
    ax_kl.set_ylim(0, KL_YMAX)
    ax_p.set_title(
        f"T = {temperature}  ({len(traces)} seed{'' if len(traces) == 1 else 's'})",
        color=TEXT, loc="left", fontsize=11,
    )
    if drop_kl > KL_YMAX:
        ax_kl.set_title(f"cost of dropping right basin = {drop_kl:.3f} (above axis)", color=TEXT_MUTED, loc="left", fontsize=9)

    for ax in [ax_p, ax_kl]:
        ax.set_facecolor(SURFACE)
        ax.set_xlim(0, EPOCHS)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(TEXT_MUTED)
        ax.tick_params(colors=TEXT_MUTED, labelcolor=TEXT, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)

for ax in axes[1]:
    ax.set_xlabel("REINFORCE epoch", color=TEXT, fontsize=10)
axes[0, 0].set_ylabel("Mass in right (local) basin", color=TEXT, fontsize=10)
axes[1, 0].set_ylabel("Reverse KL to target (nats)", color=TEXT, fontsize=10)

# One legend with every identity (top row has the target, bottom row the MLE and drop-cost lines)
handles, labels = [], []
for ax in [axes[0, 2], axes[1, 2]]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.95), ncol=5, frameon=False, fontsize=10)
fig.suptitle(
    f"Scratch vs MLE warm start, Δf = {DELTA_F} (median, 25–75 % over seeds)",
    color=TEXT, x=0.01, y=0.99, ha="left", fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(FIGURE_PATH, dpi=200, facecolor=SURFACE)
fig.savefig(FIGURE_PATH.with_suffix(".pdf"), facecolor=SURFACE)

print(f"\nSaved {SUMMARY_PATH.name} and {FIGURE_PATH.name} (+ .pdf) in {DATA_DIR}")
