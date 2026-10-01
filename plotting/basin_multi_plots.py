import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Settings
DATA_DIR = Path(__file__).resolve().parents[1] / "experiments" / "continuous_optimization"
RUN_PREFIX = "basin2d_4min"
TEMPERATURE = 0.2
SEED = 42
EPOCHS_SHOWN = [0, 100, 500, 2000]
# Same as basin_2d_multi_trace.py
CENTERS = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]
DELTAS = [0.0, 0.1, 0.2, 0.3]
GRID_POINTS = 50
EVAL_SIZE = 16384

ROWS = [
    ("noT", "Without temperature (plain REINFORCE)"),
    (f"T{TEMPERATURE}", f"With Boltzmann temperature T = {TEMPERATURE}"),
]
FIGURE_PATH = DATA_DIR / f"{RUN_PREFIX}_noT_vs_T{TEMPERATURE}.png"

REALNVP_COLOR = "#eb6834"
DENSITY_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
AXIS = "#c3c2b7"
GRID = "#e4e3df"
SURFACE = "#ffffff"


def read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


# Share of eval samples per grid cell, indexed [x1, x2]
def load_density(run, epoch):
    density = np.zeros((GRID_POINTS, GRID_POINTS))

    for row in read_csv(DATA_DIR / f"{RUN_PREFIX}_{run}_grid.csv"):
        if int(row["seed"]) == SEED and int(row["epoch"]) == epoch:
            density[int(row["i"]), int(row["j"])] = int(row["count"]) / EVAL_SIZE

    return density


# Basin index on a fine grid, for drawing basin boundaries
def basin_map(resolution=400):
    axis = np.linspace(-1, 1, resolution)
    X1, X2 = np.meshgrid(axis, axis)
    quadratics = np.stack(
        [(X1 - c1) ** 2 + (X2 - c2) ** 2 + d for (c1, c2), d in zip(CENTERS, DELTAS)]
    )
    return axis, quadratics.argmin(axis=0)


def style_axis(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=TEXT_MUTED, labelsize=8)


densities = {(run, epoch): load_density(run, epoch) for run, _ in ROWS for epoch in EPOCHS_SHOWN}
finals = {run: read_csv(DATA_DIR / f"{RUN_PREFIX}_{run}_final.csv") for run, _ in ROWS}
target = [float(finals[ROWS[1][0]][0][f"target_p_{k}"]) for k in range(len(CENTERS))]

cmap = LinearSegmentedColormap.from_list("density", DENSITY_RAMP)
cmap.set_bad(SURFACE)
norm = LogNorm(vmin=1 / EVAL_SIZE, vmax=max(d.max() for d in densities.values()))

step = 2.0 / (GRID_POINTS - 1)
extent = [-1 - step / 2, 1 + step / 2, -1 - step / 2, 1 + step / 2]
axis, labels = basin_map()


# Figure: density snapshots per row, final basin mass on the right
fig = plt.figure(figsize=(15.5, 7.4))
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(
    2, 6, width_ratios=[1, 1, 1, 1, 0.05, 1.35],
    left=0.04, right=0.985, top=0.84, bottom=0.08, wspace=0.18, hspace=0.45,
)

for row_index, (run, row_title) in enumerate(ROWS):
    for col, epoch in enumerate(EPOCHS_SHOWN):
        ax = fig.add_subplot(gs[row_index, col])
        density = np.ma.masked_equal(densities[(run, epoch)], 0)

        image = ax.imshow(density.T, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
        ax.contour(axis, axis, labels, levels=[0.5, 1.5, 2.5], colors=TEXT_MUTED, linewidths=0.7)
        ax.scatter(*zip(*CENTERS), s=14, color=TEXT, edgecolors=SURFACE, linewidths=0.8, zorder=3)

        if col == 0:
            for (c1, c2), d in zip(CENTERS, DELTAS):
                ax.annotate(f"f = {d:g}", (c1, c2), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=8, color=TEXT)
            ax.text(-1.02, 1.28, row_title, transform=ax.transData, fontsize=12, color=TEXT, ha="left")

        ax.set_title(f"Epoch {epoch}", color=TEXT_MUTED, fontsize=10, loc="left")
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        if col > 0:
            ax.set_yticklabels([])
        if row_index == 0:
            ax.set_xticklabels([])
        style_axis(ax)

    # Final basin mass over all seeds (100k fresh samples per seed)
    ax = fig.add_subplot(gs[row_index, 5])
    p = np.array([[float(r[f"p_{k}"]) for k in range(len(CENTERS))] for r in finals[run]])
    x = np.arange(len(CENTERS))

    ax.bar(x, p.mean(axis=0), width=0.55, color=REALNVP_COLOR, zorder=2)
    ax.vlines(x, p.min(axis=0), p.max(axis=0), color=TEXT_MUTED, linewidth=1.2, zorder=3)
    if run != "noT":
        ax.hlines(target, x - 0.38, x + 0.38, color=TEXT, linewidth=2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"f = {d:g}" for d in DELTAS], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of samples", color=TEXT, fontsize=9)
    ax.set_title(f"Basin mass at epoch 2000 ({len(p)} seeds)", color=TEXT_MUTED, fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    style_axis(ax)

colorbar_ax = fig.add_subplot(gs[:, 4])
colorbar = fig.colorbar(image, cax=colorbar_ax)
colorbar.set_label("Share of samples per grid cell (log)", color=TEXT, fontsize=9)
colorbar.outline.set_visible(False)
colorbar.ax.tick_params(colors=AXIS, labelcolor=TEXT_MUTED, labelsize=8)

fig.legend(
    handles=[Patch(color=REALNVP_COLOR, label="RealNVP"), Line2D([], [], color=TEXT, linewidth=2, label="Boltzmann target")],
    loc="upper right", frameon=False, fontsize=10,
)
fig.suptitle(
    f"Where RealNVP puts its samples on a landscape with 4 minima (density: seed {SEED}, {EVAL_SIZE} samples)",
    color=TEXT, x=0.04, ha="left", fontsize=13,
)
fig.savefig(FIGURE_PATH, dpi=200, facecolor=SURFACE)
fig.savefig(FIGURE_PATH.with_suffix(".pdf"), facecolor=SURFACE)

print(f"Saved {FIGURE_PATH.name} (+ .pdf) in {DATA_DIR}")
