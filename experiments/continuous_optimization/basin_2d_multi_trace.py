import csv
import math
import sys

import torch
import torch.nn as nn


# Settings
# None = standardized REINFORCE on f (control), else free energy E[f] + T * E[log q]
TEMPERATURE = float(sys.argv[1]) if len(sys.argv) > 1 else None
# Basin k: minimum at CENTERS[k] with f = DELTAS[k]; basin 0 is global
CENTERS = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]
DELTAS = [0.0, 0.1, 0.2, 0.3]
NUM_BASINS = len(CENTERS)
SEEDS = list(range(42, 52))
NUM_LAYERS = 4
BATCH_SIZE = 1024
EPOCHS = 2000
LR = 1e-4
MAX_GRAD_NORM = 5.0
EVAL_EVERY = 20
EVAL_SIZE = 16384
EVAL_SEED = 123
# Grid histogram of the eval batch at these epochs (multiples of EVAL_EVERY), for plots only
SNAPSHOT_EPOCHS = [0, 100, 200, 500, 1000, 2000]
FINAL_SIZE = 100_000
FINAL_SEED_OFFSET = 1000
OCCUPIED_THRESHOLD = 0.01
RETAIN_FRACTION = 0.5
NUM_THREADS = 1

RUN_NAME = f"basin2d_{NUM_BASINS}min"
RUN_NAME += f"_T{TEMPERATURE}" if TEMPERATURE is not None else "_noT"

TRACE_PATH = f"{RUN_NAME}_trace.csv"
FINAL_PATH = f"{RUN_NAME}_final.csv"
GRID_PATH = f"{RUN_NAME}_grid.csv"

torch.set_num_threads(NUM_THREADS)


# 2D objective: f = min_k ||x - CENTERS[k]||^2 + DELTAS[k]
def quadratics(x):
    centers = torch.tensor(CENTERS, dtype=x.dtype)
    deltas = torch.tensor(DELTAS, dtype=x.dtype)

    return ((x[:, None, :] - centers[None]) ** 2).sum(dim=2) + deltas[None]


def f(x):
    return quadratics(x).min(dim=1).values


# Basin k = index of the active quadratic (ties go to the lower index)
def basin_label(x):
    return quadratics(x).argmin(dim=1)


# Nearest point on the 50-point grid in [-1, 1]
def discretize(y):
    step = 2.0 / 49

    idx = torch.round((y + 1.0) / step)
    idx = torch.clamp(idx, 0, 49)

    return -1.0 + idx * step


# Sample count per grid cell of discretized points (i = x1 index, j = x2 index), nonzero cells only
def grid_counts(x):
    step = 2.0 / 49

    idx = torch.round((x + 1.0) / step).long()
    cells, counts = torch.unique(idx, dim=0, return_counts=True)

    return [(i, j, count) for (i, j), count in zip(cells.tolist(), counts.tolist())]


# Exact Boltzmann target on the 50x50 grid: mass ∝ cell area * exp(-f / T),
# edge cells have half width because tanh keeps y inside (-1, 1)
def boltzmann_target(temperature):
    step = 2.0 / 49
    grid = -1.0 + torch.arange(50, dtype=torch.float64) * step
    width = torch.full((50,), step, dtype=torch.float64)
    width[0] = width[-1] = step / 2

    X, Y = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    cell = (width[:, None] * width[None, :]).flatten()

    objective = f(points)
    mass = torch.softmax(torch.log(cell) - objective / temperature, dim=0)
    labels = basin_label(points)

    target = {f"target_p_{k}": mass[labels == k].sum().item() for k in range(NUM_BASINS)}
    target["target_mean_f"] = (mass * objective).sum().item()

    return target


# RealNVP model (same as objective_2d_gaussian.py)
class CouplingLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.s_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.t_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x1 = x[:, 0:1]
        x2 = x[:, 1:2]

        s = self.s_net(x1)
        t = self.t_net(x1)

        y2 = x2 * torch.exp(s) + t

        return torch.cat([x1, y2], dim=1), s.squeeze(1)

    def inverse(self, y):
        y1 = y[:, 0:1]
        y2 = y[:, 1:2]

        s = self.s_net(y1)
        t = self.t_net(y1)

        x2 = (y2 - t) * torch.exp(-s)

        return torch.cat([y1, x2], dim=1), -s.squeeze(1)


class RealNVP(nn.Module):
    def __init__(self, num_layers):
        super().__init__()

        self.layers = nn.ModuleList(
            [CouplingLayer() for _ in range(num_layers)]
        )

    def forward(self, x):
        y = x
        log_det_total = torch.zeros(x.shape[0])

        for layer in self.layers:
            y, log_det = layer(y)
            log_det_total += log_det
            y = torch.flip(y, dims=[1])

        # Keep the final solution inside (-1, 1)^2
        y = torch.tanh(y)
        log_det_total += torch.log(1.0 - y**2 + 1e-6).sum(dim=1)

        return y, log_det_total

    def inverse(self, y):
        eps = 1e-6

        y = torch.clamp(y, -1 + eps, 1 - eps)
        x = torch.atanh(y)
        log_det_inverse = -torch.log(1.0 - y**2 + eps).sum(dim=1)

        for layer in reversed(self.layers):
            x = torch.flip(x, dims=[1])
            x, log_det = layer.inverse(x)
            log_det_inverse += log_det

        return x, log_det_inverse


def gaussian_log_prob(r):
    return -0.5 * (r**2 + math.log(2.0 * math.pi)).sum(dim=1)


# Basin measurement on discretized points
def basin_stats(x):
    objective = f(x)
    labels = basin_label(x)

    row = {
        "mean_f": objective.mean().item(),
        "best_f": objective.min().item(),
    }

    for k in range(NUM_BASINS):
        mask = labels == k
        points = x[mask]
        count = points.shape[0]

        row[f"p_{k}"] = mask.float().mean().item()

        if count > 0:
            row[f"unique_{k}"] = torch.unique(points, dim=0).shape[0]
            row[f"mean_f_{k}"] = objective[mask].mean().item()
            centered = points - points.mean(dim=0)
            row[f"spread_{k}"] = centered.pow(2).sum(dim=1).mean().sqrt().item()
        else:
            row[f"unique_{k}"] = 0
            row[f"mean_f_{k}"] = float("nan")
            row[f"spread_{k}"] = float("nan")

    row["occupied_basins"] = sum(
        int(row[f"p_{k}"] > OCCUPIED_THRESHOLD) for k in range(NUM_BASINS)
    )

    return row


STAT_FIELDS = (
    ["mean_f", "best_f"]
    + [f"p_{k}" for k in range(NUM_BASINS)]
    + ["occupied_basins"]
    + [f"unique_{k}" for k in range(NUM_BASINS)]
    + [f"mean_f_{k}" for k in range(NUM_BASINS)]
    + [f"spread_{k}" for k in range(NUM_BASINS)]
)


def run_seed(seed, trace_writer, grid_writer):
    torch.manual_seed(seed)

    model = RealNVP(NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    eval_r = torch.randn(EVAL_SIZE, 2, generator=eval_gen)

    # Inverse check at initialization
    with torch.no_grad():
        eval_y, _ = model(eval_r)
        r_back, _ = model.inverse(eval_y)
        inverse_error = (eval_r - r_back).abs().max().item()

    print(f"\nSeed {seed} | max inverse error at init = {inverse_error:.3e}")

    # epoch = number of completed updates when evaluated
    for epoch in range(EPOCHS + 1):
        # Validation (fixed latent batch)
        if epoch % EVAL_EVERY == 0:
            with torch.no_grad():
                eval_y, _ = model(eval_r)
                eval_x = discretize(eval_y)
                row = basin_stats(eval_x)

            trace_writer.writerow({"seed": seed, "epoch": epoch, **row})

            if epoch in SNAPSHOT_EPOCHS:
                grid_writer.writerows([(seed, epoch, i, j, count) for i, j, count in grid_counts(eval_x)])

            if epoch % 200 == 0:
                p_text = " ".join(f"{row[f'p_{k}']:.4f}" for k in range(NUM_BASINS))
                unique_text = "/".join(str(row[f"unique_{k}"]) for k in range(NUM_BASINS))
                print(f"Epoch {epoch:4d} | mean f = {row['mean_f']:.5f} | p = {p_text} | unique = {unique_text}")

        if epoch == EPOCHS:
            break

        # REINFORCE step
        r = torch.randn(BATCH_SIZE, 2)
        y, _ = model(r)

        y_fixed = y.detach()
        r_inverse, inverse_log_det = model.inverse(y_fixed)
        log_prob_y = gaussian_log_prob(r_inverse) + inverse_log_det

        with torch.no_grad():
            objective = f(discretize(y))

            if TEMPERATURE is None:
                weights = (objective - objective.mean()) / (objective.std() + 1e-8)
            else:
                # Free energy: centered only, no standardization, so T stays fixed
                cost = objective + TEMPERATURE * log_prob_y
                weights = cost - cost.mean()

        loss = (weights * log_prob_y).mean()

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at seed {seed}, epoch {epoch}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    # Final evaluation (last epoch, fresh sample)
    final_gen = torch.Generator().manual_seed(FINAL_SEED_OFFSET + seed)
    final_r = torch.randn(FINAL_SIZE, 2, generator=final_gen)

    with torch.no_grad():
        final_y, _ = model(final_r)
        final_row = basin_stats(discretize(final_y))

    return {"seed": seed, **final_row}


# Main
TARGET = boltzmann_target(TEMPERATURE) if TEMPERATURE is not None else {}

print(f"CENTERS = {CENTERS}, DELTAS = {DELTAS}, TEMPERATURE = {TEMPERATURE}")

if TARGET:
    target_text = " ".join(f"{TARGET[f'target_p_{k}']:.4f}" for k in range(NUM_BASINS))
    print(f"Target: p = {target_text}, mean f = {TARGET['target_mean_f']:.5f}")

final_rows = []

with open(TRACE_PATH, "w", newline="") as trace_file, open(GRID_PATH, "w", newline="") as grid_file:
    trace_writer = csv.DictWriter(trace_file, fieldnames=["seed", "epoch"] + STAT_FIELDS)
    trace_writer.writeheader()
    grid_writer = csv.writer(grid_file)
    grid_writer.writerow(["seed", "epoch", "i", "j", "count"])

    for seed in SEEDS:
        final_rows.append(run_seed(seed, trace_writer, grid_writer))
        trace_file.flush()
        grid_file.flush()

# Distance to target: total variation over basin masses; basin k retained if p_k >= RETAIN_FRACTION * target
if TARGET:
    for row in final_rows:
        row["tv_to_target"] = 0.5 * sum(
            abs(row[f"p_{k}"] - TARGET[f"target_p_{k}"]) for k in range(NUM_BASINS)
        )
        row["retained_basins"] = sum(
            int(row[f"p_{k}"] >= RETAIN_FRACTION * TARGET[f"target_p_{k}"]) for k in range(NUM_BASINS)
        )

extra_fields = ["tv_to_target", "retained_basins"] if TARGET else []

with open(FINAL_PATH, "w", newline="") as final_file:
    final_writer = csv.DictWriter(final_file, fieldnames=["seed"] + STAT_FIELDS + extra_fields + list(TARGET))
    final_writer.writeheader()
    final_writer.writerows([{**row, **TARGET} for row in final_rows])


# Summary
print(f"\nFinal ({FINAL_SIZE} fresh samples, last epoch), DELTAS = {DELTAS}, TEMPERATURE = {TEMPERATURE}")
print("seed | mean f   | p per basin                     | occupied | TV     | retained")

for row in final_rows:
    p_text = " ".join(f"{row[f'p_{k}']:.4f}" for k in range(NUM_BASINS))
    line = f"{row['seed']:4d} | {row['mean_f']:.5f} | {p_text} | {row['occupied_basins']:8d}"
    if TARGET:
        line += f" | {row['tv_to_target']:.4f} | {row['retained_basins']}/{NUM_BASINS}"
    print(line)

print()
for count in range(NUM_BASINS, 0, -1):
    seeds = sum(row["occupied_basins"] == count for row in final_rows)
    print(f"{count} basins occupied (> {OCCUPIED_THRESHOLD}): {seeds}/{len(final_rows)}")

if TARGET:
    all_retained = sum(row["retained_basins"] == NUM_BASINS for row in final_rows)
    mean_tv = sum(row["tv_to_target"] for row in final_rows) / len(final_rows)
    mean_mean_f = sum(row["mean_f"] for row in final_rows) / len(final_rows)

    print(f"All {NUM_BASINS} basins retained (p_k >= {RETAIN_FRACTION} * target): {all_retained}/{len(final_rows)}")
    print("\nbasin | delta | mean p  | target p")
    for k in range(NUM_BASINS):
        mean_p = sum(row[f"p_{k}"] for row in final_rows) / len(final_rows)
        print(f"{k:5d} | {DELTAS[k]:.3f} | {mean_p:.4f}  | {TARGET[f'target_p_{k}']:.4f}")
    print(f"\nMean TV to target: {mean_tv:.4f}")
    print(f"Mean f:            {mean_mean_f:.5f} (target {TARGET['target_mean_f']:.5f})")
print(f"\nSaved {TRACE_PATH}, {FINAL_PATH} and {GRID_PATH}")
