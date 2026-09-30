import csv
import math
import sys

import torch
import torch.nn as nn


# Settings
DELTA_F = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
# None = standardized REINFORCE on f (steps 1-2), else free energy E[f] + T * E[log q]
TEMPERATURE = float(sys.argv[2]) if len(sys.argv) > 2 else None
SEEDS = list(range(42, 52))
NUM_LAYERS = 4
BATCH_SIZE = 1024
EPOCHS = 2000
LR = 1e-4
MAX_GRAD_NORM = 5.0
EVAL_EVERY = 20
EVAL_SIZE = 16384
EVAL_SEED = 123
FINAL_SIZE = 100_000
FINAL_SEED_OFFSET = 1000
OCCUPIED_THRESHOLD = 0.01
NUM_THREADS = 1

RUN_NAME = f"basin2d_delta{DELTA_F}"
if TEMPERATURE is not None:
    RUN_NAME += f"_T{TEMPERATURE}"

TRACE_PATH = f"{RUN_NAME}_trace.csv"
FINAL_PATH = f"{RUN_NAME}_final.csv"

torch.set_num_threads(NUM_THREADS)


# 2D objective: left minimum at (-0.5, -0.5) with f = 0,
# right minimum at (0.5, -0.5) with f = DELTA_F
def f(x):
    x1 = x[:, 0]
    x2 = x[:, 1]

    q1 = (x1 + 0.5) ** 2 + (x2 + 0.5) ** 2
    q2 = (x1 - 0.5) ** 2 + (x2 + 0.5) ** 2 + DELTA_F

    return torch.minimum(q1, q2)


# 0 = left basin (q1 active), 1 = right basin (q2 active)
def basin_label(x):
    x1 = x[:, 0]
    x2 = x[:, 1]

    q1 = (x1 + 0.5) ** 2 + (x2 + 0.5) ** 2
    q2 = (x1 - 0.5) ** 2 + (x2 + 0.5) ** 2 + DELTA_F

    return (q2 < q1).long()


# Nearest point on the 50-point grid in [-1, 1]
def discretize(y):
    step = 2.0 / 49

    idx = torch.round((y + 1.0) / step)
    idx = torch.clamp(idx, 0, 49)

    return -1.0 + idx * step


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

    return {
        "target_p_left": mass[labels == 0].sum().item(),
        "target_p_right": mass[labels == 1].sum().item(),
        "target_mean_f": (mass * objective).sum().item(),
    }


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

    for basin, name in [(0, "left"), (1, "right")]:
        mask = labels == basin
        points = x[mask]
        count = points.shape[0]

        row[f"p_{name}"] = mask.float().mean().item()

        if count > 0:
            row[f"unique_{name}"] = torch.unique(points, dim=0).shape[0]
            row[f"mean_f_{name}"] = objective[mask].mean().item()
            centered = points - points.mean(dim=0)
            row[f"spread_{name}"] = centered.pow(2).sum(dim=1).mean().sqrt().item()
        else:
            row[f"unique_{name}"] = 0
            row[f"mean_f_{name}"] = float("nan")
            row[f"spread_{name}"] = float("nan")

    row["occupied_basins"] = (
        int(row["p_left"] > OCCUPIED_THRESHOLD)
        + int(row["p_right"] > OCCUPIED_THRESHOLD)
    )

    return row


STAT_FIELDS = [
    "mean_f",
    "best_f",
    "p_left",
    "p_right",
    "occupied_basins",
    "unique_left",
    "unique_right",
    "mean_f_left",
    "mean_f_right",
    "spread_left",
    "spread_right",
]


def run_seed(seed, trace_writer):
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
                row = basin_stats(discretize(eval_y))

            trace_writer.writerow({"seed": seed, "epoch": epoch, **row})

            if epoch % 200 == 0:
                print(
                    f"Epoch {epoch:4d} | "
                    f"mean f = {row['mean_f']:.5f} | "
                    f"p_left = {row['p_left']:.4f} | "
                    f"p_right = {row['p_right']:.4f} | "
                    f"unique L/R = {row['unique_left']}/{row['unique_right']}"
                )

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

print(f"DELTA_F = {DELTA_F}, TEMPERATURE = {TEMPERATURE}")

if TARGET:
    print(
        f"Target: p_left = {TARGET['target_p_left']:.4f}, "
        f"p_right = {TARGET['target_p_right']:.4f}, "
        f"mean f = {TARGET['target_mean_f']:.5f}"
    )

final_rows = []

with open(TRACE_PATH, "w", newline="") as trace_file:
    trace_writer = csv.DictWriter(trace_file, fieldnames=["seed", "epoch"] + STAT_FIELDS)
    trace_writer.writeheader()

    for seed in SEEDS:
        final_rows.append(run_seed(seed, trace_writer))
        trace_file.flush()

with open(FINAL_PATH, "w", newline="") as final_file:
    final_writer = csv.DictWriter(final_file, fieldnames=["seed"] + STAT_FIELDS + list(TARGET))
    final_writer.writeheader()
    final_writer.writerows([{**row, **TARGET} for row in final_rows])


# Summary
print(f"\nFinal ({FINAL_SIZE} fresh samples, last epoch), DELTA_F = {DELTA_F}, TEMPERATURE = {TEMPERATURE}")
print("seed | mean f   | p_left  | p_right | occupied | unique L/R")

for row in final_rows:
    print(
        f"{row['seed']:4d} | {row['mean_f']:.5f} | "
        f"{row['p_left']:.4f}  | {row['p_right']:.4f}  | "
        f"{row['occupied_basins']:8d} | "
        f"{row['unique_left']}/{row['unique_right']}"
    )

both = sum(row["occupied_basins"] == 2 for row in final_rows)
only_left = sum(row["occupied_basins"] == 1 and row["p_left"] > OCCUPIED_THRESHOLD for row in final_rows)
only_right = sum(row["occupied_basins"] == 1 and row["p_right"] > OCCUPIED_THRESHOLD for row in final_rows)

print(f"\nBoth basins: {both}/{len(final_rows)}")
print(f"Only left:   {only_left}/{len(final_rows)}")
print(f"Only right:  {only_right}/{len(final_rows)}")

if TARGET:
    mean_p_right = sum(row["p_right"] for row in final_rows) / len(final_rows)
    mean_mean_f = sum(row["mean_f"] for row in final_rows) / len(final_rows)
    print(f"\nMean p_right: {mean_p_right:.4f} (target {TARGET['target_p_right']:.4f})")
    print(f"Mean f:       {mean_mean_f:.5f} (target {TARGET['target_mean_f']:.5f})")
print(f"\nSaved {TRACE_PATH} and {FINAL_PATH}")
