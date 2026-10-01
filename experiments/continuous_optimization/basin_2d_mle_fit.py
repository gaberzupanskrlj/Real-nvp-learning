import csv
import math
import sys

import torch
import torch.nn as nn


# Settings
DELTA_F = float(sys.argv[1])
TEMPERATURE = float(sys.argv[2])
SEEDS = list(range(42, 52))
NUM_LAYERS = 4
BATCH_SIZE = 1024
MAX_GRAD_NORM = 5.0

# REINFORCE on free energy, same as basin_2d_trace.py
EPOCHS = 2000
LR = 1e-4
EVAL_EVERY = 20

# Maximum likelihood fit on exact target samples
MLE_EPOCHS = 5000
MLE_LR = 1e-3
MLE_EVAL_EVERY = 100

EVAL_SIZE = 16384
EVAL_SEED = 123
TARGET_EVAL_SEED = 456
FINAL_SIZE = 100_000
FINAL_SEED_OFFSET = 1000
OCCUPIED_THRESHOLD = 0.01
NUM_THREADS = 1

# scratch = REINFORCE from default init (reproduces basin_2d_trace.py)
# mle     = maximum likelihood fit, same init
# warm    = REINFORCE starting from the mle fit
PHASES = ["scratch", "mle", "warm"]

RUN_NAME = f"basin2d_mle_delta{DELTA_F}_T{TEMPERATURE}"
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


# Exact Boltzmann target: continuous density p(y) ∝ exp(-f(discretize(y)) / T),
# constant inside each grid cell; edge cells have half width because tanh
# keeps y inside (-1, 1). Same target as boltzmann_target in basin_2d_trace.py
def build_target(temperature):
    step = 2.0 / 49
    grid = -1.0 + torch.arange(50, dtype=torch.float64) * step
    low = torch.clamp(grid - step / 2, min=-1.0)
    width = torch.clamp(grid + step / 2, max=1.0) - low

    X, Y = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    cell = (width[:, None] * width[None, :]).flatten()

    objective = f(points)
    labels = basin_label(points)
    log_weight = torch.log(cell) - objective / temperature
    log_z = torch.logsumexp(log_weight, dim=0)
    mass = torch.exp(log_weight - log_z)

    target = {
        "low": low,
        "width": width,
        "mass": mass,
        "log_z": log_z.item(),
        # E_p[log p], p is constant inside a cell
        "neg_entropy": (mass * (-objective / temperature - log_z)).sum().item(),
        "mean_f": (mass * objective).sum().item(),
    }

    for basin in [0, 1]:
        mask = labels == basin
        target[f"log_z_{basin}"] = torch.logsumexp(log_weight[mask], dim=0).item()
        target[f"p_{basin}"] = mass[mask].sum().item()

    return target


# Exact target samples: pick a cell by its mass, then uniform inside the cell
def sample_target(target, n, generator=None):
    idx = torch.multinomial(target["mass"], n, replacement=True, generator=generator)
    i = idx // 50
    j = idx % 50
    u = torch.rand(n, 2, generator=generator, dtype=torch.float64)

    y1 = target["low"][i] + target["width"][i] * u[:, 0]
    y2 = target["low"][j] + target["width"][j] * u[:, 1]

    return torch.stack([y1, y2], dim=1).float()


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

KL_FIELDS = ["reverse_kl", "kl_allocation", "kl_shape", "forward_kl"]


# Basin stats plus exact KL to the target.
# reverse KL(q || p) = E_q[log q + f / T] + log Z
#                    = allocation (basin masses vs target masses)
#                    + shape (sum over basins of q_b * KL within the basin)
# forward KL(p || q) = E_p[log p] - E_p[log q] on fixed target samples
def evaluate(model, r, target_y, target):
    with torch.no_grad():
        y, forward_log_det = model(r)
        log_q = gaussian_log_prob(r) - forward_log_det
        x = discretize(y)
        row = basin_stats(x)

        cost = (log_q + f(x) / TEMPERATURE).double()
        labels = basin_label(x)
        allocation = 0.0
        shape = 0.0

        for basin in [0, 1]:
            mask = labels == basin
            q_b = mask.double().mean().item()

            if q_b > 0:
                allocation += q_b * math.log(q_b / target[f"p_{basin}"])
                shape += q_b * (
                    cost[mask].mean().item() - math.log(q_b) + target[f"log_z_{basin}"]
                )

        row["reverse_kl"] = cost.mean().item() + target["log_z"]
        row["kl_allocation"] = allocation
        row["kl_shape"] = shape

        r_target, inverse_log_det = model.inverse(target_y)
        log_q_target = gaussian_log_prob(r_target) + inverse_log_det
        row["forward_kl"] = target["neg_entropy"] - log_q_target.double().mean().item()

    return row


def print_row(seed, phase, epoch, row):
    print(
        f"[{seed} {phase:7s}] epoch {epoch:4d} | "
        f"mean f = {row['mean_f']:.5f} | "
        f"p_right = {row['p_right']:.4f} | "
        f"KL rev = {row['reverse_kl']:.4f} "
        f"(alloc {row['kl_allocation']:.4f}, shape {row['kl_shape']:.4f}) | "
        f"KL fwd = {row['forward_kl']:.4f}"
    )


# REINFORCE on free energy E[f] + T * E[log q] (same as basin_2d_trace.py)
def reinforce(model, seed, phase, trace_writer, target, eval_r, target_y):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS + 1):
        if epoch % EVAL_EVERY == 0:
            row = evaluate(model, eval_r, target_y, target)
            trace_writer.writerow({"seed": seed, "phase": phase, "epoch": epoch, **row})

            if epoch % 500 == 0:
                print_row(seed, phase, epoch, row)

        if epoch == EPOCHS:
            break

        r = torch.randn(BATCH_SIZE, 2)
        y, _ = model(r)

        y_fixed = y.detach()
        r_inverse, inverse_log_det = model.inverse(y_fixed)
        log_prob_y = gaussian_log_prob(r_inverse) + inverse_log_det

        with torch.no_grad():
            objective = f(discretize(y))

            # Free energy: centered only, no standardization, so T stays fixed
            cost = objective + TEMPERATURE * log_prob_y
            weights = cost - cost.mean()

        loss = (weights * log_prob_y).mean()

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at seed {seed}, {phase}, epoch {epoch}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()


# Maximum likelihood on fresh exact target samples (no REINFORCE, no reverse KL)
def fit_mle(model, seed, trace_writer, target, eval_r, target_y):
    optimizer = torch.optim.Adam(model.parameters(), lr=MLE_LR)

    for epoch in range(MLE_EPOCHS + 1):
        if epoch % MLE_EVAL_EVERY == 0:
            row = evaluate(model, eval_r, target_y, target)
            trace_writer.writerow({"seed": seed, "phase": "mle", "epoch": epoch, **row})

            if epoch % 1000 == 0:
                print_row(seed, "mle", epoch, row)

        if epoch == MLE_EPOCHS:
            break

        y = sample_target(target, BATCH_SIZE)
        r, inverse_log_det = model.inverse(y)
        loss = -(gaussian_log_prob(r) + inverse_log_det).mean()

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at seed {seed}, mle, epoch {epoch}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()


def run_seed(seed, trace_writer, target, eval_r, target_y):
    # Same fresh final sample for all phases (paired)
    final_gen = torch.Generator().manual_seed(FINAL_SEED_OFFSET + seed)
    final_r = torch.randn(FINAL_SIZE, 2, generator=final_gen)
    rows = []

    print(f"\nSeed {seed}")

    # Scratch: same RNG stream as basin_2d_trace.py
    torch.manual_seed(seed)
    model = RealNVP(NUM_LAYERS)
    reinforce(model, seed, "scratch", trace_writer, target, eval_r, target_y)
    rows.append({"seed": seed, "phase": "scratch", **evaluate(model, final_r, target_y, target)})

    # MLE: same init as scratch
    torch.manual_seed(seed)
    model = RealNVP(NUM_LAYERS)
    fit_mle(model, seed, trace_writer, target, eval_r, target_y)
    rows.append({"seed": seed, "phase": "mle", **evaluate(model, final_r, target_y, target)})

    # Warm: REINFORCE from the MLE fit. Re-drawing the init consumes the
    # same random numbers, so the batch stream is identical to scratch
    torch.manual_seed(seed)
    RealNVP(NUM_LAYERS)
    reinforce(model, seed, "warm", trace_writer, target, eval_r, target_y)
    rows.append({"seed": seed, "phase": "warm", **evaluate(model, final_r, target_y, target)})

    return rows


# Main
TARGET = build_target(TEMPERATURE)
DROP_RIGHT_KL = -math.log(TARGET["p_0"])

eval_r = torch.randn(EVAL_SIZE, 2, generator=torch.Generator().manual_seed(EVAL_SEED))
target_y = sample_target(TARGET, EVAL_SIZE, torch.Generator().manual_seed(TARGET_EVAL_SEED))

print(f"DELTA_F = {DELTA_F}, TEMPERATURE = {TEMPERATURE}")
print(
    f"Target: p_right = {TARGET['p_1']:.4f}, mean f = {TARGET['mean_f']:.5f}, "
    f"log Z = {TARGET['log_z']:.4f}"
)
print(f"Reverse KL of dropping the right basin (exact left-only target): {DROP_RIGHT_KL:.4f}")

# Sampler check: target samples should reproduce the target basin mass and mean f
target_x = discretize(target_y)
print(
    f"Sampler check ({EVAL_SIZE} samples): "
    f"p_right = {basin_label(target_x).float().mean().item():.4f}, "
    f"mean f = {f(target_x).mean().item():.5f}"
)

final_rows = []

with open(TRACE_PATH, "w", newline="") as trace_file:
    trace_writer = csv.DictWriter(
        trace_file, fieldnames=["seed", "phase", "epoch"] + STAT_FIELDS + KL_FIELDS
    )
    trace_writer.writeheader()

    for seed in SEEDS:
        final_rows.extend(run_seed(seed, trace_writer, TARGET, eval_r, target_y))
        trace_file.flush()

TARGET_COLUMNS = {
    "target_p_left": TARGET["p_0"],
    "target_p_right": TARGET["p_1"],
    "target_mean_f": TARGET["mean_f"],
    "drop_right_kl": DROP_RIGHT_KL,
}

with open(FINAL_PATH, "w", newline="") as final_file:
    final_writer = csv.DictWriter(
        final_file,
        fieldnames=["seed", "phase"] + STAT_FIELDS + KL_FIELDS + list(TARGET_COLUMNS),
    )
    final_writer.writeheader()
    final_writer.writerows([{**row, **TARGET_COLUMNS} for row in final_rows])


# Summary
print(f"\nFinal ({FINAL_SIZE} fresh samples), DELTA_F = {DELTA_F}, TEMPERATURE = {TEMPERATURE}")
print(f"Target p_right = {TARGET['p_1']:.4f}, mean f = {TARGET['mean_f']:.5f}")
print(f"Dropping the right basin costs {DROP_RIGHT_KL:.4f} nats of reverse KL")

for phase in PHASES:
    rows = [row for row in final_rows if row["phase"] == phase]

    print(f"\n{phase}")
    print("seed | p_right | mean f  | KL rev | alloc  | shape  | KL fwd")

    for row in rows:
        print(
            f"{row['seed']:4d} | {row['p_right']:.4f}  | {row['mean_f']:.5f} | "
            f"{row['reverse_kl']:.4f} | {row['kl_allocation']:.4f} | "
            f"{row['kl_shape']:.4f} | {row['forward_kl']:.4f}"
        )

    both = sum(row["occupied_basins"] == 2 for row in rows)
    mean_kl = sum(row["reverse_kl"] for row in rows) / len(rows)
    below_drop = sum(row["reverse_kl"] < DROP_RIGHT_KL for row in rows)

    print(f"Both basins: {both}/{len(rows)}, mean reverse KL = {mean_kl:.4f}, "
          f"below drop cost: {below_drop}/{len(rows)}")

print(f"\nSaved {TRACE_PATH} and {FINAL_PATH}")
