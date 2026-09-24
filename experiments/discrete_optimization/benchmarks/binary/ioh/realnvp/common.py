import copy
import math
from dataclasses import dataclass
from typing import Iterable

import ioh
import torch
import torch.nn as nn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class RealNVPConfig:
    problem_name: str
    dimension: int = 100
    problem_instance: int = 1
    num_layers: int = 4
    hidden_dim: int = 64
    batch_size: int = 1024
    epochs: int = 1000
    learning_rate: float = 1e-4
    validation_size: int = 4096
    test_size: int = 16384
    validate_every: int = 50
    log_every: int = 50
    max_grad_norm: float = 5.0
    seeds: tuple[int, ...] = (42,)


def create_problem(config: RealNVPConfig):
    return ioh.get_problem(
        config.problem_name,
        instance=config.problem_instance,
        dimension=config.dimension,
        problem_class=ioh.ProblemClass.PBO,
    )


def discretize(y: torch.Tensor) -> torch.Tensor:
    return (y > 0).to(torch.int32)


class CouplingLayer(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int):
        super().__init__()

        if dimension % 2 != 0:
            raise ValueError("This implementation expects an even dimension.")

        self.half = dimension // 2

        self.scale_net = nn.Sequential(
            nn.Linear(self.half, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.half),
            nn.Tanh(),
        )

        self.translate_net = nn.Sequential(
            nn.Linear(self.half, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.half),
        )

    def forward(self, x: torch.Tensor):
        x1 = x[:, : self.half]
        x2 = x[:, self.half :]

        s = self.scale_net(x1)
        t = self.translate_net(x1)

        y1 = x1
        y2 = x2 * torch.exp(s) + t

        return torch.cat([y1, y2], dim=1), s.sum(dim=1)

    def inverse(self, y: torch.Tensor):
        y1 = y[:, : self.half]
        y2 = y[:, self.half :]

        s = self.scale_net(y1)
        t = self.translate_net(y1)

        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        return torch.cat([x1, x2], dim=1), -s.sum(dim=1)


class RealNVP(nn.Module):
    def __init__(self, dimension: int, layers: int, hidden_dim: int):
        super().__init__()

        self.layers = nn.ModuleList(
            [CouplingLayer(dimension, hidden_dim) for _ in range(layers)]
        )
        self.half = dimension // 2

    def _swap_halves(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, self.half :], x[:, : self.half]], dim=1)

    def forward(self, x: torch.Tensor):
        z = x
        log_det_total = x.new_zeros(x.shape[0])

        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
            z = self._swap_halves(z)

        return z, log_det_total

    def inverse(self, z: torch.Tensor):
        x = z
        log_det_total = z.new_zeros(z.shape[0])

        for layer in reversed(self.layers):
            x = self._swap_halves(x)
            x, log_det = layer.inverse(x)
            log_det_total += log_det

        return x, log_det_total


def log_probability(model: RealNVP, y: torch.Tensor) -> torch.Tensor:
    z, inverse_log_det = model.inverse(y)

    log_pz = -0.5 * (
        z.square() + math.log(2.0 * math.pi)
    ).sum(dim=1)

    return log_pz + inverse_log_det


@torch.no_grad()
def evaluate_ioh(problem, x: torch.Tensor) -> torch.Tensor:
    values = problem(x.detach().cpu().tolist())
    return torch.tensor(values, dtype=torch.float32, device=x.device)


def train_one_seed(config: RealNVPConfig, seed: int):
    print("\n" + "=" * 68)
    print(
        f"{config.problem_name} | seed={seed} | "
        f"layers={config.num_layers} | hidden={config.hidden_dim} | "
        f"batch={config.batch_size}"
    )
    print("=" * 68)

    torch.manual_seed(seed)
    problem = create_problem(config)
    problem.reset()

    model = RealNVP(
        config.dimension,
        config.num_layers,
        config.hidden_dim,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    validation = torch.randn(
        config.validation_size,
        config.dimension,
        device=DEVICE,
    )

    best_validation_mean = float("-inf")
    best_state = None
    best_epoch = 0

    for epoch in range(config.epochs + 1):
        r = torch.randn(
            config.batch_size,
            config.dimension,
            device=DEVICE,
        )

        with torch.no_grad():
            y, _ = model(r)
            x = discretize(y)
            reward = evaluate_ioh(problem, x)

            baseline = (
                reward.sum() - reward
            ) / (config.batch_size - 1)

            advantage = reward - baseline

        # Score-function / REINFORCE objective.
        # y.detach() is intentional: candidates are treated as fixed samples
        # while the model changes their probability.
        log_prob = log_probability(model, y.detach())
        loss = -(advantage * log_prob).mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        )

        optimizer.step()

        if epoch % config.log_every == 0:
            print(
                f"{epoch:5d} | "
                f"mean={reward.mean().item():.6f} | "
                f"best={reward.max().item():.6f} | "
                f"loss={loss.item():.6f}"
            )

        if epoch % config.validate_every == 0:
            with torch.no_grad():
                val_y, _ = model(validation)
                val_x = discretize(val_y)
                val_score = evaluate_ioh(problem, val_x)
                val_mean = val_score.mean().item()

            if val_mean > best_validation_mean:
                best_validation_mean = val_mean
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test = torch.randn(
        config.test_size,
        config.dimension,
        device=DEVICE,
    )

    with torch.no_grad():
        test_y, _ = model(test)
        test_x = discretize(test_y)
        test_score = evaluate_ioh(problem, test_x)

    result = {
        "seed": seed,
        "mean": test_score.mean().item(),
        "best": test_score.max().item(),
        "best_validation_mean": best_validation_mean,
        "best_epoch": best_epoch,
    }

    print(
        f"FINAL | seed={seed} | "
        f"mean={result['mean']:.6f} | "
        f"best={result['best']:.6f} | "
        f"best_epoch={best_epoch}"
    )

    return result


def run_experiment(config: RealNVPConfig):
    print(
        f"Device: {DEVICE} | Problem: {config.problem_name} | "
        f"Instance: {config.problem_instance} | Dimension: {config.dimension}"
    )

    results = [
        train_one_seed(config, seed)
        for seed in config.seeds
    ]

    best_values = torch.tensor(
        [r["best"] for r in results],
        dtype=torch.float64,
    )

    mean_values = torch.tensor(
        [r["mean"] for r in results],
        dtype=torch.float64,
    )

    print("\n" + "=" * 68)
    print("REALNVP SUMMARY")
    print("=" * 68)

    for result in results:
        print(
            f"seed={result['seed']} | "
            f"mean={result['mean']:.6f} | "
            f"best={result['best']:.6f} | "
            f"best_epoch={result['best_epoch']}"
        )

    print(f"\nMean best: {best_values.mean().item():.10f}")
    print(f"Median best: {best_values.median().item():.10f}")

    if len(results) > 1:
        print(f"Std best: {best_values.std().item():.10f}")
    else:
        print("Std best: n/a (single seed)")

    print(f"Best run: {best_values.max().item():.10f}")
    print(f"Worst run: {best_values.min().item():.10f}")
    print(f"Mean distribution mean: {mean_values.mean().item():.10f}")

    return results
