import copy
import csv
import json
import math
import os
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import ioh
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = torch.device("cpu")

@dataclass(frozen=True)
class RealNVPConfig:
    problem_name: str
    output_dir: str
    dimension: int = 100
    problem_instance: int = 1
    num_layers: int = 4
    hidden_dim: int = 64
    batch_size: int = 1024
    learning_rate: float = 1e-4
    max_evaluations: int = 4_096_000
    validation_size: int = 4096
    validate_every: int = 50
    test_size: int = 16384
    max_grad_norm: float = 5.0
    seed_start: int = 42
    n_seeds: int = 100
    workers: int = 24
    target: float | None = None

#diskretizacija
def discretize(y):
    return (y > 0).to(torch.int32)


class CouplingLayer(nn.Module):
    def __init__(self, dimension, hidden_dim):
        super().__init__()
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

    def forward(self, x):
        x1 = x[:, :self.half]
        x2 = x[:, self.half:]
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        y = torch.cat([x1, x2 * torch.exp(s) + t], dim=1)
        return y, s.sum(dim=1)

    def inverse(self, y):
        y1 = y[:, :self.half]
        y2 = y[:, self.half:]
        s = self.scale_net(y1)
        t = self.translate_net(y1)
        x = torch.cat([y1, (y2 - t) * torch.exp(-s)], dim=1)
        return x, -s.sum(dim=1)

#model
class RealNVP(nn.Module):
    def __init__(self, dimension, num_layers, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList(
            [CouplingLayer(dimension, hidden_dim) for _ in range(num_layers)]
        )
        self.half = dimension // 2

    def _swap(self, x):
        return torch.cat([x[:, self.half:], x[:, :self.half]], dim=1)

    def forward(self, x):
        z = x
        log_det_total = x.new_zeros(x.shape[0])
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det
            z = self._swap(z)
        return z, log_det_total

    def inverse(self, z):
        x = z
        log_det_total = z.new_zeros(z.shape[0])
        for layer in reversed(self.layers):
            x = self._swap(x)
            x, log_det = layer.inverse(x)
            log_det_total += log_det
        return x, log_det_total


def log_probability(model, y):
    z, inverse_log_det = model.inverse(y)
    log_pz = -0.5 * (z.square() + math.log(2.0 * math.pi)).sum(dim=1)
    return log_pz + inverse_log_det


def create_problem(config):
    return ioh.get_problem(
        config.problem_name,
        instance=config.problem_instance,
        dimension=config.dimension,
        problem_class=ioh.ProblemClass.PBO,
    )


@torch.no_grad()
def evaluate_ioh(problem, x):
    values = problem(x.detach().cpu().tolist())
    return torch.tensor(values, dtype=torch.float32, device=x.device)


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

#training
def train_one_seed(config, seed):
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

    validation_generator = torch.Generator().manual_seed(seed + 1000)
    validation = torch.randn(
        config.validation_size,
        config.dimension,
        generator=validation_generator,
        device=DEVICE,
    )

    best_validation_mean = float("-inf")
    best_state = None
    best_epoch = -1

    evaluation_count = 0
    best_so_far = float("-inf")
    best_at_evaluation = None
    target_evaluation = None
    target_time = None
    convergence = []

    started = time.perf_counter()
    epoch = 0

    while evaluation_count + config.batch_size <= config.max_evaluations:
        r = torch.randn(
            config.batch_size,
            config.dimension,
            device=DEVICE,
        )

        with torch.no_grad():
            y, _ = model(r)
            x = discretize(y)
            reward = evaluate_ioh(problem, x)

        evaluation_count += config.batch_size
        batch_best = reward.max().item()

        if batch_best > best_so_far:
            best_so_far = batch_best
            best_at_evaluation = evaluation_count

        now = time.perf_counter() - started

        if (
            config.target is not None
            and target_evaluation is None
            and best_so_far >= config.target
        ):
            target_evaluation = evaluation_count
            target_time = now

        convergence.append(
            {
                "evaluations": evaluation_count,
                "best_so_far": best_so_far,
                "elapsed_seconds": now,
            }
        )
#najpomebnejši del
        with torch.no_grad():
            baseline = (reward.sum() - reward) / (config.batch_size - 1)
            advantage = reward - baseline

        log_prob = log_probability(model, y.detach())
        loss = -(advantage * log_prob).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        )
        optimizer.step()

        if (
            epoch % config.validate_every == 0
            and evaluation_count + config.validation_size <= config.max_evaluations
        ):
            with torch.no_grad():
                val_y, _ = model(validation)
                val_x = discretize(val_y)
                val_score = evaluate_ioh(problem, val_x)

            evaluation_count += config.validation_size
            val_best = val_score.max().item()

            if val_best > best_so_far:
                best_so_far = val_best
                best_at_evaluation = evaluation_count

            now = time.perf_counter() - started

            if (
                config.target is not None
                and target_evaluation is None
                and best_so_far >= config.target
            ):
                target_evaluation = evaluation_count
                target_time = now

            convergence.append(
                {
                    "evaluations": evaluation_count,
                    "best_so_far": best_so_far,
                    "elapsed_seconds": now,
                }
            )

            val_mean = val_score.mean().item()

            if val_mean > best_validation_mean:
                best_validation_mean = val_mean
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        epoch += 1

    training_seconds = time.perf_counter() - started

    if best_state is not None:
        model.load_state_dict(best_state)

    test_generator = torch.Generator().manual_seed(seed + 2000)
    test = torch.randn(
        config.test_size,
        config.dimension,
        generator=test_generator,
        device=DEVICE,
    )

    with torch.no_grad():
        test_y, _ = model(test)
        test_x = discretize(test_y)
        test_score = evaluate_ioh(problem, test_x)

    result = {
        "seed": seed,
        "evaluations": evaluation_count,
        "best_found": best_so_far,
        "best_at_evaluation": best_at_evaluation,
        "best_validation_mean": best_validation_mean,
        "best_epoch": best_epoch,
        "test_mean": test_score.mean().item(),
        "test_best": test_score.max().item(),
        "training_seconds": training_seconds,
        "target_evaluation": target_evaluation,
        "target_time_seconds": target_time,
    }

    return result, convergence


def _run_and_save(config, seed):
    output_dir = Path(config.output_dir)
    seed_dir = output_dir / f"seed_{seed:03d}"
    result_path = seed_dir / "result.json"

    if result_path.exists():
        return _load_json(result_path)

    result, convergence = train_one_seed(config, seed)

    _write_csv(
        seed_dir / "convergence.csv",
        convergence,
        ["evaluations", "best_so_far", "elapsed_seconds"],
    )
    _write_json(result_path, result)

    return result


def run_experiment(config):
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "config.json", asdict(config))

    seeds = list(range(config.seed_start, config.seed_start + config.n_seeds))
    results = []
    pending = []

    for seed in seeds:
        result_path = output_dir / f"seed_{seed:03d}" / "result.json"
        if result_path.exists():
            results.append(_load_json(result_path))
        else:
            pending.append(seed)

    print(
        f"{config.problem_name}: {len(results)} completed, "
        f"{len(pending)} remaining, workers={config.workers}"
    )

    if pending:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(_run_and_save, config, seed): seed
                for seed in pending
            }

            for i, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                result = future.result()
                results.append(result)
                print(
                    f"[{i}/{len(pending)}] seed={seed} "
                    f"best={result['best_found']:.6f} "
                    f"time={result['training_seconds']:.1f}s"
                )

    results.sort(key=lambda row: row["seed"])

    fieldnames = [
        "seed",
        "evaluations",
        "best_found",
        "best_at_evaluation",
        "best_validation_mean",
        "best_epoch",
        "test_mean",
        "test_best",
        "training_seconds",
        "target_evaluation",
        "target_time_seconds",
    ]

    _write_csv(
        output_dir / "summary.csv",
        results,
        fieldnames,
    )

    best_values = np.array([r["best_found"] for r in results], dtype=float)
    runtimes = np.array([r["training_seconds"] for r in results], dtype=float)

    aggregate = {
        "problem": config.problem_name,
        "runs": len(results),
        "mean_best": float(best_values.mean()),
        "std_best": float(best_values.std(ddof=1)) if len(results) > 1 else 0.0,
        "median_best": float(np.median(best_values)),
        "best_run": float(best_values.max()),
        "worst_run": float(best_values.min()),
        "mean_training_seconds": float(runtimes.mean()),
        "std_training_seconds": float(runtimes.std(ddof=1)) if len(results) > 1 else 0.0,
        "median_training_seconds": float(np.median(runtimes)),
    }

    if config.target is not None:
        hits = [r for r in results if r["target_evaluation"] is not None]
        aggregate["target"] = config.target
        aggregate["target_hits"] = len(hits)
        aggregate["target_hit_rate"] = len(hits) / len(results)

        if hits:
            target_evals = np.array(
                [r["target_evaluation"] for r in hits],
                dtype=float,
            )
            target_times = np.array(
                [r["target_time_seconds"] for r in hits],
                dtype=float,
            )
            aggregate["mean_evaluations_to_target"] = float(target_evals.mean())
            aggregate["median_evaluations_to_target"] = float(np.median(target_evals))
            aggregate["mean_time_to_target_seconds"] = float(target_times.mean())
            aggregate["median_time_to_target_seconds"] = float(np.median(target_times))

    _write_json(output_dir / "aggregate.json", aggregate)

    print(json.dumps(aggregate, indent=2))
    return results
