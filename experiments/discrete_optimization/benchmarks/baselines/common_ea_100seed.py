import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import ioh
import numpy as np


@dataclass(frozen=True)
class EAConfig:
    problem_name: str
    output_dir: str
    dimension: int = 100
    problem_instance: int = 1
    max_evaluations: int = 4_096_000
    seed_start: int = 42
    n_seeds: int = 100
    workers: int = 24
    target: float | None = None


def create_problem(config):
    return ioh.get_problem(
        config.problem_name,
        instance=config.problem_instance,
        dimension=config.dimension,
        problem_class=ioh.ProblemClass.PBO,
    )


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


def run_one_seed(config, seed):
    rng = np.random.default_rng(seed)
    problem = create_problem(config)
    problem.reset()

    x = rng.integers(
        0,
        2,
        size=config.dimension,
        dtype=np.int32,
    )

    started = time.perf_counter()

    fitness = float(problem(x.tolist()))
    best = fitness
    evaluations = 1

    target_evaluation = None
    target_time = None

    convergence = [
        {
            "evaluations": evaluations,
            "best_so_far": best,
            "elapsed_seconds": time.perf_counter() - started,
        }
    ]

    if config.target is not None and best >= config.target:
        target_evaluation = evaluations
        target_time = convergence[-1]["elapsed_seconds"]

    while (
        evaluations < config.max_evaluations
        and target_evaluation is None
    ):
        child = x.copy()

        mutation_mask = (
            rng.random(config.dimension)
            < (1.0 / config.dimension)
        )

        child[mutation_mask] = 1 - child[mutation_mask]

        child_fitness = float(problem(child.tolist()))
        evaluations += 1

        if child_fitness >= fitness:
            x = child
            fitness = child_fitness

        if fitness > best:
            best = fitness
            elapsed = time.perf_counter() - started

            convergence.append(
                {
                    "evaluations": evaluations,
                    "best_so_far": best,
                    "elapsed_seconds": elapsed,
                }
            )

            if config.target is not None and best >= config.target:
                target_evaluation = evaluations
                target_time = elapsed

    elapsed = time.perf_counter() - started

    if convergence[-1]["evaluations"] != evaluations:
        convergence.append(
            {
                "evaluations": evaluations,
                "best_so_far": best,
                "elapsed_seconds": elapsed,
            }
        )

    result = {
        "seed": seed,
        "evaluations": evaluations,
        "best_found": best,
        "training_seconds": elapsed,
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

    result, convergence = run_one_seed(config, seed)

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
                    f"evals={result['evaluations']:,} "
                    f"time={result['training_seconds']:.1f}s"
                )

    results.sort(key=lambda row: row["seed"])

    _write_csv(
        output_dir / "summary.csv",
        results,
        [
            "seed",
            "evaluations",
            "best_found",
            "training_seconds",
            "target_evaluation",
            "target_time_seconds",
        ],
    )

    best_values = np.array(
        [r["best_found"] for r in results],
        dtype=float,
    )

    runtimes = np.array(
        [r["training_seconds"] for r in results],
        dtype=float,
    )

    evaluations = np.array(
        [r["evaluations"] for r in results],
        dtype=float,
    )

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
        "mean_evaluations_used": float(evaluations.mean()),
        "median_evaluations_used": float(np.median(evaluations)),
    }

    if config.target is not None:
        hits = [
            r for r in results
            if r["target_evaluation"] is not None
        ]

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

    _write_json(
        output_dir / "aggregate.json",
        aggregate,
    )

    print(json.dumps(aggregate, indent=2))

    return results
