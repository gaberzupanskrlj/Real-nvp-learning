import time
from dataclasses import dataclass

import ioh
import numpy as np


@dataclass(frozen=True)
class EAConfig:
    problem_name: str
    dimension: int = 100
    problem_instance: int = 1
    max_evaluations: int = 4_096_000
    seeds: tuple[int, ...] = tuple(range(42, 52))
    target: float | None = None
    progress_every: int = 0


def create_problem(config: EAConfig):
    return ioh.get_problem(
        config.problem_name,
        instance=config.problem_instance,
        dimension=config.dimension,
        problem_class=ioh.ProblemClass.PBO,
    )


def run_one_seed(config: EAConfig, seed: int):
    rng = np.random.default_rng(seed)
    problem = create_problem(config)
    problem.reset()

    x = rng.integers(
        0,
        2,
        size=config.dimension,
        dtype=np.int32,
    )

    fitness = float(problem(x.tolist()))
    best = fitness
    evaluations = 1
    started = time.perf_counter()

    while evaluations < config.max_evaluations:
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

        if (
            config.target is not None
            and best >= config.target
        ):
            break

        if (
            config.progress_every
            and evaluations % config.progress_every == 0
        ):
            elapsed = time.perf_counter() - started
            rate = evaluations / max(elapsed, 1e-9)
            print(
                f"seed={seed} | evals={evaluations:,} | "
                f"best={best:.6f} | {rate:,.0f} eval/s"
            )

    elapsed = time.perf_counter() - started

    return {
        "seed": seed,
        "best": best,
        "evaluations": evaluations,
        "time": elapsed,
    }


def run_experiment(config: EAConfig):
    results = []

    for index, seed in enumerate(config.seeds, start=1):
        result = run_one_seed(config, seed)
        results.append(result)

        print(
            f"FINISHED {index:2d}/{len(config.seeds)} | "
            f"seed={seed} | "
            f"best={result['best']:.6f} | "
            f"evals={result['evaluations']:,} | "
            f"time={result['time']:.1f}s"
        )

    values = np.array(
        [r["best"] for r in results],
        dtype=float,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Mean best objective: {values.mean()}")
    print(f"Median best objective: {np.median(values)}")
    print(f"Std best objective: {values.std(ddof=1) if len(values) > 1 else 0.0}")
    print(f"Best run: {values.max()}")
    print(f"Worst run: {values.min()}")

    print("\nPer-run results:")
    for result in results:
        print(
            f"seed={result['seed']} | "
            f"best={result['best']:.6f} | "
            f"evals={result['evaluations']:,} | "
            f"time={result['time']:.1f}s"
        )

    return results
