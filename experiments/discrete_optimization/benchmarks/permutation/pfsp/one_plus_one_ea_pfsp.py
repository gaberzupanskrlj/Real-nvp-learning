import csv
from pathlib import Path

import numpy as np
import torch

from objective import (
    N_JOBS,
    N_MACHINES,
    BEST_KNOWN,
    get_ta001,
    optimality_gap,
    pfsp_makespan,
)


# Settings

SEED_START = 42
N_SEEDS = 10

# Budget matched to RealNVP: 3000 epochs x 1024 + 60 validations x 4096
MAX_EVALS = 3000 * 1024 + (3000 // 50) * 4096

# Each child applies 1 + Poisson(EXTRA_MOVES_MEAN) random insertion moves.
# EXTRA_MOVES_MEAN = 0 gives randomized local search with a single move.
EXTRA_MOVES_MEAN = 1.0

CHECKPOINT_EVALS = (1_000, 10_000, 100_000, 1_000_000, MAX_EVALS)

RESULTS_CSV = Path(__file__).with_name(
    "pfsp_ta001_one_plus_one_ea_10seeds.csv"
)


# Single-schedule makespan, same recurrence as objective.pfsp_makespan

def makespan(permutation, times):
    completion = [0] * N_MACHINES

    for job in permutation:
        row = times[job]
        completion[0] += row[0]

        for machine in range(1, N_MACHINES):
            completion[machine] = (
                max(completion[machine], completion[machine - 1])
                + row[machine]
            )

    return completion[-1]


# Check the evaluator against objective.py

def check_evaluator(times):
    assert makespan(list(range(N_JOBS)), times) == 1448

    rng = np.random.default_rng(0)
    permutations = np.array(
        [rng.permutation(N_JOBS) for _ in range(1000)]
    )

    expected = pfsp_makespan(
        torch.from_numpy(permutations),
        get_ta001(),
    ).tolist()

    for permutation, value in zip(permutations.tolist(), expected):
        assert makespan(permutation, times) == value


# Insertion move: remove job at position i, reinsert at position j != i

def insertion_move(permutation, rng):
    i = int(rng.integers(N_JOBS))
    j = int(rng.integers(N_JOBS - 1))

    if j >= i:
        j += 1

    job = permutation.pop(i)
    permutation.insert(j, job)


# (1+1)-EA for one seed

def run_seed(seed, times):
    rng = np.random.default_rng(seed)

    current = rng.permutation(N_JOBS).tolist()
    current_makespan = makespan(current, times)

    best_permutation = current.copy()
    best_makespan = current_makespan
    best_found_eval = 1
    target_eval = 1 if best_makespan == BEST_KNOWN else None

    checkpoints = {}

    for evaluation in range(2, MAX_EVALS + 1):
        child = current.copy()

        for _ in range(1 + rng.poisson(EXTRA_MOVES_MEAN)):
            insertion_move(child, rng)

        child_makespan = makespan(child, times)

        # Elitist acceptance; equal moves are accepted so the search can drift on plateaus
        if child_makespan <= current_makespan:
            current = child
            current_makespan = child_makespan

            if current_makespan < best_makespan:
                best_permutation = current.copy()
                best_makespan = current_makespan
                best_found_eval = evaluation

                if target_eval is None and best_makespan == BEST_KNOWN:
                    target_eval = evaluation

        if evaluation in CHECKPOINT_EVALS:
            checkpoints[evaluation] = best_makespan

            print(
                f"    evals={evaluation:>9,}"
                f" | best={best_makespan}"
                f" | gap={optimality_gap(best_makespan).item():.4f}%",
                flush=True,
            )

    # Re-evaluate the best schedule with the objective module
    assert pfsp_makespan(
        torch.tensor(best_permutation),
        get_ta001(),
    ).item() == best_makespan

    result = {
        "seed": seed,
        "best_makespan": best_makespan,
        "best_gap_percent": optimality_gap(best_makespan).item(),
        "best_found_eval": best_found_eval,
        "target_hit": target_eval is not None,
        "target_eval": target_eval if target_eval is not None else "",
        "training_objective_evaluations": MAX_EVALS,
    }

    for evals in CHECKPOINT_EVALS:
        result[f"best_at_{evals}"] = checkpoints[evals]

    result["best_permutation"] = " ".join(map(str, best_permutation))

    return result


# Run all seeds and save the results

def main():
    times = get_ta001().long().tolist()

    check_evaluator(times)

    print("(1+1)-EA — TAILLARD TA001 PFSP")
    print()
    print(f"best-known makespan: {BEST_KNOWN}")
    print(f"mutation: 1 + Poisson({EXTRA_MOVES_MEAN}) insertion moves")
    print(f"evaluations per seed: {MAX_EVALS:,}")
    print(f"seeds: {SEED_START}...{SEED_START + N_SEEDS - 1}")

    results = []

    for run_index in range(N_SEEDS):
        seed = SEED_START + run_index

        print()
        print(f"[{run_index + 1:02d}/{N_SEEDS:02d}] seed={seed}")

        result = run_seed(seed, times)
        results.append(result)

        print(
            f"    best={result['best_makespan']}"
            f" | gap={result['best_gap_percent']:.4f}%"
            f" | best eval={result['best_found_eval']}"
            f" | target={result['target_hit']}"
        )

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best_makespans = np.array([r["best_makespan"] for r in results], dtype=float)
    best_gaps = np.array([r["best_gap_percent"] for r in results], dtype=float)
    best_found_evals = np.array([r["best_found_eval"] for r in results], dtype=float)
    target_hits = sum(r["target_hit"] for r in results)

    print()
    print("SUMMARY")
    print()
    print("Mean best makespan:", best_makespans.mean())
    print("Median best makespan:", np.median(best_makespans))
    print("Std best makespan:", best_makespans.std())
    print("Mean best gap:", best_gaps.mean(), "%")
    print("Best run:", best_makespans.min())
    print("Worst run:", best_makespans.max())
    print(f"Best-known hits: {target_hits}/{N_SEEDS}")
    print("Mean best-found eval:", best_found_evals.mean())

    for evals in CHECKPOINT_EVALS:
        values = np.array([r[f"best_at_{evals}"] for r in results], dtype=float)
        print(f"Mean best at {evals:>9,} evals:", values.mean())

    print()
    print("Results saved to:")
    print(RESULTS_CSV)


if __name__ == "__main__":
    main()
