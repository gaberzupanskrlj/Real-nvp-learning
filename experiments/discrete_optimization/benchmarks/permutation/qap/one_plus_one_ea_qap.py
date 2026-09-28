import csv
from pathlib import Path

import numpy as np
import torch

from objective import (
    N,
    OPTIMUM,
    OPTIMAL_PERMUTATION,
    get_nug20,
    optimality_gap,
    qap_cost,
)


# Settings

SEED_START = 42
N_SEEDS = 10

# Budget matched to RealNVP: 3000 epochs x 1024 + 60 validations x 4096
MAX_EVALS = 3000 * 1024 + (3000 // 50) * 4096

# Each child applies 1 + Poisson(EXTRA_MOVES_MEAN) random swap moves.
# EXTRA_MOVES_MEAN = 0 gives randomized local search with a single move.
EXTRA_MOVES_MEAN = 1.0

CHECKPOINT_EVALS = (1_000, 10_000, 100_000, 1_000_000, MAX_EVALS)

RESULTS_CSV = Path(__file__).with_name(
    "qap20_one_plus_one_ea_10seeds.csv"
)


# Single-permutation cost, same convention as objective.qap_cost:
# permutation[i] = location assigned to facility i

def cost(permutation, flow, distance):
    return int(
        (flow * distance[np.ix_(permutation, permutation)]).sum()
    )


# Check the evaluator against objective.py

def check_evaluator(flow, distance):
    assert cost(OPTIMAL_PERMUTATION.numpy(), flow, distance) == OPTIMUM

    rng = np.random.default_rng(0)
    permutations = np.array(
        [rng.permutation(N) for _ in range(1000)]
    )

    torch_flow, torch_distance = get_nug20()

    expected = qap_cost(
        torch.from_numpy(permutations),
        torch_flow,
        torch_distance,
    ).tolist()

    for permutation, value in zip(permutations, expected):
        assert cost(permutation, flow, distance) == value


# Swap move: exchange the locations of facilities i != j

def swap_move(permutation, rng):
    i = int(rng.integers(N))
    j = int(rng.integers(N - 1))

    if j >= i:
        j += 1

    permutation[i], permutation[j] = permutation[j], permutation[i]


# (1+1)-EA for one seed

def run_seed(seed, flow, distance):
    rng = np.random.default_rng(seed)

    current = rng.permutation(N)
    current_cost = cost(current, flow, distance)

    best_permutation = current.copy()
    best_cost = current_cost
    best_found_eval = 1
    target_eval = 1 if best_cost == OPTIMUM else None

    checkpoints = {}

    for evaluation in range(2, MAX_EVALS + 1):
        child = current.copy()

        for _ in range(1 + rng.poisson(EXTRA_MOVES_MEAN)):
            swap_move(child, rng)

        child_cost = cost(child, flow, distance)

        # Elitist acceptance; equal moves are accepted so the search can drift on plateaus
        if child_cost <= current_cost:
            current = child
            current_cost = child_cost

            if current_cost < best_cost:
                best_permutation = current.copy()
                best_cost = current_cost
                best_found_eval = evaluation

                if target_eval is None and best_cost == OPTIMUM:
                    target_eval = evaluation

        if evaluation in CHECKPOINT_EVALS:
            checkpoints[evaluation] = best_cost

            print(
                f"    evals={evaluation:>9,}"
                f" | best={best_cost}"
                f" | gap={optimality_gap(best_cost).item():.4f}%",
                flush=True,
            )

    # Re-evaluate the best permutation with the objective module
    torch_flow, torch_distance = get_nug20()

    assert qap_cost(
        torch.from_numpy(best_permutation),
        torch_flow,
        torch_distance,
    ).item() == best_cost

    result = {
        "seed": seed,
        "best_cost": best_cost,
        "best_gap_percent": optimality_gap(best_cost).item(),
        "best_found_eval": best_found_eval,
        "target_hit": target_eval is not None,
        "target_eval": target_eval if target_eval is not None else "",
        "training_objective_evaluations": MAX_EVALS,
    }

    for evals in CHECKPOINT_EVALS:
        result[f"best_at_{evals}"] = checkpoints[evals]

    result["best_permutation"] = " ".join(map(str, best_permutation.tolist()))

    return result


# Run all seeds and save the results

def main():
    torch_flow, torch_distance = get_nug20()
    flow = torch_flow.long().numpy()
    distance = torch_distance.long().numpy()

    check_evaluator(flow, distance)

    print("(1+1)-EA — QAPLIB NUG20")
    print()
    print(f"optimum: {OPTIMUM}")
    print(f"mutation: 1 + Poisson({EXTRA_MOVES_MEAN}) swap moves")
    print(f"evaluations per seed: {MAX_EVALS:,}")
    print(f"seeds: {SEED_START}...{SEED_START + N_SEEDS - 1}")

    results = []

    for run_index in range(N_SEEDS):
        seed = SEED_START + run_index

        print()
        print(f"[{run_index + 1:02d}/{N_SEEDS:02d}] seed={seed}")

        result = run_seed(seed, flow, distance)
        results.append(result)

        print(
            f"    best={result['best_cost']}"
            f" | gap={result['best_gap_percent']:.4f}%"
            f" | best eval={result['best_found_eval']}"
            f" | target={result['target_hit']}"
        )

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best_costs = np.array([r["best_cost"] for r in results], dtype=float)
    best_gaps = np.array([r["best_gap_percent"] for r in results], dtype=float)
    best_found_evals = np.array([r["best_found_eval"] for r in results], dtype=float)
    target_hits = sum(r["target_hit"] for r in results)

    print()
    print("SUMMARY")
    print()
    print("Mean best cost:", best_costs.mean())
    print("Median best cost:", np.median(best_costs))
    print("Std best cost:", best_costs.std())
    print("Mean best gap:", best_gaps.mean(), "%")
    print("Best run:", best_costs.min())
    print("Worst run:", best_costs.max())
    print(f"Optimum hits: {target_hits}/{N_SEEDS}")
    print("Mean best-found eval:", best_found_evals.mean())

    for evals in CHECKPOINT_EVALS:
        values = np.array([r[f"best_at_{evals}"] for r in results], dtype=float)
        print(f"Mean best at {evals:>9,} evals:", values.mean())

    print()
    print("Results saved to:")
    print(RESULTS_CSV)


if __name__ == "__main__":
    main()
