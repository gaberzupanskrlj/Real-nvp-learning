import csv
from pathlib import Path

import numpy as np
import torch


# Settings

N_CITIES = 20
TSP_INSTANCE_SEED = 12345
REFERENCE_OPTIMUM = 3.513668846
TARGET_TOLERANCE = 1e-6

SEED_START = 42
N_SEEDS = 10

# Budget matched to RealNVP: 3000 epochs x 1024 + 60 validations x 4096
MAX_EVALS = 3000 * 1024 + (3000 // 50) * 4096

# Budget of the frozen TSP-20 RealNVP baseline: 2000 epochs x 1024 + 40 validations x 4096
TSP20_BASELINE_EVALS = 2000 * 1024 + (2000 // 50) * 4096

# Each child applies 1 + Poisson(EXTRA_MOVES_MEAN) random inversion moves.
# EXTRA_MOVES_MEAN = 0 gives randomized local search with a single move (2-opt).
EXTRA_MOVES_MEAN = 1.0

CHECKPOINT_EVALS = (1_000, 10_000, 100_000, 1_000_000, TSP20_BASELINE_EVALS, MAX_EVALS)

RESULTS_CSV = Path(__file__).with_name(
    "tsp20_one_plus_one_ea_10seeds.csv"
)

# Best-so-far trace: first evaluation, every improvement, end of budget
TRACE_CSV = Path(__file__).with_name(
    "tsp20_one_plus_one_ea_trace.csv"
)


# TSP problem, same instance as realnvp_tsp20_baseline.py

def create_tsp_instance(n_cities, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n_cities, 2))


def distance_matrix(cities):
    diff = cities[:, None, :] - cities[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2))


NEXT = np.roll(np.arange(N_CITIES), -1)


def tour_length(tour, distances):
    return distances[tour, tour[NEXT]].sum()


def gap_percent(length):
    return 100.0 * (length - REFERENCE_OPTIMUM) / REFERENCE_OPTIMUM


# Check the evaluator against the RealNVP scripts' tsp_length (float32 torch)

def torch_tsp_length(cities, tours):
    ordered_cities = cities[tours]
    next_cities = torch.roll(ordered_cities, shifts=-1, dims=1)
    return torch.linalg.vector_norm(ordered_cities - next_cities, dim=2).sum(dim=1)


def check_evaluator(cities, distances):
    rng = np.random.default_rng(0)
    tours = np.array([rng.permutation(N_CITIES) for _ in range(1000)])

    expected = torch_tsp_length(
        torch.tensor(cities, dtype=torch.float32),
        torch.from_numpy(tours),
    ).numpy()

    for tour, value in zip(tours, expected):
        assert abs(tour_length(tour, distances) - value) < 1e-4


# Inversion move: reverse the tour segment between positions i < j

def inversion_move(tour, rng):
    i = int(rng.integers(N_CITIES))
    j = int(rng.integers(N_CITIES - 1))

    if j >= i:
        j += 1

    if i > j:
        i, j = j, i

    tour[i:j + 1] = tour[i:j + 1][::-1]


# (1+1)-EA for one seed

def run_seed(seed, cities, distances):
    rng = np.random.default_rng(seed)

    current = rng.permutation(N_CITIES)
    current_length = tour_length(current, distances)

    best_tour = current.copy()
    best_length = current_length
    best_found_eval = 1
    target_eval = 1 if best_length <= REFERENCE_OPTIMUM + TARGET_TOLERANCE else None

    checkpoints = {}
    trace = [{"seed": seed, "evaluations": 1, "best_so_far": best_length}]

    for evaluation in range(2, MAX_EVALS + 1):
        child = current.copy()

        for _ in range(1 + rng.poisson(EXTRA_MOVES_MEAN)):
            inversion_move(child, rng)

        child_length = tour_length(child, distances)

        # Elitist acceptance, same as the QAP and PFSP EA
        if child_length <= current_length:
            current = child
            current_length = child_length

            if current_length < best_length - 1e-12:
                best_tour = current.copy()
                best_length = current_length
                best_found_eval = evaluation

                trace.append({"seed": seed, "evaluations": evaluation, "best_so_far": best_length})

                if target_eval is None and best_length <= REFERENCE_OPTIMUM + TARGET_TOLERANCE:
                    target_eval = evaluation

        if evaluation in CHECKPOINT_EVALS:
            checkpoints[evaluation] = best_length

            print(
                f"    evals={evaluation:>9,}"
                f" | best={best_length:.6f}"
                f" | gap={gap_percent(best_length):.4f}%",
                flush=True,
            )

    trace.append({"seed": seed, "evaluations": MAX_EVALS, "best_so_far": best_length})

    # Re-evaluate the best tour with the RealNVP scripts' length
    torch_length = torch_tsp_length(
        torch.tensor(cities, dtype=torch.float32),
        torch.from_numpy(best_tour[None, :]),
    ).item()

    assert abs(torch_length - best_length) < 1e-4

    result = {
        "seed": seed,
        "best_length": round(best_length, 6),
        "best_gap_percent": gap_percent(best_length),
        "best_found_eval": best_found_eval,
        "target_hit": target_eval is not None,
        "target_eval": target_eval if target_eval is not None else "",
        "training_objective_evaluations": MAX_EVALS,
    }

    for evals in CHECKPOINT_EVALS:
        result[f"best_at_{evals}"] = round(checkpoints[evals], 6)

    result["best_tour"] = " ".join(map(str, best_tour.tolist()))

    return result, trace


# Run all seeds and save the results

def main():
    cities = create_tsp_instance(N_CITIES, TSP_INSTANCE_SEED)
    distances = distance_matrix(cities)

    check_evaluator(cities, distances)

    print("(1+1)-EA — TSP-20")
    print()
    print(f"reference optimum: {REFERENCE_OPTIMUM}")
    print(f"mutation: 1 + Poisson({EXTRA_MOVES_MEAN}) inversion moves")
    print(f"evaluations per seed: {MAX_EVALS:,}")
    print(f"seeds: {SEED_START}...{SEED_START + N_SEEDS - 1}")

    results = []
    trace = []

    for run_index in range(N_SEEDS):
        seed = SEED_START + run_index

        print()
        print(f"[{run_index + 1:02d}/{N_SEEDS:02d}] seed={seed}")

        result, seed_trace = run_seed(seed, cities, distances)
        results.append(result)
        trace.extend(seed_trace)

        print(
            f"    best={result['best_length']:.6f}"
            f" | gap={result['best_gap_percent']:.4f}%"
            f" | best eval={result['best_found_eval']}"
            f" | target={result['target_hit']}"
        )

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with open(TRACE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)

    best_lengths = np.array([r["best_length"] for r in results], dtype=float)
    best_gaps = np.array([r["best_gap_percent"] for r in results], dtype=float)
    best_found_evals = np.array([r["best_found_eval"] for r in results], dtype=float)
    target_hits = sum(r["target_hit"] for r in results)

    print()
    print("SUMMARY")
    print()
    print("Mean best length:", best_lengths.mean())
    print("Median best length:", np.median(best_lengths))
    print("Std best length:", best_lengths.std())
    print("Mean best gap:", best_gaps.mean(), "%")
    print("Best run:", best_lengths.min())
    print("Worst run:", best_lengths.max())
    print(f"Optimum hits: {target_hits}/{N_SEEDS}")
    print("Mean best-found eval:", best_found_evals.mean())

    for evals in CHECKPOINT_EVALS:
        values = np.array([r[f"best_at_{evals}"] for r in results], dtype=float)
        print(f"Mean best at {evals:>9,} evals:", values.mean())

    print()
    print("Results saved to:")
    print(RESULTS_CSV)
    print(TRACE_CSV)


if __name__ == "__main__":
    main()
