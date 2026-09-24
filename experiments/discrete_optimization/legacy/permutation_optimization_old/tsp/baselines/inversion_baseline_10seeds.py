import csv

import numpy as np


# Settings

N_CITIES = 20
TSP_INSTANCE_SEED = 12345

SEED_START = 42
N_SEEDS = 10
MAX_EVALS = 2_048_000

TARGET = 3.513668846
RESULTS_CSV = "tsp20_inversion_baseline_10seeds.csv"


# TSP problem

def create_tsp_instance(n_cities, seed):
    rng = np.random.default_rng(seed)
    cities = rng.random((n_cities, 2))

    diff = cities[:, None, :] - cities[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))

    return cities, distances


def tsp_length(distances, tour):
    next_tour = np.roll(tour, -1)
    return distances[tour, next_tour].sum()


def inversion_delta(distances, tour, i, j):
    n = len(tour)

    if i == 0 and j == n - 1:
        return 0.0

    a = tour[i - 1]
    b = tour[i]
    c = tour[j]
    d = tour[(j + 1) % n]

    old_edges = distances[a, b] + distances[c, d]
    new_edges = distances[a, c] + distances[b, d]

    return new_edges - old_edges


# Baseline

def run_seed(seed, distances):
    rng = np.random.default_rng(seed)

    current_tour = rng.permutation(N_CITIES)
    current_length = tsp_length(distances, current_tour)

    best_tour = current_tour.copy()
    best_length = current_length
    best_found_eval = 1
    target_hit_eval = None

    for evaluation in range(2, MAX_EVALS + 1):
        i = rng.integers(0, N_CITIES)
        j = rng.integers(0, N_CITIES - 1)

        if j >= i:
            j += 1

        if i > j:
            i, j = j, i

        delta = inversion_delta(distances, current_tour, i, j)

        if delta <= 0.0:
            current_tour[i:j + 1] = current_tour[i:j + 1][::-1]
            current_length += delta

            if current_length < best_length - 1e-12:
                best_tour = current_tour.copy()
                best_length = current_length
                best_found_eval = evaluation

                if target_hit_eval is None and best_length <= TARGET + 1e-9:
                    target_hit_eval = evaluation

    best_length = tsp_length(distances, best_tour)

    return {
        "seed": seed,
        "best_length": best_length,
        "gap_percent": (best_length - TARGET) / TARGET * 100,
        "best_found_eval": best_found_eval,
        "target_hit": target_hit_eval is not None,
        "target_hit_eval": target_hit_eval,
        "best_tour": best_tour.tolist(),
    }


# Run multiple seeds

def main():
    _, distances = create_tsp_instance(N_CITIES, TSP_INSTANCE_SEED)
    seeds = list(range(SEED_START, SEED_START + N_SEEDS))
    results = []

    print("TSP-20 inversion baseline")
    print(f"Seeds: {seeds[0]}-{seeds[-1]}")
    print(f"Max candidate evaluations per seed: {MAX_EVALS}")
    print()

    for i, seed in enumerate(seeds, start=1):
        print(f"[{i:02d}/{N_SEEDS:02d}] seed={seed} ...", flush=True)

        result = run_seed(seed, distances)
        results.append(result)

        target_text = (
            str(result["target_hit_eval"])
            if result["target_hit"]
            else "no"
        )

        print(
            f"    best={result['best_length']:.6f} | "
            f"gap={result['gap_percent']:.4f}% | "
            f"best found eval={result['best_found_eval']} | "
            f"target hit eval={target_text}"
        )

    fieldnames = [
        "seed",
        "best_length",
        "gap_percent",
        "best_found_eval",
        "target_hit",
        "target_hit_eval",
        "best_tour",
    ]

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    best_lengths = np.array([r["best_length"] for r in results])
    best_found_evals = np.array([r["best_found_eval"] for r in results])
    target_hits = sum(r["target_hit"] for r in results)

    print()
    print("SUMMARY")
    print(f"Best length mean ± std: {best_lengths.mean():.6f} ± {best_lengths.std(ddof=1):.6f}")
    print(f"Best length median:     {np.median(best_lengths):.6f}")
    print(f"Best length min:        {best_lengths.min():.6f}")
    print(f"Best length max:        {best_lengths.max():.6f}")
    print(f"Global optimum hits:    {target_hits}/{N_SEEDS}")
    print(f"Best-found eval median: {np.median(best_found_evals):.0f}")
    print(f"Best-found eval mean:   {best_found_evals.mean():.1f}")
    print(f"Saved results to:       {RESULTS_CSV}")


if __name__ == "__main__":
    main()
