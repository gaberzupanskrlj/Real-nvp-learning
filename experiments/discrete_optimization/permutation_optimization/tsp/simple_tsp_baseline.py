"""Minimal permutation-optimization baseline on a fixed 10-city Euclidean TSP.

This script intentionally does not use RealNVP yet.  It establishes a clean
permutation-valued objective and a simple elitist baseline that we can later
compare against a RealNVP + argsort decoder.

Run from the repository root:

    python experiments/discrete_optimization/permutation_optimization/tsp/simple_tsp_baseline.py
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np


N_CITIES = 10
INSTANCE_SEED = 12345
DEFAULT_ALGORITHM_SEED = 42
DEFAULT_BUDGET = 10_000


def make_instance(n_cities: int = N_CITIES, seed: int = INSTANCE_SEED) -> np.ndarray:
    """Create one deterministic Euclidean TSP instance in the unit square."""
    rng = np.random.default_rng(seed)
    return rng.random((n_cities, 2))


def distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Return the full pairwise Euclidean distance matrix."""
    diff = cities[:, None, :] - cities[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def tour_length(tour: np.ndarray, distances: np.ndarray) -> float:
    """Length of a closed tour, including the edge back to the first city."""
    next_city = np.roll(tour, -1)
    return float(distances[tour, next_city].sum())


def inversion_mutation(tour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reverse one randomly selected segment of the permutation."""
    n = len(tour)
    i, j = np.sort(rng.choice(n, size=2, replace=False))

    child = tour.copy()
    child[i : j + 1] = child[i : j + 1][::-1]
    return child


def run_one_plus_one_ea(
    distances: np.ndarray,
    budget: int,
    seed: int,
) -> tuple[np.ndarray, float, list[tuple[int, float]]]:
    """Simple elitist (1+1)-EA using inversion mutation."""
    rng = np.random.default_rng(seed)

    parent = rng.permutation(len(distances))
    parent_length = tour_length(parent, distances)

    best_tour = parent.copy()
    best_length = parent_length
    convergence = [(1, best_length)]

    for evaluation in range(2, budget + 1):
        child = inversion_mutation(parent, rng)
        child_length = tour_length(child, distances)

        if child_length <= parent_length:
            parent = child
            parent_length = child_length

        if parent_length < best_length:
            best_tour = parent.copy()
            best_length = parent_length

        convergence.append((evaluation, best_length))

    return best_tour, best_length, convergence


def exact_optimum(distances: np.ndarray) -> tuple[np.ndarray, float]:
    """Brute-force optimum for this tiny TSP-10 sanity-check instance.

    City 0 is fixed in the first position to remove rotational duplicates.
    This leaves 9! = 362,880 tours, which is small enough for a one-off check.
    """
    n = len(distances)
    best_tour: np.ndarray | None = None
    best_length = float("inf")

    for suffix in itertools.permutations(range(1, n)):
        tour = np.fromiter((0, *suffix), dtype=np.int64, count=n)
        length = tour_length(tour, distances)

        if length < best_length:
            best_length = length
            best_tour = tour.copy()

    assert best_tour is not None
    return best_tour, best_length


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple TSP-10 permutation baseline")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_ALGORITHM_SEED)
    parser.add_argument(
        "--skip-exact",
        action="store_true",
        help="Skip the brute-force TSP-10 optimum calculation.",
    )
    args = parser.parse_args()

    cities = make_instance()
    distances = distance_matrix(cities)

    best_tour, best_length, _ = run_one_plus_one_ea(
        distances=distances,
        budget=args.budget,
        seed=args.seed,
    )

    print("=" * 72)
    print("SIMPLE TSP-10 PERMUTATION BASELINE")
    print("=" * 72)
    print(f"instance seed:   {INSTANCE_SEED}")
    print(f"algorithm seed:  {args.seed}")
    print(f"cities:          {len(cities)}")
    print(f"budget:          {args.budget:,} objective evaluations")
    print(f"EA best length:  {best_length:.6f}")
    print(f"EA best tour:    {best_tour.tolist()}")

    if not args.skip_exact:
        optimum_tour, optimum_length = exact_optimum(distances)
        gap = 100.0 * (best_length - optimum_length) / optimum_length

        print("-" * 72)
        print(f"exact optimum:   {optimum_length:.6f}")
        print(f"optimal tour:    {optimum_tour.tolist()}")
        print(f"optimality gap:  {gap:.3f}%")
        print(f"target reached:  {np.isclose(best_length, optimum_length, atol=1e-12)}")


if __name__ == "__main__":
    main()
