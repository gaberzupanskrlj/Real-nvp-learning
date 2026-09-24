import csv

import numpy as np

import realnvp_tsp as tsp


# Settings

SEED_START = 42
N_SEEDS = 10
RESULTS_CSV = "tsp20_realnvp_10seeds.csv"


# Run multiple seeds

def main():
    seeds = list(range(SEED_START, SEED_START + N_SEEDS))
    results = []

    print("RealNVP TSP-20 multi-seed run")
    print(f"Seeds: {seeds[0]}-{seeds[-1]}")
    print(f"Cities: {tsp.N_CITIES}")
    print(f"Layers: {tsp.NUM_LAYERS}")
    print(f"Hidden dim: {tsp.HIDDEN_DIM}")
    print(f"Batch size: {tsp.BATCH_SIZE}")
    print(f"Epochs: {tsp.EPOCHS}")
    print(f"Learning rate: {tsp.LR}")
    print(f"Instance seed: {tsp.TSP_INSTANCE_SEED}")
    print()

    for i, seed in enumerate(seeds, start=1):
        print(f"[{i:02d}/{N_SEEDS:02d}] seed={seed} ...", flush=True)

        metrics = tsp.train(
            seed=seed,
            verbose=False,
            make_plots=False,
        )
        results.append(metrics)

        print(
            f"    best ever={metrics['best_length_ever']:.6f} | "
            f"test best={metrics['test_best_length']:.6f} | "
            f"test mean={metrics['test_mean_length']:.6f} | "
            f"best val epoch={metrics['best_validation_epoch']}"
        )

    fieldnames = [
        "seed",
        "best_validation_epoch",
        "best_validation_mean",
        "test_mean_length",
        "test_best_length",
        "best_length_ever",
        "best_tour",
    ]

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    best_ever = np.array([r["best_length_ever"] for r in results])
    test_best = np.array([r["test_best_length"] for r in results])
    test_mean = np.array([r["test_mean_length"] for r in results])

    print()
    print("SUMMARY")
    print(f"Best-ever mean ± std: {best_ever.mean():.6f} ± {best_ever.std(ddof=1):.6f}")
    print(f"Best-ever median:     {np.median(best_ever):.6f}")
    print(f"Best-ever min:        {best_ever.min():.6f}")
    print(f"Best-ever max:        {best_ever.max():.6f}")
    print(f"Test-best mean ± std: {test_best.mean():.6f} ± {test_best.std(ddof=1):.6f}")
    print(f"Test-mean mean ± std: {test_mean.mean():.6f} ± {test_mean.std(ddof=1):.6f}")
    print(f"Saved results to:     {RESULTS_CSV}")


if __name__ == "__main__":
    main()
