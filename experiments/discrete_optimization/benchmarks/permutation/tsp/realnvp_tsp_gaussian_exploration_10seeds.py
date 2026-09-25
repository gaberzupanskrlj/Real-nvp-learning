import csv
import json
import math
import statistics
import time

import realnvp_tsp_gaussian_exploration as experiment


# Experiment settings

EXPLORATION_FRACTIONS = [0.0, 0.10, 0.25]
SEEDS = list(range(42, 52))

REFERENCE_OPTIMUM = 3.513668846
TARGET_TOLERANCE = 1e-6

RESULTS_CSV = "tsp_gaussian_exploration_10seeds.csv"
SUMMARY_CSV = "tsp_gaussian_exploration_10seeds_summary.csv"
SUMMARY_JSON = "tsp_gaussian_exploration_10seeds_summary.json"


def configure_experiment(exploration_fraction):
    experiment.NUM_LAYERS = 4
    experiment.HIDDEN_DIM = 64

    experiment.BATCH_SIZE = 1024
    experiment.EPOCHS = 2000
    experiment.LR = 1e-4

    experiment.VALIDATION_SIZE = 4096
    experiment.VALIDATE_EVERY = 50
    experiment.TEST_SIZE = 16384
    experiment.MAX_GRAD_NORM = 5.0

    experiment.EXPLORATION_FRACTION = exploration_fraction

    experiment.N_EXPLORE = round(
        experiment.BATCH_SIZE
        * experiment.EXPLORATION_FRACTION
    )

    experiment.N_FLOW = (
        experiment.BATCH_SIZE
        - experiment.N_EXPLORE
    )

    experiment.ALPHA = (
        experiment.N_FLOW
        / experiment.BATCH_SIZE
    )

    experiment.EPSILON = (
        experiment.N_EXPLORE
        / experiment.BATCH_SIZE
    )


def mean_std(values):
    mean_value = statistics.mean(values)

    if len(values) > 1:
        std_value = statistics.stdev(values)
    else:
        std_value = 0.0

    return mean_value, std_value


def write_csv(filename, rows):
    if not rows:
        return

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    all_rows = []

    print("TSP-20 Gaussian exploration 10-seed sweep")
    print(
        f"Device: {experiment.DEVICE}"
    )
    print(
        "Fractions: "
        + ", ".join(
            str(value)
            for value in EXPLORATION_FRACTIONS
        )
    )
    print(
        f"Seeds: {SEEDS[0]}..{SEEDS[-1]}"
    )
    print()

    sweep_start = time.perf_counter()

    for exploration_fraction in EXPLORATION_FRACTIONS:
        configure_experiment(
            exploration_fraction
        )

        print(
            f"Exploration fraction "
            f"{experiment.EPSILON:.9f}"
        )

        for run_index, seed in enumerate(
            SEEDS,
            start=1,
        ):
            run_start = time.perf_counter()

            metrics = experiment.train(
                seed=seed,
                verbose=False,
                make_plots=False,
            )

            runtime_seconds = (
                time.perf_counter()
                - run_start
            )

            target_hit = (
                metrics["best_length_ever"]
                <= (
                    REFERENCE_OPTIMUM
                    + TARGET_TOLERANCE
                )
            )

            row = {
                "exploration_fraction_requested":
                    exploration_fraction,
                "exploration_fraction_actual":
                    metrics[
                        "exploration_fraction_actual"
                    ],
                "seed":
                    seed,
                "n_flow":
                    metrics["n_flow"],
                "n_explore":
                    metrics["n_explore"],
                "best_validation_epoch":
                    metrics[
                        "best_validation_epoch"
                    ],
                "best_validation_mean":
                    metrics[
                        "best_validation_mean"
                    ],
                "test_mean_length":
                    metrics[
                        "test_mean_length"
                    ],
                "test_best_length":
                    metrics[
                        "test_best_length"
                    ],
                "best_length_ever":
                    metrics[
                        "best_length_ever"
                    ],
                "best_flow_length_ever":
                    metrics[
                        "best_flow_length_ever"
                    ],
                "best_explore_length_ever":
                    metrics[
                        "best_explore_length_ever"
                    ],
                "final_mean_length":
                    metrics[
                        "final_mean_length"
                    ],
                "final_flow_mean_length":
                    metrics[
                        "final_flow_mean_length"
                    ],
                "final_explore_mean_length":
                    metrics[
                        "final_explore_mean_length"
                    ],
                "final_unique_all":
                    metrics[
                        "final_unique_all"
                    ],
                "final_unique_flow":
                    metrics[
                        "final_unique_flow"
                    ],
                "final_unique_explore":
                    metrics[
                        "final_unique_explore"
                    ],
                "final_rho_explore":
                    metrics[
                        "final_rho_explore"
                    ],
                "final_grad_norm":
                    metrics[
                        "final_grad_norm"
                    ],
                "target_hit":
                    target_hit,
                "best_source":
                    metrics[
                        "best_source"
                    ],
                "runtime_seconds":
                    runtime_seconds,
            }

            all_rows.append(row)

            print(
                f"  [{run_index:02d}/10] "
                f"seed={seed} | "
                f"test_mean="
                f"{row['test_mean_length']:.6f} | "
                f"test_best="
                f"{row['test_best_length']:.6f} | "
                f"unique_flow="
                f"{row['final_unique_flow']:.3f} | "
                f"hit="
                f"{'yes' if target_hit else 'no'} | "
                f"time="
                f"{runtime_seconds:.1f}s"
            )

        print()

    write_csv(
        RESULTS_CSV,
        all_rows,
    )

    summary_rows = []
    summary_json = {}

    for exploration_fraction in EXPLORATION_FRACTIONS:
        rows = [
            row
            for row in all_rows
            if math.isclose(
                row[
                    "exploration_fraction_requested"
                ],
                exploration_fraction,
                abs_tol=1e-12,
            )
        ]

        test_means = [
            row["test_mean_length"]
            for row in rows
        ]

        test_bests = [
            row["test_best_length"]
            for row in rows
        ]

        best_ever = [
            row["best_length_ever"]
            for row in rows
        ]

        unique_flow = [
            row["final_unique_flow"]
            for row in rows
        ]

        grad_norms = [
            row["final_grad_norm"]
            for row in rows
        ]

        runtimes = [
            row["runtime_seconds"]
            for row in rows
        ]

        (
            test_mean_mean,
            test_mean_std,
        ) = mean_std(test_means)

        (
            test_best_mean,
            test_best_std,
        ) = mean_std(test_bests)

        (
            best_ever_mean,
            best_ever_std,
        ) = mean_std(best_ever)

        (
            unique_flow_mean,
            unique_flow_std,
        ) = mean_std(unique_flow)

        (
            grad_norm_mean,
            grad_norm_std,
        ) = mean_std(grad_norms)

        (
            runtime_mean,
            runtime_std,
        ) = mean_std(runtimes)

        target_hits = sum(
            int(row["target_hit"])
            for row in rows
        )

        summary_row = {
            "exploration_fraction":
                exploration_fraction,
            "runs":
                len(rows),
            "target_hits":
                target_hits,
            "test_mean_mean":
                test_mean_mean,
            "test_mean_std":
                test_mean_std,
            "test_best_mean":
                test_best_mean,
            "test_best_std":
                test_best_std,
            "best_ever_mean":
                best_ever_mean,
            "best_ever_std":
                best_ever_std,
            "final_unique_flow_mean":
                unique_flow_mean,
            "final_unique_flow_std":
                unique_flow_std,
            "final_grad_norm_mean":
                grad_norm_mean,
            "final_grad_norm_std":
                grad_norm_std,
            "runtime_mean_seconds":
                runtime_mean,
            "runtime_std_seconds":
                runtime_std,
        }

        summary_rows.append(
            summary_row
        )

        summary_json[
            str(exploration_fraction)
        ] = summary_row

    write_csv(
        SUMMARY_CSV,
        summary_rows,
    )

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary_json,
            handle,
            indent=2,
        )

    elapsed = (
        time.perf_counter()
        - sweep_start
    )

    print("SUMMARY")
    print()

    for row in summary_rows:
        print(
            f"eps={row['exploration_fraction']:.2f} | "
            f"hits={row['target_hits']}/"
            f"{row['runs']} | "
            f"test mean="
            f"{row['test_mean_mean']:.6f} "
            f"+/- {row['test_mean_std']:.6f} | "
            f"test best="
            f"{row['test_best_mean']:.6f} "
            f"+/- {row['test_best_std']:.6f} | "
            f"unique flow="
            f"{row['final_unique_flow_mean']:.3f} "
            f"+/- "
            f"{row['final_unique_flow_std']:.3f}"
        )

    print()
    print(
        f"Total sweep time: {elapsed:.1f}s"
    )
    print(
        f"Per-run results: {RESULTS_CSV}"
    )
    print(
        f"Summary CSV: {SUMMARY_CSV}"
    )
    print(
        f"Summary JSON: {SUMMARY_JSON}"
    )


if __name__ == "__main__":
    main()
