import copy
import csv
import math
import time

import numpy as np
import torch
import torch.nn as nn

from objective import (
    BEST_KNOWN,
    N_JOBS,
    N_MACHINES,
    get_ta001,
    optimality_gap,
    pfsp_makespan,
)


# Settings

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_LAYERS = 4
HIDDEN_DIM = 64

BATCH_SIZE = 1024
EPOCHS = 3000
LR = 1e-4

VALIDATION_SIZE = 4096
VALIDATE_EVERY = 50

TEST_SIZE = 16384
MAX_GRAD_NORM = 5.0

T_START = 0.1
T_END = 0.001

SEED_START = 42
N_SEEDS = 10

TARGET_TOL = 1e-6

RESULTS_CSV = "pfsp_ta001_realnvp_kl_10seeds.csv"


# PFSP representation

def decode_permutation(y):
    return torch.argsort(
        y,
        dim=1,
    )


# RealNVP model

class CouplingLayer(nn.Module):
    def __init__(
        self,
        dimension,
        hidden_dim,
    ):
        super().__init__()

        if dimension % 2 != 0:
            raise ValueError(
                "This RealNVP implementation "
                "requires an even dimension."
            )

        self.half = dimension // 2

        self.scale_net = nn.Sequential(
            nn.Linear(
                self.half,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                self.half,
            ),
            nn.Tanh(),
        )

        self.translate_net = nn.Sequential(
            nn.Linear(
                self.half,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                self.half,
            ),
        )

    def forward(self, x):
        x1 = x[:, :self.half]
        x2 = x[:, self.half:]

        s = self.scale_net(x1)
        t = self.translate_net(x1)

        y2 = (
            x2 * torch.exp(s)
            + t
        )

        y = torch.cat(
            [x1, y2],
            dim=1,
        )

        log_det = s.sum(
            dim=1
        )

        return y, log_det

    def inverse(self, y):
        y1 = y[:, :self.half]
        y2 = y[:, self.half:]

        s = self.scale_net(y1)
        t = self.translate_net(y1)

        x2 = (
            y2 - t
        ) * torch.exp(-s)

        x = torch.cat(
            [y1, x2],
            dim=1,
        )

        log_det = -s.sum(
            dim=1
        )

        return x, log_det


class RealNVP(nn.Module):
    def __init__(
        self,
        dimension,
        num_layers,
        hidden_dim,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                CouplingLayer(
                    dimension,
                    hidden_dim,
                )
                for _ in range(
                    num_layers
                )
            ]
        )

        self.half = (
            dimension // 2
        )

    def _swap(self, x):
        return torch.cat(
            [
                x[:, self.half:],
                x[:, :self.half],
            ],
            dim=1,
        )

    def forward(self, x):
        y = x

        log_det_total = (
            x.new_zeros(
                x.shape[0]
            )
        )

        for layer in self.layers:
            y, log_det = layer(y)

            log_det_total = (
                log_det_total
                + log_det
            )

            y = self._swap(y)

        return (
            y,
            log_det_total,
        )

    def inverse(self, y):
        x = y

        log_det_total = (
            y.new_zeros(
                y.shape[0]
            )
        )

        for layer in reversed(
            self.layers
        ):
            x = self._swap(x)

            x, log_det = (
                layer.inverse(x)
            )

            log_det_total = (
                log_det_total
                + log_det
            )

        return (
            x,
            log_det_total,
        )


def gaussian_log_prob(z):
    return -0.5 * (
        z.square()
        + math.log(
            2.0 * math.pi
        )
    ).sum(
        dim=1
    )


def log_probability(
    model,
    y,
):
    z, inverse_log_det = (
        model.inverse(y)
    )

    return (
        gaussian_log_prob(z)
        + inverse_log_det
    )


# Annealed KL schedule

def temperature(epoch):
    if EPOCHS <= 1:
        return T_END

    progress = (
        epoch
        / (EPOCHS - 1)
    )

    return (
        T_START
        * (
            T_END
            / T_START
        ) ** progress
    )


# Training

def train(
    seed,
    verbose=True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    processing_times = get_ta001(
        device=DEVICE,
    )

    model = RealNVP(
        dimension=N_JOBS,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
    ).to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    validation_generator = (
        torch.Generator(
            device=DEVICE
        ).manual_seed(
            seed + 1000
        )
    )

    validation_z = torch.randn(
        VALIDATION_SIZE,
        N_JOBS,
        generator=(
            validation_generator
        ),
        device=DEVICE,
    )

    best_validation_mean = (
        float("inf")
    )

    best_validation_best = (
        float("inf")
    )

    best_state = None
    best_epoch = -1

    optimization_best_makespan = (
        float("inf")
    )

    optimization_best_permutation = (
        None
    )

    best_found_epoch = -1

    run_start = time.perf_counter()

    if verbose:
        print()
        print("=" * 78)
        print(
            f"PFSP Ta001 KL | "
            f"seed={seed} | "
            f"T={T_START}->{T_END}"
        )
        print("=" * 78)

        print(
            f"device={DEVICE} | "
            f"jobs={N_JOBS} | "
            f"machines={N_MACHINES} | "
            f"best_known={BEST_KNOWN}"
        )

        print(
            f"layers={NUM_LAYERS} | "
            f"hidden={HIDDEN_DIM} | "
            f"batch={BATCH_SIZE} | "
            f"epochs={EPOCHS}"
        )

    for epoch in range(
        EPOCHS
    ):
        z = torch.randn(
            BATCH_SIZE,
            N_JOBS,
            device=DEVICE,
        )

        # Keep differentiable for KL
        y, forward_log_det = (
            model(z)
        )

        with torch.no_grad():
            permutations = (
                decode_permutation(y)
            )

            makespans = (
                pfsp_makespan(
                    permutations,
                    processing_times,
                )
            )

            rewards = -makespans

            reward_sum = (
                rewards.sum()
            )

            baseline = (
                reward_sum
                - rewards
            ) / (
                BATCH_SIZE - 1
            )

            advantage = (
                rewards
                - baseline
            )

            advantage = (
                advantage
                - advantage.mean()
            ) / (
                advantage.std(
                    unbiased=False
                )
                + 1e-8
            )

        batch_best_index = int(
            torch.argmin(
                makespans
            ).item()
        )

        batch_best_makespan = (
            makespans[
                batch_best_index
            ].item()
        )

        if (
            batch_best_makespan
            < optimization_best_makespan
        ):
            optimization_best_makespan = (
                batch_best_makespan
            )

            optimization_best_permutation = (
                permutations[
                    batch_best_index
                ]
                .detach()
                .clone()
            )

            best_found_epoch = epoch

        # REINFORCE term

        log_prob = log_probability(
            model,
            y.detach(),
        )

        reinforce_loss = -(
            advantage.detach()
            * log_prob
        ).mean()

        # Annealed KL(q || N(0, I))

        log_q_y = (
            gaussian_log_prob(z)
            - forward_log_det
        )

        log_p0_y = (
            gaussian_log_prob(y)
        )

        kl = (
            log_q_y
            - log_p0_y
        ).mean()

        T = temperature(epoch)

        loss = (
            reinforce_loss
            + T * kl
        )

        optimizer.zero_grad()
        loss.backward()

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )
        )

        optimizer.step()

        # Validation checkpoint

        if (
            epoch
            % VALIDATE_EVERY
            == 0
            or epoch
            == EPOCHS - 1
        ):
            with torch.no_grad():
                val_y, _ = model(
                    validation_z
                )

                val_permutations = (
                    decode_permutation(
                        val_y
                    )
                )

                val_makespans = (
                    pfsp_makespan(
                        val_permutations,
                        processing_times,
                    )
                )

                val_mean = (
                    val_makespans
                    .mean()
                    .item()
                )

                val_best = (
                    val_makespans
                    .min()
                    .item()
                )

                best_validation_best = min(
                    best_validation_best,
                    val_best,
                )

                unique_fraction = (
                    torch.unique(
                        permutations,
                        dim=0,
                    ).shape[0]
                    / BATCH_SIZE
                )

                train_mean = (
                    makespans
                    .mean()
                    .item()
                )

                train_std = (
                    makespans
                    .std(
                        unbiased=False
                    )
                    .item()
                )

            if (
                val_mean
                < best_validation_mean
            ):
                best_validation_mean = (
                    val_mean
                )

                best_epoch = epoch

                best_state = (
                    copy.deepcopy(
                        model.state_dict()
                    )
                )

            if verbose:
                print(
                    f"{epoch:5d} | "
                    f"mean={train_mean:.3f} | "
                    f"best="
                    f"{optimization_best_makespan:.0f} | "
                    f"val_mean={val_mean:.3f} | "
                    f"val_best={val_best:.0f} | "
                    f"unique={unique_fraction:.3f} | "
                    f"std={train_std:.3f} | "
                    f"grad={float(grad_norm):.4f} | "
                    f"T={T:.6f} | "
                    f"KL={kl.item():.4f}"
                )

    # Test best checkpoint

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    test_generator = (
        torch.Generator(
            device=DEVICE
        ).manual_seed(
            seed + 2000
        )
    )

    test_z = torch.randn(
        TEST_SIZE,
        N_JOBS,
        generator=test_generator,
        device=DEVICE,
    )

    with torch.no_grad():
        test_y, _ = model(
            test_z
        )

        test_permutations = (
            decode_permutation(
                test_y
            )
        )

        test_makespans = (
            pfsp_makespan(
                test_permutations,
                processing_times,
            )
        )

        unique_permutations, counts = (
            torch.unique(
                test_permutations,
                dim=0,
                return_counts=True,
            )
        )

        mode_index = int(
            torch.argmax(
                counts
            ).item()
        )

        mode_permutation = (
            unique_permutations[
                mode_index
            ]
        )

        mode_makespan = (
            pfsp_makespan(
                mode_permutation,
                processing_times,
            )[0].item()
        )

        mode_fraction = (
            counts[
                mode_index
            ].item()
            / TEST_SIZE
        )

        test_best_index = int(
            torch.argmin(
                test_makespans
            ).item()
        )

        test_best_makespan = (
            test_makespans[
                test_best_index
            ].item()
        )

        test_best_permutation = (
            test_permutations[
                test_best_index
            ]
            .detach()
            .clone()
        )

        test_fraction_optimal = (
            (
                test_makespans
                <= (
                    BEST_KNOWN
                    + TARGET_TOL
                )
            )
            .float()
            .mean()
            .item()
        )

        test_unique_permutations = (
            unique_permutations
            .shape[0]
        )

    best_makespan_including_test = min(
        optimization_best_makespan,
        test_best_makespan,
    )

    if (
        optimization_best_makespan
        <= test_best_makespan
    ):
        best_permutation = (
            optimization_best_permutation
        )
    else:
        best_permutation = (
            test_best_permutation
        )

    optimum_hit = (
        best_makespan_including_test
        <= (
            BEST_KNOWN
            + TARGET_TOL
        )
    )

    gap_percent = float(
        optimality_gap(
            best_makespan_including_test
        ).item()
    )

    runtime_seconds = (
        time.perf_counter()
        - run_start
    )

    result = {
        "seed":
            seed,
        "best_validation_epoch":
            best_epoch,
        "best_validation_mean":
            best_validation_mean,
        "best_validation_best":
            best_validation_best,
        "test_mean_makespan":
            test_makespans.mean().item(),
        "test_best_makespan":
            test_best_makespan,
        "test_mode_makespan":
            mode_makespan,
        "test_mode_fraction":
            mode_fraction,
        "test_fraction_optimal":
            test_fraction_optimal,
        "test_unique_permutations":
            test_unique_permutations,
        "optimization_best_makespan":
            optimization_best_makespan,
        "best_found_epoch":
            best_found_epoch,
        "best_makespan_including_test":
            best_makespan_including_test,
        "gap_percent":
            gap_percent,
        "optimum_hit":
            optimum_hit,
        "runtime_seconds":
            runtime_seconds,
        "best_permutation":
            (
                best_permutation
                .cpu()
                .tolist()
            ),
    }

    if verbose:
        print()
        print("=" * 78)
        print("FINAL RESULT")
        print("=" * 78)

        print(
            "Best validation epoch:      "
            f"{best_epoch}"
        )

        print(
            "Best validation mean:       "
            f"{best_validation_mean:.3f}"
        )

        print(
            "Best validation makespan:   "
            f"{best_validation_best:.0f}"
        )

        print(
            "Test mean makespan:         "
            f"{result['test_mean_makespan']:.3f}"
        )

        print(
            "Test best makespan:         "
            f"{test_best_makespan:.0f}"
        )

        print(
            "Test mode makespan:         "
            f"{mode_makespan:.0f}"
        )

        print(
            "Test mode fraction:         "
            f"{100.0 * mode_fraction:.2f}%"
        )

        print(
            "Optimal test samples:       "
            f"{100.0 * test_fraction_optimal:.2f}%"
        )

        print(
            "Test unique permutations:   "
            f"{test_unique_permutations}"
        )

        print(
            "Optimization best makespan: "
            f"{optimization_best_makespan:.0f}"
        )

        print(
            "Best makespan incl. test:   "
            f"{best_makespan_including_test:.0f}"
        )

        print(
            "Optimality gap:             "
            f"{gap_percent:.4f}%"
        )

        print(
            "Optimum hit:                "
            f"{optimum_hit}"
        )

        print(
            "Runtime:                    "
            f"{runtime_seconds:.1f}s"
        )

        print(
            "Best permutation:           "
            f"{result['best_permutation']}"
        )

    return result


# Run 10 seeds

def run_many_seeds():
    seeds = list(
        range(
            SEED_START,
            SEED_START + N_SEEDS,
        )
    )

    results = []

    print("=" * 78)
    print(
        "REALNVP PFSP TA001 | "
        "ANNEALED KL | "
        "10 SEEDS"
    )
    print("=" * 78)

    print(
        f"device={DEVICE} | "
        f"seeds={seeds[0]}-"
        f"{seeds[-1]} | "
        f"layers={NUM_LAYERS} | "
        f"hidden={HIDDEN_DIM}"
    )

    print(
        f"batch={BATCH_SIZE} | "
        f"epochs={EPOCHS} | "
        f"lr={LR} | "
        f"T={T_START}->{T_END}"
    )

    for index, seed in enumerate(
        seeds,
        start=1,
    ):
        print()
        print(
            f"[{index:02d}/"
            f"{len(seeds):02d}] "
            f"seed={seed}"
        )

        result = train(
            seed,
            verbose=True,
        )

        results.append(
            result
        )

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    best_makespans = np.array(
        [
            result[
                "best_makespan_including_test"
            ]
            for result in results
        ],
        dtype=float,
    )

    test_means = np.array(
        [
            result[
                "test_mean_makespan"
            ]
            for result in results
        ],
        dtype=float,
    )

    test_bests = np.array(
        [
            result[
                "test_best_makespan"
            ]
            for result in results
        ],
        dtype=float,
    )

    gaps = np.array(
        [
            result[
                "gap_percent"
            ]
            for result in results
        ],
        dtype=float,
    )

    runtimes = np.array(
        [
            result[
                "runtime_seconds"
            ]
            for result in results
        ],
        dtype=float,
    )

    optimum_hits = sum(
        int(
            result[
                "optimum_hit"
            ]
        )
        for result in results
    )

    final_mode_optimum_hits = sum(
        int(
            result[
                "test_mode_makespan"
            ]
            <= (
                BEST_KNOWN
                + TARGET_TOL
            )
        )
        for result in results
    )

    mean_optimal_test_share = (
        100.0
        * np.mean(
            [
                result[
                    "test_fraction_optimal"
                ]
                for result in results
            ]
        )
    )

    print()
    print("=" * 78)
    print("10-SEED SUMMARY")
    print("=" * 78)

    print(
        "Mean best makespan:          "
        f"{best_makespans.mean():.3f}"
    )

    print(
        "Std best makespan:           "
        f"{best_makespans.std(ddof=1):.3f}"
    )

    print(
        "Median best makespan:        "
        f"{np.median(best_makespans):.3f}"
    )

    print(
        "Best run:                    "
        f"{best_makespans.min():.0f}"
    )

    print(
        "Worst run:                   "
        f"{best_makespans.max():.0f}"
    )

    print(
        "Mean optimality gap:         "
        f"{gaps.mean():.4f}%"
    )

    print(
        "Optimum hits:                "
        f"{optimum_hits}/"
        f"{len(results)}"
    )

    print(
        "Final mode = optimum:        "
        f"{final_mode_optimum_hits}/"
        f"{len(results)}"
    )

    print(
        "Mean final generator:        "
        f"{test_means.mean():.3f}"
    )

    print(
        "Std final generator:         "
        f"{test_means.std(ddof=1):.3f}"
    )

    print(
        "Mean final test best:        "
        f"{test_bests.mean():.3f}"
    )

    print(
        "Mean optimal test share:     "
        f"{mean_optimal_test_share:.2f}%"
    )

    print(
        "Mean runtime:                "
        f"{runtimes.mean():.1f}s"
    )

    print(
        "Saved results to:            "
        f"{RESULTS_CSV}"
    )


if __name__ == "__main__":
    run_many_seeds()