import copy
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from objective import (
    N_JOBS,
    BEST_KNOWN,
    get_ta001,
    optimality_gap,
    pfsp_makespan,
)


# Settings

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DIMENSION = N_JOBS

NUM_LAYERS = 4
HIDDEN_DIM = 64

BATCH_SIZE = 1024
EPOCHS = 3000
LR = 1e-4

VALIDATION_SIZE = 4096
VALIDATE_EVERY = 50

TEST_SIZE = 16384
MAX_GRAD_NORM = 5.0

SEED_START = 42
N_SEEDS = 10

LOG_EVERY = 100

RESULTS_CSV = Path(__file__).with_name(
    "pfsp20_realnvp_10seeds.csv"
)


# Reproducibility

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# RealNVP coupling layer

class CouplingLayer(nn.Module):

    def __init__(
        self,
        dimension: int,
        hidden_dim: int,
        flip: bool = False,
    ):
        super().__init__()

        assert dimension % 2 == 0

        self.half = dimension // 2
        self.flip = flip

        self.s_net = nn.Sequential(
            nn.Linear(self.half, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.half),
            nn.Tanh(),
        )

        self.t_net = nn.Sequential(
            nn.Linear(self.half, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.half),
        )

    def forward(self, x):

        if not self.flip:
            x_a = x[:, :self.half]
            x_b = x[:, self.half:]

            s = self.s_net(x_a)
            t = self.t_net(x_a)

            y_a = x_a
            y_b = x_b * torch.exp(s) + t

            y = torch.cat(
                [y_a, y_b],
                dim=1,
            )

        else:
            x_b = x[:, :self.half]
            x_a = x[:, self.half:]

            s = self.s_net(x_a)
            t = self.t_net(x_a)

            y_b = x_b * torch.exp(s) + t
            y_a = x_a

            y = torch.cat(
                [y_b, y_a],
                dim=1,
            )

        log_det = s.sum(dim=1)

        return y, log_det

    def inverse(self, y):

        if not self.flip:
            y_a = y[:, :self.half]
            y_b = y[:, self.half:]

            s = self.s_net(y_a)
            t = self.t_net(y_a)

            x_a = y_a
            x_b = (y_b - t) * torch.exp(-s)

            x = torch.cat(
                [x_a, x_b],
                dim=1,
            )

        else:
            y_b = y[:, :self.half]
            y_a = y[:, self.half:]

            s = self.s_net(y_a)
            t = self.t_net(y_a)

            x_b = (y_b - t) * torch.exp(-s)
            x_a = y_a

            x = torch.cat(
                [x_b, x_a],
                dim=1,
            )

        log_det_inverse = -s.sum(dim=1)

        return x, log_det_inverse


# RealNVP model

class RealNVP(nn.Module):

    def __init__(
        self,
        dimension: int,
        num_layers: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                CouplingLayer(
                    dimension=dimension,
                    hidden_dim=hidden_dim,
                    flip=(i % 2 == 1),
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, z):

        y = z

        total_log_det = torch.zeros(
            z.shape[0],
            device=z.device,
        )

        for layer in self.layers:
            y, log_det = layer(y)
            total_log_det += log_det

        return y, total_log_det

    def inverse(self, y):

        z = y

        total_log_det = torch.zeros(
            y.shape[0],
            device=y.device,
        )

        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            total_log_det += log_det

        return z, total_log_det


# Gaussian base distribution

def gaussian_log_prob(z):

    log_two_pi = math.log(
        2.0 * math.pi
    )

    return (
        -0.5
        * (
            z.pow(2)
            + log_two_pi
        )
    ).sum(dim=1)


# Probability under the flow

def log_probability(
    model,
    y,
):

    z, inverse_log_det = model.inverse(y)

    return (
        gaussian_log_prob(z)
        + inverse_log_det
    )


# Continuous random keys -> job permutation

def discretize(y):

    return torch.argsort(
        y,
        dim=1,
    )


# Sample permutations from the current model

@torch.no_grad()
def sample_model(
    model,
    sample_size,
):

    z = torch.randn(
        sample_size,
        DIMENSION,
        device=DEVICE,
    )

    y, _ = model(z)

    return discretize(y)


# Find the first occurrence of a value in a batch

def first_index_equal(
    values,
    target,
):

    indices = torch.nonzero(
        values == target,
        as_tuple=False,
    )

    if len(indices) == 0:
        return None

    return indices[0].item()


# Train one independent seed

def train_one_seed(seed):

    set_seed(seed)

    processing_times = get_ta001(
        device=DEVICE,
    )

    model = RealNVP(
        dimension=DIMENSION,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    best_makespan = float("inf")
    best_gap = float("inf")
    best_found_eval = None

    target_hit = False
    target_eval = None

    objective_evaluations = 0

    best_validation_mean = float("inf")
    best_validation_epoch = None

    best_state = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        z = torch.randn(
            BATCH_SIZE,
            DIMENSION,
            device=DEVICE,
        )

        y, _ = model(z)

        with torch.no_grad():

            permutations = discretize(y)

            makespans = pfsp_makespan(
                permutations,
                processing_times,
            )

        evaluations_before_batch = (
            objective_evaluations
        )

        objective_evaluations += BATCH_SIZE

        batch_best = makespans.min().item()

        if batch_best < best_makespan:

            index = first_index_equal(
                makespans,
                batch_best,
            )

            best_makespan = batch_best

            best_gap = optimality_gap(
                best_makespan
            ).item()

            best_found_eval = (
                evaluations_before_batch
                + index
                + 1
            )

        if not target_hit:

            target_index = first_index_equal(
                makespans,
                BEST_KNOWN,
            )

            if target_index is not None:

                target_hit = True

                target_eval = (
                    evaluations_before_batch
                    + target_index
                    + 1
                )

        # Leave-one-out REINFORCE advantage

        rewards = -makespans

        reward_sum = rewards.sum()

        baseline = (
            reward_sum - rewards
        ) / (
            BATCH_SIZE - 1
        )

        advantage = rewards - baseline

        advantage = (
            advantage
            - advantage.mean()
        ) / (
            advantage.std(
                unbiased=False
            )
            + 1e-8
        )

        log_prob = log_probability(
            model,
            y.detach(),
        )

        loss = -(
            advantage.detach()
            * log_prob
        ).mean()

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer.step()

        # Validation and checkpoint selection

        if epoch % VALIDATE_EVERY == 0:

            model.eval()

            with torch.no_grad():

                validation_permutations = sample_model(
                    model,
                    VALIDATION_SIZE,
                )

                validation_makespans = pfsp_makespan(
                    validation_permutations,
                    processing_times,
                )

            validation_before = (
                objective_evaluations
            )

            objective_evaluations += (
                VALIDATION_SIZE
            )

            validation_mean = (
                validation_makespans
                .mean()
                .item()
            )

            validation_best = (
                validation_makespans
                .min()
                .item()
            )

            if validation_best < best_makespan:

                index = first_index_equal(
                    validation_makespans,
                    validation_best,
                )

                best_makespan = validation_best

                best_gap = optimality_gap(
                    best_makespan
                ).item()

                best_found_eval = (
                    validation_before
                    + index
                    + 1
                )

            if not target_hit:

                target_index = first_index_equal(
                    validation_makespans,
                    BEST_KNOWN,
                )

                if target_index is not None:

                    target_hit = True

                    target_eval = (
                        validation_before
                        + target_index
                        + 1
                    )

            if validation_mean < best_validation_mean:

                best_validation_mean = validation_mean
                best_validation_epoch = epoch

                best_state = copy.deepcopy(
                    model.state_dict()
                )

        if (
            epoch % LOG_EVERY == 0
            or epoch == 1
        ):

            print(
                f"    epoch={epoch:4d}"
                f" | loss={loss.item(): .6f}"
                f" | batch mean={makespans.mean().item():.2f}"
                f" | best={best_makespan:.0f}"
                f" | gap={best_gap:.4f}%"
                f" | evals={objective_evaluations:,}"
            )

    # Evaluate the best validation checkpoint

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():

        test_permutations = sample_model(
            model,
            TEST_SIZE,
        )

        test_makespans = pfsp_makespan(
            test_permutations,
            processing_times,
        )

    final_mean_makespan = (
        test_makespans.mean().item()
    )

    final_best_makespan = (
        test_makespans.min().item()
    )

    final_mean_gap = optimality_gap(
        final_mean_makespan
    ).item()

    final_best_gap = optimality_gap(
        final_best_makespan
    ).item()

    final_optimum_fraction = (
        (
            test_makespans
            == BEST_KNOWN
        )
        .float()
        .mean()
        .item()
    )

    unique_permutations = (
        torch.unique(
            test_permutations,
            dim=0,
        )
        .shape[0]
    )

    unique_fraction = (
        unique_permutations
        / TEST_SIZE
    )

    retention_loss = (
        final_best_makespan
        - best_makespan
    )

    return {
        "seed": seed,
        "best_makespan": best_makespan,
        "best_gap_percent": best_gap,
        "best_found_eval": best_found_eval,
        "target_hit": target_hit,
        "target_eval": (
            target_eval
            if target_eval is not None
            else ""
        ),
        "best_validation_epoch": best_validation_epoch,
        "best_validation_mean": best_validation_mean,
        "final_mean_makespan": final_mean_makespan,
        "final_mean_gap_percent": final_mean_gap,
        "final_best_makespan": final_best_makespan,
        "final_best_gap_percent": final_best_gap,
        "retention_loss": retention_loss,
        "final_optimum_fraction": final_optimum_fraction,
        "final_unique_permutations": unique_permutations,
        "final_unique_fraction": unique_fraction,
        "training_objective_evaluations": objective_evaluations,
    }


# Run all seeds and save the results

def main():

    print("REALNVP — TAILLARD TA001 PFSP")
    print()

    print(f"device: {DEVICE}")
    print(f"dimension: {DIMENSION}")
    print(f"best-known makespan: {BEST_KNOWN}")

    print(
        f"architecture: "
        f"{NUM_LAYERS} layers, "
        f"hidden={HIDDEN_DIM}"
    )

    print(f"batch size: {BATCH_SIZE}")
    print(f"epochs: {EPOCHS}")
    print(f"learning rate: {LR}")
    print(f"validation size: {VALIDATION_SIZE}")
    print(f"test size: {TEST_SIZE}")

    print(
        f"seeds: "
        f"{SEED_START}..."
        f"{SEED_START + N_SEEDS - 1}"
    )

    results = []

    for run_index in range(N_SEEDS):

        seed = SEED_START + run_index

        print()
        print(
            f"[{run_index + 1:02d}/"
            f"{N_SEEDS:02d}] "
            f"seed={seed}"
        )

        result = train_one_seed(seed)

        results.append(result)

        print(
            f"    best="
            f"{result['best_makespan']:.0f}"
            f" | gap="
            f"{result['best_gap_percent']:.4f}%"
            f" | best eval="
            f"{result['best_found_eval']}"
            f" | target="
            f"{result['target_hit']}"
        )

        print(
            f"    final mean="
            f"{result['final_mean_makespan']:.3f}"
            f" | final best="
            f"{result['final_best_makespan']:.0f}"
            f" | retention loss="
            f"{result['retention_loss']:.0f}"
        )

        print(
            f"    unique="
            f"{result['final_unique_permutations']}"
            f"/{TEST_SIZE}"
        )

    fieldnames = list(
        results[0].keys()
    )

    with open(
        RESULTS_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    best_makespans = np.array(
        [
            r["best_makespan"]
            for r in results
        ],
        dtype=float,
    )

    best_gaps = np.array(
        [
            r["best_gap_percent"]
            for r in results
        ],
        dtype=float,
    )

    final_means = np.array(
        [
            r["final_mean_makespan"]
            for r in results
        ],
        dtype=float,
    )

    retention_losses = np.array(
        [
            r["retention_loss"]
            for r in results
        ],
        dtype=float,
    )

    unique_counts = np.array(
        [
            r["final_unique_permutations"]
            for r in results
        ],
        dtype=float,
    )

    target_hits = sum(
        r["target_hit"]
        for r in results
    )

    print()
    print("SUMMARY")
    print()

    print(
        "Mean best makespan:",
        best_makespans.mean(),
    )

    print(
        "Median best makespan:",
        np.median(best_makespans),
    )

    print(
        "Std best makespan:",
        best_makespans.std(),
    )

    print(
        "Mean best gap:",
        best_gaps.mean(),
        "%",
    )

    print(
        "Best run:",
        best_makespans.min(),
    )

    print(
        "Worst run:",
        best_makespans.max(),
    )

    print(
        f"Best-known hits: "
        f"{target_hits}/{N_SEEDS}"
    )

    print(
        "Mean final generator makespan:",
        final_means.mean(),
    )

    print(
        "Mean retention loss:",
        retention_losses.mean(),
    )

    print(
        "Mean final unique permutations:",
        unique_counts.mean(),
    )

    print()
    print("Results saved to:")
    print(RESULTS_CSV)


if __name__ == "__main__":
    main()