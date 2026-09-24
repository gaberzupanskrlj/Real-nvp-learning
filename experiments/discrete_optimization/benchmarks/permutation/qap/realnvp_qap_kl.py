import copy
import csv
from pathlib import Path

import numpy as np
import torch

from objective import (
    OPTIMUM,
    get_nug20,
    optimality_gap,
    qap_cost,
)

from realnvp_qap_baseline import (
    RealNVP,
    discretize,
    gaussian_log_prob,
    log_probability,
    set_seed,
)


# ============================================================
# Settings
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DIMENSION = 20

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


# ============================================================
# KL temperature
# ============================================================

T_START = 0.1
T_END = 0.001

RESULTS_CSV = Path(__file__).with_name(
    "qap20_realnvp_kl_10seeds.csv"
)


def kl_temperature(
    epoch: int,
) -> float:
    """
    Geometric annealing:
        epoch 1      -> 0.1
        epoch 3000   -> 0.001
    """

    if EPOCHS <= 1:
        return T_END

    progress = (
        (epoch - 1)
        / (EPOCHS - 1)
    )

    return (
        T_START
        * (T_END / T_START) ** progress
    )


@torch.no_grad()
def sample_model(
    model: RealNVP,
    sample_size: int,
):

    z = torch.randn(
        sample_size,
        DIMENSION,
        device=DEVICE,
    )

    y, _ = model(z)

    return discretize(y)


def first_index_equal(
    values: torch.Tensor,
    target: float,
):

    indices = torch.nonzero(
        values == target,
        as_tuple=False,
    )

    if len(indices) == 0:
        return None

    return indices[0].item()


def train_one_seed(
    seed: int,
):

    set_seed(seed)

    flow, distance = get_nug20(
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

    best_cost = float("inf")
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

        y, forward_log_det = model(z)

        with torch.no_grad():

            permutations = discretize(y)

            costs = qap_cost(
                permutations,
                flow,
                distance,
            )

        evaluations_before_batch = (
            objective_evaluations
        )

        objective_evaluations += BATCH_SIZE

        batch_best_cost = costs.min().item()

        if batch_best_cost < best_cost:

            index = first_index_equal(
                costs,
                batch_best_cost,
            )

            best_cost = batch_best_cost
            best_gap = optimality_gap(
                best_cost
            ).item()

            best_found_eval = (
                evaluations_before_batch
                + index
                + 1
            )

        if not target_hit:

            target_index = first_index_equal(
                costs,
                OPTIMUM,
            )

            if target_index is not None:

                target_hit = True

                target_eval = (
                    evaluations_before_batch
                    + target_index
                    + 1
                )

        # ----------------------------------------------------
        # REINFORCE term
        # ----------------------------------------------------

        rewards = -costs

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

        reinforce_loss = -(
            advantage.detach()
            * log_prob
        ).mean()

        # ----------------------------------------------------
        # KL(q_theta(y) || N(0,I))
        # ----------------------------------------------------
        #
        # q(y) = p(z) / |det J_f(z)|
        #
        # log q(y) = log p(z) - log|det J_f(z)|
        #
        # y is not detached in the KL term so gradients flow
        # through the transformation.
        #

        log_q_y = (
            gaussian_log_prob(z)
            - forward_log_det
        )

        log_standard_normal_y = (
            gaussian_log_prob(y)
        )

        kl_loss = (
            log_q_y
            - log_standard_normal_y
        ).mean()

        temperature = kl_temperature(
            epoch
        )

        loss = (
            reinforce_loss
            + temperature * kl_loss
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer.step()

        if epoch % VALIDATE_EVERY == 0:

            model.eval()

            with torch.no_grad():

                validation_permutations = sample_model(
                    model,
                    VALIDATION_SIZE,
                )

                validation_costs = qap_cost(
                    validation_permutations,
                    flow,
                    distance,
                )

            validation_before = (
                objective_evaluations
            )

            objective_evaluations += VALIDATION_SIZE

            validation_mean = (
                validation_costs
                .mean()
                .item()
            )

            validation_best = (
                validation_costs
                .min()
                .item()
            )

            if validation_best < best_cost:

                index = first_index_equal(
                    validation_costs,
                    validation_best,
                )

                best_cost = validation_best

                best_gap = optimality_gap(
                    best_cost
                ).item()

                best_found_eval = (
                    validation_before
                    + index
                    + 1
                )

            if not target_hit:

                target_index = first_index_equal(
                    validation_costs,
                    OPTIMUM,
                )

                if target_index is not None:

                    target_hit = True

                    target_eval = (
                        validation_before
                        + target_index
                        + 1
                    )

            # Same checkpoint criterion as baseline.
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
                f" | T={temperature:.6f}"
                f" | loss={loss.item(): .6f}"
                f" | reinforce={reinforce_loss.item(): .6f}"
                f" | KL={kl_loss.item(): .6f}"
                f" | mean={costs.mean().item():.2f}"
                f" | best={best_cost:.0f}"
                f" | gap={best_gap:.4f}%"
                f" | evals={objective_evaluations:,}"
            )

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():

        test_permutations = sample_model(
            model,
            TEST_SIZE,
        )

        test_costs = qap_cost(
            test_permutations,
            flow,
            distance,
        )

    final_mean_cost = test_costs.mean().item()
    final_best_cost = test_costs.min().item()

    final_mean_gap = optimality_gap(
        final_mean_cost
    ).item()

    final_best_gap = optimality_gap(
        final_best_cost
    ).item()

    final_optimum_fraction = (
        (
            test_costs
            == OPTIMUM
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
        final_best_cost
        - best_cost
    )

    return {
        "seed": seed,
        "best_cost": best_cost,
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
        "final_mean_cost": final_mean_cost,
        "final_mean_gap_percent": final_mean_gap,
        "final_best_cost": final_best_cost,
        "final_best_gap_percent": final_best_gap,
        "retention_loss": retention_loss,
        "final_optimum_fraction": final_optimum_fraction,
        "final_unique_permutations": unique_permutations,
        "final_unique_fraction": unique_fraction,
        "training_objective_evaluations": objective_evaluations,
    }


def main():

    print("=" * 72)
    print("REALNVP + ANNEALED KL — QAPLIB NUG20")
    print("=" * 72)

    print()
    print(f"device: {DEVICE}")
    print(f"dimension: {DIMENSION}")
    print(f"known optimum: {OPTIMUM}")

    print()
    print(
        f"architecture: "
        f"{NUM_LAYERS} layers, "
        f"hidden={HIDDEN_DIM}"
    )

    print(f"batch size: {BATCH_SIZE}")
    print(f"epochs: {EPOCHS}")
    print(f"learning rate: {LR}")

    print()
    print(
        f"KL temperature: "
        f"{T_START} -> {T_END}"
    )
    print("KL schedule: geometric")

    print()
    print(
        f"validation size: "
        f"{VALIDATION_SIZE}"
    )
    print(
        f"test size: "
        f"{TEST_SIZE}"
    )

    print(
        f"seeds: "
        f"{SEED_START}..."
        f"{SEED_START + N_SEEDS - 1}"
    )

    print()

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

        print()

        print(
            f"    best="
            f"{result['best_cost']:.0f}"
            f" | gap="
            f"{result['best_gap_percent']:.4f}%"
            f" | best eval="
            f"{result['best_found_eval']}"
            f" | target="
            f"{result['target_hit']}"
        )

        print(
            f"    final mean="
            f"{result['final_mean_cost']:.3f}"
            f" | final best="
            f"{result['final_best_cost']:.0f}"
            f" | retention loss="
            f"{result['retention_loss']:.0f}"
        )

        print(
            f"    P(optimum)="
            f"{result['final_optimum_fraction']:.6f}"
            f" | unique="
            f"{result['final_unique_permutations']}"
            f"/{TEST_SIZE}"
            f" "
            f"({result['final_unique_fraction']:.6f})"
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

    best_costs = np.array(
        [
            r["best_cost"]
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
            r["final_mean_cost"]
            for r in results
        ],
        dtype=float,
    )

    final_best = np.array(
        [
            r["final_best_cost"]
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
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print()
    print(
        "Mean best cost:",
        best_costs.mean(),
    )
    print(
        "Median best cost:",
        np.median(best_costs),
    )
    print(
        "Std best cost:",
        best_costs.std(),
    )

    print()
    print(
        "Mean best gap:",
        best_gaps.mean(),
        "%",
    )
    print(
        "Median best gap:",
        np.median(best_gaps),
        "%",
    )

    print()
    print(
        "Best run:",
        best_costs.min(),
    )
    print(
        "Worst run:",
        best_costs.max(),
    )

    print()
    print(
        f"Global optimum hits: "
        f"{target_hits}/{N_SEEDS}"
    )

    print()
    print(
        "Mean final generator cost:",
        final_means.mean(),
    )
    print(
        "Mean final generator best:",
        final_best.mean(),
    )

    print()
    print(
        "Mean retention loss:",
        retention_losses.mean(),
    )
    print(
        "Median retention loss:",
        np.median(retention_losses),
    )

    print()
    print(
        "Mean final unique permutations:",
        unique_counts.mean(),
    )
    print(
        "Median final unique permutations:",
        np.median(unique_counts),
    )

    print()
    print("Results saved to:")
    print(RESULTS_CSV)


if __name__ == "__main__":
    main()
