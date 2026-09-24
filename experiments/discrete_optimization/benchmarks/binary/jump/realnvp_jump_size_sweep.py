import copy
import math
import sys
from pathlib import Path

import torch

from objective import jump_fitness


# Import existing RealNVP architecture from the IOH benchmark.
IOH_REALNVP_DIR = (
    Path(__file__).resolve().parents[1]
    / "ioh"
    / "realnvp"
)

sys.path.insert(0, str(IOH_REALNVP_DIR))

from common_100seed import RealNVP, discretize, log_probability  # noqa: E402


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DIMENSION = 100

NUM_LAYERS = 4
HIDDEN_DIM = 64

BATCH_SIZE = 1024
LEARNING_RATE = 1e-4

MAX_EVALUATIONS = 4_096_000

VALIDATION_SIZE = 4096
VALIDATE_EVERY = 50

TEST_SIZE = 16384
MAX_GRAD_NORM = 5.0

SEED_START = 42
N_SEEDS = 10

JUMP_SIZES = [2,4,5,6,7,8, 16, 25]
TARGET = 100.0


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_jump(x, jump_size):
    return jump_fitness(x, jump_size=jump_size)


# ---------------------------------------------------------------------
# Single seed
# ---------------------------------------------------------------------

def train_one_seed(seed, jump_size):
    torch.manual_seed(seed)

    model = RealNVP(
        DIMENSION,
        NUM_LAYERS,
        HIDDEN_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    validation_generator = torch.Generator(device=DEVICE)
    validation_generator.manual_seed(seed + 1000)

    validation = torch.randn(
        VALIDATION_SIZE,
        DIMENSION,
        generator=validation_generator,
        device=DEVICE,
    )

    best_validation_mean = float("-inf")
    best_state = None
    best_epoch = -1

    best_so_far = float("-inf")
    best_correct = 0

    evaluation_count = 0
    target_evaluation = None

    epoch = 0

    while evaluation_count + BATCH_SIZE <= MAX_EVALUATIONS:

        # -------------------------------------------------------------
        # Sample
        # -------------------------------------------------------------

        z = torch.randn(
            BATCH_SIZE,
            DIMENSION,
            device=DEVICE,
        )

        with torch.no_grad():
            y, _ = model(z)
            x = discretize(y)

            reward = evaluate_jump(x, jump_size)

            target_pattern = (
                torch.arange(
                    DIMENSION,
                    device=DEVICE,
                    dtype=torch.int32,
                )
                % 2
            )

            correct = (x == target_pattern).sum(dim=1)

        evaluation_count += BATCH_SIZE

        batch_best = reward.max().item()
        batch_max_correct = correct.max().item()

        if batch_best > best_so_far:
            best_so_far = batch_best

        if batch_max_correct > best_correct:
            best_correct = batch_max_correct

        if (
            target_evaluation is None
            and best_so_far >= TARGET
        ):
            target_evaluation = evaluation_count

        # -------------------------------------------------------------
        # REINFORCE
        # -------------------------------------------------------------

        with torch.no_grad():
            baseline = (
                reward.sum() - reward
            ) / (BATCH_SIZE - 1)

            advantage = reward - baseline

        log_prob = log_probability(
            model,
            y.detach(),
        )

        loss = -(
            advantage * log_prob
        ).mean()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer.step()

        # -------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------

        if (
            epoch % VALIDATE_EVERY == 0
            and evaluation_count + VALIDATION_SIZE
            <= MAX_EVALUATIONS
        ):
            with torch.no_grad():

                val_y, _ = model(validation)
                val_x = discretize(val_y)

                val_reward = evaluate_jump(val_x, jump_size)

                val_correct = (
                    val_x == target_pattern
                ).sum(dim=1)

            evaluation_count += VALIDATION_SIZE

            val_mean = val_reward.mean().item()
            val_best = val_reward.max().item()
            val_max_correct = val_correct.max().item()

            if val_best > best_so_far:
                best_so_far = val_best

            if val_max_correct > best_correct:
                best_correct = val_max_correct

            if (
                target_evaluation is None
                and best_so_far >= TARGET
            ):
                target_evaluation = evaluation_count

            if val_mean > best_validation_mean:
                best_validation_mean = val_mean
                best_epoch = epoch
                best_state = copy.deepcopy(
                    model.state_dict()
                )

            print(
                f"epoch={epoch:4d} | "
                f"evals={evaluation_count:8d} | "
                f"train_best={batch_best:6.1f} | "
                f"best={best_so_far:6.1f} | "
                f"max_correct={best_correct:3d}/100 | "
                f"val_mean={val_mean:8.3f} | "
                f"val_best={val_best:6.1f}"
            )

        if target_evaluation is not None:
            print(
                "\nGLOBAL OPTIMUM FOUND "
                f"after {target_evaluation:,} evaluations"
            )
            break

        epoch += 1

    # -----------------------------------------------------------------
    # Final generator test
    # -----------------------------------------------------------------

    if best_state is not None:
        model.load_state_dict(best_state)

    test_generator = torch.Generator(device=DEVICE)
    test_generator.manual_seed(seed + 2000)

    test_z = torch.randn(
        TEST_SIZE,
        DIMENSION,
        generator=test_generator,
        device=DEVICE,
    )

    with torch.no_grad():

        test_y, _ = model(test_z)
        test_x = discretize(test_y)

        test_reward = evaluate_jump(test_x, jump_size)

        target_pattern = (
            torch.arange(
                DIMENSION,
                device=DEVICE,
                dtype=torch.int32,
            )
            % 2
        )

        test_correct = (
            test_x == target_pattern
        ).sum(dim=1)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(f"seed:                  {seed}")
    print(f"best objective:        {best_so_far}")
    print(f"max correct bits:      {best_correct}/100")
    print(f"target evaluation:     {target_evaluation}")
    print(f"best checkpoint epoch: {best_epoch}")
    print(f"validation mean:       {best_validation_mean:.6f}")

    print()
    print(f"test mean:             {test_reward.mean().item():.6f}")
    print(f"test best:             {test_reward.max().item():.6f}")

    print(
        f"test max correct:      "
        f"{test_correct.max().item()}/100"
    )

    local_optimum = DIMENSION - jump_size

    local_optimum_fraction = (
        test_correct == local_optimum
    ).float().mean().item()

    optimum_fraction = (
        test_correct == 100
    ).float().mean().item()

    print(
        f"P(correct={local_optimum}):      "
        f"{local_optimum_fraction:.6f}"
    )

    print(
        f"P(correct=100):        "
        f"{optimum_fraction:.6f}"
    )


if __name__ == "__main__":
    for seed in range(SEED_START, SEED_START + N_SEEDS):
        train_one_seed(
            seed=seed,
            jump_size=25,
        )