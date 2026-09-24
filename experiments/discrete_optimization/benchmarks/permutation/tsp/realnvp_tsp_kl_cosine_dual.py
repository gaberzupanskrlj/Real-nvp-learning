import argparse
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# SETTINGS
# ============================================================

# Keep the TSP experiments on the same hardware setup as the
# existing cosine/elite and KL runs.
DEVICE = torch.device("cpu")

N_CITIES = 20
NUM_LAYERS = 8
HIDDEN_DIM = 64

BATCH_SIZE = 4096
EPOCHS = 5000

LR_START = 1e-4
LR_END = 1e-5

VALIDATION_SIZE = 4096
VALIDATE_EVERY = 50

TEST_SIZE = 16384
MAX_GRAD_NORM = 5.0

T_START = 0.1
T_END = 0.001

ELITE_FRACTION = 0.01
CHECKPOINT_EPS = 1e-8

SEED_START = 42
N_SEEDS = 10
TSP_INSTANCE_SEED = 12345

# Used only for evaluation/reporting, never for training or
# checkpoint selection.
REFERENCE_OPTIMUM = 3.513668846
TARGET_TOL = 1e-5


# ============================================================
# TSP PROBLEM
# ============================================================

def create_tsp_instance(n_cities, seed):
    rng = np.random.default_rng(seed)
    cities = rng.random((n_cities, 2))

    return torch.tensor(
        cities,
        dtype=torch.float32,
        device=DEVICE,
    )


def decode_permutation(y):
    return torch.argsort(y, dim=1)


@torch.no_grad()
def tsp_length(cities, tours):
    ordered_cities = cities[tours]

    next_cities = torch.roll(
        ordered_cities,
        shifts=-1,
        dims=1,
    )

    edge_lengths = torch.linalg.vector_norm(
        ordered_cities - next_cities,
        dim=2,
    )

    return edge_lengths.sum(dim=1)


@torch.no_grad()
def canonical_tours(tours):
    """
    Canonical representation of a symmetric closed TSP tour.

    Rotations and reversed versions of the same cycle are treated
    as the same tour.
    """
    n = tours.shape[1]

    start = (
        (tours == 0)
        .to(torch.int64)
        .argmax(dim=1)
    )

    positions = (
        start[:, None]
        + torch.arange(
            n,
            device=tours.device,
        )[None, :]
    ) % n

    canon = torch.gather(
        tours,
        1,
        positions,
    )

    reverse = torch.cat(
        [
            canon[:, :1],
            canon[:, 1:].flip(1),
        ],
        dim=1,
    )

    use_reverse = (
        canon[:, 1]
        > canon[:, -1]
    )[:, None]

    return torch.where(
        use_reverse,
        reverse,
        canon,
    )


@torch.no_grad()
def validation_statistics(lengths, tours):
    sorted_lengths = torch.sort(
        lengths
    ).values

    elite_k = max(
        1,
        int(
            ELITE_FRACTION
            * lengths.shape[0]
        ),
    )

    canonical = canonical_tours(
        tours
    )

    unique_fraction = (
        torch.unique(
            canonical,
            dim=0,
        ).shape[0]
        / tours.shape[0]
    )

    fraction_optimal = (
        (
            lengths
            <= REFERENCE_OPTIMUM
            + TARGET_TOL
        )
        .float()
        .mean()
        .item()
    )

    return {
        "mean": lengths.mean().item(),
        "elite_mean": (
            sorted_lengths[:elite_k]
            .mean()
            .item()
        ),
        "best": sorted_lengths[0].item(),
        "p95": torch.quantile(
            lengths,
            0.95,
        ).item(),
        "fraction_optimal": (
            fraction_optimal
        ),
        "unique_fraction": (
            unique_fraction
        ),
    }


# ============================================================
# REALNVP
# ============================================================

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

        self.half = (
            dimension // 2
        )

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

        inverse_log_det = (
            -s.sum(dim=1)
        )

        return (
            x,
            inverse_log_det,
        )


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

            log_det_total += (
                log_det
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

            log_det_total += (
                log_det
            )

        return (
            x,
            log_det_total,
        )


def log_probability(model, y):
    z, inverse_log_det = (
        model.inverse(y)
    )

    log_pz = -0.5 * (
        z.square()
        + math.log(
            2.0 * math.pi
        )
    ).sum(dim=1)

    return (
        log_pz
        + inverse_log_det
    )


# ============================================================
# SCHEDULES
# ============================================================

def schedule_progress(
    epoch,
    epochs,
):
    if epochs <= 1:
        return 1.0

    return (
        epoch
        / (epochs - 1)
    )


def temperature(
    epoch,
    epochs,
):
    """
    Cosine annealing:
        T_START -> T_END
    """
    progress = schedule_progress(
        epoch,
        epochs,
    )

    return (
        T_END
        + 0.5
        * (T_START - T_END)
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )


def learning_rate_at(
    epoch,
    epochs,
    lr_schedule,
):
    """
    constant:
        1e-4

    cosine:
        1e-4 -> 1e-5
    """
    if lr_schedule == "constant":
        return LR_START

    if lr_schedule != "cosine":
        raise ValueError(
            f"Unknown LR schedule: "
            f"{lr_schedule}"
        )

    progress = schedule_progress(
        epoch,
        epochs,
    )

    return (
        LR_END
        + 0.5
        * (LR_START - LR_END)
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def discovery_checkpoint_better(
    stats,
    best_elite_mean,
    best_mean,
):
    """
    Primary criterion:
        lower top-1% validation mean

    Tie-breaker:
        lower overall validation mean
    """
    elite_improved = (
        stats["elite_mean"]
        < best_elite_mean
        - CHECKPOINT_EPS
    )

    elite_tied = (
        abs(
            stats["elite_mean"]
            - best_elite_mean
        )
        <= CHECKPOINT_EPS
    )

    mean_improved_on_tie = (
        elite_tied
        and stats["mean"]
        < best_mean
        - CHECKPOINT_EPS
    )

    return (
        elite_improved
        or mean_improved_on_tie
    )


def concentration_checkpoint_better(
    stats,
    best_mean,
    best_p95,
):
    """
    Primary criterion:
        lower overall validation mean

    Tie-breaker:
        lower 95th-percentile tour length

    This checkpoint is the main one for the goal:
    "make almost every generated tour good".
    """
    mean_improved = (
        stats["mean"]
        < best_mean
        - CHECKPOINT_EPS
    )

    mean_tied = (
        abs(
            stats["mean"]
            - best_mean
        )
        <= CHECKPOINT_EPS
    )

    p95_improved_on_tie = (
        mean_tied
        and stats["p95"]
        < best_p95
        - CHECKPOINT_EPS
    )

    return (
        mean_improved
        or p95_improved_on_tie
    )


# ============================================================
# TEST EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_checkpoint(
    model,
    state_dict,
    test_z,
    cities,
):
    model.load_state_dict(
        state_dict
    )

    test_y, _ = model(
        test_z
    )

    test_tours = (
        decode_permutation(
            test_y
        )
    )

    test_lengths = tsp_length(
        cities,
        test_tours,
    )

    canonical = canonical_tours(
        test_tours
    )

    unique_tours, counts = (
        torch.unique(
            canonical,
            dim=0,
            return_counts=True,
        )
    )

    mode_index = int(
        torch.argmax(counts)
    )

    mode_tour = (
        unique_tours[
            mode_index
        ]
    )

    mode_length = tsp_length(
        cities,
        mode_tour[None],
    ).item()

    mode_fraction = (
        counts.max().item()
        / TEST_SIZE
    )

    best_index = int(
        torch.argmin(
            test_lengths
        )
    )

    best_length = (
        test_lengths[
            best_index
        ].item()
    )

    best_tour = (
        test_tours[
            best_index
        ].clone()
    )

    fraction_optimal = (
        (
            test_lengths
            <= REFERENCE_OPTIMUM
            + TARGET_TOL
        )
        .float()
        .mean()
        .item()
    )

    return {
        "test_mean_length": (
            test_lengths
            .mean()
            .item()
        ),
        "test_std_length": (
            test_lengths
            .std(
                unbiased=False
            )
            .item()
        ),
        "test_p95_length": (
            torch.quantile(
                test_lengths,
                0.95,
            )
            .item()
        ),
        "test_best_length": (
            best_length
        ),
        "test_mode_length": (
            mode_length
        ),
        "test_mode_fraction": (
            mode_fraction
        ),
        "test_fraction_optimal": (
            fraction_optimal
        ),
        "test_unique_tours": int(
            unique_tours.shape[0]
        ),
        "test_best_tour": (
            canonical_tours(
                best_tour[None]
            )[0]
            .cpu()
            .tolist()
        ),
    }


def add_prefixed_metrics(
    target,
    prefix,
    metrics,
):
    for key, value in (
        metrics.items()
    ):
        target[
            f"{prefix}_{key}"
        ] = value


# ============================================================
# TRAINING
# ============================================================

def train(
    seed,
    epochs,
    lr_schedule,
    output_dir,
    verbose=True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cities = create_tsp_instance(
        N_CITIES,
        TSP_INSTANCE_SEED,
    )

    model = RealNVP(
        dimension=N_CITIES,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_START,
    )

    validation_generator = (
        torch.Generator(
            device=DEVICE
        )
        .manual_seed(
            seed + 1000
        )
    )

    validation_z = torch.randn(
        VALIDATION_SIZE,
        N_CITIES,
        generator=(
            validation_generator
        ),
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Two checkpoints
    # --------------------------------------------------------

    discovery_state = None
    discovery_epoch = -1

    discovery_best_elite_mean = (
        float("inf")
    )
    discovery_best_mean = (
        float("inf")
    )
    discovery_stats = None

    concentration_state = None
    concentration_epoch = -1

    concentration_best_mean = (
        float("inf")
    )
    concentration_best_p95 = (
        float("inf")
    )
    concentration_stats = None

    # --------------------------------------------------------
    # Best solution discovered anywhere during optimization
    # --------------------------------------------------------

    optimization_best_length = (
        float("inf")
    )

    optimization_best_tour = None
    best_found_epoch = -1

    best_validation_tour_seen = (
        float("inf")
    )

    training_evaluations = 0
    validation_evaluations = 0

    diagnostics = []

    if verbose:
        print()
        print("=" * 78)
        print(
            f"TSP V3 | seed={seed} | "
            f"annealed KL + "
            f"{lr_schedule} LR | "
            f"dual checkpoint"
        )
        print("=" * 78)

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    for epoch in range(
        epochs
    ):
        learning_rate = (
            learning_rate_at(
                epoch,
                epochs,
                lr_schedule,
            )
        )

        for group in (
            optimizer.param_groups
        ):
            group["lr"] = (
                learning_rate
            )

        # Keep forward pass differentiable because KL uses
        # the reparameterized flow sample.
        z = torch.randn(
            BATCH_SIZE,
            N_CITIES,
            device=DEVICE,
        )

        y, log_det = model(z)

        with torch.no_grad():
            tours = (
                decode_permutation(
                    y
                )
            )

            lengths = tsp_length(
                cities,
                tours,
            )

            reward = -lengths

            baseline = (
                reward.sum()
                - reward
            ) / (
                BATCH_SIZE - 1
            )

            advantage = (
                reward
                - baseline
            )

        training_evaluations += (
            BATCH_SIZE
        )

        # ----------------------------------------------------
        # Best solution found during optimization
        # ----------------------------------------------------

        batch_best_index = int(
            torch.argmin(
                lengths
            )
        )

        batch_best_length = (
            lengths[
                batch_best_index
            ].item()
        )

        if (
            batch_best_length
            < optimization_best_length
        ):
            optimization_best_length = (
                batch_best_length
            )

            optimization_best_tour = (
                tours[
                    batch_best_index
                ]
                .detach()
                .clone()
            )

            best_found_epoch = epoch

        # ----------------------------------------------------
        # REINFORCE term
        # ----------------------------------------------------

        log_prob = log_probability(
            model,
            y.detach(),
        )

        reinforce_loss = -(
            advantage
            * log_prob
        ).mean()

        # ----------------------------------------------------
        # KL(q_theta || N(0, I))
        #
        # Up to theta-independent constants:
        #
        # KL = E[
        #       -log|det J|
        #       + 0.5 ||y||^2
        #      ]
        # ----------------------------------------------------

        T = temperature(
            epoch,
            epochs,
        )

        kl = (
            -log_det
            + 0.5
            * y.square()
            .sum(dim=1)
        ).mean()

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

        # ====================================================
        # VALIDATION
        # ====================================================

        if (
            epoch
            % VALIDATE_EVERY
            == 0
            or epoch
            == epochs - 1
        ):
            with torch.no_grad():
                val_y, _ = model(
                    validation_z
                )

                val_tours = (
                    decode_permutation(
                        val_y
                    )
                )

                val_lengths = (
                    tsp_length(
                        cities,
                        val_tours,
                    )
                )

                val_stats = (
                    validation_statistics(
                        val_lengths,
                        val_tours,
                    )
                )

            validation_evaluations += (
                VALIDATION_SIZE
            )

            best_validation_tour_seen = min(
                best_validation_tour_seen,
                val_stats["best"],
            )

            # -----------------------------------------------
            # Discovery checkpoint:
            # best top-1% validation mean.
            # -----------------------------------------------

            if discovery_checkpoint_better(
                val_stats,
                discovery_best_elite_mean,
                discovery_best_mean,
            ):
                discovery_best_elite_mean = (
                    val_stats[
                        "elite_mean"
                    ]
                )

                discovery_best_mean = (
                    val_stats[
                        "mean"
                    ]
                )

                discovery_epoch = (
                    epoch
                )

                discovery_state = (
                    copy.deepcopy(
                        model.state_dict()
                    )
                )

                discovery_stats = (
                    dict(val_stats)
                )

            # -----------------------------------------------
            # Concentration checkpoint:
            # best overall validation mean.
            # p95 is the tie-breaker.
            # -----------------------------------------------

            if concentration_checkpoint_better(
                val_stats,
                concentration_best_mean,
                concentration_best_p95,
            ):
                concentration_best_mean = (
                    val_stats["mean"]
                )

                concentration_best_p95 = (
                    val_stats["p95"]
                )

                concentration_epoch = (
                    epoch
                )

                concentration_state = (
                    copy.deepcopy(
                        model.state_dict()
                    )
                )

                concentration_stats = (
                    dict(val_stats)
                )

            row = {
                "epoch": epoch,
                "training_evaluations": (
                    training_evaluations
                ),
                "validation_evaluations": (
                    validation_evaluations
                ),
                "learning_rate": (
                    learning_rate
                ),
                "temperature": T,
                "loss": loss.item(),
                "reinforce_loss": (
                    reinforce_loss.item()
                ),
                "kl": kl.item(),
                "grad_norm_before_clip": (
                    float(grad_norm)
                ),
                "train_mean_length": (
                    lengths.mean().item()
                ),
                "train_std_length": (
                    lengths.std(
                        unbiased=False
                    ).item()
                ),
                "optimization_best_length": (
                    optimization_best_length
                ),
                "val_mean": (
                    val_stats["mean"]
                ),
                "val_elite_mean": (
                    val_stats[
                        "elite_mean"
                    ]
                ),
                "val_best": (
                    val_stats["best"]
                ),
                "val_p95": (
                    val_stats["p95"]
                ),
                "val_fraction_optimal": (
                    val_stats[
                        "fraction_optimal"
                    ]
                ),
                "val_unique_fraction": (
                    val_stats[
                        "unique_fraction"
                    ]
                ),
                "discovery_checkpoint_epoch": (
                    discovery_epoch
                ),
                "concentration_checkpoint_epoch": (
                    concentration_epoch
                ),
            }

            diagnostics.append(
                row
            )

            if verbose:
                print(
                    f"{epoch:5d} | "
                    f"mean="
                    f"{row['train_mean_length']:.6f} | "
                    f"best="
                    f"{optimization_best_length:.6f} | "
                    f"val_mean="
                    f"{val_stats['mean']:.6f} | "
                    f"val_elite="
                    f"{val_stats['elite_mean']:.6f} | "
                    f"val_p95="
                    f"{val_stats['p95']:.6f} | "
                    f"val_opt="
                    f"{100 * val_stats['fraction_optimal']:.2f}% | "
                    f"unique="
                    f"{val_stats['unique_fraction']:.3f} | "
                    f"grad="
                    f"{float(grad_norm):.4f} | "
                    f"lr="
                    f"{learning_rate:.2e} | "
                    f"T="
                    f"{T:.5f}"
                )

    # ========================================================
    # SAVE DIAGNOSTICS
    # ========================================================

    diagnostics_path = (
        output_dir
        / "diagnostics.csv"
    )

    with diagnostics_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                diagnostics[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            diagnostics
        )

    # ========================================================
    # FRESH TEST SET
    #
    # Both checkpoints are evaluated on the SAME fresh test
    # sample so the comparison is paired and less noisy.
    # ========================================================

    test_generator = (
        torch.Generator(
            device=DEVICE
        )
        .manual_seed(
            seed + 2000
        )
    )

    test_z = torch.randn(
        TEST_SIZE,
        N_CITIES,
        generator=test_generator,
        device=DEVICE,
    )

    discovery_test = (
        evaluate_checkpoint(
            model,
            discovery_state,
            test_z,
            cities,
        )
    )

    concentration_test = (
        evaluate_checkpoint(
            model,
            concentration_state,
            test_z,
            cities,
        )
    )

    # ========================================================
    # RESULT
    # ========================================================

    optimum_hit = (
        optimization_best_length
        <= REFERENCE_OPTIMUM
        + TARGET_TOL
    )

    optimization_gap_percent = (
        (
            optimization_best_length
            - REFERENCE_OPTIMUM
        )
        / REFERENCE_OPTIMUM
        * 100.0
    )

    result = {
        "experiment": (
            "tsp_v3_kl_cosine_dual_checkpoint"
            if lr_schedule == "cosine"
            else
            "tsp_v3_kl_constant_dual_checkpoint"
        ),
        "seed": seed,
        "epochs": epochs,
        "lr_schedule": (
            lr_schedule
        ),
        "initial_lr": (
            LR_START
        ),
        "final_lr": (
            learning_rate_at(
                epochs - 1,
                epochs,
                lr_schedule,
            )
        ),
        "temperature_start": (
            T_START
        ),
        "temperature_end": (
            T_END
        ),
        "elite_fraction": (
            ELITE_FRACTION
        ),
        "training_evaluations": (
            training_evaluations
        ),
        "validation_evaluations": (
            validation_evaluations
        ),
        "test_evaluations_per_checkpoint": (
            TEST_SIZE
        ),
        "optimization_best_length": (
            optimization_best_length
        ),
        "best_found_epoch": (
            best_found_epoch
        ),
        "best_validation_tour_seen": (
            best_validation_tour_seen
        ),
        "optimization_gap_percent": (
            optimization_gap_percent
        ),
        "optimum_hit": (
            optimum_hit
        ),
        "optimization_best_tour": (
            canonical_tours(
                optimization_best_tour[
                    None
                ]
            )[0]
            .cpu()
            .tolist()
        ),
        "discovery_checkpoint_epoch": (
            discovery_epoch
        ),
        "discovery_val_mean": (
            discovery_stats["mean"]
        ),
        "discovery_val_elite_mean": (
            discovery_stats[
                "elite_mean"
            ]
        ),
        "discovery_val_best": (
            discovery_stats["best"]
        ),
        "discovery_val_p95": (
            discovery_stats["p95"]
        ),
        "discovery_val_fraction_optimal": (
            discovery_stats[
                "fraction_optimal"
            ]
        ),
        "concentration_checkpoint_epoch": (
            concentration_epoch
        ),
        "concentration_val_mean": (
            concentration_stats[
                "mean"
            ]
        ),
        "concentration_val_elite_mean": (
            concentration_stats[
                "elite_mean"
            ]
        ),
        "concentration_val_best": (
            concentration_stats[
                "best"
            ]
        ),
        "concentration_val_p95": (
            concentration_stats[
                "p95"
            ]
        ),
        "concentration_val_fraction_optimal": (
            concentration_stats[
                "fraction_optimal"
            ]
        ),
        "n_cities": N_CITIES,
        "instance_seed": (
            TSP_INSTANCE_SEED
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "validation_size": (
            VALIDATION_SIZE
        ),
        "validate_every": (
            VALIDATE_EVERY
        ),
        "test_size": (
            TEST_SIZE
        ),
        "num_layers": (
            NUM_LAYERS
        ),
        "hidden_dim": (
            HIDDEN_DIM
        ),
        "max_grad_norm": (
            MAX_GRAD_NORM
        ),
        "device": str(
            DEVICE
        ),
        "torch_version": (
            torch.__version__
        ),
    }

    add_prefixed_metrics(
        result,
        "discovery",
        discovery_test,
    )

    add_prefixed_metrics(
        result,
        "concentration",
        concentration_test,
    )

    summary_path = (
        output_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    if verbose:
        print()
        print("=" * 78)
        print("FINAL RESULT")
        print("=" * 78)

        print(
            f"Optimization best:              "
            f"{optimization_best_length:.6f}"
        )
        print(
            f"Optimum hit:                    "
            f"{optimum_hit}"
        )
        print()
        print(
            "DISCOVERY CHECKPOINT "
            "(top-1% validation mean)"
        )
        print(
            f"  epoch:                        "
            f"{discovery_epoch}"
        )
        print(
            f"  val elite mean:               "
            f"{discovery_stats['elite_mean']:.6f}"
        )
        print(
            f"  val mean:                     "
            f"{discovery_stats['mean']:.6f}"
        )
        print(
            f"  test mean:                    "
            f"{discovery_test['test_mean_length']:.6f}"
        )
        print(
            f"  test p95:                     "
            f"{discovery_test['test_p95_length']:.6f}"
        )
        print(
            f"  test mode length:             "
            f"{discovery_test['test_mode_length']:.6f}"
        )
        print(
            f"  test mode fraction:           "
            f"{100 * discovery_test['test_mode_fraction']:.2f}%"
        )
        print(
            f"  optimal test samples:         "
            f"{100 * discovery_test['test_fraction_optimal']:.2f}%"
        )
        print()
        print(
            "CONCENTRATION CHECKPOINT "
            "(overall validation mean)"
        )
        print(
            f"  epoch:                        "
            f"{concentration_epoch}"
        )
        print(
            f"  val mean:                     "
            f"{concentration_stats['mean']:.6f}"
        )
        print(
            f"  val p95:                      "
            f"{concentration_stats['p95']:.6f}"
        )
        print(
            f"  test mean:                    "
            f"{concentration_test['test_mean_length']:.6f}"
        )
        print(
            f"  test p95:                     "
            f"{concentration_test['test_p95_length']:.6f}"
        )
        print(
            f"  test mode length:             "
            f"{concentration_test['test_mode_length']:.6f}"
        )
        print(
            f"  test mode fraction:           "
            f"{100 * concentration_test['test_mode_fraction']:.2f}%"
        )
        print(
            f"  optimal test samples:         "
            f"{100 * concentration_test['test_fraction_optimal']:.2f}%"
        )

    return result


# ============================================================
# MULTI-SEED RUN
# ============================================================

def run_many_seeds(
    seeds,
    epochs,
    lr_schedule,
    output_dir,
    verbose,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "experiment": (
            "RealNVP TSP-20 V3"
        ),
        "description": (
            "Annealed KL + configurable LR schedule "
            "+ dual validation checkpoints"
        ),
        "device": str(
            DEVICE
        ),
        "threads": (
            torch.get_num_threads()
        ),
        "seeds": seeds,
        "n_cities": (
            N_CITIES
        ),
        "instance_seed": (
            TSP_INSTANCE_SEED
        ),
        "epochs": epochs,
        "batch_size": (
            BATCH_SIZE
        ),
        "validation_size": (
            VALIDATION_SIZE
        ),
        "validate_every": (
            VALIDATE_EVERY
        ),
        "test_size": (
            TEST_SIZE
        ),
        "lr_schedule": (
            lr_schedule
        ),
        "lr_start": (
            LR_START
        ),
        "lr_end": (
            LR_END
        ),
        "temperature_start": (
            T_START
        ),
        "temperature_end": (
            T_END
        ),
        "elite_fraction": (
            ELITE_FRACTION
        ),
        "reference_optimum_evaluation_only": (
            REFERENCE_OPTIMUM
        ),
        "target_tolerance": (
            TARGET_TOL
        ),
        "torch_version": (
            torch.__version__
        ),
    }

    (
        output_dir
        / "run_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        )
        + "\n"
    )

    print("=" * 78)
    print(
        "REALNVP TSP-20 V3 | "
        "ANNEALED KL + "
        f"{lr_schedule.upper()} LR | "
        "DUAL CHECKPOINT"
    )
    print("=" * 78)

    print(
        f"device={DEVICE} | "
        f"threads="
        f"{torch.get_num_threads()} | "
        f"seeds={seeds[0]}-{seeds[-1]} | "
        f"layers={NUM_LAYERS} | "
        f"hidden={HIDDEN_DIM}"
    )

    print(
        f"batch={BATCH_SIZE} | "
        f"epochs={epochs} | "
        f"lr={LR_START}->{LR_END if lr_schedule == 'cosine' else LR_START} | "
        f"T={T_START}->{T_END}"
    )

    results = []

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

        seed_dir = (
            output_dir
            / f"seed_{seed}"
        )

        result = train(
            seed=seed,
            epochs=epochs,
            lr_schedule=lr_schedule,
            output_dir=seed_dir,
            verbose=verbose,
        )

        results.append(
            result
        )

    comparison_path = (
        output_dir
        / "comparison.csv"
    )

    with comparison_path.open(
        "w",
        newline="",
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

    # ========================================================
    # 10-SEED SUMMARY
    # ========================================================

    optimization_best = np.array(
        [
            r[
                "optimization_best_length"
            ]
            for r in results
        ]
    )

    optimum_hits = sum(
        r["optimum_hit"]
        for r in results
    )

    discovery_test_means = np.array(
        [
            r[
                "discovery_test_mean_length"
            ]
            for r in results
        ]
    )

    concentration_test_means = np.array(
        [
            r[
                "concentration_test_mean_length"
            ]
            for r in results
        ]
    )

    discovery_optimal_share = np.array(
        [
            r[
                "discovery_test_fraction_optimal"
            ]
            for r in results
        ]
    )

    concentration_optimal_share = np.array(
        [
            r[
                "concentration_test_fraction_optimal"
            ]
            for r in results
        ]
    )

    discovery_mode_optimum_hits = sum(
        r[
            "discovery_test_mode_length"
        ]
        <= REFERENCE_OPTIMUM
        + TARGET_TOL
        for r in results
    )

    concentration_mode_optimum_hits = sum(
        r[
            "concentration_test_mode_length"
        ]
        <= REFERENCE_OPTIMUM
        + TARGET_TOL
        for r in results
    )

    print()
    print("=" * 78)
    print("MULTI-SEED SUMMARY")
    print("=" * 78)

    print(
        f"Optimum discovery hits:        "
        f"{optimum_hits}/"
        f"{len(results)}"
    )

    print(
        f"Mean optimization best:        "
        f"{optimization_best.mean():.6f}"
    )

    print()
    print(
        "DISCOVERY CHECKPOINT"
    )

    print(
        f"Mean test mean:                "
        f"{discovery_test_means.mean():.6f}"
    )

    print(
        f"Mean optimal test share:       "
        f"{100 * discovery_optimal_share.mean():.2f}%"
    )

    print(
        f"Final mode = optimum:          "
        f"{discovery_mode_optimum_hits}/"
        f"{len(results)}"
    )

    print()
    print(
        "CONCENTRATION CHECKPOINT"
    )

    print(
        f"Mean test mean:                "
        f"{concentration_test_means.mean():.6f}"
    )

    print(
        f"Mean optimal test share:       "
        f"{100 * concentration_optimal_share.mean():.2f}%"
    )

    print(
        f"Final mode = optimum:          "
        f"{concentration_mode_optimum_hits}/"
        f"{len(results)}"
    )

    print()
    print(
        f"Saved comparison:              "
        f"{comparison_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "RealNVP TSP-20 V3: "
            "annealed KL + cosine/constant LR "
            "+ discovery and concentration checkpoints."
        )
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(
            range(
                SEED_START,
                SEED_START
                + N_SEEDS,
            )
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr-schedule",
        choices=[
            "cosine",
            "constant",
        ],
        default="cosine",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error(
            "epochs must be positive"
        )

    if args.threads < 1:
        parser.error(
            "threads must be positive"
        )

    torch.set_num_threads(
        args.threads
    )

    if args.output_dir is None:
        if (
            args.lr_schedule
            == "cosine"
        ):
            output_dir = Path(
                "results/"
                "permutation_optimization/"
                "tsp20/"
                "kl_cosine_dual_checkpoint"
            )
        else:
            output_dir = Path(
                "results/"
                "permutation_optimization/"
                "tsp20/"
                "kl_constant_dual_checkpoint"
            )
    else:
        output_dir = (
            args.output_dir
        )

    run_many_seeds(
        seeds=args.seeds,
        epochs=args.epochs,
        lr_schedule=(
            args.lr_schedule
        ),
        output_dir=output_dir,
        verbose=not args.quiet,
    )
