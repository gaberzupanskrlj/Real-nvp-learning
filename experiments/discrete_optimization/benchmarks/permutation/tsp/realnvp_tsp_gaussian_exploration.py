import copy
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# Settings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CITIES = 20
NUM_LAYERS = 4
HIDDEN_DIM = 64

BATCH_SIZE = 1024
EPOCHS = 2000
LR = 1e-4

VALIDATION_SIZE = 4096
VALIDATE_EVERY = 50
TEST_SIZE = 16384
MAX_GRAD_NORM = 5.0

SEED = 42
TSP_INSTANCE_SEED = 12345

EXPLORATION_FRACTION = 0.0
BASELINE_RATE = 0.05

N_EXPLORE = round(BATCH_SIZE * EXPLORATION_FRACTION)
N_FLOW = BATCH_SIZE - N_EXPLORE

ALPHA = N_FLOW / BATCH_SIZE
EPSILON = N_EXPLORE / BATCH_SIZE


# TSP problem

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


# RealNVP model

class CouplingLayer(nn.Module):
    def __init__(self, dimension, hidden_dim):
        super().__init__()

        if dimension % 2 != 0:
            raise ValueError(
                "This RealNVP implementation requires an even dimension."
            )

        self.half = dimension // 2

        self.scale_net = nn.Sequential(
            nn.Linear(self.half, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.half),
            nn.Tanh(),
        )

        self.translate_net = nn.Sequential(
            nn.Linear(self.half, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.half),
        )

    def forward(self, x):
        x1 = x[:, :self.half]
        x2 = x[:, self.half:]

        s = self.scale_net(x1)
        t = self.translate_net(x1)

        y1 = x1
        y2 = x2 * torch.exp(s) + t

        return (
            torch.cat([y1, y2], dim=1),
            s.sum(dim=1),
        )

    def inverse(self, y):
        y1 = y[:, :self.half]
        y2 = y[:, self.half:]

        s = self.scale_net(y1)
        t = self.translate_net(y1)

        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        return (
            torch.cat([x1, x2], dim=1),
            -s.sum(dim=1),
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
                for _ in range(num_layers)
            ]
        )

        self.half = dimension // 2

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
        log_det_total = x.new_zeros(
            x.shape[0]
        )

        for layer in self.layers:
            y, log_det = layer(y)
            log_det_total += log_det
            y = self._swap(y)

        return y, log_det_total

    def inverse(self, y):
        x = y
        log_det_total = y.new_zeros(
            y.shape[0]
        )

        for layer in reversed(self.layers):
            x = self._swap(x)
            x, log_det = layer.inverse(x)
            log_det_total += log_det

        return x, log_det_total


# Gaussian density

def gaussian_log_prob(z):
    return -0.5 * (
        z.square()
        + math.log(2.0 * math.pi)
    ).sum(dim=1)


# Mixture density at fixed output samples

def mixture_log_probs(
    model,
    y,
    exploration_fraction,
):
    eps = float(exploration_fraction)

    if not 0.0 <= eps < 1.0:
        raise ValueError(
            "Require 0 <= exploration_fraction < 1"
        )

    y = y.detach()

    r, inverse_log_det = model.inverse(y)

    if (
        r.shape != y.shape
        or inverse_log_det.shape != y.shape[:1]
    ):
        raise ValueError(
            "Unexpected inverse output or logdet shape"
        )

    log_q = (
        gaussian_log_prob(r)
        + inverse_log_det
    )

    log_p0_y = gaussian_log_prob(y)

    if eps == 0.0:
        log_m = log_q

    else:
        alpha = 1.0 - eps

        log_m = torch.logaddexp(
            math.log(alpha) + log_q,
            math.log(eps) + log_p0_y,
        )

    return (
        log_q,
        log_p0_y,
        log_m,
        inverse_log_det,
    )


# Plotting

def plot_convergence(
    history_mean,
    history_best,
    filename="tsp_gaussian_exploration_convergence.png",
):
    epochs = np.arange(
        len(history_mean)
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history_mean,
        label="Mean sampled tour length",
    )

    plt.plot(
        epochs,
        history_best,
        label="Best-so-far tour length",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Tour length")

    plt.title(
        "RealNVP Gaussian exploration on TSP-20"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
    )

    plt.close()


def plot_tour_comparison(
    cities,
    best_tour,
    best_length,
    test_best_tour,
    test_best_length,
    filename="tsp_gaussian_exploration_tours.png",
):
    cities = cities.detach().cpu().numpy()

    best_tour = np.asarray(
        best_tour.detach().cpu()
    )

    test_best_tour = np.asarray(
        test_best_tour.detach().cpu()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    tours = [
        (
            best_tour,
            best_length,
            "Best ever",
        ),
        (
            test_best_tour,
            test_best_length,
            "Final test best",
        ),
    ]

    for ax, (
        tour,
        length,
        title,
    ) in zip(axes, tours):

        ordered = cities[tour]

        closed = np.vstack(
            [
                ordered,
                ordered[0],
            ]
        )

        ax.plot(
            closed[:, 0],
            closed[:, 1],
            marker="o",
        )

        for i, (x, y) in enumerate(cities):
            ax.text(
                x,
                y,
                str(i),
                fontsize=9,
            )

        ax.set_title(
            f"{title}\nLength = {length:.6f}"
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
    )

    plt.close()


# Training

def train(
    seed=SEED,
    verbose=True,
    make_plots=True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        lr=LR,
    )

    validation_generator = torch.Generator(
        device=DEVICE
    ).manual_seed(
        seed + 1000
    )

    validation = torch.randn(
        VALIDATION_SIZE,
        N_CITIES,
        generator=validation_generator,
        device=DEVICE,
    )

    best_validation_mean = float("inf")
    best_state = None
    best_epoch = -1

    best_length = float("inf")
    best_tour = None
    best_source = None

    best_flow_length = float("inf")
    best_explore_length = float("inf")

    baseline = 0.0

    history_mean = []
    history_best = []

    for epoch in range(EPOCHS + 1):

        optimizer.zero_grad(
            set_to_none=True
        )

        # Generate flow and Gaussian exploration candidates.
        # No sampling graph is retained.

        with torch.no_grad():

            r_initial = torch.randn(
                N_FLOW,
                N_CITIES,
                device=DEVICE,
            )

            y_flow, _ = model(
                r_initial
            )

            y_explore = torch.randn(
                N_EXPLORE,
                N_CITIES,
                device=DEVICE,
            )

            y = torch.cat(
                (
                    y_flow,
                    y_explore,
                ),
                dim=0,
            ).detach()

            if not torch.isfinite(y).all():
                raise FloatingPointError(
                    "Non-finite generated candidates"
                )

            tours = decode_permutation(y)

            lengths = tsp_length(
                cities,
                tours,
            ).detach()

            if lengths.shape != (
                BATCH_SIZE,
            ):
                raise ValueError(
                    "Objective must return shape [BATCH_SIZE]"
                )

            if not torch.isfinite(
                lengths
            ).all():
                raise FloatingPointError(
                    "Non-finite objective costs"
                )

        # Keep the best candidate independently of the model.

        batch_best_index = int(
            lengths.argmin().item()
        )

        batch_best_length = (
            lengths[
                batch_best_index
            ].item()
        )

        if (
            batch_best_index
            < N_FLOW
        ):
            batch_best_source = "flow"
        else:
            batch_best_source = "exploration"

        if (
            batch_best_length
            < best_length
        ):
            best_length = (
                batch_best_length
            )

            best_tour = (
                tours[
                    batch_best_index
                ]
                .detach()
                .clone()
            )

            best_source = (
                batch_best_source
            )

        # Re-invert every fixed output candidate with
        # autograd enabled.

        (
            log_q,
            log_p0_y,
            log_m,
            inverse_log_det,
        ) = mixture_log_probs(
            model,
            y,
            EPSILON,
        )

        if not (
            torch.isfinite(
                log_q
            ).all()
            and torch.isfinite(
                log_m
            ).all()
        ):
            raise FloatingPointError(
                "Non-finite model log-density"
            )

        # Expected-cost score estimator from the document.
        # The baseline is the value from before this batch.

        centered_cost = (
            lengths
            - float(baseline)
        )

        loss = (
            centered_cost.detach()
            * log_m
        ).mean() / ALPHA

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite loss"
            )

        loss.backward()

        for parameter in model.parameters():
            if (
                parameter.grad
                is not None
                and not torch.isfinite(
                    parameter.grad
                ).all()
            ):
                raise FloatingPointError(
                    "Non-finite parameter gradient"
                )

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )
        )

        # Diagnostics refer to the model and candidates
        # before this optimizer update.

        with torch.no_grad():

            rho = torch.exp(
                math.log(ALPHA)
                + log_q.detach()
                - log_m.detach()
            )

            mean_length = (
                lengths.mean().item()
            )

            flow_lengths = lengths[:N_FLOW]

            mean_flow_length = (
                flow_lengths
                .mean()
                .item()
            )

            flow_best_length = (
                flow_lengths
                .min()
                .item()
            )

            if flow_best_length < best_flow_length:
                best_flow_length = flow_best_length

            if N_EXPLORE > 0:

                explore_lengths = lengths[N_FLOW:]

                mean_explore_length = (
                    explore_lengths
                    .mean()
                    .item()
                )

                explore_best_length = (
                    explore_lengths
                    .min()
                    .item()
                )

                if explore_best_length < best_explore_length:
                    best_explore_length = explore_best_length

                mean_explore_responsibility = (
                    rho[N_FLOW:]
                    .mean()
                    .item()
                )

            else:
                mean_explore_length = None
                explore_best_length = None
                mean_explore_responsibility = None

            next_baseline = (
                (
                    1.0
                    - BASELINE_RATE
                )
                * float(baseline)
                + BASELINE_RATE
                * mean_length
            )

            all_unique_fraction = (
                torch.unique(
                    tours,
                    dim=0,
                ).shape[0]
                / BATCH_SIZE
            )

            flow_unique_fraction = (
                torch.unique(
                    tours[:N_FLOW],
                    dim=0,
                ).shape[0]
                / N_FLOW
            )

            if N_EXPLORE > 0:
                explore_unique_fraction = (
                    torch.unique(
                        tours[N_FLOW:],
                        dim=0,
                    ).shape[0]
                    / N_EXPLORE
                )
            else:
                explore_unique_fraction = None

        optimizer.step()

        # The running baseline is updated only after
        # the current gradient has been computed.

        baseline = next_baseline

        history_mean.append(
            mean_length
        )

        history_best.append(
            best_length
        )

        # Validation stays flow-only, as in the frozen baseline.

        if epoch % VALIDATE_EVERY == 0:

            with torch.no_grad():

                validation_y, _ = model(
                    validation
                )

                validation_tours = (
                    decode_permutation(
                        validation_y
                    )
                )

                validation_lengths = (
                    tsp_length(
                        cities,
                        validation_tours,
                    )
                )

                validation_mean = (
                    validation_lengths
                    .mean()
                    .item()
                )

                validation_best_index = (
                    torch.argmin(
                        validation_lengths
                    )
                )

                validation_best_length = (
                    validation_lengths[
                        validation_best_index
                    ].item()
                )

                if (
                    validation_best_length
                    < best_length
                ):
                    best_length = (
                        validation_best_length
                    )

                    best_tour = (
                        validation_tours[
                            validation_best_index
                        ]
                        .detach()
                        .clone()
                    )

                    best_source = (
                        "validation"
                    )

            if (
                validation_mean
                < best_validation_mean
            ):
                best_validation_mean = (
                    validation_mean
                )

                best_epoch = epoch

                best_state = copy.deepcopy(
                    model.state_dict()
                )

        if (
            verbose
            and epoch % 50 == 0
        ):
            explore_mean_text = (
                f"{mean_explore_length:.6f}"
                if mean_explore_length
                is not None
                else "none"
            )

            explore_rho_text = (
                f"{mean_explore_responsibility:.6f}"
                if mean_explore_responsibility
                is not None
                else "none"
            )

            if N_EXPLORE > 0:
                explore_best_text = (
                    f"{explore_best_length:.6f}"
                )
                explore_unique_text = (
                    f"{explore_unique_fraction:.3f}"
                )
            else:
                explore_best_text = "none"
                explore_unique_text = "none"

            print(
                f"{epoch:5d} | "
                f"mean={mean_length:.6f} | "
                f"flow={mean_flow_length:.6f} | "
                f"explore={explore_mean_text} | "
                f"best={best_length:.6f} | "
                f"source={best_source} | "
                f"best_flow={flow_best_length:.6f} | "
                f"best_explore={explore_best_text} | "
                f"rho_exp={explore_rho_text} | "
                f"unique_all={all_unique_fraction:.3f} | "
                f"unique_flow={flow_unique_fraction:.3f} | "
                f"unique_explore={explore_unique_text} | "
                f"loss={loss.item():.6f} | "
                f"baseline={baseline:.6f} | "
                f"grad={float(grad_norm):.4f}"
            )

    # Restore the flow checkpoint chosen by validation mean.

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    test_generator = torch.Generator(
        device=DEVICE
    ).manual_seed(
        seed + 2000
    )

    test_r = torch.randn(
        TEST_SIZE,
        N_CITIES,
        generator=test_generator,
        device=DEVICE,
    )

    with torch.no_grad():

        test_y, _ = model(
            test_r
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

    test_best_index = torch.argmin(
        test_lengths
    )

    test_best_length = (
        test_lengths[
            test_best_index
        ].item()
    )

    test_best_tour = (
        test_tours[
            test_best_index
        ]
        .detach()
        .clone()
    )

    test_mean_length = (
        test_lengths.mean().item()
    )

    if test_best_length < best_length:

        best_length = (
            test_best_length
        )

        best_tour = (
            test_best_tour
            .detach()
            .clone()
        )

        best_source = "test"

    metrics = {
        "seed": seed,
        "exploration_fraction_requested":
            EXPLORATION_FRACTION,
        "exploration_fraction_actual":
            EPSILON,
        "n_flow": N_FLOW,
        "n_explore": N_EXPLORE,
        "best_validation_epoch":
            best_epoch,
        "best_validation_mean":
            best_validation_mean,
        "test_mean_length":
            test_mean_length,
        "test_best_length":
            test_best_length,
        "best_length_ever":
            best_length,
        "best_flow_length_ever":
            best_flow_length,
        "best_explore_length_ever":
            (
                best_explore_length
                if N_EXPLORE > 0
                else None
            ),
        "best_source":
            best_source,
        "best_tour":
            best_tour
            .detach()
            .cpu()
            .tolist(),
    }

    if verbose:

        print()
        print("FINAL RESULT")

        print(
            f"Flow samples per batch:       "
            f"{N_FLOW}"
        )

        print(
            f"Exploration samples per batch:"
            f" {N_EXPLORE}"
        )

        print(
            f"Actual exploration fraction:  "
            f"{EPSILON:.9f}"
        )

        print(
            f"Best validation epoch:         "
            f"{best_epoch}"
        )

        print(
            f"Best validation mean:          "
            f"{best_validation_mean:.6f}"
        )

        print(
            f"Test mean length:               "
            f"{test_mean_length:.6f}"
        )

        print(
            f"Test best length:               "
            f"{test_best_length:.6f}"
        )

        print(
            f"Best length ever:               "
            f"{best_length:.6f}"
        )

        print(
            f"Best flow length ever:          "
            f"{best_flow_length:.6f}"
        )

        if N_EXPLORE > 0:
            print(
                f"Best exploration length ever:   "
                f"{best_explore_length:.6f}"
            )
        else:
            print(
                "Best exploration length ever:   none"
            )

        print(
            f"Best source:                    "
            f"{best_source}"
        )

        print(
            f"Best tour:                      "
            f"{best_tour.detach().cpu().tolist()}"
        )

    if make_plots:

        plot_convergence(
            history_mean,
            history_best,
        )

        plot_tour_comparison(
            cities,
            best_tour,
            best_length,
            test_best_tour,
            test_best_length,
        )

    return metrics


if __name__ == "__main__":
    train()