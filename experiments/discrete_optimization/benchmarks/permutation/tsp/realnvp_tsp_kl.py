import copy
import csv
import math

import numpy as np
import torch
import torch.nn as nn


#spremeljikve

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CITIES = 50
NUM_LAYERS = 8
HIDDEN_DIM = 128

BATCH_SIZE = 4096
EPOCHS = 3000
LR = 1e-4

VALIDATION_SIZE = 4096
VALIDATE_EVERY = 50

TEST_SIZE = 16384
MAX_GRAD_NORM = 5.0

T_START = 0.1
T_END = 0.001

SEED_START = 42
N_SEEDS = 1
TSP_INSTANCE_SEED = 12345

REFERENCE_OPTIMUM = 5.207124420405269
TARGET_TOL = 1e-5

RESULTS_CSV = "tsp20_realnvp_kl_10seeds.csv"


# TSP problem

def create_tsp_instance(n_cities, seed):
    rng = np.random.default_rng(seed)
    cities = rng.random((n_cities, 2))
    return torch.tensor(cities, dtype=torch.float32, device=DEVICE)


def decode_permutation(y):
    return torch.argsort(y, dim=1)


@torch.no_grad()
def tsp_length(cities, tours):
    ordered_cities = cities[tours]
    next_cities = torch.roll(ordered_cities, shifts=-1, dims=1)
    edge_lengths = torch.linalg.vector_norm(
        ordered_cities - next_cities,
        dim=2,
    )
    return edge_lengths.sum(dim=1)


@torch.no_grad()
def canonical_tours(tours):
    """
    Canonical representation of a symmetric closed TSP tour:
    rotate so city 0 is first and choose one direction consistently.
    """
    n = tours.shape[1]

    start = (tours == 0).int().argmax(dim=1)
    index = (
        start[:, None]
        + torch.arange(n, device=tours.device)[None, :]
    ) % n

    canon = torch.gather(tours, 1, index)

    flip = canon[:, 1] > canon[:, -1]
    canon[flip, 1:] = canon[flip, 1:].flip(1)

    return canon


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

        y2 = x2 * torch.exp(s) + t

        return torch.cat([x1, y2], dim=1), s.sum(dim=1)

    def inverse(self, y):
        y1 = y[:, :self.half]
        y2 = y[:, self.half:]

        s = self.scale_net(y1)
        t = self.translate_net(y1)

        x2 = (y2 - t) * torch.exp(-s)

        return torch.cat([y1, x2], dim=1), -s.sum(dim=1)


class RealNVP(nn.Module):
    def __init__(self, dimension, num_layers, hidden_dim):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                CouplingLayer(dimension, hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.half = dimension // 2

    def _swap(self, x):
        return torch.cat(
            [x[:, self.half:], x[:, :self.half]],
            dim=1,
        )

    def forward(self, x):
        y = x
        log_det_total = x.new_zeros(x.shape[0])

        for layer in self.layers:
            y, log_det = layer(y)
            log_det_total += log_det
            y = self._swap(y)

        return y, log_det_total

    def inverse(self, y):
        x = y
        log_det_total = y.new_zeros(y.shape[0])

        for layer in reversed(self.layers):
            x = self._swap(x)
            x, log_det = layer.inverse(x)
            log_det_total += log_det

        return x, log_det_total


def log_probability(model, y):
    z, inverse_log_det = model.inverse(y)

    log_pz = -0.5 * (
        z.square()
        + math.log(2.0 * math.pi)
    ).sum(dim=1)

    return log_pz + inverse_log_det


# Annealed KL schedule

def temperature(epoch):
    if EPOCHS <= 1:
        return T_END

    progress = epoch / (EPOCHS - 1)

    return (
        T_END
        + 0.5
        * (T_START - T_END)
        * (1.0 + math.cos(math.pi * progress))
    )


# Training

def train(seed, verbose=True):
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
    ).manual_seed(seed + 1000)

    validation = torch.randn(
        VALIDATION_SIZE,
        N_CITIES,
        generator=validation_generator,
        device=DEVICE,
    )

    best_validation_mean = float("inf")
    best_validation_best = float("inf")
    best_state = None
    best_epoch = -1

    optimization_best_length = float("inf")
    optimization_best_tour = None
    best_found_epoch = -1

    if verbose:
        print()
        print("=" * 75)
        print(
            f"TSP KL | seed={seed} | "
            f"annealed KL T={T_START}->{T_END}"
        )
        print("=" * 75)

    for epoch in range(EPOCHS):
        z = torch.randn(
            BATCH_SIZE,
            N_CITIES,
            device=DEVICE,
        )

        # Keep this forward pass differentiable:
        # the KL term uses the reparameterized flow sample.
        y, log_det = model(z)

        with torch.no_grad():
            tours = decode_permutation(y)
            lengths = tsp_length(cities, tours)
            reward = -lengths

            baseline = (
                reward.sum() - reward
            ) / (BATCH_SIZE - 1)

            advantage = reward - baseline

        # Best solution found during optimization
        batch_best_index = int(torch.argmin(lengths))
        batch_best_length = lengths[batch_best_index].item()

        if batch_best_length < optimization_best_length:
            optimization_best_length = batch_best_length
            optimization_best_tour = tours[batch_best_index].clone()
            best_found_epoch = epoch

        # REINFORCE term
        log_prob = log_probability(
            model,
            y.detach(),
        )

        reinforce_loss = -(
            advantage * log_prob
        ).mean()

        # Annealed KL(p_theta || N(0, I))
        #
        # Up to theta-independent constants:
        #
        # KL = E[-log|det J| + 0.5 * ||y||^2]
        T = temperature(epoch)

        kl = (-log_det+ 0.5 * y.square().sum(dim=1)).mean()

        loss = reinforce_loss + T * kl

        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer.step()

        # Validation checkpoint
        if (
            epoch % VALIDATE_EVERY == 0
            or epoch == EPOCHS - 1
        ):
            with torch.no_grad():
                val_y, _ = model(validation)

                val_tours = decode_permutation(val_y)
                val_lengths = tsp_length(
                    cities,
                    val_tours,
                )

                val_mean = val_lengths.mean().item()
                val_best = val_lengths.min().item()

                best_validation_best = min(
                    best_validation_best,
                    val_best,
                )

            if val_mean < best_validation_mean:
                best_validation_mean = val_mean
                best_epoch = epoch
                best_state = copy.deepcopy(
                    model.state_dict()
                )

            if verbose:
                with torch.no_grad():
                    unique_fraction = (
                        len(
                            torch.unique(
                                canonical_tours(tours),
                                dim=0,
                            )
                        )
                        / BATCH_SIZE
                    )

                print(
                    f"{epoch:5d} | "
                    f"mean={lengths.mean().item():.6f} | "
                    f"best={optimization_best_length:.6f} | "
                    f"val_mean={val_mean:.6f} | "
                    f"val_best={val_best:.6f} | "
                    f"unique={unique_fraction:.3f} | "
                    f"std={lengths.std().item():.4f} | "
                    f"grad={float(grad_norm):.4f} | "
                    f"T={T:.5f} | "
                    f"kl={kl.item():.3f}"
                )

    # Test selected checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    test_generator = torch.Generator(
        device=DEVICE
    ).manual_seed(seed + 2000)

    test_z = torch.randn(
        TEST_SIZE,
        N_CITIES,
        generator=test_generator,
        device=DEVICE,
    )

    with torch.no_grad():
        test_y, _ = model(test_z)
        test_tours = decode_permutation(test_y)
        test_lengths = tsp_length(
            cities,
            test_tours,
        )

        canonical = canonical_tours(
            test_tours
        )

        unique_tours, counts = torch.unique(
            canonical,
            dim=0,
            return_counts=True,
        )

        mode_index = int(torch.argmax(counts))
        mode_tour = unique_tours[mode_index]

        mode_length = tsp_length(
            cities,
            mode_tour[None],
        ).item()

        mode_fraction = (
            counts.max().item()
            / TEST_SIZE
        )

        test_best_index = int(
            torch.argmin(test_lengths)
        )

        test_best_length = (
            test_lengths[
                test_best_index
            ].item()
        )

        test_best_tour = (
            test_tours[
                test_best_index
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

    best_length_including_test = min(
        optimization_best_length,
        test_best_length,
    )

    if optimization_best_length <= test_best_length:
        best_tour = optimization_best_tour
    else:
        best_tour = test_best_tour

    optimum_hit = (
        best_length_including_test
        <= REFERENCE_OPTIMUM
        + TARGET_TOL
    )

    gap_percent = (
        (
            best_length_including_test
            - REFERENCE_OPTIMUM
        )
        / REFERENCE_OPTIMUM
        * 100.0
    )

    result = {
        "seed": seed,
        "best_validation_epoch": best_epoch,
        "best_validation_mean": best_validation_mean,
        "best_validation_best": best_validation_best,
        "test_mean_length": test_lengths.mean().item(),
        "test_best_length": test_best_length,
        "test_mode_length": mode_length,
        "test_mode_fraction": mode_fraction,
        "test_fraction_optimal": fraction_optimal,
        "test_unique_tours": len(unique_tours),
        "optimization_best_length": optimization_best_length,
        "best_found_epoch": best_found_epoch,
        "best_length_including_test": best_length_including_test,
        "gap_percent": gap_percent,
        "optimum_hit": optimum_hit,
        "best_tour": canonical_tours(
            best_tour[None]
        )[0].cpu().tolist(),
    }

    if verbose:
        print()
        print("=" * 75)
        print("FINAL RESULT")
        print("=" * 75)

        print(
            f"Best validation checkpoint epoch: "
            f"{best_epoch}"
        )
        print(
            f"Best validation mean seen:        "
            f"{best_validation_mean:.6f}"
        )
        print(
            f"Best validation tour seen:        "
            f"{best_validation_best:.6f}"
        )
        print(
            f"Test mean length:                 "
            f"{result['test_mean_length']:.6f}"
        )
        print(
            f"Test best length:                 "
            f"{test_best_length:.6f}"
        )
        print(
            f"Test mode length:                 "
            f"{mode_length:.6f}"
        )
        print(
            f"Test mode fraction:               "
            f"{100 * mode_fraction:.2f}%"
        )
        print(
            f"Optimal test samples:             "
            f"{100 * fraction_optimal:.2f}%"
        )
        print(
            f"Test unique tours:                "
            f"{len(unique_tours)}"
        )
        print(
            f"Optimization best length:         "
            f"{optimization_best_length:.6f}"
        )
        print(
            f"Best length including test:       "
            f"{best_length_including_test:.6f}"
        )
        print(
            f"Optimality gap:                   "
            f"{gap_percent:.4f}%"
        )
        print(
            f"Optimum hit:                      "
            f"{optimum_hit}"
        )
        print(
            f"Best tour:                        "
            f"{result['best_tour']}"
        )

    return result


# Run multiple seeds

def run_many_seeds():
    seeds = list(
        range(
            SEED_START,
            SEED_START + N_SEEDS,
        )
    )

    results = []

    print("=" * 75)
    print("REALNVP TSP-20 | ANNEALED KL | 10 SEEDS")
    print("=" * 75)

    print(
        f"device={DEVICE} | "
        f"seeds={seeds[0]}-{seeds[-1]} | "
        f"layers={NUM_LAYERS} | "
        f"hidden={HIDDEN_DIM}"
    )

    print(
        f"batch={BATCH_SIZE} | "
        f"epochs={EPOCHS} | "
        f"lr={LR} | "
        f"T={T_START}->{T_END}"
    )

    for i, seed in enumerate(seeds, start=1):
        print()
        print(
            f"[{i:02d}/{len(seeds):02d}] "
            f"seed={seed}"
        )

        result = train(
            seed,
            verbose=True,
        )

        results.append(result)

    with open(
        RESULTS_CSV,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

    best_lengths = np.array(
        [
            r["best_length_including_test"]
            for r in results
        ]
    )

    test_means = np.array(
        [
            r["test_mean_length"]
            for r in results
        ]
    )

    gaps = np.array(
        [
            r["gap_percent"]
            for r in results
        ]
    )

    optimum_hits = sum(
        r["optimum_hit"]
        for r in results
    )

    mode_optimum_hits = sum(
        r["test_mode_length"]
        <= REFERENCE_OPTIMUM
        + TARGET_TOL
        for r in results
    )

    print()
    print("=" * 75)
    print("10-SEED SUMMARY")
    print("=" * 75)

    print(
        f"Mean best length:              "
        f"{best_lengths.mean():.6f}"
    )
    print(
        f"Std best length:               "
        f"{best_lengths.std(ddof=0):.6f}"
    )
    print(
        f"Median best length:            "
        f"{np.median(best_lengths):.6f}"
    )
    print(
        f"Best run:                      "
        f"{best_lengths.min():.6f}"
    )
    print(
        f"Worst run:                     "
        f"{best_lengths.max():.6f}"
    )
    print(
        f"Mean optimality gap:           "
        f"{gaps.mean():.4f}%"
    )
    print(
        f"Optimum hits:                  "
        f"{optimum_hits}/{len(results)}"
    )
    print(
        f"Final mode = optimum:          "
        f"{mode_optimum_hits}/{len(results)}"
    )
    print(
        f"Mean final generator length:   "
        f"{test_means.mean():.6f}"
    )
    print(
        f"Mean optimal test share:       "
        f"{100 * np.mean([r['test_fraction_optimal'] for r in results]):.2f}%"
    )
    print(
        f"Saved results to:              "
        f"{RESULTS_CSV}"
    )


if __name__ == "__main__":
    run_many_seeds()
