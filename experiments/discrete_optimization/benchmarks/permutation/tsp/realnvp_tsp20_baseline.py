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


# RealNVP model

class CouplingLayer(nn.Module):
    def __init__(self, dimension, hidden_dim):
        super().__init__()

        if dimension % 2 != 0:
            raise ValueError("This RealNVP implementation requires an even dimension.")

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

        return torch.cat([y1, y2], dim=1), s.sum(dim=1)

    def inverse(self, y):
        y1 = y[:, :self.half]
        y2 = y[:, self.half:]

        s = self.scale_net(y1)
        t = self.translate_net(y1)

        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        return torch.cat([x1, x2], dim=1), -s.sum(dim=1)


class RealNVP(nn.Module):
    def __init__(self, dimension, num_layers, hidden_dim):
        super().__init__()

        self.layers = nn.ModuleList(
            [CouplingLayer(dimension, hidden_dim) for _ in range(num_layers)]
        )
        self.half = dimension // 2

    def _swap(self, x):
        return torch.cat([x[:, self.half:], x[:, :self.half]], dim=1)

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
        z.square() + math.log(2.0 * math.pi)
    ).sum(dim=1)
    return log_pz + inverse_log_det


# Plotting

def plot_convergence(history_mean, history_best, filename="tsp_convergence.png"):
    epochs = np.arange(len(history_mean))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_mean, label="Mean sampled tour length")
    plt.plot(epochs, history_best, label="Best-so-far tour length")
    plt.xlabel("Epoch")
    plt.ylabel("Tour length")
    plt.title("RealNVP convergence on TSP-20")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_tour_comparison(
    cities,
    best_tour,
    best_length,
    test_best_tour,
    test_best_length,
    filename="tsp_tour_comparison.png",
):
    cities = cities.detach().cpu().numpy()
    best_tour = np.asarray(best_tour.detach().cpu())
    test_best_tour = np.asarray(test_best_tour.detach().cpu())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tours = [
        (best_tour, best_length, "Best ever"),
        (test_best_tour, test_best_length, "Final test best"),
    ]

    for ax, (tour, length, title) in zip(axes, tours):
        ordered = cities[tour]
        closed = np.vstack([ordered, ordered[0]])
        ax.plot(closed[:, 0], closed[:, 1], marker="o")

        for i, (x, y) in enumerate(cities):
            ax.text(x, y, str(i), fontsize=9)

        ax.set_title(f"{title}\nLength = {length:.6f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# Training

def train(seed=SEED, verbose=True, make_plots=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cities = create_tsp_instance(N_CITIES, TSP_INSTANCE_SEED)

    model = RealNVP(
        dimension=N_CITIES,
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    validation_generator = torch.Generator(device=DEVICE).manual_seed(seed + 1000)
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

    history_mean = []
    history_best = []

    for epoch in range(EPOCHS + 1):
        r = torch.randn(BATCH_SIZE, N_CITIES, device=DEVICE)

        with torch.no_grad():
            y, _ = model(r)
            tours = decode_permutation(y)
            lengths = tsp_length(cities, tours)
            reward = -lengths

        batch_best_index = torch.argmin(lengths)
        batch_best_length = lengths[batch_best_index].item()

        if batch_best_length < best_length:
            best_length = batch_best_length
            best_tour = tours[batch_best_index].detach().clone()

        if BATCH_SIZE > 1:
            baseline = (reward.sum() - reward) / (BATCH_SIZE - 1)
        else:
            baseline = reward.mean()

        advantage = reward - baseline
        advantage = advantage / (advantage.std() + 1e-8)

        log_prob = log_probability(model, y.detach())
        loss = -(advantage.detach() * log_prob).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        mean_length = lengths.mean().item()
        history_mean.append(mean_length)
        history_best.append(best_length)

        if epoch % VALIDATE_EVERY == 0:
            with torch.no_grad():
                validation_y, _ = model(validation)
                validation_tours = decode_permutation(validation_y)
                validation_lengths = tsp_length(cities, validation_tours)
                validation_mean = validation_lengths.mean().item()

                validation_best_index = torch.argmin(validation_lengths)
                validation_best_length = validation_lengths[validation_best_index].item()

                if validation_best_length < best_length:
                    best_length = validation_best_length
                    best_tour = validation_tours[validation_best_index].detach().clone()

            if validation_mean < best_validation_mean:
                best_validation_mean = validation_mean
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        if verbose and epoch % 50 == 0:
            print(
                f"{epoch:5d} | "
                f"mean length={mean_length:.6f} | "
                f"batch best={batch_best_length:.6f} | "
                f"best ever={best_length:.6f} | "
                f"loss={loss.item():.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_generator = torch.Generator(device=DEVICE).manual_seed(seed + 2000)
    test_r = torch.randn(
        TEST_SIZE,
        N_CITIES,
        generator=test_generator,
        device=DEVICE,
    )

    with torch.no_grad():
        test_y, _ = model(test_r)
        test_tours = decode_permutation(test_y)
        test_lengths = tsp_length(cities, test_tours)

    test_best_index = torch.argmin(test_lengths)
    test_best_length = test_lengths[test_best_index].item()
    test_best_tour = test_tours[test_best_index].detach().clone()
    test_mean_length = test_lengths.mean().item()

    if test_best_length < best_length:
        best_length = test_best_length
        best_tour = test_best_tour.detach().clone()

    metrics = {
        "seed": seed,
        "best_validation_epoch": best_epoch,
        "best_validation_mean": best_validation_mean,
        "test_mean_length": test_mean_length,
        "test_best_length": test_best_length,
        "best_length_ever": best_length,
        "best_tour": best_tour.detach().cpu().tolist(),
    }

    if verbose:
        print()
        print("FINAL RESULT")
        print(f"Best validation epoch: {best_epoch}")
        print(f"Best validation mean:  {best_validation_mean:.6f}")
        print(f"Test mean length:       {test_mean_length:.6f}")
        print(f"Test best length:       {test_best_length:.6f}")
        print(f"Best length ever:       {best_length:.6f}")
        print(f"Best tour:              {best_tour.detach().cpu().tolist()}")

    if make_plots:
        plot_convergence(history_mean, history_best)
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
