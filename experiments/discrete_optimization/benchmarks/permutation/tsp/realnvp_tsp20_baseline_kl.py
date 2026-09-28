import copy
import csv
import math
from pathlib import Path

import numpy as np
import torch

import realnvp_tsp20_baseline as tsp


# Settings
# Frozen TSP-20 baseline (realnvp_tsp20_baseline.py) + annealed KL, nothing else changed.

SEED_START = 42
N_SEEDS = 10

T_START = 0.1
T_END = 0.001

REFERENCE_OPTIMUM = 3.513668846
OPTIMUM_TOLERANCE = 1e-5

RESULTS_CSV = Path(__file__).with_name(
    "tsp20_realnvp_baseline_kl_10seeds.csv"
)


# KL temperature, geometric: epoch 0 -> T_START, last epoch -> T_END

def kl_temperature(epoch):
    progress = epoch / tsp.EPOCHS
    return T_START * (T_END / T_START) ** progress


def gaussian_log_prob(x):
    return -0.5 * (x.square() + math.log(2.0 * math.pi)).sum(dim=1)


# Training, same loop as tsp.train with the KL term added to the loss

def train(seed, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cities = tsp.create_tsp_instance(tsp.N_CITIES, tsp.TSP_INSTANCE_SEED)

    model = tsp.RealNVP(
        dimension=tsp.N_CITIES,
        num_layers=tsp.NUM_LAYERS,
        hidden_dim=tsp.HIDDEN_DIM,
    ).to(tsp.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=tsp.LR)

    validation_generator = torch.Generator(device=tsp.DEVICE).manual_seed(seed + 1000)
    validation = torch.randn(
        tsp.VALIDATION_SIZE,
        tsp.N_CITIES,
        generator=validation_generator,
        device=tsp.DEVICE,
    )

    best_validation_mean = float("inf")
    best_state = None
    best_epoch = -1

    best_length = float("inf")
    best_tour = None

    for epoch in range(tsp.EPOCHS + 1):
        r = torch.randn(tsp.BATCH_SIZE, tsp.N_CITIES, device=tsp.DEVICE)

        # Keep y differentiable for the KL term
        y, forward_log_det = model(r)

        with torch.no_grad():
            tours = tsp.decode_permutation(y)
            lengths = tsp.tsp_length(cities, tours)
            reward = -lengths

        batch_best_index = torch.argmin(lengths)
        batch_best_length = lengths[batch_best_index].item()

        if batch_best_length < best_length:
            best_length = batch_best_length
            best_tour = tours[batch_best_index].detach().clone()

        if tsp.BATCH_SIZE > 1:
            baseline = (reward.sum() - reward) / (tsp.BATCH_SIZE - 1)
        else:
            baseline = reward.mean()

        advantage = reward - baseline
        advantage = advantage / (advantage.std() + 1e-8)

        # REINFORCE term
        log_prob = tsp.log_probability(model, y.detach())
        reinforce_loss = -(advantage.detach() * log_prob).mean()

        # KL(q_theta(y) || N(0, I)), log q(y) = log p(z) - forward log det
        kl_loss = (gaussian_log_prob(r) - forward_log_det - gaussian_log_prob(y)).mean()

        temperature = kl_temperature(epoch)
        loss = reinforce_loss + temperature * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tsp.MAX_GRAD_NORM)
        optimizer.step()

        mean_length = lengths.mean().item()

        if epoch % tsp.VALIDATE_EVERY == 0:
            with torch.no_grad():
                validation_y, _ = model(validation)
                validation_tours = tsp.decode_permutation(validation_y)
                validation_lengths = tsp.tsp_length(cities, validation_tours)
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
                f"T={temperature:.6f} | "
                f"reinforce={reinforce_loss.item():.6f} | "
                f"KL={kl_loss.item():.6f}"
            )

    # Final evaluation, same as the baseline
    if best_state is not None:
        model.load_state_dict(best_state)

    test_generator = torch.Generator(device=tsp.DEVICE).manual_seed(seed + 2000)
    test_r = torch.randn(
        tsp.TEST_SIZE,
        tsp.N_CITIES,
        generator=test_generator,
        device=tsp.DEVICE,
    )

    with torch.no_grad():
        test_y, _ = model(test_r)
        test_tours = tsp.decode_permutation(test_y)
        test_lengths = tsp.tsp_length(cities, test_tours)

    test_best_index = torch.argmin(test_lengths)
    test_best_length = test_lengths[test_best_index].item()
    test_best_tour = test_tours[test_best_index].detach().clone()
    test_mean_length = test_lengths.mean().item()

    # Extra diagnostics, not in the stored baseline CSV
    test_unique_tours = torch.unique(test_tours, dim=0).shape[0]
    test_optimum_fraction = (
        (test_lengths <= REFERENCE_OPTIMUM + OPTIMUM_TOLERANCE).float().mean().item()
    )

    if test_best_length < best_length:
        best_length = test_best_length
        best_tour = test_best_tour.detach().clone()

    return {
        "seed": seed,
        "best_validation_epoch": best_epoch,
        "best_validation_mean": best_validation_mean,
        "test_mean_length": test_mean_length,
        "test_best_length": test_best_length,
        "best_length_ever": best_length,
        "test_unique_tours": test_unique_tours,
        "test_optimum_fraction": test_optimum_fraction,
        "best_tour": best_tour.detach().cpu().tolist(),
    }


# Run all seeds and save the results

def main():
    seeds = list(range(SEED_START, SEED_START + N_SEEDS))
    results = []

    print("RealNVP TSP-20 frozen baseline + annealed KL")
    print(f"Seeds: {seeds[0]}-{seeds[-1]}")
    print(f"Layers: {tsp.NUM_LAYERS} | hidden: {tsp.HIDDEN_DIM} | batch: {tsp.BATCH_SIZE} | epochs: {tsp.EPOCHS} | lr: {tsp.LR}")
    print(f"KL temperature: {T_START} -> {T_END}, geometric")
    print()

    for i, seed in enumerate(seeds, start=1):
        print(f"[{i:02d}/{N_SEEDS:02d}] seed={seed} ...", flush=True)

        metrics = train(seed=seed, verbose=False)
        results.append(metrics)

        print(
            f"    best ever={metrics['best_length_ever']:.6f} | "
            f"test best={metrics['test_best_length']:.6f} | "
            f"test mean={metrics['test_mean_length']:.6f} | "
            f"unique={metrics['test_unique_tours']} | "
            f"P(optimum)={metrics['test_optimum_fraction']:.4f} | "
            f"best val epoch={metrics['best_validation_epoch']}",
            flush=True,
        )

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best_lengths = np.array([r["best_length_ever"] for r in results])
    test_means = np.array([r["test_mean_length"] for r in results])
    hits = sum(r["best_length_ever"] <= REFERENCE_OPTIMUM + OPTIMUM_TOLERANCE for r in results)

    print()
    print("SUMMARY")
    print(f"Best ever mean ± std:  {best_lengths.mean():.6f} ± {best_lengths.std(ddof=1):.6f}")
    print(f"Best ever median:      {np.median(best_lengths):.6f}")
    print(f"Optimum hits:          {hits}/{N_SEEDS}")
    print(f"Test mean mean ± std:  {test_means.mean():.6f} ± {test_means.std(ddof=1):.6f}")
    print(f"Mean unique tours:     {np.mean([r['test_unique_tours'] for r in results]):.1f} / {tsp.TEST_SIZE}")
    print(f"Mean P(optimum):       {np.mean([r['test_optimum_fraction'] for r in results]):.4f}")
    print(f"Saved results to:      {RESULTS_CSV}")


if __name__ == "__main__":
    main()
    