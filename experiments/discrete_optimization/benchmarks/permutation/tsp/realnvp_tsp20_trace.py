import copy
import csv
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import torch

import realnvp_tsp20_baseline as tsp
import realnvp_tsp20_baseline_kl as tsp_kl


# Settings
# Rerun of the frozen TSP-20 baseline and clean KL scripts with a best-so-far trace.
# Same loop as tsp.train / tsp_kl.train, only logging is added.
# Usage: python realnvp_tsp20_trace.py baseline|kl

METHOD = sys.argv[1] if len(sys.argv) > 1 else "baseline"
assert METHOD in ("baseline", "kl")
USE_KL = METHOD == "kl"

SEED_START = 42
N_SEEDS = 10

# Parallel seeds (48 threads on big.ijs.si)
N_WORKERS = 10
THREADS_PER_SEED = 4

REFERENCE_OPTIMUM = tsp_kl.REFERENCE_OPTIMUM
OPTIMUM_TOLERANCE = tsp_kl.OPTIMUM_TOLERANCE

RESULTS_CSV = Path(__file__).with_name(
    f"tsp20_realnvp_{METHOD}_trace_run_10seeds.csv"
)
TRACE_CSV = Path(__file__).with_name(
    f"tsp20_realnvp_{METHOD}_trace.csv"
)


# Training

def train(seed, trace):
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

    evaluations = 0

    for epoch in range(tsp.EPOCHS + 1):
        r = torch.randn(tsp.BATCH_SIZE, tsp.N_CITIES, device=tsp.DEVICE)

        if USE_KL:
            y, forward_log_det = model(r)
        else:
            with torch.no_grad():
                y, _ = model(r)

        with torch.no_grad():
            tours = tsp.decode_permutation(y)
            lengths = tsp.tsp_length(cities, tours)
            reward = -lengths

        evaluations += tsp.BATCH_SIZE

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

        log_prob = tsp.log_probability(model, y.detach())
        loss = -(advantage.detach() * log_prob).mean()

        if USE_KL:
            kl_loss = (
                tsp_kl.gaussian_log_prob(r)
                - forward_log_det
                - tsp_kl.gaussian_log_prob(y)
            ).mean()
            loss = loss + tsp_kl.kl_temperature(epoch) * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tsp.MAX_GRAD_NORM)
        optimizer.step()

        # Trace
        trace.append({
            "seed": seed,
            "epoch": epoch,
            "phase": "train",
            "evaluations": evaluations,
            "best_so_far": best_length,
            "sample_mean": lengths.mean().item(),
            "sample_best": batch_best_length,
        })

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

            evaluations += tsp.VALIDATION_SIZE

            trace.append({
                "seed": seed,
                "epoch": epoch,
                "phase": "validation",
                "evaluations": evaluations,
                "best_so_far": best_length,
                "sample_mean": validation_mean,
                "sample_best": validation_best_length,
            })

            if validation_mean < best_validation_mean:
                best_validation_mean = validation_mean
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        if epoch % 100 == 0:
            print(
                f"seed={seed} | {epoch:5d} | "
                f"mean length={lengths.mean().item():.6f} | "
                f"best ever={best_length:.6f} | "
                f"evals={evaluations:,}",
                flush=True,
            )

    # Final evaluation, same as the frozen scripts (test samples can update best-ever)
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


def run_seed(seed):
    torch.set_num_threads(THREADS_PER_SEED)
    trace = []
    result = train(seed, trace)
    return result, trace


# Run all seeds in parallel and save the results

def main():
    seeds = list(range(SEED_START, SEED_START + N_SEEDS))

    print(f"RealNVP TSP-20 trace rerun: {METHOD}")
    print(f"Seeds: {seeds[0]}-{seeds[-1]} | workers: {N_WORKERS} x {THREADS_PER_SEED} threads")
    print(f"Layers: {tsp.NUM_LAYERS} | hidden: {tsp.HIDDEN_DIM} | batch: {tsp.BATCH_SIZE} | epochs: {tsp.EPOCHS} | lr: {tsp.LR}")
    if USE_KL:
        print(f"KL temperature: {tsp_kl.T_START} -> {tsp_kl.T_END}, geometric")
    print(flush=True)

    with multiprocessing.get_context("spawn").Pool(N_WORKERS) as pool:
        outputs = pool.map(run_seed, seeds)

    results = [result for result, _ in outputs]
    trace = [row for _, seed_trace in outputs for row in seed_trace]

    print()
    for result in results:
        print(
            f"seed={result['seed']} | "
            f"best ever={result['best_length_ever']:.6f} | "
            f"test best={result['test_best_length']:.6f} | "
            f"test mean={result['test_mean_length']:.6f} | "
            f"unique={result['test_unique_tours']} | "
            f"best val epoch={result['best_validation_epoch']}"
        )

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with open(TRACE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)

    hits = sum(r["best_length_ever"] <= REFERENCE_OPTIMUM + OPTIMUM_TOLERANCE for r in results)
    print()
    print(f"Optimum hits: {hits}/{N_SEEDS}")
    print(f"Saved results to: {RESULTS_CSV}")
    print(f"Saved trace to:   {TRACE_CSV}")


if __name__ == "__main__":
    main()
