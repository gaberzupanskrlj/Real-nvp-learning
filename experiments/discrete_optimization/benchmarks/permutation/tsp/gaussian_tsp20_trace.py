import copy
import csv
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import realnvp_tsp20_baseline as tsp
import realnvp_tsp20_baseline_kl as tsp_kl


# Settings
# Learned diagonal Gaussian random-keys baseline for TSP-20.
# Same loop and bookkeeping as realnvp_tsp20_trace.py, only RealNVP is replaced by
# y = mu + sigma * z, i.e. q(y) = N(mu, diag(sigma^2)) with 40 parameters.
# Usage:
#   python gaussian_tsp20_trace.py baseline|kl sweep   LR sweep on tuning seeds 0..2
#   python gaussian_tsp20_trace.py baseline|kl LR      main run on seeds 42..51

METHOD = sys.argv[1] if len(sys.argv) > 1 else "baseline"
assert METHOD in ("baseline", "kl")
USE_KL = METHOD == "kl"

MODE = sys.argv[2] if len(sys.argv) > 2 else "sweep"

if MODE == "sweep":
    LRS = [1e-3, 3e-3, 1e-2, 3e-2]
    SEEDS = [0, 1, 2]
    RUN_NAME = f"gaussian_{METHOD}_sweep"
else:
    LRS = [float(MODE)]
    SEEDS = list(range(42, 52))
    RUN_NAME = f"gaussian_{METHOD}_lr{LRS[0]:g}"

# Parallel jobs (48 threads on big.ijs.si)
MAX_WORKERS = 12
THREADS_PER_SEED = 3

REFERENCE_OPTIMUM = tsp_kl.REFERENCE_OPTIMUM
OPTIMUM_TOLERANCE = tsp_kl.OPTIMUM_TOLERANCE

RESULTS_CSV = Path(__file__).with_name(f"tsp20_{RUN_NAME}_results.csv")
TRACE_CSV = Path(__file__).with_name(f"tsp20_{RUN_NAME}_trace.csv")


# Diagonal Gaussian model
# Same interface as tsp.RealNVP: forward(z) -> (y, log_det), inverse(y) -> (z, log_det),
# so tsp.log_probability and the KL term are used unchanged.
# Initialised at mu = 0, sigma = 1, i.e. q(y) = N(0, I).

class DiagonalGaussian(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dimension))
        self.log_sigma = nn.Parameter(torch.zeros(dimension))

    def forward(self, z):
        y = self.mu + torch.exp(self.log_sigma) * z
        log_det = self.log_sigma.sum().expand(z.shape[0])
        return y, log_det

    def inverse(self, y):
        z = (y - self.mu) * torch.exp(-self.log_sigma)
        log_det = -self.log_sigma.sum().expand(y.shape[0])
        return z, log_det


# Training

def train(seed, lr, trace):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cities = tsp.create_tsp_instance(tsp.N_CITIES, tsp.TSP_INSTANCE_SEED)

    model = DiagonalGaussian(tsp.N_CITIES).to(tsp.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            "lr": lr,
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
                "lr": lr,
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
                f"lr={lr:g} seed={seed} | {epoch:5d} | "
                f"mean length={lengths.mean().item():.6f} | "
                f"best ever={best_length:.6f} | "
                f"mean sigma={torch.exp(model.log_sigma).mean().item():.4f} | "
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
        "lr": lr,
        "seed": seed,
        "best_validation_epoch": best_epoch,
        "best_validation_mean": best_validation_mean,
        "test_mean_length": test_mean_length,
        "test_best_length": test_best_length,
        "best_length_ever": best_length,
        "test_unique_tours": test_unique_tours,
        "test_optimum_fraction": test_optimum_fraction,
        "final_mean_sigma": torch.exp(model.log_sigma).mean().item(),
        "best_tour": best_tour.detach().cpu().tolist(),
    }


def run_job(job):
    seed, lr = job
    torch.set_num_threads(THREADS_PER_SEED)
    trace = []
    result = train(seed, lr, trace)
    return result, trace


# Run all jobs in parallel and save the results

def main():
    jobs = [(seed, lr) for lr in LRS for seed in SEEDS]
    n_workers = min(len(jobs), MAX_WORKERS)

    print(f"Diagonal Gaussian TSP-20: {METHOD} | mode: {MODE}")
    print(f"LRs: {LRS} | seeds: {SEEDS} | workers: {n_workers} x {THREADS_PER_SEED} threads")
    print(f"Batch: {tsp.BATCH_SIZE} | epochs: {tsp.EPOCHS}")
    if USE_KL:
        print(f"KL temperature: {tsp_kl.T_START} -> {tsp_kl.T_END}, geometric")
    print(flush=True)

    with multiprocessing.get_context("spawn").Pool(n_workers) as pool:
        outputs = pool.map(run_job, jobs)

    results = [result for result, _ in outputs]
    trace = [row for _, job_trace in outputs for row in job_trace]

    print()
    for result in results:
        print(
            f"lr={result['lr']:g} seed={result['seed']} | "
            f"best ever={result['best_length_ever']:.6f} | "
            f"test best={result['test_best_length']:.6f} | "
            f"test mean={result['test_mean_length']:.6f} | "
            f"unique={result['test_unique_tours']} | "
            f"sigma={result['final_mean_sigma']:.4f} | "
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

    # Summary per LR
    # Sweep selection rule: lowest mean best-ever, ties broken by lowest mean test length.
    print()
    for lr in LRS:
        lr_results = [r for r in results if r["lr"] == lr]
        hits = sum(r["best_length_ever"] <= REFERENCE_OPTIMUM + OPTIMUM_TOLERANCE for r in lr_results)
        mean_best = np.mean([r["best_length_ever"] for r in lr_results])
        mean_test = np.mean([r["test_mean_length"] for r in lr_results])
        print(
            f"lr={lr:g} | hits={hits}/{len(lr_results)} | "
            f"mean best ever={mean_best:.6f} | mean test length={mean_test:.6f}"
        )

    print()
    print(f"Saved results to: {RESULTS_CSV}")
    print(f"Saved trace to:   {TRACE_CSV}")


if __name__ == "__main__":
    main()
