# Discrete optimization benchmark

This folder contains the implementation used for the final 100-seed comparison between a **RealNVP-based discrete optimizer** and a classical **(1+1)-EA** on IOH/PBO problems.

## Problems

All benchmarks use dimension 100:

- OneMax
- LeadingOnes
- ConcatenatedTrap
- NKLandscapes
- IsingTorus

## Algorithms

### RealNVP

The optimizer learns a search distribution with affine coupling layers. Samples are thresholded into binary solutions and the model is trained with a REINFORCE-style objective using exact RealNVP log-probabilities.

```text
advantage = reward - leave-one-out baseline
loss = -(advantage * log_prob).mean()
```

Gradient clipping is set to 5.0.

### (1+1)-EA

The baseline is a standard elitist (1+1)-EA with independent bit mutation probability:

```text
p = 1 / dimension
```

An offspring is accepted when its objective value is at least as good as the current parent.

## Final benchmark protocol

Both algorithms use:

```text
seeds = 42, 43, ..., 141
n = 100
dimension = 100
maximum budget = 4,096,000 objective-function evaluations
```

For RealNVP, every objective call that can influence optimization is counted, including:

- training-batch evaluations;
- validation evaluations used for checkpoint selection.

The final test set is reported separately and is not counted toward the optimization budget.

The main comparison is:

```text
x = objective-function evaluations
y = best objective value found so far
```

Convergence curves are aggregated as:

```text
mean best-so-far ± 1 standard deviation
```

## Known targets

| Problem | Target |
|---|---:|
| OneMax | 100 |
| LeadingOnes | 100 |
| ConcatenatedTrap | 20 |
| IsingTorus | 200 |
| NKLandscapes | no fixed target |

For problems with a known target, the runner additionally records:

- target hit rate;
- evaluations to target;
- time to target.

The (1+1)-EA stops a run once the known global target is reached. In the convergence plots, that terminal optimum is forward-filled to the common 4,096,000-evaluation horizon. This does not imply extra objective calls after the optimum was found; it only represents the best-so-far value, which cannot improve beyond a known global optimum.

## Final results

| Problem | RealNVP mean best ± std | (1+1)-EA mean best ± std | RealNVP target hits | EA target hits |
|---|---:|---:|---:|---:|
| OneMax | 100.000 ± 0.000 | 100.000 ± 0.000 | 100/100 | 100/100 |
| LeadingOnes | 77.300 ± 12.356 | 100.000 ± 0.000 | 1/100 | 100/100 |
| ConcatenatedTrap | 16.002 ± 0.020 | 16.350 ± 0.256 | 0/100 | 0/100 |
| NKLandscapes | -0.30309 ± 0.00360 | -0.29265 ± 0.00093 | n/a | n/a |
| IsingTorus | 172.160 ± 7.768 | 195.000 ± 8.704 | 1/100 | 75/100 |

For detailed interpretation, see [the final benchmark report](../../../results/benchmark_100seeds/README.md).

## Output layout

```text
results/benchmark_100seeds/
├── realnvp/
│   └── <problem>/
│       ├── config.json
│       ├── summary.csv
│       ├── aggregate.json
│       └── seed_XXX/
│           ├── convergence.csv
│           └── result.json
├── one_plus_one_ea/
│   └── <problem>/
├── figures/
└── benchmark_summary.csv
```

Each seed is saved independently. Existing `result.json` files are detected on restart, so interrupted runs resume without repeating completed seeds.

Result files are first written to temporary paths and then renamed into place to reduce the chance of leaving partially written outputs after an interruption.

## Runtime methodology

The 100-seed experiments were run in parallel for throughput, typically using 24 CPU workers with one PyTorch/BLAS thread per worker.

Those wall-clock times are influenced by cache, memory-bandwidth and scheduling contention. They should not be treated as the final runtime comparison.

A separate timing experiment should use:

- the same machine;
- `workers=1`;
- identical runtime conditions;
- the same fixed seed set.

The final statistical benchmark is therefore primarily **evaluation-based**.

## Exploratory results

Older results under `results/benchmark/` are preserved as development history. They used smaller and unequal seed counts and should not be interpreted as the final statistical study.
