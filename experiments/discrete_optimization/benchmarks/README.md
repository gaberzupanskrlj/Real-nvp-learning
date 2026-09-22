# Discrete optimization benchmark scripts

This folder contains the benchmark implementation used to compare a **RealNVP-based discrete optimizer** with a classical **(1+1)-EA** on IOH/PBO problems.

The repository now contains two layers of experiments:

1. the earlier exploratory benchmark scripts;
2. the current **100-seed statistical benchmark**, designed for fair convergence comparisons.

## Problems

All current benchmark problems use dimension 100:

- OneMax
- LeadingOnes
- ConcatenatedTrap
- NKLandscapes
- IsingTorus

## Algorithms

### RealNVP

The optimizer learns a search distribution with affine coupling layers and a REINFORCE-style objective based on exact RealNVP log-probabilities.

The current training signal is

```text
advantage = reward - leave-one-out baseline
loss = -(advantage * log_prob).mean()
```

Gradient clipping is set to 5.0.

### (1+1)-EA

The baseline is a standard elitist (1+1)-EA with independent bit mutation probability

```text
p = 1 / dimension
```

and offspring accepted when its objective value is at least as good as the current parent.

## 100-seed benchmark protocol

The current benchmark uses the same seed range for both algorithms:

```text
seeds = 42, 43, ..., 141
n = 100
```

The common optimization budget is:

```text
4,096,000 objective-function evaluations
```

For RealNVP, **all objective calls that affect optimization are counted**, including:

- training-batch evaluations;
- validation evaluations used for checkpoint selection.

The final test set is reported separately and is **not counted as optimization budget**.

The primary comparison axis is therefore:

```text
x = objective-function evaluations
y = best objective value found so far
```

For each problem, convergence curves are aggregated over 100 independent seeds and plotted as:

```text
mean best-so-far ± 1 standard deviation
```

## Target metrics

For problems with a known optimum, the benchmark also records:

- target hit rate;
- evaluations to target;
- time to target.

The current targets are:

| Problem | Target |
|---|---:|
| OneMax | 100 |
| LeadingOnes | 100 |
| ConcatenatedTrap | 20 |
| IsingTorus | 200 |
| NKLandscapes | no fixed target |

## Runtime methodology

The 100-seed experiments are executed in parallel for throughput, typically with 24 CPU workers and one PyTorch/BLAS thread per worker.

Because parallel workers compete for cache, memory bandwidth and other shared CPU resources, their wall-clock times are **not treated as the final algorithmic runtime comparison**.

A separate timing study should use:

- the same machine;
- `workers=1`;
- identical runtime conditions;
- a smaller fixed seed set.

The primary statistical benchmark remains evaluation-based.

## Output layout

The 100-seed runs write results under:

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
└── figures/
```

Each seed is saved independently, allowing interrupted experiments to resume without repeating completed runs.

Results are first written to temporary files and then renamed into place so that an interrupted process is less likely to leave a partially written result file.

## Current completed result

The first completed 100-seed comparison is **OneMax**.

| Metric | RealNVP | (1+1)-EA |
|---|---:|---:|
| Runs | 100 | 100 |
| Target hits | 100 | 100 |
| Mean best | 100.0 | 100.0 |
| Mean evaluations to target | 134,574.08 | 1,027.44 |
| Median evaluations to target | 135,168 | 962.5 |

Both algorithms reached the optimum in every run, while the (1+1)-EA required far fewer objective evaluations on this simple incremental problem.

LeadingOnes and the remaining benchmarks are currently being evaluated under the same protocol.

## Exploratory results

The earlier results under `results/benchmark/` were useful for model development, but they used unequal seed counts and should not be interpreted as the final statistical comparison.

They are preserved for reproducibility and to show the development history of the RealNVP optimizer.
