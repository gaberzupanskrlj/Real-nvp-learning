# RealNVP Learning

A research/learning repository for experimenting with **RealNVP normalizing flows** in PyTorch, from density estimation to continuous and discrete black-box optimization.

This branch contains a completed 100-seed benchmark comparison between a **RealNVP-based discrete optimizer** and a classical **(1+1)-EA**.

## Discrete optimization benchmark

Five 100-dimensional IOH/PBO problems are included:

- OneMax
- LeadingOnes
- ConcatenatedTrap
- NKLandscapes
- IsingTorus

The benchmark compares:

- a RealNVP search distribution trained with a REINFORCE-style objective;
- a standard elitist (1+1)-EA with bit mutation probability `1 / n`.

The final study uses:

```text
100 seeds per algorithm/problem
seeds 42 ... 141
dimension = 100
maximum budget = 4,096,000 objective evaluations
```

That gives **1,000 benchmark runs** in total.

The primary convergence metric is:

```text
best objective found so far vs. objective-function evaluations
```

Curves are aggregated as **mean ± 1 standard deviation** across 100 independent seeds.

## Final 100-seed results

Higher objective values are better for all five problems.

| Problem | RealNVP mean best ± std | (1+1)-EA mean best ± std | RealNVP target hits | EA target hits |
|---|---:|---:|---:|---:|
| OneMax | 100.000 ± 0.000 | 100.000 ± 0.000 | 100/100 | 100/100 |
| LeadingOnes | 77.300 ± 12.356 | 100.000 ± 0.000 | 1/100 | 100/100 |
| ConcatenatedTrap | 16.002 ± 0.020 | 16.350 ± 0.256 | 0/100 | 0/100 |
| NKLandscapes | -0.30309 ± 0.00360 | -0.29265 ± 0.00093 | n/a | n/a |
| IsingTorus | 172.160 ± 7.768 | 195.000 ± 8.704 | 1/100 | 75/100 |

Known targets are 100 for OneMax, 100 for LeadingOnes, 20 for ConcatenatedTrap and 200 for IsingTorus.

### Main findings

- **OneMax:** both algorithms are reliable, but the (1+1)-EA is far more sample-efficient. Mean evaluations to target are 1,027 for the EA versus 134,574 for RealNVP.
- **LeadingOnes:** RealNVP shows strong seed-to-seed instability and reaches the optimum in only 1/100 runs.
- **ConcatenatedTrap:** both methods struggle, but RealNVP almost deterministically collapses to the same deceptive local optimum near 16.
- **NKLandscapes:** the EA is consistently better and substantially less variable. Its mean is even better than the best RealNVP run observed in the 100-seed experiment.
- **IsingTorus:** the strongest reliability gap; RealNVP reaches 200 in 1/100 runs while the EA reaches it in 75/100.

The current **RealNVP + REINFORCE + threshold discretization** setup does not outperform the (1+1)-EA on these five benchmarks. The value of the study is that it reveals several distinct failure modes rather than a single generic weakness.

## Full report

- [Final 100-seed benchmark report](results/benchmark_100seeds/README.md)
- [Machine-readable benchmark summary](results/benchmark_100seeds/benchmark_summary.csv)
- [Benchmark methodology](experiments/discrete_optimization/benchmarks/README.md)

## Benchmark implementation

The active benchmark code is organized under:

```text
experiments/discrete_optimization/benchmarks/
├── realnvp/
└── baselines/
```

The 100-seed runner stores per-seed convergence data, per-seed result metadata, aggregate statistics and resumable output under:

```text
results/benchmark_100seeds/
```

The plotting workflow creates one convergence figure per problem under:

```text
results/benchmark_100seeds/figures/
```

## Runtime note

The main experiments were run in parallel for throughput. Parallel wall-clock times are affected by shared CPU resources and are not treated as the final algorithmic runtime comparison.

A separate timing experiment should use identical hardware with `workers=1` for both algorithms.

## Earlier exploratory benchmark

Older exploratory results are preserved under `results/benchmark/`. They used fewer and unequal numbers of seeds and should not be confused with the final 100-seed study.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## Other experiments

The repository also contains earlier density-estimation and continuous-optimization experiments, while legacy discrete scripts are preserved so the development of the optimizer remains traceable.
