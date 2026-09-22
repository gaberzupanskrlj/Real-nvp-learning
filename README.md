# RealNVP Learning

A research/learning repository for experimenting with **RealNVP normalizing flows** in PyTorch, from density estimation to continuous and discrete black-box optimization.

This branch focuses on a structured benchmark comparison between a **RealNVP-based discrete optimizer** and a classical **(1+1)-EA**.

## Project structure

```text
Real-nvp-learning/
├── experiments/
│   ├── density_estimation/
│   ├── continuous_optimization/
│   └── discrete_optimization/
│       ├── benchmarks/
│       │   ├── realnvp/
│       │   └── baselines/
│       └── legacy/
├── data/
├── results/
├── plotting/
├── notes/
├── README.md
├── requirements.txt
└── .gitignore
```

## Discrete optimization benchmark

Five 100-dimensional IOH/PBO problems are currently included:

- OneMax
- LeadingOnes
- ConcatenatedTrap
- NKLandscapes
- IsingTorus

The benchmark compares:

- a RealNVP search distribution trained with a REINFORCE-style objective;
- a standard elitist (1+1)-EA with bit mutation probability `1 / n`.

The earlier exploratory experiments are preserved under `results/benchmark/`. They were useful during development but used unequal numbers of seeds and incomplete objective-evaluation accounting.

The current study replaces that exploratory comparison with a controlled **100-seed statistical benchmark**.

### Current 100-seed protocol

Both algorithms use the same seed range:

```text
42 ... 141
```

with:

```text
100 independent runs per algorithm/problem
4,096,000 objective-function evaluations maximum
dimension = 100
```

The primary convergence metric is:

```text
best objective found so far vs. objective-function evaluations
```

For RealNVP, training and validation objective calls are both included in the optimization budget because validation is used for checkpoint selection. The final test set is reported separately and is not counted toward the optimization budget.

For each problem, convergence is aggregated over all seeds using:

```text
mean best-so-far ± 1 standard deviation
```

For problems with a known optimum, the benchmark also records target hit rate, evaluations to target and time to target.

### Completed 100-seed result: OneMax

The first full comparison is complete:

| Metric | RealNVP | (1+1)-EA |
|---|---:|---:|
| Runs | 100 | 100 |
| Target hits | 100 | 100 |
| Mean best | 100.0 | 100.0 |
| Mean evaluations to target | 134,574.08 | 1,027.44 |
| Median evaluations to target | 135,168 | 962.5 |

Both methods reached the optimum in every run. On OneMax, the (1+1)-EA reached it with substantially fewer objective evaluations.

LeadingOnes and the remaining benchmark problems are being evaluated under the same protocol.

### Runtime note

The 100-seed experiments are run in parallel to improve throughput. Parallel wall-clock times are affected by shared CPU resources and are therefore **not used as the final runtime comparison**.

A separate timing experiment should use the same hardware with `workers=1` and identical conditions for both algorithms.

For the full benchmark methodology and output format, see:

- [Discrete optimization benchmark protocol](experiments/discrete_optimization/benchmarks/README.md)

## Benchmark outputs

The statistical benchmark stores per-seed and aggregate outputs under:

```text
results/benchmark_100seeds/
```

Each seed records its own convergence curve and result metadata, which allows interrupted experiments to resume without rerunning completed seeds.

Plots compare the algorithms using objective evaluations on the x-axis rather than epochs or optimizer steps.

## Earlier exploratory benchmark

The older benchmark results remain available for development history:

- [Benchmark report](results/benchmark/README.md)
- [Raw results](results/benchmark/raw_results.md)
- [Benchmark summary CSV](results/benchmark/benchmark_summary.csv)
- [Original plotting script](plotting/plot_benchmark_comparison.py)

Those results should be treated as exploratory rather than as the final statistical comparison.

## Other experiments

### Density estimation

- Ring distribution transformed with RealNVP.
- Historical Two Moons experiment.

### Continuous optimization

- 2D multimodal objective on a discretized 50×50 grid.
- Uniform and Gaussian base-distribution variants.
- 10D extensions projected into a 2D objective space.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

Older exploratory scripts are intentionally preserved under `legacy/` so that changes in the RealNVP optimization method remain traceable without cluttering the active benchmark code.
