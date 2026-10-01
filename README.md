# RealNVP Learning

A research/learning repository for experimenting with **RealNVP normalizing flows** in PyTorch, from density estimation to continuous and discrete black-box optimization.

The current discrete-optimization work contains a completed 100-seed binary benchmark and permutation studies on **TSP, QAP and PFSP**.

> **Project status: concluded October 2026.** Start with the handoff: [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) gives the story, main findings and next steps; [`PROJECT_HANDOFF_TECHNICAL.md`](PROJECT_HANDOFF_TECHNICAL.md) has the detailed results, protocol and pitfalls. The 2D basin study is described in [`experiments/continuous_optimization/README.md`](experiments/continuous_optimization/README.md).

## Binary optimization benchmark

Five 100-dimensional IOH/PBO problems are compared using RealNVP + REINFORCE and a standard elitist (1+1)-EA:

- OneMax
- LeadingOnes
- ConcatenatedTrap
- NKLandscapes
- IsingTorus

Protocol:

```text
100 seeds per algorithm/problem
seeds 42 ... 141
dimension = 100
maximum budget = 4,096,000 objective evaluations
```

### Final 100-seed results

Higher objective values are better for all five problems.

| Problem | RealNVP mean best ± std | (1+1)-EA mean best ± std | RealNVP target hits | EA target hits |
|---|---:|---:|---:|---:|
| OneMax | 100.000 ± 0.000 | 100.000 ± 0.000 | 100/100 | 100/100 |
| LeadingOnes | 77.300 ± 12.356 | 100.000 ± 0.000 | 1/100 | 100/100 |
| ConcatenatedTrap | 16.002 ± 0.020 | 16.350 ± 0.256 | 0/100 | 0/100 |
| NKLandscapes | -0.30309 ± 0.00360 | -0.29265 ± 0.00093 | n/a | n/a |
| IsingTorus | 172.160 ± 7.768 | 195.000 ± 8.704 | 1/100 | 75/100 |

Known targets are 100 for OneMax, 100 for LeadingOnes, 20 for ConcatenatedTrap and 200 for IsingTorus.

- **OneMax:** both algorithms are reliable, but the (1+1)-EA is far more sample-efficient. Mean evaluations to target are 1,027 for the EA versus 134,574 for RealNVP.
- **LeadingOnes:** RealNVP shows strong seed-to-seed instability and reaches the optimum in only 1/100 runs.
- **ConcatenatedTrap:** both methods struggle, but RealNVP almost deterministically collapses to the same deceptive local optimum near 16.
- **NKLandscapes:** the EA is consistently better and substantially less variable.
- **IsingTorus:** RealNVP reaches 200 in 1/100 runs while the EA reaches it in 75/100.

The current threshold-discretized RealNVP setup does not outperform the (1+1)-EA on these benchmarks. The useful result is the separation of failure modes: sample inefficiency, seed instability, deceptive-basin collapse, lower-quality plateaus and low optimum-hit reliability.

- [Final binary report](results/benchmark_100seeds/README.md)
- [Machine-readable summary](results/benchmark_100seeds/benchmark_summary.csv)
- [Benchmark methodology](experiments/discrete_optimization/benchmarks/README.md)

## Permutation optimization

Permutation experiments use a shared random-key representation:

```text
latent sample -> RealNVP -> continuous keys -> argsort -> permutation -> objective
```

### Latest result: RealNVP + KL vs (1+1)-EA convergence

The current comparison is in [`results/permutation_optimization/anytime/`](results/permutation_optimization/anytime/). Both methods log their best-so-far objective value against the number of objective evaluations, 10 seeds (42–51) each, with a budget of 3,317,760 evaluations per seed (RealNVP TSP-20: 2,216,960, the frozen 2000-epoch protocol). RealNVP evaluations include its validation samples.

| Problem | Reference | (1+1)-EA median final best | RealNVP + KL median final best | Runs reaching reference (EA / RealNVP) |
|---|---:|---:|---:|---:|
| TSP-20 | optimum 3.513669 | 3.513669 | 3.513669 | 10/10 / 8/10 |
| QAP Nug20 | optimum 2570 | 2621 | 2802 | 0/10 / 0/10 |
| PFSP Ta001 | best-known 1278 | 1278 | 1295.5 | 10/10 / 0/10 |

![TSP-20 convergence](results/permutation_optimization/anytime/tsp20_clean_convergence.png)
![QAP Nug20 convergence](results/permutation_optimization/anytime/qap_nug20_clean_convergence.png)
![PFSP Ta001 convergence](results/permutation_optimization/anytime/pfsp_ta001_clean_convergence.png)

The figures are made by [`plotting/clean_anytime_plots.py`](plotting/clean_anytime_plots.py) from the `*_trace.csv` files in the same folder. The per-problem sections below and the `tsp20/`, `qap/` and `pfsp/` result folders hold the earlier experiments.

### TSP

The TSP folder now separates frozen TSP-20 scripts from TSP-50 scaling experiments with explicit filenames.

- [TSP experiments](experiments/discrete_optimization/benchmarks/permutation/tsp/README.md)
- [TSP-20 results](results/permutation_optimization/tsp20/README.md)
- [Gaussian exploration ablation](results/permutation_optimization/tsp20/gaussian_exploration/README.md)
- [Mode-collapse investigation](experiments/discrete_optimization/legacy/permutation_optimization_old/tsp/diagnostics/MODE_COLLAPSE_ANALYSIS_2026-09-23.md)

### QAP

QAP uses the standard QAPLIB Nug20 instance and compares the frozen RealNVP baseline with annealed KL.

- [QAP experiments](experiments/discrete_optimization/benchmarks/permutation/qap/README.md)
- [QAP results](results/permutation_optimization/qap/README.md)

### PFSP

PFSP uses Taillard Ta001 and exposes an important distinction: preserving many different permutations does not necessarily preserve useful objective variation.

- [PFSP experiments](experiments/discrete_optimization/benchmarks/permutation/pfsp/README.md)

## Active structure

```text
experiments/discrete_optimization/benchmarks/
├── binary/
│   ├── ioh/
│   │   ├── realnvp/
│   │   └── baselines/
│   └── jump/
└── permutation/
    ├── tsp/
    ├── qap/
    └── pfsp/
```

Final aggregate results live under `results/`; older exploratory implementations are kept under `experiments/discrete_optimization/legacy/`.

## Runtime note

The main experiments were run in parallel for throughput. Parallel wall-clock times are affected by shared CPU resources and are not treated as the final algorithmic runtime comparison.

A separate timing experiment should use identical hardware with `workers=1` for both algorithms.

## Earlier exploratory benchmark

Older exploratory binary results are preserved under `results/benchmark/`. They used fewer and unequal numbers of seeds and should not be confused with the final 100-seed study.

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

Earlier density-estimation experiments remain under `experiments/density_estimation/`. `experiments/continuous_optimization/` holds the 2D basin study (see its README) next to older continuous toy scripts.
