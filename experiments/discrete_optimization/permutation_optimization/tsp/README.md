# TSP permutation optimization

This directory contains the permutation-valued part of the RealNVP optimization project.

The benchmark problem is a **symmetric Euclidean TSP with 20 cities**. City coordinates are sampled once in the unit square and kept fixed with `TSP_INSTANCE_SEED = 12345`.

## Representation

RealNVP remains continuous. A generated vector `y` is decoded into a valid permutation using random keys:

```python
tour = torch.argsort(y, dim=1)
```

The objective is the closed-tour length.

## Active structure

```text
tsp/
├── README.md
├── realnvp_tsp_baseline_v1.py
├── realnvp_tsp_cosine_elite.py
├── realnvp_tsp_kl.py
├── plot_method_comparison.py
├── baselines/
│   └── inversion_baseline_10seeds.py
├── diagnostics/
│   └── MODE_COLLAPSE_ANALYSIS_2026-09-23.md
└── archive/
    └── legacy/
        ├── README.md
        ├── realnvp_tsp.py
        ├── realnvp_tsp_10seeds.py
        └── simple_tsp_baseline.py
```

## Methods

### Frozen RealNVP baseline

`realnvp_tsp_baseline_v1.py` is the immutable reference implementation:

- 4 coupling layers
- hidden dimension 64
- batch size 1024
- 2000 epochs
- fixed learning rate `1e-4`
- Gaussian base
- leave-one-out REINFORCE
- standardized advantage
- validation-mean checkpoint

### Cosine LR + elite checkpoint

`realnvp_tsp_cosine_elite.py` tests two interventions:

- 8-layer / batch-4096 diagnostic-capacity setup
- cosine learning-rate decay `1e-4 -> 1e-5`
- checkpoint selected using the best 1% of validation samples

Across 10 seeds it discovers the reference optimum in **8/10** runs, but final generator quality is unstable.

### Annealed KL

`realnvp_tsp_kl.py` adds an annealed KL regularizer to the REINFORCE objective:

```text
loss = reinforce_loss + T * KL(q_theta || N(0, I))
T: 0.1 -> 0.001
```

Across 10 seeds it discovers the reference optimum in **10/10** runs and gives a much more stable final generator, but the final distribution retains the optimum as its mode in only **1/10** runs.

### Classical reference

`baselines/inversion_baseline_10seeds.py` is an elitist inversion local-search baseline.

Its fast delta evaluation counts candidate inversion moves rather than full black-box tour recomputations, so its evaluation count is not directly equivalent to RealNVP sampled-tour evaluations.

## Main conclusion so far

The experiments separate two behaviors:

1. **discovery** — whether training ever finds the global optimum;
2. **retention** — whether the learned generator assigns substantial final probability mass to that optimum.

Cosine + elite improves discovery but is unstable at retention. Annealed KL reaches 10/10 discovery and stabilizes the final generator, but 9/10 runs still concentrate on the same near-optimal basin around length `3.522437`.

Detailed 10-seed results and comparison figures are under [the TSP-20 results directory](../../../../results/permutation_optimization/tsp20/).

## Run

From the repository root:

```bash
python experiments/discrete_optimization/permutation_optimization/tsp/realnvp_tsp_baseline_v1.py
```

```bash
python experiments/discrete_optimization/permutation_optimization/tsp/realnvp_tsp_cosine_elite.py --seeds 42 43 44 45 46 47 48 49 50 51
```

```bash
python experiments/discrete_optimization/permutation_optimization/tsp/realnvp_tsp_kl.py
```

To regenerate the method-level comparison figures:

```bash
python experiments/discrete_optimization/permutation_optimization/tsp/plot_method_comparison.py
```

Exploratory or superseded scripts should be kept in an archive rather than mixed with the active methods.
