# TSP permutation optimization

This directory contains the permutation-valued part of the RealNVP optimization project.

The benchmark problem is a **symmetric Euclidean TSP with 20 cities**. City coordinates are sampled once in the unit square and kept fixed with `TSP_INSTANCE_SEED = 12345`.

## Representation

RealNVP remains continuous. A generated vector `y` is decoded into a valid permutation using random keys:

```python
tour = torch.argsort(y, dim=1)
```

The objective is the closed-tour length. The frozen RealNVP baseline uses reward `-tour_length` with a REINFORCE-style score-function estimator and a leave-one-out baseline.

## Active structure

```text
tsp/
├── README.md
├── realnvp_tsp_baseline_v1.py
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

### Authoritative baseline

`realnvp_tsp_baseline_v1.py` is the frozen reference implementation. It should not be modified when testing new ideas.

Baseline configuration:

| Parameter | Value |
|---|---:|
| Cities | 20 |
| Instance seed | 12345 |
| RealNVP layers | 4 |
| Hidden dimension | 64 |
| Batch size | 1024 |
| Epochs | 2000 |
| Learning rate | 1e-4 |
| Reference optimum | 3.513668846 |

### Classical reference

`baselines/inversion_baseline_10seeds.py` runs an elitist inversion local-search baseline for seeds 42–51.

Its fast delta evaluation counts candidate inversion moves rather than full black-box tour recomputations, so its evaluation count should not be interpreted as directly equivalent to RealNVP's sampled-tour count.

### Diagnostics

`diagnostics/MODE_COLLAPSE_ANALYSIS_2026-09-23.md` records the focused investigation that showed an important distinction:

> RealNVP can discover the global optimum, while the learned sampling distribution can later lose diversity and concentrate around a worse tour basin.

The document separates established observations from hypotheses and keeps the exploratory rank/multiplicity interventions out of the main method list.

## Current published TSP-20 result

Across the existing 10-seed comparison:

| Metric | RealNVP baseline | Inversion baseline |
|---|---:|---:|
| Seeds | 10 | 10 |
| Global optimum hits | 3/10 | 4/10 |
| Mean best length | 3.586519 | 3.524057 |
| Std best length | 0.211928 | 0.017981 |
| Median best length | 3.522437 | 3.522437 |
| Best run | 3.513669 | 3.513669 |
| Worst run | 4.189561 | 3.573708 |
| Mean optimality gap | 2.0733% | 0.2956% |

Detailed results are under [`results/permutation_optimization/tsp20/`](../../../../results/permutation_optimization/tsp20/).

## Run

From the repository root:

```bash
python experiments/discrete_optimization/permutation_optimization/tsp/realnvp_tsp_baseline_v1.py
```

```bash
python experiments/discrete_optimization/permutation_optimization/tsp/baselines/inversion_baseline_10seeds.py
```

## Naming rule for new experiments

New methods should use descriptive names based on the intervention, not scratch version numbers. For example:

```text
realnvp_tsp_<method_name>.py
```

Each method should keep the frozen problem definition and clearly document which training, exploration, checkpointing or regularization choice changed.

Exploratory or superseded scripts should be moved to an archive rather than left beside the active methods.
