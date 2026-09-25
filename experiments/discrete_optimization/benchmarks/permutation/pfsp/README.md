# PFSP — Taillard Ta001

This directory contains the permutation flow-shop scheduling benchmark used for the RealNVP permutation-optimization experiments.

## Benchmark

The first benchmark instance is **Taillard Ta001**:

```text
jobs = 20
machines = 5
best-known makespan = 1278
```

A solution is a permutation of the 20 jobs. The same job order is used on all five machines, and every job visits the machines in the fixed order:

```text
M1 -> M2 -> M3 -> M4 -> M5
```

The optimization variable is therefore only the **job permutation**. The machine count is part of the objective landscape, not an additional decision dimension.

## Objective

For a permutation `pi`, let `p[j, m]` be the processing time of job `j` on machine `m`.

Completion times follow the standard PFSP recurrence:

```text
C[i, m] =
    max(C[i - 1, m], C[i, m - 1])
    + p[pi[i], m]
```

with zero boundary conditions.

The objective is to minimize the makespan:

```text
C_max = completion time of the final job on the final machine
```

For the RealNVP optimizer:

```python
reward = -makespan
```

## Representation

The representation matches the existing TSP and QAP experiments:

```text
z ~ N(0, I)
    ↓
RealNVP
    ↓
y ∈ R^20
    ↓
argsort(y)
    ↓
permutation of 20 jobs
    ↓
PFSP makespan
```

This keeps the continuous model and random-key decoder fixed while changing the permutation landscape.

## Sanity references

```text
identity permutation makespan = 1448
standard NEH makespan         = 1286
best-known Ta001 makespan     = 1278
```

The NEH heuristic is included as a benchmark sanity/reference method, not as the final evaluation-budget baseline.

## Experimental protocol

The controlled comparison uses:

```text
dimension = 20
decoder = argsort
architecture = 4 RealNVP coupling layers, hidden = 64
batch size = 1024
epochs = 3000
learning rate = 1e-4
validation size = 4096
test size = 16384
seeds = 42..51
```

The two compared methods are:

1. RealNVP + REINFORCE baseline
2. RealNVP + annealed KL

The annealed-KL run uses `T = 0.1 -> 0.001`.

## 10-seed results

| Metric | Baseline | Annealed KL |
| --- | ---: | ---: |
| Mean best makespan | 1295.8 | 1293.9 |
| Std best makespan | 2.040 | 2.644 |
| Median best makespan | 1296.5 | 1294.5 |
| Best run | 1290 | 1289 |
| Worst run | 1297 | 1297 |
| Best-known hits | 0/10 | 0/10 |
| Mean final generator makespan | 1297.003 | 1297.006 |
| Mean final unique permutations | 1550.0 / 16384 | 13754.4 / 16384 |
| Final unique fraction | 9.46% | 83.95% |

## Main observation

Annealed KL strongly increases permutation diversity and slightly improves the best solutions found, but it does **not** move the final generator away from the same objective plateau near makespan 1297.

The important distinction is therefore:

```text
permutation diversity != objective diversity
```

The baseline often loses both permutation diversity and reward variation. Annealed KL preserves many distinct permutations, yet many of those permutations still map to the same makespan. Once a training batch has nearly constant makespan, the REINFORCE advantage signal becomes negligible even though permutation diversity can remain high.

This makes Ta001 useful as a diagnostic case for objective-space plateaus rather than only classical permutation mode collapse.

## Files

Committed benchmark module:

- `objective.py` — Ta001 processing times, batched PyTorch makespan evaluator, reward, optimality gap, slow reference evaluator, NEH heuristic, and sanity tests.

Training scripts and 10-seed result CSVs have been generated locally and are the next artifacts to commit:

- `realnvp_pfsp_baseline.py`
- `realnvp_pfsp_kl.py`
- `results/permutation_optimization/pfsp/pfsp_ta001_realnvp_baseline_10seeds.csv`
- `results/permutation_optimization/pfsp/pfsp_ta001_realnvp_kl_10seeds.csv`

## Run objective sanity checks

From the repository root:

```bash
python experiments/discrete_optimization/benchmarks/permutation/pfsp/objective.py
```

Expected final line:

```text
All Ta001 checks passed.
```
