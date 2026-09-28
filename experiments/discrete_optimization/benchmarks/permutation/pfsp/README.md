# PFSP — Taillard Ta001

This directory contains the RealNVP permutation flow-shop scheduling benchmark on **Taillard Ta001**.

## Benchmark

```text
jobs = 20
machines = 5
best-known makespan = 1278
```

A solution is a permutation of the 20 jobs. The same job order is used on all five machines.

For a permutation `pi`, completion times follow:

```text
C[i, m] = max(C[i - 1, m], C[i, m - 1]) + p[pi[i], m]
```

The objective is the final makespan and RealNVP uses `reward = -makespan`.

## Representation

```text
z ~ N(0, I)
    ↓
RealNVP
    ↓
y ∈ R^20
    ↓
argsort(y)
    ↓
job permutation
    ↓
PFSP makespan
```

## Sanity references

```text
identity permutation makespan = 1448
standard NEH makespan         = 1286
best-known Ta001 makespan     = 1278
```

## Experimental protocol

```text
dimension = 20
architecture = 4 RealNVP coupling layers, hidden = 64
batch size = 1024
epochs = 3000
learning rate = 1e-4
validation size = 4096
test size = 16384
seeds = 42..51
```

Compared methods:

1. RealNVP + REINFORCE baseline
2. RealNVP + annealed KL, T = 0.1 -> 0.001

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

Annealed KL strongly increases permutation diversity and slightly improves best-found solutions, but the final generator still sits near the same makespan plateau. This is the key diagnostic result:

```text
permutation diversity != objective diversity
```

## Files

Active benchmark code:

- `objective.py` — Ta001 data, batched and slow evaluators, optimality gap, NEH heuristic and sanity tests.
- `realnvp_pfsp_baseline.py` — frozen 10-seed RealNVP + REINFORCE baseline.
- `realnvp_pfsp_kl.py` — matched annealed-KL variant.

Committed results:

- `results/permutation_optimization/pfsp/pfsp_ta001_realnvp_baseline_10seeds.csv`
- `results/permutation_optimization/pfsp/pfsp_ta001_realnvp_kl_10seeds.csv`

Run sanity checks with:

```bash
python experiments/discrete_optimization/benchmarks/permutation/pfsp/objective.py
```
