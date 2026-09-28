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
3. Budget-matched (1+1)-EA, 1 + Poisson(1) insertion moves, 3,317,760 evaluations per seed

## 10-seed results

Best makespan is best-ever over training and validation evaluations for both RealNVP variants. Std uses ddof = 1. Retention loss = final best makespan − best-ever makespan.

| Metric | Baseline | Annealed KL | (1+1)-EA |
| --- | ---: | ---: | ---: |
| Mean best makespan | 1295.8 | 1293.1 | 1278.0 |
| Mean best gap | 1.39% | 1.18% | 0.00% |
| Std best makespan | 2.150 | 2.961 | 0.000 |
| Median best makespan | 1296.5 | 1293.5 | 1278.0 |
| Best run | 1290 | 1288 | 1278 |
| Worst run | 1297 | 1297 | 1278 |
| Best-known hits | 0/10 | 0/10 | 10/10 |
| Mean final generator makespan | 1297.003 | 1297.006 | — |
| Mean retention loss | 1.2 | 3.9 | — |
| Mean final unique permutations | 1550.0 / 16384 | 13754.4 / 16384 | — |
| Final unique fraction | 9.46% | 83.95% | — |

Annealed KL is better than the baseline on 8/10 paired seeds (1 tie, 1 worse). It strongly increases permutation diversity and slightly improves best-found solutions, but the final generator sits on the same 1297 plateau. Its retention loss is larger because it finds a better best-ever and then returns to that plateau. This is the key diagnostic result:

```text
permutation diversity != objective diversity
```

The (1+1)-EA reaches 1278 on every seed after 10.7k–105k evaluations (median about 40k) and beats both RealNVP variants on 10/10 paired seeds. Retention and diversity do not exist for a single-solution EA.

`realnvp_pfsp_kl.py` keeps bookkeeping that differs from the baseline: the validation best is stored separately (`best_validation_best`) and does not update `optimization_best_makespan`, it runs 61 validations instead of 60 (+4096 evaluations), and its CSV has no `retention_loss`, `best_found_eval` or `target_eval` columns. The KL values above are computed from the CSV as `min(optimization_best_makespan, best_validation_best)` and `test_best_makespan − best-ever`. Reading `optimization_best_makespan` alone gives 1293.9.

## Files

Active benchmark code:

- `objective.py` — Ta001 data, batched and slow evaluators, optimality gap, NEH heuristic and sanity tests.
- `realnvp_pfsp_baseline.py` — frozen 10-seed RealNVP + REINFORCE baseline.
- `realnvp_pfsp_kl.py` — matched annealed-KL variant.
- `one_plus_one_ea_pfsp.py` — budget-matched (1+1)-EA baseline.

Committed results:

- `results/permutation_optimization/pfsp/pfsp_ta001_realnvp_baseline_10seeds.csv`
- `results/permutation_optimization/pfsp/pfsp_ta001_realnvp_kl_10seeds.csv`
- `results/permutation_optimization/pfsp/pfsp_ta001_one_plus_one_ea_10seeds.csv`

Run sanity checks with:

```bash
python experiments/discrete_optimization/benchmarks/permutation/pfsp/objective.py
```
