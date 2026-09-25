# PFSP — Taillard Ta001

This directory contains the permutation flow-shop scheduling benchmark used for the next RealNVP permutation-optimization experiment.

## Benchmark

The first benchmark instance is **Taillard Ta001**:

```text
jobs = 20
machines = 5
best-known makespan = 1278
```

A solution is a permutation of the 20 jobs.

The same job order is used on all five machines. Every job visits the machines in the fixed order:

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

Lower is better.

For the RealNVP optimizer the reward will therefore be:

```python
reward = -makespan
```

## Representation

The RealNVP representation will deliberately match the existing TSP and QAP experiments:

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

## Current files

- `objective.py` — Ta001 processing times, batched PyTorch makespan evaluator, reward, optimality gap, slow reference evaluator, NEH heuristic, and sanity tests.

The RealNVP training scripts will be added only after the benchmark module is verified.

## Sanity references

The implementation checks several deterministic reference values:

```text
identity permutation makespan = 1448
standard NEH makespan         = 1286
best-known Ta001 makespan     = 1278
```

The NEH heuristic is included as a benchmark sanity/reference method, not as the final evaluation-budget baseline.

## Run

From the repository root:

```bash
python experiments/discrete_optimization/benchmarks/permutation/pfsp/objective.py
```

Expected final line:

```text
All Ta001 checks passed.
```

## Planned experiment

The first controlled comparison should mirror the QAP protocol:

```text
dimension = 20
decoder = argsort
seeds = 42..51

1. RealNVP + REINFORCE baseline
2. RealNVP + annealed KL
```

The main question is whether the anti-collapse behavior observed on TSP and QAP transfers to a structurally different scheduling landscape.
