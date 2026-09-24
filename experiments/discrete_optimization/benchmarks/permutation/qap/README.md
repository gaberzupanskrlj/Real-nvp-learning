# QAP — QAPLIB Nug20

This directory contains the RealNVP permutation-optimization experiments for the **Quadratic Assignment Problem (QAP)**.

## Benchmark

The benchmark is **QAPLIB Nug20**:

```text
dimension = 20
known global optimum = 2570
```

The objective is:

[
C(pi)=sum_{i,j}F_{ij}D_{pi(i),pi(j)}
]

where the implementation uses:

```text
permutation[i] = location assigned to facility i
```

QAPLIB publishes the known optimal permutation in the inverse orientation. `objective.py` converts it into the convention used by the RealNVP decoder and verifies that the resulting cost is exactly 2570.

## Representation

Both RealNVP variants use the same continuous-to-permutation decoder as the TSP experiments:

```python
permutation = torch.argsort(y, dim=1)
```

so the model learns a continuous distribution over score vectors while the objective is evaluated on the induced discrete permutations.

## Files

- `objective.py` — Nug20 matrices, known optimum, permutation-convention conversion, vectorized QAP objective, and sanity tests.
- `realnvp_qap_baseline.py` — frozen 10-seed RealNVP + REINFORCE baseline.
- `realnvp_qap_kl.py` — same setup with annealed KL regularization.

## Frozen experiment settings

```text
dimension = 20
layers = 4
hidden = 64
batch = 1024
epochs = 3000
learning rate = 1e-4
validation = 4096 every 50 epochs
test = 16384
gradient clipping = 5.0
seeds = 42..51
```

The baseline and KL experiments deliberately keep all of these settings identical.

The KL run uses:

```text
T_START = 0.1
T_END = 0.001
schedule = geometric
```

and minimizes:

[
L=L_{mathrm{REINFORCE}}+T(t)D_{mathrm{KL}}(q_	heta(y)|N(0,I)).
]

## Run

First verify the benchmark:

```bash
python experiments/discrete_optimization/benchmarks/permutation/qap/objective.py
```

Expected key output:

```text
computed optimal cost:
2570.0

optimality gap:
0.0

All Nug20 checks passed.
```

Run the baseline:

```bash
python experiments/discrete_optimization/benchmarks/permutation/qap/realnvp_qap_baseline.py
```

Run annealed KL:

```bash
python experiments/discrete_optimization/benchmarks/permutation/qap/realnvp_qap_kl.py
```

## Result summary

Across seeds 42–51:

| Metric | Baseline | Annealed KL |
|---|---:|---:|
| Mean best cost | 2805.8 | **2794.8** |
| Mean best gap | 9.1751% | **8.7471%** |
| Global optimum hits | 0/10 | 0/10 |
| Mean final cost | 2871.819 | **2838.144** |
| Mean retention loss | 66.0 | **43.2** |
| Mean final unique permutations | 4.2 | **11.2** |

The main QAP finding is that annealed KL **consistently reduces final distribution collapse** and improves retention, while its search-quality improvement is smaller and seed-dependent. KL has more final unique permutations than the baseline on all 10 paired seeds.

Detailed per-seed results and interpretation are stored in:

```text
results/permutation_optimization/qap/
```
