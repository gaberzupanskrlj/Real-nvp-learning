# QAP-20 — QAPLIB Nug20 results

This folder contains the 10-seed RealNVP study on the **Quadratic Assignment Problem (QAP)** using the standard **QAPLIB Nug20** instance.

## Problem

QAP assigns (n) facilities to (n) locations. For a permutation (pi),

[
C(pi)=sum_{i,j} F_{ij}D_{pi(i),pi(j)}
]

where:

- (F_{ij}) is the flow/interaction between facilities (i) and (j);
- (D_{ab}) is the distance between locations (a) and (b);
- lower cost is better.

For Nug20:

```text
dimension = 20
known global optimum = 2570
```

The implementation uses the convention:

```text
permutation[i] = location assigned to facility i
```

The published QAPLIB optimum uses the inverse orientation, so the stored reference permutation is inverted before evaluation. The sanity test verifies that both the vectorized and slow implementations return exactly `2570`.

## RealNVP representation

The continuous flow output is converted into a permutation with:

```python
permutation = torch.argsort(y, dim=1)
```

The optimization pipeline is:

```text
z ~ N(0,I)
    ↓
RealNVP
    ↓
y ∈ R^20
    ↓
argsort
    ↓
permutation
    ↓
Nug20 cost
    ↓
reward = -cost
```

This deliberately mirrors the TSP permutation experiments: the decoder stays the same while the objective landscape changes.

## Frozen protocol

Both QAP experiments use:

```text
layers = 4
hidden = 64
batch size = 1024
epochs = 3000
learning rate = 1e-4
validation size = 4096
validation every = 50 epochs
test size = 16384
gradient clipping = 5.0
seeds = 42..51
objective evaluations per run = 3,317,760
```

Checkpoint selection is identical in both variants: the model state with the **lowest validation mean cost** is restored for the final generator test.

## Methods

### RealNVP baseline

The baseline uses standardized leave-one-out REINFORCE:

```text
reward = -cost
advantage = reward - leave-one-out baseline
loss = -(advantage * log q_theta(y)).mean()
```

### RealNVP + annealed KL

The KL experiment changes only the training loss:

[
L=L_{mathrm{REINFORCE}}+T(t)D_{mathrm{KL}}left(q_	heta(y),|,N(0,I)ight)
]

with geometric annealing:

```text
T_START = 0.1
T_END   = 0.001
```

The goal is to delay premature concentration of the learned distribution and test whether the anti-collapse effect observed on TSP transfers to another permutation problem.

## 10-seed comparison

| Metric | Baseline | Annealed KL |
|---|---:|---:|
| Seeds | 10 | 10 |
| Global optimum hits | 0/10 | 0/10 |
| Mean best cost | 2805.8 | **2794.8** |
| Median best cost | 2795 | **2786** |
| Std best cost | 38.09 | **32.57** |
| Mean best gap | 9.1751% | **8.7471%** |
| Best single run | **2742** | 2754 |
| Mean eval of best discovery | **681,192** | 1,627,101 |
| Mean final generator cost | 2871.819 | **2838.144** |
| Mean retention loss | 66.0 | **43.2** |
| Median retention loss | 70 | **44** |
| Mean unique permutations / 16,384 | 4.2 | **11.2** |
| Median unique permutations | 4 | **11** |

### Per-seed discovery comparison

KL obtains a lower best-ever cost on **6/10 seeds** and a higher cost on 4/10 seeds. Its average best cost improves by 11 points, but the gain is much less uniform than the diversity effect.

### Mode collapse

The baseline final generator is extremely concentrated:

```text
unique permutations per 16,384 samples:
6, 4, 4, 3, 2, 4, 6, 6, 1, 6
```

The KL variant remains concentrated, but it is consistently broader:

```text
11, 8, 9, 12, 10, 17, 12, 13, 11, 9
```

KL has more final unique permutations in **10/10 paired seeds**.

Seed 50 is a particularly clear baseline collapse case: the baseline final generator produces only **1 unique permutation in 16,384 samples**, while the KL version produces 11.

### Discovery vs retention

The baseline loses an average of **66 cost points** between the best solution ever discovered and the best solution present in the restored final generator. Annealed KL reduces this to **43.2**.

KL achieves perfect retention on seeds 49 and 50:

```text
seed 49: best ever = 2788, final best = 2788
seed 50: best ever = 2848, final best = 2848
```

The price is later discovery. The baseline finds its run-best solution after about 0.68M objective evaluations on average, while KL requires about 1.63M. This is consistent with KL maintaining exploration for longer before the distribution contracts.

## Interpretation

The Nug20 experiment supports three observations:

1. **RealNVP can learn substantially better-than-random QAP permutations, but the plain REINFORCE baseline collapses to only a handful of permutations.**
2. **Annealed KL consistently reduces distribution collapse and improves retention.**
3. **Search-quality improvement is positive on average but seed-dependent; KL does not solve Nug20 and no run reaches the global optimum.**

Together with the earlier TSP result, QAP provides a second permutation landscape where annealed KL improves the behavior of the learned search distribution. This is cross-problem evidence for the anti-collapse mechanism, but not yet evidence that the method is universal. A third structurally different permutation benchmark such as PFSP or LOP is the natural next test.

## Files

- `qap20_realnvp_10seeds.csv` — exact baseline results for seeds 42–51.
- `qap20_realnvp_kl_10seeds.csv` — exact annealed-KL results for seeds 42–51.
- experiment code: `../../../experiments/discrete_optimization/benchmarks/permutation/qap/`
