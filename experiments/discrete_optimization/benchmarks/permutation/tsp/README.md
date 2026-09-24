# TSP permutation optimization

This folder contains the first permutation-valued experiments for the discrete RealNVP project.

The problem is a symmetric Euclidean Traveling Salesman Problem (TSP). City coordinates are sampled once in the unit square and kept fixed with `TSP_INSTANCE_SEED = 12345`.

## Representation

RealNVP stays continuous. A generated vector `y` is converted into a valid permutation with a random-key decoder:

```python
tour = torch.argsort(y, dim=1)
```

The TSP objective is the length of the closed tour. Training uses reward `-tour_length` with the same REINFORCE / leave-one-out baseline used in the binary experiments.

## Files

- `realnvp_tsp.py` — single RealNVP TSP-20 run with convergence and tour plots.
- `realnvp_tsp_10seeds.py` — RealNVP TSP-20 experiment for seeds 42–51.
- `inversion_baseline_10seeds.py` — simple elitist inversion local-search baseline for seeds 42–51.
- `simple_tsp_baseline.py` — initial TSP-10 sanity-check baseline kept for reference.

## Current TSP-20 setup

| Parameter | Value |
|---|---:|
| Cities | 20 |
| Instance seed | 12345 |
| RealNVP layers | 4 |
| Hidden dimension | 64 |
| Batch size | 1024 |
| Epochs | 2000 |
| Learning rate | 1e-4 |
| Training seeds | 42–51 |
| Reference optimum | 3.513668846 |

## Run

From the repository root:

```bash
python experiments/discrete_optimization/benchmarks/permutation/tsp/realnvp_tsp_10seeds.py
```

```bash
python experiments/discrete_optimization/benchmarks/permutation/tsp/realnvp_tsp_10seeds.py
```

```bash
python experiments/discrete_optimization/benchmarks/permutation/tsp/inversion_baseline_10seeds.py
```

## Current result

On the fixed TSP-20 instance, RealNVP reached the reference optimum in 3/10 runs. The inversion baseline reached it in 4/10 runs. Both had median best tour length `3.522437`, while the inversion baseline was substantially more stable across seeds.

The fast inversion implementation uses the fact that an inversion changes only two boundary edges in a symmetric TSP. Its `best_found_eval` therefore counts candidate inversion moves, not full black-box objective recomputations. Use this distinction when making runtime or evaluation-cost claims.

Detailed results are stored under `results/permutation_optimization/tsp20/`.
