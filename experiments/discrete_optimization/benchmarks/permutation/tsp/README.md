# TSP permutation optimization

This folder contains the active permutation-valued TSP experiments for the discrete RealNVP project.

The problem is a symmetric Euclidean Traveling Salesman Problem (TSP). City coordinates are sampled once in the unit square and kept fixed with `TSP_INSTANCE_SEED = 12345`.

## Representation

RealNVP stays continuous. A generated vector `y` is converted into a valid permutation with a random-key decoder:

```python
tour = torch.argsort(y, dim=1)
```

The TSP objective is the length of the closed tour. The active experiments study both search quality and the quality/diversity of the learned permutation generator.

## Active files

### Baselines

- `realnvp_tsp_10seeds.py` — frozen RealNVP TSP-20 baseline for seeds 42–51.
- `inversion_baseline_10seeds.py` — classical elitist inversion-search baseline for seeds 42–51.

### RealNVP variants

- `realnvp_tsp_cosine_elite.py` — cosine learning-rate schedule with elite checkpointing.
- `realnvp_tsp_kl.py` — annealed-KL anti-collapse experiment.
- `realnvp_tsp_kl_cosine_dual.py` — combined KL/cosine diagnostic variant.
- `realnvp_tsp_gaussian_exploration.py` — expected-cost optimization with fixed Gaussian output-space exploration.
- `realnvp_tsp_gaussian_exploration_10seeds.py` — 10-seed sweep for exploration fractions 0.00, 0.10 and 0.25.

### Reference

- `simple_tsp_baseline.py` — earlier small TSP sanity-check baseline kept for reference.

## Frozen TSP-20 baseline setup

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

Individual experimental variants may change optimization settings. Their settings should be read from the corresponding script and result documentation rather than assumed to match the frozen baseline.

## Run

From the repository root:

```bash
python experiments/discrete_optimization/benchmarks/permutation/tsp/realnvp_tsp_10seeds.py
```

```bash
python experiments/discrete_optimization/benchmarks/permutation/tsp/inversion_baseline_10seeds.py
```

For individual RealNVP variants, run the corresponding script from the same directory.

## Established TSP-20 results

On the fixed TSP-20 instance, the frozen RealNVP baseline reached the reference optimum in 3/10 runs. The inversion baseline reached it in 4/10 runs. Both had median best tour length `3.522437`, while the inversion baseline was substantially more stable across seeds.

Completed cosine+elite, annealed-KL and Gaussian-exploration results, together with learned-generator diagnostics, are documented under:

```text
results/permutation_optimization/tsp20/
```

The Gaussian-exploration sweep shows a clear diversity increase with larger exploration fractions, but only a small change in optimum-hit rate and a substantial generator-quality penalty at high exploration. The completed annealed-KL experiment remains the stronger observed TSP variant, while using a larger optimization budget.

The inversion implementation uses the fact that an inversion changes only two boundary edges in a symmetric TSP. Its `best_found_eval` therefore counts candidate inversion moves, not full black-box objective recomputations. This distinction matters for evaluation-cost and runtime comparisons.

## Research organization

Active, reproducible experiment scripts stay in this folder under descriptive method names. Older exploratory versions and diagnostic code are kept under the legacy permutation-optimization tree so the active TSP folder remains readable.

New result sets should be stored under `results/permutation_optimization/tsp20/` with their own method-specific subdirectory when appropriate.
