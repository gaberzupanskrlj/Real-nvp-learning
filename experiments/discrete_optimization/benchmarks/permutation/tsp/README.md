# TSP permutation optimization

This folder contains the active permutation-valued TSP experiments for the discrete RealNVP project.

RealNVP remains continuous and uses a random-key decoder:

```python
tour = torch.argsort(y, dim=1)
```

## Active files

### Frozen TSP-20 study

- `realnvp_tsp20_baseline.py` — frozen 20-city RealNVP baseline implementation.
- `realnvp_tsp_10seeds.py` — seeds 42–51 runner for the frozen baseline.
- `realnvp_tsp20_cosine_elite.py` — TSP-20 cosine learning-rate + elite-checkpoint experiment.
- `realnvp_tsp20_kl.py` — original TSP-20 annealed-KL experiment used for the committed 10-seed results.
- `realnvp_tsp_kl_cosine_dual.py` — TSP-20 combined KL/cosine diagnostic variant.
- `realnvp_tsp_gaussian_exploration.py` — TSP-20 fixed Gaussian output-space exploration.
- `realnvp_tsp_gaussian_exploration_10seeds.py` — TSP-20 exploration sweep.
- `inversion_baseline_10seeds.py` — classical inversion-search baseline.

The historical committed TSP-20 cosine/elite source blob is corrupted. `realnvp_tsp20_cosine_elite.py` restores the recorded 20-city configuration using the active cosine+elite implementation. Existing TSP-20 result CSVs are unchanged.

### TSP-50 scaling

- `realnvp_tsp50_cosine_elite.py` — current 50-city cosine+elite scaling experiment.
- `realnvp_tsp50_kl.py` — current 50-city annealed-KL scaling experiment.

Dimension-specific filenames prevent later scaling work from silently replacing code associated with the frozen TSP-20 results.

## Frozen TSP-20 baseline

| Parameter | Value |
|---|---:|
| Cities | 20 |
| Instance seed | 12345 |
| RealNVP layers | 4 |
| Hidden dimension | 64 |
| Batch size | 1024 |
| Epochs | 2000 |
| Learning rate | 1e-4 |
| Seeds | 42–51 |
| Reference optimum | 3.513668846 |

## Run

Baseline:

```bash
python experiments/discrete_optimization/benchmarks/permutation/tsp/realnvp_tsp_10seeds.py
```

TSP-20 annealed KL:

```bash
python experiments/discrete_optimization/benchmarks/permutation/tsp/realnvp_tsp20_kl.py
```

TSP-20 cosine + elite:

```bash
python experiments/discrete_optimization/benchmarks/permutation/tsp/realnvp_tsp20_cosine_elite.py --seeds 42 43 44 45 46 47 48 49 50 51
```

TSP-50 scaling should use only the explicitly named `realnvp_tsp50_*.py` scripts.

## Established TSP-20 results

The frozen baseline reached the reference optimum in 3/10 runs and the inversion baseline in 4/10. Completed cosine+elite, annealed-KL and Gaussian-exploration results are stored under:

```text
results/permutation_optimization/tsp20/
```

The completed annealed-KL experiment reached the reference optimum during optimization in 10/10 seeds, while the final learned generator usually concentrated near a slightly worse tour. The Gaussian exploration sweep preserved diversity more strongly but did not produce the same search reliability.

## Organization rule

Frozen result-producing scripts keep the problem size in their filename. Scaling experiments use separate scripts rather than modifying frozen TSP-20 implementations. Older exploratory versions and diagnostics remain under the legacy tree.
