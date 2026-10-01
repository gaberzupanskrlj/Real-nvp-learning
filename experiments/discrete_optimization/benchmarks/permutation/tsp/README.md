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
- `realnvp_tsp20_baseline_kl.py` — clean KL test: frozen TSP-20 baseline + annealed KL (T 0.1 → 0.001), nothing else changed; paired with the frozen baseline.
- `realnvp_tsp_kl_cosine_dual.py` — TSP-20 combined KL/cosine diagnostic variant.
- `realnvp_tsp_gaussian_exploration.py` — TSP-20 fixed Gaussian output-space exploration.
- `realnvp_tsp_gaussian_exploration_10seeds.py` — TSP-20 exploration sweep.
- `one_plus_one_ea_tsp20.py` — budget-matched (1+1)-EA baseline, 1 + Poisson(1) inversion moves, same protocol as the QAP and PFSP EA.

### Trace reruns and controls (30. 9. – 1. 10.)

- `realnvp_tsp20_trace.py baseline|kl` — reruns of the frozen baseline and the clean KL script with a best-so-far trace (used for the anytime comparison with the EA).
- `realnvp_tsp20_boltzmann_trace.py` — the KL trace run without advantage standardization (fixed Boltzmann target).
- `gaussian_tsp20_trace.py baseline|kl sweep|<LR>` — learned diagonal Gaussian random keys instead of RealNVP; LR sweep on seeds 0–2, then main run on seeds 42–51.
- `realnvp_tsp20_lr_trace.py baseline|kl sweep|<LR>` — the same LR sweep for RealNVP (copy of the Gaussian script with the frozen RealNVP model).

The original TSP-20 cosine/elite script was never committed: the file saved in commit `823efda` contains only placeholder text. The experiment used the same code as the TSP-50 script with `N_CITIES = 20`, so `realnvp_tsp20_cosine_elite.py` is that script with `N_CITIES = 20`. Its settings and output columns match `results/permutation_optimization/tsp20/cosine_elite/comparison.csv`. Existing TSP-20 result CSVs are unchanged.

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

The frozen baseline reached the reference optimum in 3/10 runs. An earlier single-inversion local search (4/10) was removed from the active folder; it stalled in a 2-opt local optimum within about 1,000 evaluations on every seed. The old script remains in the legacy tree. Completed cosine+elite, annealed-KL and Gaussian-exploration results are stored under:

```text
results/permutation_optimization/tsp20/
```

The completed annealed-KL experiment reached the reference optimum during optimization in 10/10 seeds, while the final learned generator usually concentrated near a slightly worse tour. The Gaussian exploration sweep preserved diversity more strongly but did not produce the same search reliability.

## Organization rule

Frozen result-producing scripts keep the problem size in their filename. Scaling experiments use separate scripts rather than modifying frozen TSP-20 implementations. Older exploratory versions and diagnostics remain under the legacy tree.
