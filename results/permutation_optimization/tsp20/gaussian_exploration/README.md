# Gaussian exploration on TSP-20

This directory contains the completed 10-seed Gaussian-exploration ablation on the fixed Euclidean TSP-20 instance.

## Setup

The sweep keeps the frozen small TSP architecture and training budget fixed:

```text
N_CITIES = 20
TSP_INSTANCE_SEED = 12345
NUM_LAYERS = 4
HIDDEN_DIM = 64
BATCH_SIZE = 1024
EPOCHS = 2000
LR = 1e-4
seeds = 42 ... 51
reference optimum = 3.513668846
```

Three output-space Gaussian exploration fractions were tested: `0.00`, `0.10`, and `0.25`. Because sample counts must be integers, the actual `0.10` mixture uses 922 flow samples and 102 Gaussian samples per batch, i.e. `epsilon = 0.099609375`.

## Results

| Exploration epsilon | Optimum hits | Final test mean | Final test best | Final flow unique fraction | Final raw grad norm |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 4/10 | 3.725264 ± 0.239355 | 3.724022 ± 0.238924 | 0.003320 ± 0.001608 | 0.916 ± 0.728 |
| 0.10 | 5/10 | 3.689891 ± 0.190429 | 3.651728 ± 0.184295 | 0.024403 ± 0.011493 | 44.556 ± 18.788 |
| 0.25 | 5/10 | 4.129849 ± 0.512785 | 3.642931 ± 0.182076 | 0.225391 ± 0.258593 | 60.242 ± 28.383 |

The strongest and most repeatable effect is on diversity. Mean final flow uniqueness rises from `0.0033` with no Gaussian exploration to `0.0244` at epsilon `0.10` and `0.2254` at epsilon `0.25`.

This extra diversity does not translate into a clear improvement in optimum discovery: the hit count changes only from 4/10 to 5/10. At epsilon `0.25`, the mean final generator quality is substantially worse even though the final test batch more often contains very good tours.

The exploration responsibility is effectively zero at the end of training in the exploration runs, while raw gradient norms remain much larger than in the no-exploration baseline. Gaussian exploration therefore changes the optimization trajectory and delays concentration, but does not remain a strong direct gradient source late in training.

## Relation to annealed KL

The completed annealed-KL TSP experiment reached the reference optimum in 10/10 seeds and achieved mean final test mean `3.524477 ± 0.005491`. In the completed experiments, annealed KL therefore gives better observed search reliability and final-generator quality than the Gaussian-exploration sweep.

The optimization budgets are not matched: this Gaussian sweep uses batch 1024 for 2000 epochs, while the completed annealed-KL runs use batch 4096 for 3000 epochs. The comparison should therefore be treated as an observed method comparison, not a sample-efficiency ranking.

## Files

- `comparison.csv` — exact per-seed results for all 30 runs.
- `summary.csv` — aggregate mean/std statistics for the three exploration settings.
- `summary.json` — machine-readable aggregate summary.

The plotting script used during the experiment is kept outside this results directory; figures can be regenerated directly from `comparison.csv`.
