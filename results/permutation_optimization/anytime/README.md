# Convergence comparison: RealNVP + KL vs (1+1)-EA

Best-so-far objective value against the number of objective evaluations for RealNVP + annealed KL and a budget-matched (1+1)-EA on three permutation problems. This is the current, evaluation-matched comparison; the `tsp20/`, `qap/` and `pfsp/` folders next to it hold earlier experiments.

## Protocol

- 10 seeds per method and problem (42–51).
- Budget: 3,317,760 objective evaluations per seed (3000 × 1024 training samples + 60 × 4096 validation samples). RealNVP on TSP-20 runs the frozen 2000-epoch protocol and stops at 2,216,960.
- RealNVP evaluations include validation samples; the final test samples are not counted.
- RealNVP + KL is the baseline with only the annealed-KL term added (T = 0.1 → 0.001, geometric).
- (1+1)-EA: 1 + Poisson(1) moves per child (TSP inversion, QAP swap, PFSP insertion), accept if not worse.

## Results

| Problem | Reference | (1+1)-EA median final best | RealNVP + KL median final best | Runs reaching reference (EA / RealNVP) |
|---|---:|---:|---:|---:|
| TSP-20 | optimum 3.513669 | 3.513669 | 3.513669 | 10/10 / 8/10 |
| QAP Nug20 | optimum 2570 | 2621 | 2802 | 0/10 / 0/10 |
| PFSP Ta001 | best-known 1278 | 1278 | 1295.5 | 10/10 / 0/10 |

## Files

| File | Produced by |
|---|---|
| `tsp20_realnvp_kl_{trace,10seeds}.csv` | `permutation/tsp/realnvp_tsp20_trace.py kl` |
| `qap20_realnvp_kl_{trace,10seeds}.csv` | `permutation/qap/realnvp_qap_kl.py` |
| `pfsp_ta001_realnvp_baseline_kl_{trace,10seeds}.csv` | `permutation/pfsp/realnvp_pfsp_baseline_kl.py` |
| `*_one_plus_one_ea_{trace,10seeds}.csv` | `permutation/{tsp,qap,pfsp}/one_plus_one_ea_*.py` |
| `*_clean_convergence.{png,pdf}` | `plotting/clean_anytime_plots.py` |

Script paths are relative to `experiments/discrete_optimization/benchmarks/`. The `*_10seeds.csv` files hold per-seed final results.

Trace columns: `seed, evaluations, best_so_far` for both methods. RealNVP traces add `epoch, phase, sample_mean, sample_best` and have one row after every training batch and validation. EA traces have one row at the first evaluation, at every improvement and at the end of the budget.

To recreate the figures, run from the repository root:

```bash
python plotting/clean_anytime_plots.py
```
