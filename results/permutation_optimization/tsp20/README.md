# TSP-20 results

Fixed Euclidean instance:

```text
N_CITIES = 20
TSP_INSTANCE_SEED = 12345
reference optimum = 3.513668846
```

This directory is the main results entry point for the TSP study.

## Current 10-seed comparison

| Metric | RealNVP baseline | Inversion baseline |
|---|---:|---:|
| Seeds | 10 | 10 |
| Global optimum hits | 3/10 | 4/10 |
| Mean best length | 3.586519 | 3.524057 |
| Std best length | 0.211928 | 0.017981 |
| Median best length | 3.522437 | 3.522437 |
| Best run | 3.513669 | 3.513669 |
| Worst run | 4.189561 | 3.573708 |
| Mean optimality gap | 2.0733% | 0.2956% |

The current RealNVP baseline frequently discovers excellent tours, but its performance is less stable across seeds. The diagnostic work also shows that best-ever search quality and the quality of the final learned generator must be reported separately: a good tour can be discovered and later receive very little sampled probability mass.

## Files

- `summary.csv` — compact method-level summary intended for quick comparison and plotting.
- `comparison_10seeds.csv` — per-seed values from the existing RealNVP-vs-inversion experiment.

Future named methods should add their own result directories and then append one row to `summary.csv`. This keeps raw outputs method-specific while preserving one simple table for the overall study.

## Evaluation note

The RealNVP baseline uses batches of 1024 sampled tours for 2000 epochs, giving about 2,048,000 training tour evaluations per seed, excluding validation and final testing.

The inversion baseline uses an exact O(1) delta update based on the two boundary edges changed by an inversion. Its `best_found_eval` is therefore a count of candidate moves, not an apples-to-apples count of full black-box objective-function calls.

For this reason, the current table is primarily a **solution-quality and reliability comparison**, not a definitive sample-efficiency comparison.

## Related documentation

- [TSP experiment overview](../../../experiments/discrete_optimization/permutation_optimization/tsp/README.md)
- [Mode-collapse investigation](../../../experiments/discrete_optimization/permutation_optimization/tsp/diagnostics/MODE_COLLAPSE_ANALYSIS_2026-09-23.md)
