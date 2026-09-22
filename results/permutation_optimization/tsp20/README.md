# TSP-20 results

Fixed Euclidean instance: `N_CITIES = 20`, `TSP_INSTANCE_SEED = 12345`.

Reference optimum used in these experiments: `3.513668846`.

## 10-seed comparison

| Metric | RealNVP | Inversion baseline |
|---|---:|---:|
| Seeds | 10 | 10 |
| Global optimum hits | 3/10 | 4/10 |
| Mean best length | 3.586519 | 3.524057 |
| Std best length | 0.211928 | 0.017981 |
| Median best length | 3.522437 | 3.522437 |
| Best run | 3.513669 | 3.513669 |
| Worst run | 4.189561 | 3.573708 |
| Mean optimality gap | 2.0733% | 0.2956% |

RealNVP frequently finds excellent tours, but the current training is less stable across seeds and can converge to a worse final distribution after having visited a better tour. The inversion baseline is much more consistent on this small Euclidean TSP instance.

## Evaluation note

RealNVP uses batches of 1024 sampled tours for 2000 epochs, giving about 2,048,000 training tour evaluations per seed, excluding validation and final testing.

The fast inversion baseline uses an exact O(1) delta update based on the two boundary edges changed by an inversion. Its `best_found_eval` is therefore a count of candidate moves, not an apples-to-apples count of full black-box objective function calls.

See `comparison_10seeds.csv` for the per-seed values.
