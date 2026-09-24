# TSP-50 scaling experiment

This directory records the first scaling check from the fixed Euclidean TSP setup at 20 cities to **50 cities**.

## Instance

- Cities: 50
- Instance seed: `12345`
- Coordinates: NumPy `default_rng(12345)`, uniform in ([0,1]^2)
- Exact reference optimum: **5.207124420405269**
- Reference optimum was verified independently with an exact mixed-integer TSP formulation using iterative subtour-elimination constraints.

These are currently **single-seed scaling runs**, so they should not be interpreted as a 10-seed benchmark.

## Results

| Method | Best found | Correct gap to optimum | Final test mean | Final-test mean gap |
|---|---:|---:|---:|---:|
| Cosine LR + elite checkpoint | 6.031545 | 15.8326% | 6.099992 | 17.1470% |
| Annealed KL | 5.430757 | 4.2947% | 5.442529 | 4.5208% |

The annealed-KL run is substantially closer to the exact TSP-50 optimum than the cosine/elite run in this preliminary scaling comparison.

## Important correction to the raw KL output

The KL run printed:

```text
Optimality gap: 54.5609%
```

That number is **not a valid TSP-50 gap**. The run reused the TSP-20 reporting constant `REFERENCE_OPTIMUM = 3.513668846`.

Using the correct TSP-50 optimum,

```text
(5.430757 - 5.207124420405269) / 5.207124420405269
= 0.042947...
```

so the corrected best-found gap is **4.2947%**.

The raw `test_fraction_optimal` is still 0% for this run because no final test sample reached the exact optimum, but future TSP-50 scripts should update the reference optimum before reporting gap-based metrics.

## Annealed-KL final generator snapshot

- Best validation checkpoint epoch: 2950
- Best validation mean: 5.441950
- Best validation tour: 5.430757
- Test mean: 5.442529
- Test best: 5.430757
- Test mode: 5.430757
- Test mode fraction: 47.05%
- Test unique tours: 346
- Best optimization length: 5.430757
- Corrected optimality gap: 4.2947%

The mode fraction of only 47.05% and 346 unique final tours show that the TSP-50 KL generator remains materially more diverse than the highly concentrated TSP-20 runs.
