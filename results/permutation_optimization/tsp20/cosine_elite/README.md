# Cosine LR + elite checkpoint

Official 10-seed TSP-20 experiment for seeds 42-51.

Configuration:

- fixed TSP instance seed: `12345`
- 20 cities
- 8 RealNVP coupling layers
- hidden dimension: 64
- batch size: 4096
- 3000 epochs
- learning rate: cosine decay from `1e-4` to `1e-5`
- checkpoint criterion: mean tour length of the best 1% of validation samples
- final test size: 16384

## 10-seed result

- optimum discovery: **8/10**
- mean best-ever length: **3.515422**
- sample standard deviation of best-ever length: **0.003697**
- mean optimality gap: **0.0499%**
- mean final test mean length: **3.746350**
- sample standard deviation of final test mean: **0.251125**
- final test batch contained the reference optimum in **3/10** runs

The method substantially improves best-ever search quality over the frozen RealNVP baseline, but final generator quality is unstable across seeds. This is evidence that discovery and retention are separate issues.

`comparison.csv` contains the exact per-seed outputs from the completed run.
