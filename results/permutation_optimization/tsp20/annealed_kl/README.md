# Annealed KL

Official 10-seed TSP-20 annealed-KL experiment for seeds 42-51.

Configuration:

- fixed TSP instance seed: `12345`
- 20 cities
- 8 RealNVP coupling layers
- hidden dimension: 64
- batch size: 4096
- 3000 epochs
- fixed Adam learning rate: `1e-4`
- temperature schedule: `T = 0.1 -> 0.001`
- objective: REINFORCE + annealed `KL(q_theta || N(0,I))`
- checkpoint criterion: lowest validation mean
- final test size: 16384

## 10-seed result

- optimum discovery: **10/10**
- mean best-ever length: **3.513669**
- mean optimality gap: approximately **0.0000%**
- mean final test mean length: **3.524477**
- sample standard deviation of final test mean: **0.005491**
- final test batch contained the reference optimum in **1/10** runs
- final mode was the reference optimum in **1/10** runs

Seed 48 is the one successful retention case: its final mode is the optimum and about 98.96% of its final test samples are optimal. The other nine runs have 0% optimal final-test samples, so the 9.90% mean optimal-sample share is driven almost entirely by seed 48.

The experiment therefore strongly improves discovery and run-to-run generator stability, but does not reliably retain the globally optimal mode.

`comparison.csv` contains the exact per-seed outputs from the completed run.
