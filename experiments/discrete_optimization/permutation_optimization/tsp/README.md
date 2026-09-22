# Permutation optimization

This directory starts the next discrete-search-space experiment after the
binary IOH/PBO benchmark.

The first problem is deliberately small and self-contained:

- **problem:** symmetric Euclidean Traveling Salesman Problem (TSP);
- **size:** 10 cities;
- **instance:** fixed 2D coordinates generated with seed `12345`;
- **solution representation:** a permutation of the city indices;
- **objective:** minimize the length of the closed tour;
- **baseline:** elitist **(1+1)-EA** with inversion mutation;
- **sanity check:** an exact brute-force optimum is computed by fixing city 0
  and enumerating the remaining `9!` orderings.

## Run

From the repository root:

```bash
python experiments/discrete_optimization/permutation_optimization/tsp/simple_tsp_baseline.py
```

Use a different optimization budget or algorithm seed with:

```bash
python experiments/discrete_optimization/permutation_optimization/tsp/simple_tsp_baseline.py --budget 50000 --seed 43
```

For a quick run without the exact brute-force check:

```bash
python experiments/discrete_optimization/permutation_optimization/tsp/simple_tsp_baseline.py --skip-exact
```

## Why this comes first

The goal of this script is only to verify the permutation-valued problem and a
simple baseline before introducing the normalizing flow.

The intended next experiment keeps the existing RealNVP model continuous and
replaces binary thresholding with a random-key decoder:

```python
tour = torch.argsort(y, dim=1)
```

Every continuous sample then maps to a valid TSP permutation without duplicate
cities or a repair step.
