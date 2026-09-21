# Discrete optimization benchmark scripts

This folder contains the cleaned benchmark implementation used to compare the RealNVP search distribution against a classical **(1+1)-EA**.

## Layout

```text
benchmarks/
├── realnvp/
│   ├── common.py
│   ├── onemax.py
│   ├── leading_ones.py
│   ├── concatenated_trap.py
│   ├── nk_landscapes.py
│   └── ising_torus.py
└── baselines/
    ├── one_plus_one_ea.py
    ├── onemax.py
    ├── leading_ones.py
    ├── concatenated_trap.py
    ├── nk_landscapes.py
    └── ising_torus.py
```

The small problem files only contain configuration. Shared algorithm code lives in `common.py` and `one_plus_one_ea.py`, so bug fixes do not have to be copied into five separate scripts.

## Running

From the repository root:

```bash
python experiments/discrete_optimization/benchmarks/realnvp/nk_landscapes.py
python experiments/discrete_optimization/benchmarks/baselines/nk_landscapes.py
```

The current configurations mirror the exploratory runs recorded in `results/benchmark/` as closely as possible. They are intentionally explicit so hyperparameters can be changed in one place per problem.

## Important comparison note

The current study is exploratory. RealNVP and the EA do not yet have the same number of independent seeds on every benchmark, and RealNVP validation calls add extra objective evaluations beyond the nominal training batch × epoch count.

For final statistical reporting, use equal seed counts and account for all objective calls.
