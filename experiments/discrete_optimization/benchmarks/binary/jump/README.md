# Jump-100 experiments

Binary Jump experiments for studying the ability of the RealNVP + REINFORCE
optimizer to cross deceptive fitness valleys.

## Problem

Dimension:

    n = 100

The target is the Nevergrad-style alternating binary pattern:

    0, 1, 0, 1, ...

For jump size `m`, solutions improve normally until

    correct = n - m

which forms a local optimum.

Solutions inside the valley receive worse fitness until the global optimum
at `correct = n`.

## Jump-size diagnostic

A controlled single-seed experiment was run with:

    seed = 42
    jump sizes = [2, 4, 5, 6, 7, 8, 16, 25]

The RealNVP architecture and training configuration were kept fixed.

Observed transition:

    m <= 6 : global optimum discovered
    m >= 7 : global optimum not discovered

For seed 42, the empirical transition therefore occurs between m=6 and m=7.

The most informative failure is Jump-7:

    local optimum = 93
    maximum correct coordinates ever sampled = 99
    global optimum = not found
    final generator concentrated almost entirely near the local optimum

This suggests that the model can explore deeply into the fitness valley,
but REINFORCE assigns those samples lower reward than the local optimum and
eventually suppresses them.

For small valleys (m=2..6), another failure mode appears:
the global optimum is discovered during training but is not retained by the
validation-mean checkpoint.

This mirrors the discovery-versus-retention behavior observed in the TSP
experiments.

## Files

- `objective.py` - Jump objective
- `realnvp_jump_size_sweep.py` - fixed-seed difficulty sweep
- `jump_size_sweep_seed42.csv` - sweep results
- `realnvp_100seeds.py` - standard Jump-25 RealNVP benchmark
- `one_plus_one_ea_100seeds.py` - classical baseline
