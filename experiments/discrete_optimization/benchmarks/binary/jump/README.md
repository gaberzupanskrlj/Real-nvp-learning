# Jump-100 experiments

Binary Jump experiments study whether the RealNVP + REINFORCE optimizer can cross a deceptive fitness valley.

## Problem

```text
dimension = 100
target = 0, 1, 0, 1, ...
```

For jump size `m`, solutions improve until `correct = n - m`, which forms a local optimum. Solutions inside the valley receive worse fitness until the global optimum at `correct = n`.

## Jump-size diagnostic

A controlled single-seed experiment used:

```text
seed = 42
jump sizes = [2, 4, 5, 6, 7, 8, 16, 25]
```

The RealNVP architecture and training configuration were kept fixed across jump sizes.

Observed transition:

```text
m <= 6 : global optimum discovered
m >= 7 : global optimum not discovered
```

Jump-7 is especially informative: the model sampled solutions with up to 99 correct coordinates but ultimately concentrated around the local optimum at 93. This indicates that valley samples are generated, but their lower reward causes REINFORCE to suppress them.

For smaller valleys (m = 2..6), the global optimum can be discovered during training and later lost by validation-mean checkpoint selection, matching the discovery-versus-retention issue seen in TSP.

## Files

- `objective.py` — Jump objective.
- `realnvp_jump_size_sweep.py` — fixed-seed difficulty sweep.

Sweep results are stored under `results/binary_problems/jump/`.

The former zero-byte 100-seed placeholder runners were removed. A future full Jump benchmark should add actual implementations rather than empty entry points.
