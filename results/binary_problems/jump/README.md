# Binary Problems — Jump-100

This folder records the main findings from the Jump-function experiments with the fixed RealNVP + REINFORCE baseline.

## Setup

- Dimension: 100
- Binary target: alternating pattern `0,1,0,1,...`
- RealNVP baseline: 4 coupling layers, hidden size 64
- Batch size: 1024
- Learning rate: 1e-4
- Evaluation budget: 4,096,000
- Standard Jump benchmark: `m = 25`
- Jump-25 local optimum: 75 correct coordinates
- Global optimum: 100 correct coordinates

## Jump-size sweep — seed 42

A controlled single-seed sweep was run with the same model initialization, optimizer settings, and evaluation budget while changing only the jump size.

| m | Local optimum | Max correct ever | Global optimum | Evaluations to target |
|---:|---:|---:|:---:|---:|
| 2 | 98 | 100 | yes | 238,592 |
| 4 | 96 | 100 | yes | 239,616 |
| 5 | 95 | 100 | yes | 245,760 |
| 6 | 94 | 100 | yes | 245,760 |
| 7 | 93 | 99 | no | — |
| 8 | 92 | 98 | no | — |
| 16 | 84 | 94 | no | — |
| 25 | 75 | 88 | no | — |

For seed 42, the empirical transition occurs between `m=6` and `m=7`.

The important observation is that failure at `m=7` is not caused by an inability to generate near-optimal solutions: the model sampled a 99/100 solution, but the global optimum was never sampled. Because intermediate valley solutions receive worse reward than the local optimum, REINFORCE suppresses them and the learned distribution eventually concentrates around the local optimum.

For `m=2..6`, the global optimum was discovered during training, but the validation-mean checkpoint did not retain it. This exposes a second failure mode: discovery without retention.

## Jump-25 — 10-seed baseline

Seeds 42–51 were run with `m=25`.

| Metric | Result |
|---|---:|
| Global optimum hits | 0/10 |
| Best objective | 75.0 in 10/10 runs |
| Mean max-correct-ever | 86.5 / 100 |
| Median max-correct-ever | 86 / 100 |
| Range max-correct-ever | 86–88 / 100 |
| Maximum observed | 88 / 100 |
| Final P(correct=100) | 0 in 10/10 runs |
| Median final P(correct=75) | 0.99594 |

Distribution of maximum correct coordinates sampled during training:

- 86/100: 7 seeds
- 87/100: 1 seed
- 88/100: 2 seeds
- 100/100: 0 seeds

The behavior is therefore highly consistent across seeds. The model moves into the deceptive valley early in training, but solutions deeper in the valley receive lower reward than the local optimum. Training then pushes probability mass back toward the local-optimum region.

Two runs (seeds 43 and 50) retained noticeably more final diversity than the others, but neither discovered the global optimum. This suggests that complete mode collapse is not required for failure: even a broader final distribution does not bridge the 25-bit fitness valley under the current learning signal.

## Main conclusion

The Jump experiments expose two distinct limitations of the current RealNVP + REINFORCE optimizer:

1. **Discovery failure:** for sufficiently wide deceptive valleys, the reward signal actively suppresses samples that move toward the global optimum before the optimum itself is reached.
2. **Retention failure:** for small valleys, the global optimum can be discovered but later lost because checkpoint selection favors higher average validation fitness rather than preserving rare best solutions.

These findings are closely related to the discovery-versus-retention behavior observed in the TSP experiments.
