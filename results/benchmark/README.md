# Benchmark comparison

This directory contains the current comparison between the RealNVP-based discrete optimizer and a classical **(1+1)-EA** on five 100-dimensional IOH/PBO-style benchmark problems.

## Benchmarks

| Problem | RealNVP | (1+1)-EA | Main observation |
|---|---:|---:|---|
| OneMax | best 100 / 100 | best 100 / 100 | Both reach the optimum |
| LeadingOnes | best 97 / 100 | best 100 / 100 | EA reaches the optimum much more efficiently |
| ConcatenatedTrap | best 16 / 20 | best 16.2 / 20 | Both struggle with the deceptive structure |
| NKLandscapes | mean best -0.304662 | mean best -0.292663 | EA is clearly better; higher is better |
| IsingTorus | mean best 173.33 / 200 | mean best 194 / 200 | EA reaches 200 in 7/10 runs |

## Overview

![Normalized benchmark overview](figures/benchmark_overview.svg)

The overview uses the **best observed solution as a percentage of the known optimum**. NKLandscapes is excluded because a useful finite optimum was not available from the IOH problem object.

## NKLandscapes

![NKLandscapes comparison](figures/nk_landscapes.svg)

RealNVP statistics are based on 3 seeds and the EA statistics on 10 seeds. The comparison is therefore exploratory rather than a final significance test.

## IsingTorus

![IsingTorus comparison](figures/ising_torus.svg)

For IsingTorus, the EA reached the global optimum of 200 in 7 of 10 runs. The best RealNVP run reached 180.

## Interpretation

The current experiments suggest that RealNVP can learn a strong search distribution on simple separable structure such as OneMax, but it tends to collapse into suboptimal regions on more structured or rugged landscapes. The (1+1)-EA is especially strong on problems where local mutation can preserve already-good structure while making incremental improvements.

These results should be treated as an exploratory benchmark. A final experiment should use the same evaluation accounting and the same number of independent runs for both methods.

## Files

- `benchmark_summary.csv` — compact numerical summary.
- `raw_results.md` — recorded multi-seed outputs used for the comparison.
- `figures/` — GitHub-renderable SVG figures.
- `../../plotting/plot_benchmark_comparison.py` — script for reproducing PNG versions locally.
