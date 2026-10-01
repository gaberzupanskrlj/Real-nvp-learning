# 2D basin study: when does RealNVP keep more than one minimum?

A controlled toy problem for the retention failure seen on TSP-20 (the optimum is found but the final distribution does not generate it). The target is known exactly, so we can separate model capacity, the optimum of the loss and training dynamics.

**Result.** With a temperature, RealNVP spreads its mass over several basins close to the Boltzmann target. At low temperature (T = 0.05) it loses the rare basin when trained from scratch on 10/10 seeds. This is training dynamics, not capacity and not the optimum of the loss: started from a maximum-likelihood fit of the target, the same REINFORCE keeps both basins on 10/10 seeds with a 3–4× lower reverse KL. Once a basin empties there are no samples in it and REINFORCE gets no signal to bring it back. The density shape inside a basin is capacity-limited (`tanh` on `s` bounds the log-determinant).

## Setup

- 2D objective: global minimum at (−0.5, −0.5) with f = 0, second minimum at (0.5, −0.5) with f = Δf.
- No temperature: standardized REINFORCE on f. With T: REINFORCE on the free energy E[f] + T · E[log q], i.e. reverse KL to the Boltzmann target ∝ exp(−f/T), centred advantage.
- RealNVP as in `objective_2d_gaussian.py` (correct score-function estimator), Adam LR 1e-4, 2000 epochs, seeds 42–51, `NUM_THREADS = 1`.
- No plateau restore: checkpointing on the lowest mean f may favour a more concentrated distribution, so it is switched off as a possible confounder.
- Reported on 100k fresh samples at the last epoch. Basin label on the discretized x; a basin is "occupied" at ≥ 1 % mass (fixed before the runs).

## Steps and results

| Step | Script and arguments | Setting | Result |
| --- | --- | --- | --- |
| 1 | `basin_2d_trace.py 0.2` | Δf = 0.2, no T | 10/10 global basin only; the right basin drops below 1 % at epoch 100–140 |
| 2 | `basin_2d_trace.py 0.0` | Δf = 0, no T | 3/10 both, 3 left, 4 right; decided by epoch ~200, an empty basin never returns |
| 3 | `basin_2d_trace.py 0.2 <T>` | T ∈ {1, 0.5, 0.3, 0.2, 0.1, 0.05} | T ≥ 0.2: 10/10 both, p_right / target 0.99–1.00. T = 0.1: 10/10 both, ratio 0.885, not converged at 2000. T = 0.05: 0/10 |
| 4 | `basin_2d_multi_trace.py [T]` | 4 minima, f = 0 / 0.1 / 0.2 / 0.3 | no T: 10/10 global only. With T: all 4 basins on 10/10 at every T, TV to target 0.003 (T = 1) … 0.020 (T = 0.1) |
| 5 | `basin_2d_mle_fit.py 0.2 <T>` | T ∈ {0.2, 0.1, 0.05}; phases scratch / mle / warm | T = 0.05: scratch 0/10 both (reverse KL 0.029–0.037); MLE fit 10/10 both (forward KL 0.028); warm 10/10 both (reverse KL 0.0074–0.0099, p_right 0.0148–0.0165 vs target 0.0179) |
| 6 | `basin_2d_anneal.py 0.2 0.2 0.05` | T 0.2 → 0.05 geometric over 1000 epochs, then 1000 at 0.05 | p_right 0.0005–0.0087 (scratch ≤ 0.0003), 0/10 above 1 %; reverse KL 0.021–0.030, lower than scratch and higher than warm on 10/10 |

At T = 0.05 dropping the right basin costs 0.018 nats of reverse KL. The scratch phase of step 5 is bit-identical to the step 3 run at T = 0.05; step 6 is paired with it (same init, batch stream, LR and epochs, only the T schedule differs).

Limitations: fixed LR 1e-4 and 2000 epochs; at T = 0.2 / 0.1 scratch has not converged (KL 2–4× above warm). Mean f is above target on every seed (+0.003–0.006), so the in-basin shape is not exact. Whether the same mechanism drives collapse on permutation problems is untested.

## Files

Each run writes `<run name>_trace.csv` (per-epoch masses, mean f, KL) and `<run name>_final.csv` (100k-sample evaluation per seed) next to the script:

- `basin2d_delta{Δf}[_T{T}]_*` — steps 1–3
- `basin2d_4min_{noT,T{T}}_*` — step 4
- `basin2d_mle_delta0.2_T{T}_*` — step 5
- `basin2d_anneal_delta0.2_T0.2to0.05_*` — step 6

Plots (they read and write this folder, from any working directory):

- `plotting/basin_temperature_plots.py` → `basin2d_delta0.2_temperature_{p_right.png,summary.csv}` (step 3)
- `plotting/basin_mle_plots.py` → `basin2d_mle_delta0.2_{p_right_kl.png,summary.csv}` (step 5)
- `plotting/basin_multi_plots.py` (2D densities for step 4) needs a rerun of step 4 with grid snapshots; not run yet.

## Older scripts in this folder

`objective_2d_baseline.py`, `objective_2d_experimental.py`, `objective_2d_test.py` and `objective_10d_baseline.py` use `loss = -(weights * log_det).mean()`, which is not a score-function estimator. Do not build on them; use `objective_2d_gaussian.py` or the `basin_2d_*.py` scripts.
