# TSP-20 RealNVP: mode-collapse investigation — 2026-09-23

## Executive summary

The main result of today's work is that the current TSP-20 failure mode is no longer best described as "the model cannot find the optimum".

On seed 44 the RealNVP search distribution **does find the known optimum** `3.513668846`, but the learned distribution later loses permutation diversity and collapses onto a worse tour basin. The original leave-one-out REINFORCE signal gives relatively little directional influence to the best samples during the critical phase, while the bulk of the batch often determines the parameter update. Rank shaping changes this behavior, but by itself does not solve the underlying loss of exploration.

The strongest current interpretation is therefore:

> **Discovery is possible; retention and continued exploration are the unresolved problems.**

This is consistent with the mode-collapse behavior described in Chapter 7 of the supervisor's thesis for a closely related RealNVP + `argsort` TSP method. That chapter also identifies premature concentration of the sampling distribution as the main limitation and suggests explicit exploration as future work.

**Decision after today's experiments:** pause further ad-hoc tuning of rank weights, archive weights, multiplicity exponents, etc. Get supervisor input before selecting the next principled direction.

---

## 1. Current TSP-20 setting

The diagnostics discussed here use the current larger TSP experiment:

| Parameter | Value |
|---|---:|
| Problem | symmetric Euclidean TSP |
| Cities | 20 |
| TSP instance seed | 12345 |
| Reference optimum | `3.513668846` |
| Decoder | full random-key `argsort(y)` |
| RealNVP layers | 8 |
| Hidden dimension | 64 |
| Batch size | 4096 |
| Epochs | 3000 |
| Optimizer | Adam |
| Initial LR | `1e-4` |
| LR schedule | cosine, ending near `1e-5` |
| Validation size | 4096 |
| Test size | 16384 |
| Gradient clipping | 5.0 |

The baseline training objective is the score-function / REINFORCE estimator with a leave-one-out baseline:

```python
with torch.no_grad():
    baseline = (reward.sum() - reward) / (BATCH_SIZE - 1)
    advantage = reward - baseline

log_prob = log_probability(model, y.detach())
loss = -(advantage * log_prob).mean()
```

with `reward = -tour_length`.

This estimator is mathematically valid. The problem investigated today is not a simple sign/detach bug.

---

## 2. Baseline behavior: the optimum is found and then lost by the distribution

Seed 44 is the most informative failure case.

The optimization process finds the global optimum as a best-ever solution, but the current sampling distribution later concentrates around a worse tour. The final checkpoint does not generate the optimum.

Final baseline seed-44 result:

| Metric | Result |
|---|---:|
| Optimization best-ever | **3.513669** |
| Best checkpoint epoch | 350 |
| Best validation elite mean | 3.522437 |
| Final test mean | 3.810318 |
| Final test best | 3.522437 |
| Final test unique-tour fraction | 0.047546 |

The distinction between **best-ever search solution** and **what the learned generator still samples** is essential.

The model has enough representational capacity to discover the optimum. The failure is that a discovery does not necessarily become a stable high-probability mode of the final generator.

---

## 3. New gradient-signal diagnostics

A diagnostic version of the baseline was created without changing the training objective.

The training batch is partitioned by tour quality into four disjoint groups:

- `top1`: best 1%
- `next9`: next 9%
- `middle80`: middle 80%
- `bottom10`: worst 10%

For each group we compute its contribution to the original score-function gradient and project it onto the total update direction:

```text
projection_share(group)
    = dot(g_group, g_total) / ||g_total||^2
```

The four projection shares sum to approximately 1. A negative value means that group's gradient is partially opposed to the final total update direction.

### Critical seed-44 observations

| Epoch | mean length | unique fraction | top 1% | next 9% | middle 80% | bottom 10% |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.6253 | 1.000 | +0.039 | +0.296 | +0.340 | +0.325 |
| 200 | 4.7784 | 0.771 | +0.038 | +0.364 | +0.609 | -0.011 |
| 250 | 4.3293 | 0.478 | **+0.010** | +0.189 | **+0.696** | +0.105 |
| 300 | 3.9888 | 0.225 | **-0.072** | +0.692 | +0.346 | +0.034 |
| 350 | 3.8100 | 0.080 | -0.022 | +0.074 | +0.270 | **+0.678** |
| 450 | 3.6576 | 0.016 | +0.055 | -0.090 | +0.226 | **+0.808** |
| 550 | 3.5855 | 0.005 | +0.025 | +0.010 | +0.113 | **+0.853** |
| 850 | 3.5741 | 0.001 | +0.003 | +0.004 | +0.010 | **+0.983** |

At epoch 250, the best 1% of samples contributes only about **1% of the final gradient direction**, while the middle 80% contributes about **70%**.

At epoch 300 the top 1% contribution is negative relative to the final update direction. This does **not** mean that good tours receive a negative reward. It means the gradient preferred by the elite samples is in conflict with the aggregate direction produced by the rest of the batch.

Later, as the distribution concentrates, the update becomes dominated by rejecting the relatively poor tail rather than strongly pulling probability toward rare exceptional tours.

### Signal collapse

The best-sample advantage also collapses:

```text
epoch 250:  best advantage ≈ +0.807
epoch 500:  best advantage ≈ +0.034
epoch 700:  best advantage ≈ +0.00073
epoch 1550: best advantage ≈ +0.00002
```

By epochs where the batch becomes essentially one tour, all rewards are equal, so:

```text
advantage -> 0
gradient  -> 0
```

For example at epoch 2050 the batch mean is exactly `3.573707`, the unique fraction is effectively zero, and the gradient is zero.

### Interpretation

This supports the following descriptive sequence:

1. the search distribution is initially broad;
2. very good tours are discovered;
3. individual elite discoveries have limited influence compared with the aggregate batch;
4. the distribution concentrates in a large, reasonably good basin;
5. diversity falls;
6. better tours stop being sampled;
7. reward variance disappears;
8. the score-function signal vanishes;
9. the model becomes stuck.

This is evidence for a **credit-assignment / exploration problem**, but it is not a proof that weak elite weighting is the sole causal mechanism.

---

## 4. Rank-weighted experiment V1

The first intervention replaced the numerical leave-one-out advantage with a tie-aware rank utility while preserving the model, decoder, optimizer, learning-rate schedule and checkpoint rule.

The intent was to make relative quality matter more strongly:

```text
good sample rank -> strong positive signal
bad sample rank  -> negative signal
```

Equal tour lengths receive the same rank utility.

### Result

The elite part of the batch immediately became much more influential.

At epoch 250:

```text
mean     = 3.952025
unique   = 0.209
top1     = +0.422
next9    = +0.490
```

The top 10% therefore accounts for about 91% of the update direction at that point.

However, seed 44 **did not discover the global optimum** in this run. It quickly exploited the `3.522437` basin instead.

By epoch 600:

```text
mean   = 3.538484
best   = 3.522437
unique = 0.008
```

### Interpretation

Rank shaping fixed one observed problem — weak elite influence — but created much stronger selection pressure and accelerated premature exploitation.

Conclusion:

> **Stronger selection pressure alone is not a solution to mode collapse.**

---

## 5. Unique-cycle rank experiment V2

The next hypothesis was that repeated samples of the same decoded permutation create a multiplicity bias.

If a good local tour is sampled 1000 times, ordinary sample-based rank weighting effectively gives it 1000 votes. V2 therefore canonicalized TSP cycles under rotation and reversal and ranked **unique cycles** instead of raw samples.

Each unique cycle received one total rank vote, split over all of its occurrences.

### Early result: clear improvement in exploration

This version rediscovered the global optimum by epoch 200:

```text
epoch 200
mean   = 4.451972
best   = 3.513669
unique = 0.669
dom    = 0.012
```

Compared with V1, this is strong evidence that multiplicity of already-dominant tours can accelerate premature concentration.

### Late result: gradient-scale instability

V2 also multiplied the unique-cycle signal by `B / U` to keep the loss equivalent to an average over unique cycles.

As `U` became small, this amplification became extreme:

```text
epoch 300: grad ≈   27
epoch 400: grad ≈   77
epoch 650: grad ≈  472
epoch 750: grad ≈ 6226
epoch 800: grad ≈ 2172
```

Gradient clipping prevented the optimizer from literally applying these magnitudes, but the raw estimator became extremely unstable.

At epoch 750:

```text
mean = 3.522449
dom  = 1.000
grad = 6226
```

and by epoch 850:

```text
mean   = 3.522438
val    = 3.522438
dom    = 1.000
unique = 0
grad   = 0
```

The distribution had fully collapsed to the `3.522438` mode.

### Interpretation

V2 produced the most encouraging discovery behavior of the rank variants, but the full multiplicity correction plus `B/U` scaling introduced a severe late-stage gradient-scale problem.

This is useful, but still not a complete solution.

---

## 6. Partial multiplicity correction V3

V3 tested an intermediate correction:

```text
sample weight ~ 1 / sqrt(count)
```

rather than:

```text
V1: no multiplicity correction
V2: 1 / count
```

The `B/U` amplification was also removed.

### Result

The global optimum was again found by epoch 200:

```text
epoch 200
mean   = 4.373688
best   = 3.513669
unique = 0.579
dom    = 0.022
grad   = 3.75
```

The catastrophic V2 gradient explosion was largely avoided, but convergence became slower and less attractive:

```text
epoch 700:  mean = 3.649169, dom = 0.484
epoch 1000: mean = 3.639496, dom = 0.448
epoch 1100: mean = 3.543872, dom = 0.890
```

### Interpretation

V3 changed two things relative to V2 — multiplicity weighting and global signal scaling — so it is not a clean ablation of the V2 failure.

The experiment nevertheless shows that simply interpolating the multiplicity exponent is unlikely to be a satisfying research answer.

---

## 7. V2.1: the clean ablation prepared but not yet treated as the main direction

A cleaner V2.1 was prepared:

- keep V2's full `1 / count` unique-cycle correction;
- keep rank temperature `0.10`;
- remove only the `B/U` amplification.

For a fixed batch, removing `B/U` preserves the V2 gradient direction and changes only its scale.

This is a much cleaner test than V3.

However, after reviewing the supervisor's thesis chapter, the decision is to **pause further tuning until supervisor input**, rather than continue a chain of local fixes without a clear methodological target.

---

## 8. Connection to the supervisor's earlier TSP work

Chapter 7 of the provided thesis uses a very closely related methodology:

- RealNVP as a continuous sampling distribution;
- `argsort` as the discretization for permutations;
- symmetric TSP experiments;
- Adam optimization;
- uniform latent samples from `[-1, 1]^n`;
- 4 RealNVP layers;
- batch size 1024.

The chapter reports that the method learns useful TSP structure and improves substantially over random search, but does not converge close to the optimum.

Most importantly, the analysis reports that after convergence the learned distribution has a very narrow peak, almost no probability around it, and a nearly singular Jacobian. The thesis identifies this as **premature concentration / mode collapse**.

The proposed future-work direction is an explicit exploration strategy analogous to reinforcement learning: sample mainly from the learned distribution but, with some probability, also introduce samples from a uniform distribution.

That is almost exactly the phenomenon independently observed in today's TSP-20 diagnostics.

### Why this matters

The agreement between the two investigations makes several simple explanations less convincing:

- it is probably not only a Gaussian-vs-uniform latent issue;
- it is probably not only a 4-layer-vs-8-layer capacity issue;
- it is probably not only the exact batch size;
- it is probably not a trivial implementation bug in the current experiment.

Different versions of the same general RealNVP + discretization idea show the same qualitative failure:

> **the learned search distribution eventually becomes too concentrated and stops exploring.**

---

## 9. What is established vs. what is still uncertain

### Supported by today's experiments

- RealNVP can discover the global optimum on the fixed TSP-20 instance.
- Discovery does not guarantee that the final learned distribution retains probability mass on that optimum.
- Permutation diversity can collapse rapidly.
- In the original REINFORCE estimator, the top 1% of the batch can have very small or even conflicting directional influence during the critical phase.
- Rank shaping strongly increases elite influence.
- Stronger rank selection can accelerate premature exploitation.
- Correcting multiplicity across repeated tours changes the behavior materially.
- Full unique-cycle correction with `B/U` scaling can produce extremely large raw gradients.
- Once the batch collapses to a single objective value, centered score-function signals vanish.

### Not established

- We have **not** proven that the original REINFORCE estimator is fundamentally unsuitable.
- We have **not** proven that rank weighting is the correct replacement.
- We have **not** shown that a particular entropy, KL, mutation, mixture or off-policy method will solve the problem.
- We have **not** established that the random-key `argsort` representation itself is the fundamental cause.
- A single seed is useful diagnostically but is not enough for a final algorithmic claim.
- The experimental rank variants should not yet be presented as a new method; they are diagnostic interventions.

---

## 10. Recommended next step

Do **not** continue blindly tuning:

- rank temperature;
- archive weight;
- archive size;
- multiplicity exponent;
- arbitrary reward scaling.

The next step should be a short technical discussion with the supervisor.

### Questions to resolve with the supervisor

1. Is the intended research goal to **solve** the mode-collapse problem, or is careful diagnosis/comparison of mitigation strategies already a valid project contribution?
2. Given the thesis's future-work note, should the next experiment use an explicit **exploration mixture**?
3. If external/uniform samples are introduced, how should the gradient be treated?
   - importance correction / off-policy estimator;
   - separate exploration samples used only for discovery;
   - replay / elite fitting;
   - another principled estimator?
4. Would an entropy or KL/trust-region constraint be more appropriate than external random exploration?
5. Should diversity be controlled in continuous `y` space, permutation space, or both?
6. Is preserving the current general-purpose nature of the method more important than achieving stronger TSP-specific performance?
7. Which diagnostics should be considered primary evidence:
   - unique permutation fraction;
   - dominant-cycle share;
   - entropy estimate;
   - Jacobian conditioning;
   - objective variance;
   - gradient contribution by quality group?

---

## 11. Current working hypothesis

The most useful current hypothesis is:

> The TSP difficulty is primarily an **exploration-retention failure of the learned search distribution**. The score-function weighting affects how quickly the failure occurs and which basin wins, but changing the selection signal alone has not removed the underlying tendency to prematurely concentrate.

The next experiment should therefore be selected to test a **principled exploration-control mechanism**, ideally after supervisor feedback.

---

## 12. Practical status at the end of 2026-09-23

### Keep as authoritative baseline

- original RealNVP TSP setup and frozen benchmark results;
- best-ever solution tracked separately from final generator quality.

### Keep as diagnostics

- gradient contribution split: top 1%, next 9%, middle 80%, bottom 10%;
- unique-tour fraction;
- dominant-cycle fraction;
- raw gradient norm;
- best-sample signal;
- validation elite / validation best.

### Treat as experimental, not final

- rank-weighted V1;
- unique-cycle rank V2;
- partial multiplicity V3;
- V2.1 without `B/U` scaling.

### Immediate action

**Pause algorithm changes and get supervisor input on the exploration/mode-collapse direction.**

This is now a clearly defined research problem, not a debugging mystery.
