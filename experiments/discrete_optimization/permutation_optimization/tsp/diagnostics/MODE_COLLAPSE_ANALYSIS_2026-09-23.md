# TSP-20 RealNVP: mode-collapse investigation — 2026-09-23

## Executive summary

The main result of today's work is that the current TSP-20 failure mode is no longer best described as "the model cannot find the optimum".

On seed 44 the RealNVP search distribution **does find the known optimum** `3.513668846`, but the learned distribution later loses permutation diversity and collapses onto a worse tour basin. The original leave-one-out REINFORCE diagnostics show small or opposing elite-group projections onto the total raw gradient at some critical epochs. These projections describe the current gradient, not Adam's actual parameter step, and do not by themselves establish that elite samples are underweighted. Rank shaping changes this behavior, but by itself does not solve the underlying loss of exploration.

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

The optimization process finds the global optimum as a best-ever solution, but the current sampling distribution later concentrates around a worse tour. The reported final test samples from the selected checkpoint do not include the optimum. This finite-sample observation does not establish that its probability is exactly zero.

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

For each group we compute its contribution to the original score-function loss gradient and project it onto the total raw gradient. For this decomposition, group contributions must use the same full-batch normalization so that they sum to `g_total`:

```text
projection_share(group)
    = dot(g_group, g_total) / ||g_total||^2
```

When `g_total` is nonzero, the four projection shares sum to approximately 1. A negative value means that group's raw gradient is partially opposed to the total raw gradient. These shares are projections, not probabilities, and may lie outside [0, 1]. They are undefined for a zero total gradient and can be sensitive to numerical error when its norm is very small.

Adam combines gradient history with coordinate-wise second-moment scaling, so its actual parameter step generally differs from the current raw gradient direction. Record `delta_theta = theta_after_step - theta_before_step` separately before drawing conclusions about the optimizer's applied update. See the [PyTorch Adam algorithm](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html).

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

At epoch 250, the best 1% of samples has a projection share of about **1% of the total raw gradient**, while the middle 80% has a share of about **70%**. A group containing 1% of samples contributing 1% of this projection is not, by itself, evidence of insufficient weighting. Group size, gradient magnitude, alignment, and the intended optimization objective all matter.

At epoch 300 the top 1% contribution is negative relative to the total raw gradient. This does **not** mean that good tours receive a negative advantage; rewards themselves are already negative tour lengths. It means the elite group's raw gradient conflicts with the aggregate raw gradient. It does not directly establish how Adam's step changes the probability of an elite tour.

Later, as the distribution concentrates, the bottom 10% accounts for most of the projection onto the total raw gradient. This is consistent with a signal dominated by penalizing the relatively poor tail, but it does not measure the change in discrete probability assigned to rare exceptional tours.

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

For example at epoch 2050 the reported batch mean is `3.573707`, the displayed unique fraction rounds to zero, and the reported gradient is zero. One unique cycle in a batch of 4096 has fraction `1 / 4096`, not literally zero. With equal rewards, centered advantages vanish in exact arithmetic; floating-point residuals can remain. Adam's momentum can also produce a nonzero parameter step after the current raw gradient becomes zero.

### Interpretation

This supports the following descriptive sequence:

1. the search distribution is initially broad;
2. very good tours are discovered;
3. elite-group gradients sometimes have small or opposing projections onto the aggregate raw gradient;
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

The top 10% therefore accounts for about 91% of the projection onto the total raw gradient at that point, not necessarily 91% of Adam's parameter step.

However, seed 44 **did not discover the global optimum** in this run. It quickly exploited the `3.522437` basin instead.

By epoch 600:

```text
mean   = 3.538484
best   = 3.522437
unique = 0.008
```

### Interpretation

Rank shaping increased elite-group projection shares and changed selection pressure. In this reported seed it was followed by earlier concentration on a suboptimal tour. This does not establish that the original weighting was incorrect or that increasing elite influence resolves the retention problem.

Conclusion:

> **Stronger selection pressure alone is not a solution to mode collapse.**

---

## 5. Unique-cycle rank experiment V2

The next hypothesis was that repeated samples of already-common cycles reinforce concentration under the rank-weighted objective. This is a hypothesis about optimization dynamics, not an established bias in the original on-policy REINFORCE estimator.

Repeated tours are expected when sampling from the learned distribution: their occurrence frequency reflects the probability mass assigned to them. Ordinary sample-based weighting includes every occurrence. V2 instead canonicalized TSP cycles under rotation and reversal and ranked **unique cycles** rather than raw samples.

Each unique cycle received one total rank vote, split over all of its occurrences. This reweights the sampled distribution and changes the training signal; it is not automatically a correction to an invalid estimator.

### Early result: encouraging discovery in the reported seed

This version rediscovered the global optimum by epoch 200:

```text
epoch 200
mean   = 4.451972
best   = 3.513669
unique = 0.669
dom    = 0.012
```

Compared with V1, this is consistent with the hypothesis that multiplicity-sensitive weighting affects concentration. The single-seed comparison does not isolate multiplicity as the cause, particularly because V2 also uses the `B/U` normalization described below.

### Late result: large raw gradient norms

V2 also multiplied the unique-cycle signal by `B / U` to keep the loss equivalent to an average over unique cycles.

As `U` became small, this amplification became extreme:

```text
epoch 300: grad ≈   27
epoch 400: grad ≈   77
epoch 650: grad ≈  472
epoch 750: grad ≈ 6226
epoch 800: grad ≈ 2172
```

These are large raw gradient norms. Clipping bounds the gradient passed to Adam, and Adam further transforms it using optimizer state. The reported norms therefore do not establish equally large or harmful parameter steps. Record clipping frequency and actual step norms before attributing optimization instability to the raw magnitudes.

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

V2 showed encouraging discovery behavior in the reported seed and very large late-stage raw gradients. For a fixed batch, `B/U` directly amplifies the gradient, but these observations do not isolate its role in the eventual collapse from unique-cycle weighting, density-score magnitudes, and optimizer dynamics.

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

The very large raw gradient norms reported for V2 were largely avoided, while concentration developed more slowly in this run:

```text
epoch 700:  mean = 3.649169, dom = 0.484
epoch 1000: mean = 3.639496, dom = 0.448
epoch 1100: mean = 3.543872, dom = 0.890
```

### Interpretation

V3 changed two things relative to V2 — multiplicity weighting and global signal scaling — so it is not a clean ablation of the V2 failure.

The experiment does not identify whether multiplicity weighting or signal scaling explains the difference. It therefore motivates a controlled ablation rather than selecting a multiplicity exponent from this run alone.

---

## 7. V2.1: the clean ablation prepared but not yet treated as the main direction

A cleaner V2.1 was prepared:

- keep V2's full `1 / count` unique-cycle correction;
- keep rank temperature `0.10`;
- remove only the `B/U` amplification.

For a fixed batch with weights treated as detached, removing `B/U` preserves the raw V2 gradient direction and changes only its scale. That does not imply identical Adam steps or training trajectories: clipping and optimizer history must also be considered.

If this ablation is run, compare V2 and V2.1 with identical initialization, architecture, rank temperature, checkpoint rule, and objective-evaluation budget. Record:

- raw gradient norm and the fraction of updates that activate clipping;
- gradient norm after clipping;
- actual parameter-step norm, `||theta_after_step - theta_before_step||`;
- unique-cycle count and dominant-cycle share at a fixed diagnostic sample size;
- best-ever tour length, sampled tour quality, and sampling frequency of previously discovered good tours.

Use matched seeds and account for any diagnostic evaluations used to make optimization decisions. This is a cleaner test of gradient scaling than V3, not a commitment to rank weighting as the final method.

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
- In the original REINFORCE estimator, the top 1% can have small or opposing projections onto the total raw gradient during the critical phase.
- Rank shaping increases elite-group projection shares in the reported comparison.
- Stronger rank selection can accelerate premature exploitation.
- Reweighting repeated tours by cycle multiplicity changes the observed behavior; this is not evidence that ordinary on-policy sampling is biased.
- Full unique-cycle correction with `B/U` scaling can produce extremely large raw gradients.
- Once the batch collapses to a single objective value, centered score-function signals vanish.

### Not established

- We have **not** proven that the original REINFORCE estimator is fundamentally unsuitable.
- We have **not** proven that rank weighting is the correct replacement.
- We have **not** established that small elite projection shares imply insufficient weighting, or that those shares describe Adam's actual step.
- We have **not** isolated `B/U` scaling as the cause of collapse or established harmful parameter-step sizes from raw gradient norms.
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

The next step should be a short technical discussion with the supervisor, centered on this question:

> **How can we maintain access to alternative tours while still concentrating probability on good solutions?**

The reported rank experiments make exploration control a stronger next direction than simply increasing elite or archive weights. Retention and exploration remain hypotheses to test through one clearly specified mechanism at a time.

For an exploration mixture, define the learning rule before implementation:

- **Exploration for discovery only:** external samples can improve the stored best tour, but do not directly train the flow if they are excluded from its update.
- **Exploration used for learning:** specify the target objective, the distribution that generated the candidates, and a valid estimator or explicit auxiliary fitting objective. External samples must not silently be treated as fresh draws from the current flow. Importance weighting is one possible approach, not an automatic requirement for every possible formulation.

Also specify whether exploration is introduced in latent space, continuous key space, or permutation space; these are different interventions. Evaluate diversity after decoding. Increased continuous entropy alone need not increase tour diversity, since positive scaling of all keys preserves their ordering.

Keep the baseline fixed and use the same total objective-evaluation budget. Record best-ever quality separately from generator quality. A primary success criterion should be whether the proposed mechanism preserves or improves the sampling frequency of previously discovered good tours (or a predefined quality threshold) without sacrificing search quality. Estimate this with fixed-size independent probes; count probes in the optimization budget when they influence decisions. A finite probe with no hits does not prove zero probability.

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

- raw-gradient contribution split: top 1%, next 9%, middle 80%, bottom 10%;
- unique-tour fraction;
- dominant-cycle fraction;
- raw gradient norm; add post-clipping norms, clipping frequency, and actual parameter-step norms when testing V2.1;
- sampling frequency of previously discovered good tours, measured with a fixed probe budget;
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
