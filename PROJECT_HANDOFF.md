# RealNVP for discrete optimization: project handoff

Gaber Zupan Škrlj, Jožef Stefan Institute, 1 October 2026

## Introduction

This document is for whoever takes over the project. In September 2026 I investigated at IJS whether a normalizing flow model, specifically RealNVP, can serve as an optimization algorithm for discrete problems. It describes the idea I started with, how the work developed, what I found and where, in my opinion, the work should continue.

The document is intentionally short. All numbers, experiment settings and technical details are in [`PROJECT_HANDOFF_TECHNICAL.md`](PROJECT_HANDOFF_TECHNICAL.md), and the 2D study has its own description in [`experiments/continuous_optimization/README.md`](experiments/continuous_optimization/README.md). I suggest reading this document first and using the technical one as a reference once you start working with the code.

## The idea

Classical optimization algorithms, for example evolutionary ones, maintain one or a few current solutions and improve them step by step. The idea of the project was different: instead of individual solutions, we learn an entire probability distribution over solutions. We sample candidates from it, evaluate them and shift the distribution towards the better ones. If this worked, the model could keep several good regions of the solution space at once, which is an interesting property for hard problems with many local optima.

RealNVP generates continuous vectors. A discrete solution is obtained by rounding the vector (binary problems) or by sorting it (permutations, so-called random keys). Since this step is not differentiable, the model is trained with REINFORCE, which only needs the value of the objective function, not its gradient.

## How the work went

I first tested the method on five standard binary problems and compared it with a simple (1+1) evolutionary algorithm. The method worked, but it was worse or much slower everywhere. It did, however, show an interesting pattern: the model often found good solutions, and then its distribution narrowed down to a single one, not necessarily the best.

The main part of the work was on permutation problems: TSP-20, QAP and PFSP. On TSP the same pattern showed up very clearly. The model often found the optimal tour during training, but by the end of training it almost never generated it anymore. I called this the difference between *discovering* and *retaining* a solution, and it became the central question of the project.

Several attempts to make the model retain the optimum followed: a different learning-rate schedule, extra exploration, and a KL term with a temperature that keeps the distribution wider. The KL term improved discovery on TSP and increased diversity on all three problems, but none of the attempts solved retention of the optimum.

For a fair comparison I then ran a (1+1) evolutionary algorithm with the same number of evaluations on all three permutation problems. The result was clear and is the main conclusion of the comparative part.

Finally, I wanted to understand why the model loses good solutions. I ran two controls. The first replaces RealNVP with the simplest possible model, a diagonal Gaussian distribution, to see whether the problem lies in the flow itself. The second is a 2D problem with two minima, where the target distribution is known exactly, so we can tell whether the model lacks capacity or whether the problem is in the training itself.

## What I found

On none of the tested problems was RealNVP better than the simple evolutionary algorithm. On the permutation problems the algorithm reached an equally good or better solution on all ten seeds, and on TSP it found the optimum about 50 times faster (median over seeds).

| Problem | RealNVP + KL | (1+1)-EA |
| --- | --- | --- |
| TSP-20 | optimum on 8/10 seeds | optimum on 10/10 seeds |
| QAP Nug20 | on average about 9 % above the optimum | on average about 2 % above the optimum |
| PFSP Ta001 | optimum on 0/10 seeds | optimum on 10/10 seeds |

The second result is the difference between discovery and retention. The model finds the optimum, but the final distribution as a rule does not contain it. None of the tried fixes solved this reliably; some older runs occasionally retained the optimum.

The third result is that the problem is not specific to RealNVP. A Gaussian model with only 40 parameters did about as well as RealNVP on TSP-20, in some respects even better. The extra expressiveness of the flow therefore did not help. The learning rate was chosen with the same procedure for both models; RealNVP does not retain the optimum even at its best learning rate.

The fourth and, in my view, most interesting result comes from the 2D study. The model can represent a distribution with several minima, and when it starts from the correct distribution, it also keeps it during training. When it is trained from scratch, however, it loses the rare minimum, even though the solution with both minima would be better by the very criterion it optimizes. The reason is simple: once there are no more samples in some region, training gets no information about it and cannot return there. The problem is therefore in the training dynamics, not in the model's capacity. Whether the same mechanism causes the loss of the optimum on TSP has not been tested yet.

## Where I would continue

First I would check whether the 2D finding also holds for permutations: start TSP training from a distribution that already contains the optimum and see whether the model keeps it. If it does, the problem there is also in the training dynamics. Also worthwhile are a comparison with CMA-ES on the same representation, the Gaussian model on QAP and PFSP, and more repetitions of the Jump_k experiment. A detailed list is at the end of the technical document.

I would not continue by adding new architectures or heuristic fixes until it is clear whether the bottleneck is the training or the problem representation.

## Getting started

1. Read this document, then the section "Pitfalls for a successor" in [`PROJECT_HANDOFF_TECHNICAL.md`](PROJECT_HANDOFF_TECHNICAL.md). It contains things that are not obvious from the code and have already led to wrong conclusions.
2. The code of the permutation experiments is in `experiments/discrete_optimization/benchmarks/permutation/`, the results in `results/permutation_optimization/`, the 2D study in `experiments/continuous_optimization/`.
3. Run scripts from their own folder with `~/Real-nvp-learning/.venv/bin/python`. CUDA does not work on the big.ijs.si server; everything runs on the CPU.
