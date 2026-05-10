# Baseline Uncertainty Methods: One-Page Comparison

**Methods covered:** Deep Ensembles · MC Dropout · Bayesian Variational Inference (mean-field)

---

## 1. Deep Ensembles (Lakshminarayanan et al., 2017)

Train M independent copies of the same architecture from different random seeds. At inference, the predictive mean is the member average and the predictive variance is the empirical variance across members.

**Compute cost.** Training is M× the cost of a single model with no added algorithmic overhead. Wall-clock scales linearly with M. Inference requires M forward passes, which can be parallelised across GPUs but serialises on CPU. For M = 5 neural ODE members on CPU at ~60 s/member: ~5 min total training, ~5× inference overhead.

**Calibration claims.** Empirically the best-calibrated of the three methods across standard benchmarks (Ovadia et al., 2019). Diversity is achieved through loss-surface multimodality; members genuinely explore different modes. However, calibration is *empirical*, not distribution-free — coverage at a chosen nominal level is not guaranteed in finite samples.

**Known failure modes.**
- *Computational budget:* M full training runs; prohibitive for large surrogates (e.g. weather GNN).
- *Correlated members:* shared architecture + data → members can collapse to similar solutions, understating uncertainty in low-data regimes.
- *Overconfident tails:* variance underestimates uncertainty on out-of-distribution inputs; ensemble spread narrows where no member has seen data.
- *No coverage guarantee:* a 90%-nominal interval can empirically cover 75% on shifted data with no theoretical recourse.

---

## 2. MC Dropout (Gal & Ghahramani, 2016)

Retain dropout layers active at test time and draw T stochastic forward passes. Interpret the sample mean and variance as approximate posterior predictive moments under a Bernoulli approximate posterior.

**Compute cost.** Zero added training cost (dropout is standard regularisation). Inference requires T forward passes through a single model — typically T = 30–100. Cheaper than ensembles by a factor of ~M/T when T < M × (epochs per member). Memory footprint is a single model.

**Calibration claims.** Theoretically motivated as variational inference with a Bernoulli approximate posterior, but the approximation is loose. In practice calibration is competitive with ensembles for in-distribution data but degrades faster under covariate shift. Dropout rate p is a sensitivity parameter that requires tuning; wrong p → systematic over- or under-dispersion.

**Known failure modes.**
- *Hyperparameter sensitivity:* predictive variance scales with p and T; both require validation-set tuning, which costs calibration data.
- *Architecture coupling:* dropout must be inserted before every weight layer; awkward for neural ODEs where the dynamics network is small and dropout collapses it.
- *Underestimated epistemic uncertainty:* the Bernoulli family is a weak approximate posterior; uncertainty is systematically underestimated for complex posteriors.
- *No coverage guarantee:* same limitation as ensembles — intervals carry no finite-sample validity claim.

---

## 3. Bayesian Variational Inference — Mean-Field (MFVI)

Replace each weight W with a Gaussian q(W) = N(μ, σ²) and minimise the ELBO: E_q[log p(y|x,W)] − KL(q ∥ p). At inference draw T weight samples and aggregate.

**Compute cost.** Training requires two parameters per weight (μ, σ), roughly doubling parameter memory. The reparameterisation trick makes the ELBO gradient unbiased but each minibatch gradient is noisier than a deterministic gradient. Convergence often requires 2–5× more epochs. Inference overhead matches MC Dropout (T passes). Libraries: Pyro, TyXe (for neural ODEs), or manual Flipout layers.

**Calibration claims.** Principled posterior approximation with explicit prior; uncertainty has a Bayesian interpretation. Calibration can be better than MC Dropout when the prior is well-specified, but mean-field (diagonal covariance) ignores weight correlations. For neural ODEs the posterior is highly non-Gaussian, making MFVI a poor approximation in high-dimensional parameter space.

**Known failure modes.**
- *Mean-field pathology:* diagonal covariance severely underestimates posterior uncertainty in directions of high weight correlation; intervals narrow where they should widen.
- *Prior sensitivity:* results depend on the choice of prior variance σ₀; uninformative priors tend to over-regularise while tight priors impose strong shrinkage.
- *Optimisation instability:* KL annealing required to avoid posterior collapse early in training; adds a schedule hyperparameter.
- *Neural ODE incompatibility:* weight-sampling at every ODE solver step is expensive; combining MFVI with adaptive-step solvers (dopri5) is non-trivial.
- *No coverage guarantee:* same as above.

---

## Summary Table

| Property | Deep Ensembles | MC Dropout | Bayesian VI (MFVI) |
|---|---|---|---|
| **Train overhead** | M × single-model | None | ~2–5× single-model |
| **Inference overhead** | M forward passes | T forward passes | T forward passes |
| **Calibration (in-dist)** | Best empirically | Good | Variable, prior-sensitive |
| **Calibration (OOD)** | Moderate | Degrades faster | Often poor |
| **Coverage guarantee** | None | None | None |
| **Neural ODE fit** | Excellent | Awkward | Difficult |
| **Key failure mode** | Correlated members | Dropout rate tuning | Mean-field collapse |

**Bottom line for this project.** Deep ensembles are the recommended baseline: strongest empirical calibration, straightforward to implement on top of `PKNeuralODE`, and the dominant comparison point in the literature. MC Dropout is the cheapest ablation. MFVI is worth including as a principled foil but expect mean-field pathology on the neural ODE dynamics network.

**Why all three still need conformal wrapping.** None provides a distribution-free coverage guarantee. Split conformal prediction (already implemented in `src/conformal.py`) converts any of the three into a method with finite-sample valid coverage at a chosen α — that is the thesis of this project.

---

*References:* Lakshminarayanan et al. (NeurIPS 2017); Gal & Ghahramani (ICML 2016); Blundell et al. (ICML 2015); Ovadia et al. (NeurIPS 2019); Angelopoulos & Bates (2023 tutorial).
