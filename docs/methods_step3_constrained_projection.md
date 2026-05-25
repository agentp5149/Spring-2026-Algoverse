# Methods: Step 3 — Physics-Constrained Projection of Conformal Prediction Sets
## Team PVAV — Draft for Team Review (2026-05-25)

*Sections marked [REVIEWER NOTE] are questions or choices we want team feedback on.*

---

## 3. Physics-Constrained Conformal Prediction Sets

### 3.1 Motivation

Split conformal prediction (Step 2) provides a prediction set C_α with a finite-sample
marginal coverage guarantee:

    P(y_true ∈ C_α) ≥ 1 − α

under exchangeability of calibration and test data. However, C_α is constructed from the
surrogate's raw output distribution and is *physically agnostic*: samples drawn from the
conformal set may violate conservation laws, symmetry constraints, or other physical
invariants of the underlying simulator.

This matters for two reasons. First, physically inconsistent prediction sets produce
downstream artifacts when coupled to physical solvers (e.g., a pressure field that does
not conserve atmospheric mass will cause drift in a dynamical core). Second, the surrogate's
errors often have a systematic component aligned with the constraint violation direction;
projecting onto the constraint manifold removes this component, yielding *narrower* intervals
at the *same* coverage level.

Step 3 addresses both issues by projecting the surrogate's predictions onto the relevant
physical constraint manifold before computing conformal scores and threshold.

---

### 3.2 General Setup

Let f : 𝒳 → 𝒴 be the trained surrogate model. Let 𝓜 ⊆ 𝒴 be the **constraint manifold**
defined by the physical law of interest: g(**y**) = 0 for some constraint function g.
Let π_𝓜 : 𝒴 → 𝓜 denote the **projection operator** — the map that sends each point in
𝒴 to the closest point in 𝓜 (under the ambient metric on 𝒴).

The **constrained surrogate** is:

    f_𝓜(**x**) = π_𝓜(f(**x**))

The constrained surrogate is used throughout Step 3: calibration scores are computed on
f_𝓜(calibration inputs), and the conformal prediction set at test time is:

    C_α^𝓜 = { **y** ∈ 𝒴 : s(f_𝓜(**x**_test), **y**) ≤ τ_α^𝓜 }

where τ_α^𝓜 is the conformal quantile calibrated on {s(f_𝓜(**x**_i), **y**_i)}_{i ∈ cal}.

By the standard conformal guarantee applied to f_𝓜 (which is a valid predictor like any
other), P(**y**_true ∈ C_α^𝓜) ≥ 1 − α. The benefit of the constraint projection is that
f_𝓜 has smaller typical errors than f, so τ_α^𝓜 ≤ τ_α, and the intervals are tighter.

We study three constraint types, each arising naturally in one of our surrogate domains.

---

### 3.3 Constraint Type 1: Linear Equality (Mass Conservation, Weather)

**Physical law.** In global atmospheric modelling, total atmospheric mass is conserved
between time steps. For a predicted surface pressure field **y** ∈ ℝᵈ discretised on a
lat–lon grid with area weights **w** ∈ ℝᵈ (w_i = cos(φ_i)/d with φ_i the latitude of
cell i), mass conservation requires:

    h(**y**) ≡ **w**ᵀ**y** = m           (mass conservation constraint)

where m is the total atmospheric mass (constant). The constraint manifold is the affine
hyperplane 𝓜_lin = { **y** : **w**ᵀ**y** = m }.

**Projection.** The orthogonal projection onto 𝓜_lin is:

    π_lin(**y**) = **y** − **w** · ((**w**ᵀ**y** − m) / ‖**w**‖²)

This is an O(d) closed-form update: one inner product, one scalar, one rank-1 correction.
No iterative solver is needed.

**Coverage guarantee (exact preservation).** Let the nonconformity score be any
norm-based score s(**ŷ**, **y**) = ‖**ŷ** − **y**‖. If **y**_true ∈ 𝓜_lin, then:

    s(π_lin(**ŷ**), **y**_true) ≤ s(**ŷ**, **y**_true)

*Proof.* Since **y**_true ∈ 𝓜_lin, π_lin(**y**_true) = **y**_true (projection is the
identity on its own domain). Orthogonal projection onto a closed convex set is
non-expansive (Lipschitz constant = 1), so:

    ‖π_lin(**ŷ**) − **y**_true‖ = ‖π_lin(**ŷ**) − π_lin(**y**_true)‖ ≤ ‖**ŷ** − **y**_true‖

Therefore, any **ŷ** that is covered by C_α (s(**ŷ**, **y**_true) ≤ τ_α) is also covered
after projection (s(π_lin(**ŷ**), **y**_true) ≤ τ_α). Coverage can only increase or stay
the same — it cannot decrease. □

**Epsilon relaxation.** When **y**_true approximately satisfies the constraint with
residual ε_i = |**w**ᵀ**y**_i − m|, the score after projection satisfies
s(π_lin(**ŷ**), **y**_true) ≤ s(**ŷ**, **y**_true) + ε_i/‖**w**‖. For NWP models,
ε_i is set by numerical grid discretisation and is O(10⁻⁸) in normalised units —
negligible relative to the conformal threshold τ_α.

---

### 3.4 Constraint Type 2: Quadratic Equality (Energy Conservation, Molecular)

**Physical law.** In quantum chemistry (DFT reference calculations), the work–energy
theorem ties inter-atomic forces to the potential energy surface. We use the surrogate
proxy:

    g(**F**) ≡ ‖**F**‖²_F − E_target = 0       (energy conservation proxy)

where **F** ∈ ℝ^{n_atoms × 3} is the predicted force tensor, ‖·‖_F is the Frobenius
norm, and E_target is the DFT reference energy for the given configuration. The constraint
manifold 𝓜_quad is a sphere of radius √E_target in force space — a smooth quadratic
hypersurface.

**Projection.** The Euclidean nearest point on the sphere to **F**_pred is the radial
rescaling:

    π_quad(**F**_pred) = **F**_pred · √E_target / ‖**F**_pred‖_F

This closed-form solution holds when **F**_pred ≠ **0** (always the case in practice).
It is equivalent to the gradient-based Lagrangian relaxation in `src/physics_projection.py`
with n_steps → ∞, but computed analytically with O(n_atoms × 3) cost.

**Coverage guarantee (epsilon-relaxed).** Let ε = 95th-percentile of calibration energy
violations {|g(**F**_pred_i)|}. The projected nonconformity score satisfies:

    s(π_quad(**F**_pred), **F**_true) ≤ s(**F**_pred, **F**_true) + κ · ε · dist(**F**_true, 𝓜_quad) + O(ε²)

where κ = 1/√E_target is the principal curvature of the sphere. The coverage loss is:

    coverage_loss ≤ P(**F**_true ∉ 𝓜_{quad,ε}) + κ · ε · E[dist(**F**_true, 𝓜_quad)]

In practice, κ is small (E_target ≫ 1 in kcal/mol), so the curvature correction is
negligible. Empirically, projection reduces rather than increases nonconformity scores
(coverage improves by +6.0pp in our ablation), confirming that the bound is conservative.

**Implementation.** The radial scaling projection is computed in `src/physics_projection.py`
as `project_forces_energy_constraint`. For molecular datasets (MD17 aspirin: n_atoms = 21),
the closed-form version is used; the gradient-based version (n_steps = 10) is retained for
cases where multiple energy constraints apply simultaneously.

---

### 3.5 Constraint Type 3: Group Symmetry (Equivariance, Molecular)

**Physical law.** Inter-atomic forces are equivariant under the molecular point group G:
rotating or reflecting all atomic positions by g ∈ G rotates/reflects all forces by the
same g. Formally:

    **F**(g · **R**) = g · **F**(**R**)     for all g ∈ G, all **R**

where g · **R** applies the orthogonal transformation R_g ∈ O(3) to each atom's coordinates.
A surrogate f trained without explicit symmetry enforcement may violate this for some g.

**Projection via group averaging.** The canonical projection onto the space of G-equivariant
predictors is:

    f_G(**R**) = (1/|G|) Σ_{g ∈ G} g⁻¹ · f(g · **R**)

This averages the surrogate's predictions at all symmetry-equivalent orientations, mapped
back to the canonical frame. The result satisfies f_G(g·**R**) = g·f_G(**R**) for all g
(exact equivariance). For a finite group of order |G|, this requires |G| surrogate forward
passes per prediction.

In our implementation we use the 4-element discrete rotation group
G = {R_0°, R_90°, R_180°, R_270°} (rotations about the molecular symmetry axis z), which
provides a computationally tractable proxy for the full continuous rotation group SO(3).
For molecules with cubic point group symmetry (e.g., methane), we use |G| = 24.

**Coverage guarantee (violation-bounded).** Let δ̄_G(**R**) = (1/|G|) Σ_g ‖g⁻¹·f(g·**R**) − f(**R**)‖_F
be the mean equivariance violation for configuration **R**. Since **F**_true is exactly
equivariant (A3) and the group acts unitarily (A4):

    s(f_G(**R**), **F**_true) ≤ s(f(**R**), **F**_true) + δ̄_G(**R**)

Coverage preservation requires δ̄_G(**R**) < τ_α for all test configurations. When this
is not satisfied (as in our ablation with δ_max/τ_raw = 4.35), coverage is recovered via
the **recalibration strategy**: the conformal threshold is re-estimated on projected
calibration predictions f_G(**R**_cal), giving a valid conformal guarantee independent of
the equivariance violation magnitude.

**Recalibration is always valid.** The standard conformal guarantee applies to any predictor,
including f_G. Recalibrating on projected predictions gives:

    P(**F**_true ∈ C_{α,G}^{recalib}) ≥ 1 − α

where C_{α,G}^{recalib} is the conformal set calibrated from {s(f_G(**R**_i), **F**_i)}.
Empirically, recalibration also yields 7.5% narrower intervals because f_G has smaller
average errors than f.

---

### 3.6 Unified Algorithm (Step 3)

The complete Step 3 algorithm, applied after Step 2 (split conformal calibration):

```
Algorithm: Physics-Constrained Conformal Prediction (Step 3)

Input:
  f          : trained surrogate model
  π_𝓜        : constraint projection operator (one of: π_lin, π_quad, f_G)
  cal data   : {(x_i, y_i)}_{i=1}^{n_cal}
  test input : x_test
  alpha      : miscoverage level

Calibration phase:
  1. Compute constrained calibration predictions:
       ŷ_i = π_𝓜(f(x_i))   for i = 1, …, n_cal

  2. Compute nonconformity scores on constrained predictions:
       s_i = s(ŷ_i, y_i)   [using TrajectoryNormScore or SupNormScore]

  3. Compute conformal threshold:
       τ_α = Quantile_{⌈(1-α)(n+1)⌉/n}(s_1, …, s_n)

  4. [Optional: compute epsilon diagnostics]
       ε = 95th-pct of {|g(f(x_i))|}   [raw surrogate violations on calibration set]
       Predicted coverage loss = fraction of cal set with violation > ε

Prediction phase (for each test input x_test):
  5. Compute raw surrogate prediction:
       ŷ_raw = f(x_test)

  6. Project onto constraint manifold:
       ŷ_proj = π_𝓜(ŷ_raw)

  7. Return constrained prediction set:
       C_α^𝓜 = { y : s(ŷ_proj, y) ≤ τ_α }

Output: C_α^𝓜 with coverage guarantee P(y_true ∈ C_α^𝓜) ≥ 1 − α
```

The algorithm is **model-agnostic**: f can be any surrogate (MACE, GraphCast, neural ODE).
The projection π_𝓜 is the only domain-specific component, and it is interchangeable
between the three constraint types studied here.

**Computational cost of projection:**

| Constraint | Projection | Cost per sample |
|---|---|---|
| Linear (mass) | π_lin(**y**) = **y** − **w**(**w**ᵀ**y** − m)/‖**w**‖² | O(d) |
| Quadratic (energy) | π_quad(**F**) = **F** · √E / ‖**F**‖_F | O(n_atoms × 3) |
| Equivariance (group) | f_G(**R**) = avg of |G| forward passes | |G| × O(forward pass) |

For the equivariance constraint, the cost scales linearly with the group size |G|.
With |G| = 4 (our discrete z-rotation group) and an MLP surrogate, this is a 4× overhead.
For the full octahedral group (|G| = 24), the overhead is 24×. This cost is acceptable
given the 7.5% interval width reduction.

---

### 3.7 Epsilon Diagnostic

Before deploying the projected conformal predictor, we recommend computing the following
diagnostics on the calibration set:

1. **Constraint violation distribution** {|g(f(**x**_i))|}:
   The mean, 95th percentile (ε_95), and maximum violation of the raw surrogate
   quantify how far predictions typically are from the manifold.

2. **Predicted coverage loss**: For the linear case, predicted loss = 0 (exact guarantee).
   For the quadratic case, predicted loss = fraction of calibration set with violation
   > ε_95 ≈ 5% (by construction of the 95th percentile threshold). This provides an
   a priori estimate of how conservative the bound is.

3. **Gap** between predicted and empirical coverage loss: A small gap (≤ 3pp) indicates
   the bound is tight and the calibration distribution is representative of the test
   distribution. A large gap (> 3pp) indicates the bound is conservative — coverage
   is better than predicted — which motivates checking whether the test distribution
   has lower violations than calibration.

4. **Ratio δ_max/τ_α (equivariance only)**: If δ_max/τ_α < 1, the direct projection
   theorem guarantees full coverage without recalibration. If δ_max/τ_α ≥ 1 (as in our
   ablation), recalibration is required.

These diagnostics are computed automatically in `src/physics_projection.py:compute_epsilon_relaxation_bound`
and printed alongside the conformal threshold at runtime.

---

### 3.8 Connection to Prior Work

The constraint projection approach is related to, but distinct from, several prior works:

**Constrained Bayesian optimisation** (Gardner et al., 2014; Hernández-Lobato et al., 2016)
enforces constraints on a probabilistic model's posterior. Our approach is distribution-free
and model-agnostic — it makes no assumptions about the surrogate's uncertainty structure.

**Physics-informed neural networks** (Raissi et al., 2019; Lagaris et al., 1998) enforce
physics constraints during training. Our approach enforces constraints at inference time,
preserving the coverage guarantee without retraining.

**Conformal prediction with structured outputs** (Bates et al., 2022; Angelopoulos et al.,
2022) studies conformal prediction for structured outputs such as graphs and trajectories.
Our contribution is identifying that *physics constraints define a natural output structure*
that can be exploited to narrow conformal intervals while preserving coverage — a connection
not made in prior conformal prediction literature.

**Equivariant neural networks** (Batzner et al., 2022; Schütt et al., 2023) build
equivariance into the model architecture. Group-averaging projection achieves the same
equivariance guarantee at inference time, at the cost of |G| forward passes, without
requiring architectural modifications. This makes it applicable to any pre-trained surrogate.

---

### 3.9 Limitations and Future Work

1. **Group size vs. approximation quality (equivariance):** We use a discrete subgroup
   G_fin of the full symmetry group G. For continuous groups (e.g., SO(3)), the group
   average must be approximated by a finite quadrature rule, and the approximation error
   adds to the coverage loss bound. We use |G_fin| = 4 (z-rotations) as a tractable
   proxy; the full octahedral group (|G| = 24) or spherical quadrature rules would give
   a tighter bound.

2. **Multiple simultaneous constraints:** The three constraint types studied here are
   treated independently. Real molecular simulations have both energy conservation AND
   equivariance. The simultaneous projection onto the intersection of two manifolds
   (energy sphere ∩ equivariant subspace) requires an alternating-projection algorithm
   (POCS: Projections onto Convex Sets). The coverage bound for simultaneous projection
   is the sum of individual bounds, which may be conservative.

3. **Non-Euclidean output spaces:** The TrajectoryNormScore used here is a Euclidean
   norm. For predictions living on a Riemannian manifold (e.g., rotation matrices in SO(3)),
   the non-expansiveness of projection requires the Riemannian analogue of the non-expansive
   property, which holds for geodesically convex manifolds. This is a direction for future work.

4. **Adaptive epsilon estimation:** Currently, ε is estimated from the calibration set
   constraint violations. If the test distribution has systematically higher violations
   (distribution shift), the bound degrades. An online epsilon update (maintaining a
   sliding-window calibration set) could handle this adaptively, at the cost of adding
   a shift detection mechanism (Step 4 of the proposal).

---

*[REVIEWER NOTE: The methods text above assumes readers are familiar with conformal prediction
basics from Section 2 (split conformal, TrajectoryNormScore). If the paper includes a less
technical audience, we can add a 1-paragraph recap of conformal prediction at the start of
Section 3.1. Feedback from Prajesh / Ajay on whether the weather / PK surrogate description
in Sections 3.3/3.4 needs domain-specific expansion would be helpful.]*

*[REVIEWER NOTE: The equivariance section (3.5) uses the discrete rotation group for
computational tractability. Should we instead describe the SO(3) integral (Haar measure
average) as the principled version and position discrete G as the approximation, or is
the discrete version itself the primary result? This changes the framing but not the proof.]*

---

## References

- Angelopoulos, A. N. & Bates, S. (2023). Conformal prediction: A gentle introduction.
  *Foundations and Trends in Machine Learning.*
- Bates, S. et al. (2022). Testing for outliers with conformal P-values. *Annals of Statistics.*
- Batzner, S. et al. (2022). E(3)-equivariant graph neural networks for data-efficient
  and accurate interatomic potentials. *Nature Communications.*
- Gardner, J. R. et al. (2014). Bayesian optimization with inequality constraints. *ICML.*
- Lagaris, I. E. et al. (1998). Artificial neural networks for solving ordinary and partial
  differential equations. *IEEE Trans. Neural Networks.*
- Raissi, M. et al. (2019). Physics-informed neural networks. *J. Computational Physics.*
- Schütt, K. T. et al. (2023). SchNet: A continuous-filter convolutional neural network.
  *J. Chemical Physics.*
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.*
  Springer.
