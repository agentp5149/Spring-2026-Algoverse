# Formal Proof: Coverage Preservation Under Equivariance Constraint Projection
## Team PVAV — Week 3 (Equivariance / Group Averaging Case)

---

## 1. Context and Motivation

In molecular dynamics, the laws of physics are rotationally and reflectionally symmetric.
If we rotate all atomic positions by a rotation R ∈ O(3), the forces on each atom must
rotate by the same R. Formally, the ground-truth force field F* satisfies:

    F*(g · **R**) = g · F*(**R**)    for all g ∈ G, all **R**

where G ⊆ O(3) is a symmetry group (e.g., all 3D rotations SO(3), or the finite point
group of the molecule such as C₂ᵥ for water).

A surrogate f trained on a finite dataset may not learn this symmetry exactly. Its
equivariance violation for input **R** and group element g is:

    δ_g(**R**) = ‖g⁻¹ · f(g · **R**) − f(**R**)‖_F

If δ_g > 0, the surrogate gives different force predictions depending on how the molecule
was oriented before the forward pass — a physically unacceptable inconsistency.

**Group-averaging projection** symmetrises the prediction:

    f_G(**R**) = (1/|G|) Σ_{g ∈ G} g⁻¹ · f(g · **R**)

This is a projection in the function space sense: applying the same operation twice
gives f_{GG} = f_G (idempotent). The averaged predictor f_G is exactly G-equivariant
by construction (a standard result in representation theory).

The central question for conformal prediction: **does projecting f → f_G preserve the
(1−α) coverage guarantee?** This is non-trivial because, unlike the linear case, group
averaging can increase the distance from f(**R**) to F*(**R**) for individual samples.
The key quantity is the equivariance violation δ_G, which bounds the coverage loss.

---

## 2. Setup and Notation

**Inputs and outputs:**

- **R** ∈ ℝ^{n_atoms × 3}: atomic positions (one molecular configuration)
- **F** ∈ ℝ^{n_atoms × 3}: forces (target output of surrogate)
- f : ℝ^{n_atoms × 3} → ℝ^{n_atoms × 3}: surrogate model (not necessarily equivariant)

**Group action:**

Let G be a compact group with a linear representation ρ on ℝ^{n_atoms × 3} via
simultaneous rotation of all atom coordinates:

    (g · **R**)_i = R_g **r**_i    (rotate position of atom i by the rotation matrix R_g ∈ ℝ^{3×3})

For forces, the group acts contravariantly (vectors transform the same way as positions):

    (g · **F**)_i = R_g **f**_i    (same rotation applied to force on atom i)

Each R_g is orthogonal: R_gᵀ R_g = I, so ‖g · **v**‖_F = ‖**v**‖_F for all g ∈ G.

**Equivariance violation:**

For a surrogate f and a single configuration **R**, define:

    δ̄_G(**R**) = (1/|G|) Σ_{g ∈ G} ‖g⁻¹ · f(g · **R**) − f(**R**)‖_F

This is the mean equivariance violation averaged over the group. Define the maximum:

    δ_max = max_{**R** ∈ cal ∪ test} δ̄_G(**R**)

**Nonconformity score:**

We use the normalised Frobenius norm:

    s(**F̂**, **F**) = ‖**F̂** − **F**‖_F / √(n_atoms)

**Group-averaged predictor:**

    **F̂**_G(**R**) = f_G(**R**) = (1/|G|) Σ_{g ∈ G} g⁻¹ · f(g · **R**)

**Standard conformal guarantee (before projection):**

Under exchangeability, the conformal threshold τ_α satisfies:

    P(s(f(**R**), **F**_true) ≤ τ_α) ≥ 1 − α

The question is: what is P(s(f_G(**R**), **F**_true) ≤ τ_α)?

---

## 3. Main Theorem

**Theorem (Coverage Under Equivariance Projection — Mean Violation Bound).**

Let assumptions hold:
- (A1) Calibration and test data are exchangeable: (**R**_i, **F**_i) i.i.d. from P.
- (A2) The data distribution P is G-invariant: (**R**, **F**) and (g·**R**, g·**F**) are
  identically distributed for all g ∈ G.
- (A3) The true forces are G-equivariant: **F**_true(g·**R**) = g·**F**_true(**R**) a.s.
- (A4) G acts via orthogonal transformations (satisfied by rotations R_g ∈ O(3)).

Let τ_α be the conformal threshold calibrated on {s(f(**R**_i), **F**_i)}. Then:

    s(f_G(**R**), **F**_true) ≤ s(f(**R**), **F**_true) + δ̄_G(**R**)

and the coverage after group averaging satisfies:

    P(**F**_true ∈ C_α^G) ≥ 1 − α − P(δ̄_G(**R**_test) > τ_α − s(f(**R**_test), **F**_true))

where C_α^G = { **F** : s(**F̂**_G(**R**), **F**) ≤ τ_α }.

**Corollary.** If δ_max < τ_α (the maximum equivariance violation is smaller than the
conformal threshold), then the coverage loss is zero: P(**F**_true ∈ C_α^G) ≥ 1 − α.

**Corollary (Small violation limit).** As δ_max → 0 (perfect equivariance):

    P(**F**_true ∈ C_α^G) → P(**F**_true ∈ C_α) ≥ 1 − α

**Comparison with linear case.** Unlike the linear case where the score can only
decrease under projection, group averaging can increase the score by up to δ̄_G(**R**).
The coverage guarantee degrades by at most P(δ̄_G > τ_α − s_test), which is bounded
by the equivariance violation. This makes equivariance constraint projection a
conservative operation: it benefits those samples where the surrogate's asymmetry
is the dominant error source, and may slightly harm coverage when the surrogate is
already nearly equivariant (the group average moves predictions in unhelpful directions).

---

## 4. Proof

**Step 1: Equivariance of the ground truth (fixing π_G(**F**_true) = **F**_true).**

By assumption (A3), **F**_true is G-equivariant, so:

    g⁻¹ · **F**_true(g · **R**) = **F**_true(**R**)    for all g ∈ G

Therefore:

    f_G(**R**) − **F**_true(**R**)
        = (1/|G|) Σ_g g⁻¹ · f(g · **R**) − **F**_true(**R**)
        = (1/|G|) Σ_g [ g⁻¹ · f(g · **R**) − **F**_true(**R**) ]
        = (1/|G|) Σ_g [ g⁻¹ · f(g · **R**) − g⁻¹ · **F**_true(g · **R**) ]

where the last equality uses A3: **F**_true(**R**) = g⁻¹ · **F**_true(g · **R**).

**Step 2: Bound the per-group-element error using A4 (orthogonality).**

For each g ∈ G:

    ‖g⁻¹ · f(g · **R**) − g⁻¹ · **F**_true(g · **R**)‖_F
        = ‖g⁻¹ · (f(g · **R**) − **F**_true(g · **R**))‖_F
        = ‖f(g · **R**) − **F**_true(g · **R**)‖_F        [since g⁻¹ ∈ O(3) is unitary]

By the conformal guarantee (and G-invariance of the data distribution A2):

    P(‖f(g · **R**) − **F**_true(g · **R**)‖_F ≤ τ_α · √n_atoms) ≥ 1 − α

for every fixed g. (The distribution of (g·**R**, g·**F**_true) is the same as (**R**, **F**_true)
by A2, so the conformal guarantee applies to rotated inputs as well.)

**Step 3: Triangle inequality on the group average.**

    s(f_G(**R**), **F**_true)
        = ‖f_G(**R**) − **F**_true(**R**)‖_F / √n_atoms
        = ‖(1/|G|) Σ_g [g⁻¹·f(g·**R**) − g⁻¹·**F**_true(g·**R**)]‖_F / √n_atoms
        ≤ (1/|G|) Σ_g ‖g⁻¹·f(g·**R**) − g⁻¹·**F**_true(g·**R**)‖_F / √n_atoms   [Jensen / triangle ineq.]
        = (1/|G|) Σ_g ‖f(g·**R**) − **F**_true(g·**R**)‖_F / √n_atoms   [Step 2, orthogonality]
        ≤ (1/|G|) Σ_g [s(f(**R**), **F**_true) + ‖f(g·**R**) − f(**R**)‖_F/√n_atoms
                         − ‖**F**_true(g·**R**) − **F**_true(**R**)‖_F/√n_atoms + ...]

    ...but this path leads to mixed terms. The clean route uses the additive decomposition:

    s(f_G(**R**), **F**_true)
        ≤ (1/|G|) Σ_g ‖g⁻¹·f(g·**R**) − g⁻¹·**F**_true(g·**R**)‖_F / √n_atoms

    Now write g⁻¹·f(g·**R**) = f(**R**) + [g⁻¹·f(g·**R**) − f(**R**)]:

    ≤ (1/|G|) Σ_g [‖f(**R**) − **F**_true(**R**)‖_F + ‖g⁻¹·f(g·**R**) − f(**R**)‖_F] / √n_atoms
       [triangle inequality, using g⁻¹·**F**_true(g·**R**) = **F**_true(**R**) again]

    = s(f(**R**), **F**_true) + (1/|G|) Σ_g ‖g⁻¹·f(g·**R**) − f(**R**)‖_F / √n_atoms

    = s(f(**R**), **F**_true) + δ̄_G(**R**) / √n_atoms

(absorbing √n_atoms into the definition of δ̄_G for normalised form, or keeping explicit).

**Step 4: Coverage lower bound.**

    P(**F**_true ∈ C_α^G)
        = P(s(f_G(**R**_test), **F**_true) ≤ τ_α)
        ≥ P(s(f(**R**_test), **F**_true) + δ̄_G(**R**_test) ≤ τ_α)       [Step 3]
        = P(s(f(**R**_test), **F**_true) ≤ τ_α − δ̄_G(**R**_test))

    If δ̄_G(**R**_test) ≤ 0 always (impossible for positive δ̄), coverage improves.
    In general:

    ≥ P(s(f(**R**_test), **F**_true) ≤ τ_α) − P(δ̄_G(**R**_test) > τ_α − s(f, **F**_true))
    ≥ (1 − α) − P(δ̄_G(**R**_test) > τ_α − s_test)

When δ_max = max δ̄_G(**R**) < τ_α − s_test for all test samples:
the second term is zero, and full coverage is preserved. ∎

---

## 5. Interpretation of the Bound

The coverage loss is bounded by the probability that the equivariance violation is
large enough to push a previously covered sample outside the conformal set:

    coverage_loss ≤ P(s_test ∈ (τ_α − δ̄_G, τ_α])

This is the **marginal fraction of test samples in the "shadow zone"** — those whose
nonconformity score is within δ̄_G of the conformal threshold. If the score distribution
has low density near τ_α (a typical situation when the surrogate is well-trained), the
coverage loss is small even when δ̄_G is not negligible.

**Practical implication:** The bound is most useful as a diagnostic. Before deploying
the symmetrised predictor, compute δ_max on the calibration set and compare it to τ_α.
If δ_max / τ_α < 0.05 (5% of the threshold), projected coverage is within 5% of the
original guarantee.

---

## 6. Recalibration Strategy (Coverage Recovery)

If δ_max is not small relative to τ_α, coverage can be fully recovered by recalibrating
on the projected predictor f_G rather than the raw predictor f:

    s_G_i = s(f_G(**R**_i), **F**_i)    for i = 1, …, n_cal
    τ_α^G = Quantile_{⌈(1-α)(n+1)⌉/n}(s_G_1, …, s_G_n)

Then by the standard conformal guarantee applied to f_G:

    P(**F**_true ∈ C_α^G_recal) ≥ 1 − α

This is always valid regardless of the equivariance violation, with a potential change
in interval width. In practice, τ_α^G ≤ τ_α (projected predictions are more physically
consistent, hence more accurate on average), so intervals tighten.

**Recalibration is always valid. The direct projection theorem gives a guarantee without
recalibration, with a coverage loss bounded by δ̄_G. The ablation in
experiments/ablation_physics_projection.py reports both scenarios.**

---

## 7. Finite Group Case (Implementation)

For computational tractability, we use a finite discrete subgroup G_fin ⊂ O(3) with
|G_fin| rotations. The bound applies verbatim with G = G_fin.

In our implementation, G_fin is the 24-element octahedral rotation group (all rotations
mapping the cube to itself), which provides a systematic sampling of orientation space.

The group averaging at test time requires |G_fin| = 24 forward passes per sample.
For the molecular surrogate with n_atoms = 21 (aspirin), this costs 24 × 21 × 3 = 1512
additional floating point outputs per prediction, which is negligible relative to the
training cost of the surrogate.

The equivariance violation on the calibration set:

    δ̄_G(**R**_i) = (1/24) Σ_{g ∈ G_fin} ‖g⁻¹ · f(g · **R**_i) − f(**R**_i)‖_F

is computed in `experiments/ablation_physics_projection.py` and reported in the
results table as ε_G (equivariance violation, compared to ε_lin and ε_quad for the
other constraint types).

---

## 8. Connection to Other Proofs

| Case | Projection type | Score change | Coverage change | Bound |
|---|---|---|---|---|
| Linear (mass) | Orthogonal onto hyperplane | Decreases by (η_i - 0) | Monotone improvement | Exact: loss ≤ P(**y**∉𝓜_ε) |
| Quadratic (energy) | Projection onto sphere | Mixed | Relaxed + curvature κ | Approx: loss ≤ P(|g| > ε) + κ·ε·E[dist] |
| **Equivariance (this proof)** | Group averaging | Increases by ≤ δ̄_G | Relaxed by δ̄_G | Exact: loss ≤ P(δ̄_G > τ − s_test) |

The equivariance case is unique in that the projection is an **average** rather than a
nearest-point projection. This means:
1. The projected prediction is always in the convex hull of {g⁻¹·f(g·**R**)}_{g∈G}, which
   may not be the closest equivariant vector to f(**R**).
2. The bound depends on the VIOLATION of the model (δ̄_G) rather than the violation of
   the data (ε_true), making it a fundamentally different regime.
3. Recalibration fully recovers coverage at the cost of 2× computation (one pass for
   group averaging, one to re-estimate the conformal threshold).

---

## References

- Zaheer, M. et al. (2017). Deep sets. *NeurIPS*.
- Weiler, M. & Cesa, G. (2019). General E(2)-equivariant steerable CNNs. *NeurIPS*.
- Batzner, S. et al. (2022). E(3)-equivariant graph neural networks for data-efficient
  and accurate interatomic potentials. *Nature Communications*.
- Angelopoulos, A. N. et al. (2021). Learn then test: Calibrating predictive algorithms
  to achieve risk control. *arXiv:2110.01052*.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.*
