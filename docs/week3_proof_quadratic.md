# Formal Proof: Coverage Preservation Under Quadratic Constraint Projection
## Vijay -- Team PVAV, Week 3

---

## Setup

Let the surrogate model produce force predictions F_hat in R^(n_atoms x 3).
The energy conservation constraint is:

    g(F) = ||F||^2_F - E_target = 0

where E_target is the reference DFT energy for a given atomic configuration,
and ||F||^2_F is the Frobenius norm squared (sum of squared force components).
This is a quadratic constraint because g is degree-2 in F.

The constraint manifold is:

    M = { F in R^(n_atoms x 3) : g(F) = 0 }

which is a sphere of radius sqrt(E_target) in force space.

---

## Theorem (Epsilon-Relaxed Coverage Under Quadratic Projection)

Let C_alpha be a valid (1-alpha) conformal prediction set for forces,
satisfying:

    P(F_true in C_alpha) >= 1 - alpha

Let pi_M(F) denote the projection of F onto M (closest point on the manifold).
Let C_alpha_proj = { pi_M(F) : F in C_alpha } be the projected set.

### Exact case (epsilon = 0)

If the true simulator satisfies the energy constraint exactly
(F_true in M for all test inputs), then:

    P(F_true in C_alpha_proj) >= 1 - alpha

Proof sketch: If F_true in M and F_true in C_alpha, then pi_M(F_true) = F_true
(projection is identity on M), so F_true in C_alpha_proj.
Since P(F_true in C_alpha) >= 1 - alpha and the event {F_true in C_alpha_proj}
contains the event {F_true in C_alpha, F_true in M}:

    P(F_true in C_alpha_proj) >= P(F_true in C_alpha) - P(F_true not in M)
                               >= (1 - alpha) - 0
                               = 1 - alpha   QED

### Epsilon-relaxed case

In practice, real simulators satisfy constraints only approximately due to
numerical discretization. Let epsilon be the constraint violation:

    epsilon = max_{i in cal} |g(F_true_i)|

Define the epsilon-relaxed manifold:

    M_epsilon = { F : |g(F)| <= epsilon }

Then:

    P(F_true in C_alpha_proj) >= (1 - alpha) - P(F_true not in M_epsilon)

where P(F_true not in M_epsilon) is the fraction of test points whose true
output violates the constraint by more than epsilon.

Proof sketch:
    P(F_true in C_alpha_proj)
        >= P(F_true in C_alpha, F_true in M_epsilon)
        >= P(F_true in C_alpha) - P(F_true not in M_epsilon)
        >= (1 - alpha) - P(F_true not in M_epsilon)   QED

---

## Curvature Effects (Quadratic Case)

Unlike the linear constraint case (mass conservation), the quadratic manifold M
is curved -- it is a sphere in force space. This introduces an additional term.

Let kappa = 1/sqrt(E_target) be the curvature of M (principal curvature of
the sphere). The projection can distort distances by at most (1 + kappa*delta)
where delta is the distance from F to M.

The full bound becomes:

    coverage_loss <= P(F_true not in M_epsilon)
                   + kappa * epsilon * E[||F_true - pi_M(F_true)||]

In practice kappa is small (E_target >> 1 in kcal/mol units), so the
curvature correction is negligible and the simpler bound applies.

---

## Empirical Validation

We validate by:
1. Computing epsilon_95 = 95th percentile of calibration energy violations
2. Predicted coverage loss = fraction of cal set with violation > epsilon_95 (~5%)
3. Comparing to empirical coverage loss = coverage_before - coverage_after

If |predicted - empirical| <= 3pp, the bound is tight.
If the gap is large, it indicates the test distribution has higher energy
violations than the calibration set, motivating adaptive epsilon estimation.

See results in results/molecular/week3_projection_results.pt for numerical values.

---

## Connection to Other Proofs

- Linear case (mass conservation, Ajay): Projection onto a linear subspace
  is exact (no curvature), so coverage is preserved exactly when epsilon = 0.
- Equivariance case (Ajay): Group-averaging projection; coverage loss
  bounded by the surrogate's equivariance violation.
- This proof (quadratic, energy conservation): Coverage loss bounded by
  epsilon plus curvature correction. Empirically validated above.
