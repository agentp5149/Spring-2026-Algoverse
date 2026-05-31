# Coverage Bound Under Bounded Shift and Stochastic Dominance

This note derives the coverage guarantee used for adaptive shift detection and interval
widening. The goal is to make precise how much conformal coverage can degrade when the
test distribution is not exchangeable with the calibration distribution, but the degree
of shift is bounded by a measurable shift score.

## Setup

Let

$$
S(x,y) = s(\hat f(x), y)
$$

be a nonconformity score, where smaller scores mean better agreement between the
surrogate prediction and the simulator or observed truth.

Let

$$
S_1,\ldots,S_n \sim P_{\mathrm{cal}}
$$

be calibration scores. For a target miscoverage level $\alpha$, split conformal prediction
chooses the empirical quantile

$$
\hat q_\alpha = S_{(k)}, \qquad
k = \lceil (n+1)(1-\alpha) \rceil ,
$$

with the usual convention that $\hat q_\alpha = \infty$ if $k=n+1$. In the implemented
finite-sample setting, $k \le n$ for the nominal levels used in the experiments.

The conformal prediction set is

$$
C_\alpha(x) = \{y : S(x,y) \le \hat q_\alpha\}.
$$

Under exchangeability between calibration and test scores,

$$
\mathbb P_{P_{\mathrm{cal}}}\{Y \in C_\alpha(X)\}
= \mathbb P_{P_{\mathrm{cal}}}\{S(X,Y) \le \hat q_\alpha\}
\ge 1-\alpha .
$$

The question is what remains true when the test score is drawn from a shifted
distribution $Q_s$ indexed by a shift magnitude $s$.

## Assumption 1: Bounded CDF Deviation

Let $F_{\mathrm{cal}}(t)$ and $F_s(t)$ denote the CDFs of the calibration and shifted
test nonconformity scores:

$$
F_{\mathrm{cal}}(t) = \mathbb P_{P_{\mathrm{cal}}}(S \le t),
\qquad
F_s(t) = \mathbb P_{Q_s}(S \le t).
$$

Assume there exists a nondecreasing function $\varepsilon(s) \in [0,1]$ such that, for
all thresholds $t$,

$$
F_s(t) \ge F_{\mathrm{cal}}(t) - \varepsilon(s).
$$

This is a bounded-shift assumption on the score distribution. It says that the shifted
test distribution may place more mass at high-error scores, but at any threshold it can
lose at most $\varepsilon(s)$ probability mass relative to the calibration distribution.

Equivalently, the shifted score distribution is not allowed to be arbitrarily worse than
the calibration score distribution. A convenient worst-case choice is to take
$\varepsilon(s)$ as an upper bound on the Kolmogorov distance between the score CDFs:

$$
\sup_t \left(F_{\mathrm{cal}}(t) - F_s(t)\right)_+ \le \varepsilon(s).
$$

## Theorem 1: Coverage Under Bounded Shift

Suppose the bounded CDF deviation assumption holds for shift magnitude $s$. Then the
unwidened conformal set satisfies

$$
\mathbb P_{Q_s}\{Y \in C_\alpha(X)\}
\ge
1-\alpha-\varepsilon(s).
$$

More precisely, conditioning on the calibration set,

$$
\mathbb P_{Q_s}\{Y \in C_\alpha(X) \mid S_1,\ldots,S_n\}
= F_s(\hat q_\alpha)
\ge F_{\mathrm{cal}}(\hat q_\alpha) - \varepsilon(s).
$$

Taking expectation over the calibration set gives

$$
\mathbb E\left[
\mathbb P_{Q_s}\{Y \in C_\alpha(X) \mid S_1,\ldots,S_n\}
\right]
\ge
1-\alpha-\varepsilon(s).
$$

## Proof

The coverage event is exactly the event that the test nonconformity score falls below the
conformal threshold:

$$
Y \in C_\alpha(X)
\quad \Longleftrightarrow \quad
S(X,Y) \le \hat q_\alpha .
$$

Therefore, under the shifted distribution $Q_s$,

$$
\mathbb P_{Q_s}\{Y \in C_\alpha(X) \mid S_1,\ldots,S_n\}
= F_s(\hat q_\alpha).
$$

By the bounded CDF deviation assumption, for every threshold $t$,

$$
F_s(t) \ge F_{\mathrm{cal}}(t) - \varepsilon(s).
$$

Substituting $t=\hat q_\alpha$ gives

$$
F_s(\hat q_\alpha)
\ge
F_{\mathrm{cal}}(\hat q_\alpha) - \varepsilon(s).
$$

For a fresh calibration-distribution score $S_{n+1}\sim P_{\mathrm{cal}}$, the standard
split conformal rank argument gives

$$
\mathbb P\{S_{n+1} \le \hat q_\alpha\}
\ge 1-\alpha.
$$

Equivalently,

$$
\mathbb E[F_{\mathrm{cal}}(\hat q_\alpha)] \ge 1-\alpha.
$$

Combining the two inequalities yields

$$
\mathbb E[F_s(\hat q_\alpha)]
\ge
\mathbb E[F_{\mathrm{cal}}(\hat q_\alpha)] - \varepsilon(s)
\ge
1-\alpha-\varepsilon(s).
$$

Thus the worst-case expected coverage under shift magnitude $s$ is bounded below by

$$
\boxed{
\mathrm{Coverage}_{Q_s}
\ge
\max\{0,\;1-\alpha-\varepsilon(s)\}.
}
$$

The outer maximum only enforces that coverage is a probability and cannot be negative.

## Assumption 2: Stochastic Dominance With Additive Score Shift

A stronger and often more interpretable assumption is additive stochastic dominance.
Assume there is a nonnegative shift radius $\Delta(s)$ such that

$$
S_{Q_s} \preceq_{\mathrm{st}} S_{P_{\mathrm{cal}}} + \Delta(s),
$$

meaning

$$
F_s(t) \ge F_{\mathrm{cal}}(t-\Delta(s))
\qquad
\text{for all } t.
$$

This says that shifted test errors are stochastically no worse than calibration errors
plus an additive penalty $\Delta(s)$.

## Theorem 2: Coverage Recovery by Widening

Under additive stochastic dominance, the widened conformal set

$$
C_{\alpha,s}^{\mathrm{wide}}(x)
=
\{y : S(x,y) \le \hat q_\alpha + \Delta(s)\}
$$

satisfies

$$
\mathbb P_{Q_s}\{Y \in C_{\alpha,s}^{\mathrm{wide}}(X)\}
\ge
1-\alpha.
$$

## Proof

The widened coverage probability is

$$
\mathbb P_{Q_s}\{S(X,Y) \le \hat q_\alpha + \Delta(s)\}
= F_s(\hat q_\alpha + \Delta(s)).
$$

By additive stochastic dominance,

$$
F_s(\hat q_\alpha + \Delta(s))
\ge
F_{\mathrm{cal}}(\hat q_\alpha).
$$

Taking expectation over the calibration set and using the split conformal rank argument,

$$
\mathbb E[F_s(\hat q_\alpha + \Delta(s))]
\ge
\mathbb E[F_{\mathrm{cal}}(\hat q_\alpha)]
\ge
1-\alpha.
$$

Therefore, if the widening radius $\Delta(s)$ is a valid stochastic-dominance bound,
adaptive widening recovers nominal conformal coverage under the shifted distribution.

## Practical Plug-In Bound for the PK OOD Experiment

In the PK OOD experiment, the shift score is the standardized distance from the calibration
parameter centroid:

$$
s(x)
=
\left\|
\frac{x-\mu_{\mathrm{cal}}}{\sigma_{\mathrm{cal}}}
\right\|_2.
$$

Let $s_0$ be the 95th percentile of calibration shift scores. We use the conservative
linear degradation model

$$
\varepsilon(s)
=
\mathrm{clip}
\left(
\frac{s-s_0}{s_0},
0,
1
\right).
$$

This yields the minimum expected coverage function

$$
\boxed{
L(s)
=
\max
\left\{
0,\;
1-\alpha
-
\mathrm{clip}
\left(
\frac{s-s_0}{s_0},
0,
1
\right)
\right\}.
}
$$

Interpretation:

- If $s \le s_0$, the point is within the calibration shift envelope and the lower bound
  remains $1-\alpha$.
- If $s_0 < s < 2s_0$, the guaranteed coverage decays linearly.
- If $s \ge 2s_0$, the worst-case bound becomes vacuous, meaning the shift is too severe
  for this assumption to certify nontrivial coverage.

The interval-widening rule used in the experiment is

$$
\hat q_{\alpha,s}
=
\hat q_\alpha
\cdot
\max\left\{1,\frac{s}{s_0}\right\}.
$$

This multiplicative rule is an empirical approximation to the additive stochastic
dominance correction. In paper language, the bound should be stated as conditional on
the assumed validity of $\varepsilon(s)$ or $\Delta(s)$ as an upper bound on score-distribution
shift. The empirical OOD experiment then tests how conservative or optimistic that plug-in
assumption is for rare-metabolizer PK regimes.

## Summary

The bounded-shift guarantee is

$$
\mathrm{Coverage}_{Q_s}
\ge
\max\{0,\;1-\alpha-\varepsilon(s)\}.
$$

The stronger stochastic-dominance guarantee is

$$
\mathrm{Coverage}_{Q_s}
\left(
C_{\alpha,s}^{\mathrm{wide}}
\right)
\ge
1-\alpha
\quad
\text{when}
\quad
S_{Q_s} \preceq_{\mathrm{st}} S_{P_{\mathrm{cal}}}+\Delta(s).
$$

The first theorem gives a worst-case degradation bound. The second theorem explains when
adaptive widening can recover nominal coverage under OOD shift.
