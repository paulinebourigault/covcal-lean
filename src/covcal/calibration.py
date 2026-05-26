"""Finite-sample selective-risk calibration.

Two valid regimes are supported:

1. **bonferroni** (Eq. 7 of the paper): a Bonferroni-corrected union bound over the
   predeclared grid T. Per-cell alpha = delta / |T|. Valid under arbitrary dependence
   between cell statistics; conservative because the |T| Clopper-Pearson UBs are computed
   from the same shared cal sample (highly dependent in practice, especially under the
   nested-grid structure).

2. **dev-then-cal** (Theorem 1 of the paper): two-stage procedure. Stage 1 selects
   tau_hat on the dev split (most permissive cell whose dev empirical error <= epsilon_dev,
   with deterministic lex tie-break). Stage 2 certifies on the cal split with a single
   Clopper-Pearson UB at level alpha = delta -- no union bound. Valid under i.i.d.
   D / C / T because tau_hat is sigma(D)-measurable and therefore independent of C.

Both regimes:

* use the Clopper-Pearson exact binomial upper bound U_alpha(k, m) with the paper's
  convention U_alpha(k, 0) = 1 (vacuous when m = 0);
* return CalibrationResult.reject_all = True when no feasible threshold exists.

The grid T is fixed in the YAML config and must not be touched after calibration labels
are inspected. The optimizers are pure: they consume per-threshold (m, k) pairs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product

from scipy.stats import beta as _beta

from .types import Thresholds


def clopper_pearson_upper(k: int, m: int, alpha: float) -> float:
    """One-sided Clopper--Pearson upper bound for a Bernoulli mean.

    Returns ``U`` such that ``Pr_{X ~ Bin(m, p)}[X >= k] <= alpha`` for ``p = U``.
    Convention: ``U(k, 0, alpha) = 1`` (vacuous bound when no accepted examples).

    Edge cases:
      * ``k == 0``: ``U = 1 - alpha^(1/m)``.
      * ``k == m``: ``U = 1``.
      * otherwise: ``U = Beta_inv(1 - alpha; k + 1, m - k)``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if m < 0 or k < 0 or k > m:
        raise ValueError(f"need 0 <= k <= m, got k={k}, m={m}")
    if m == 0:
        return 1.0
    if k == m:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / m)
    return float(_beta.ppf(1.0 - alpha, k + 1, m - k))


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Outcome of the Eq. (7) maximum-coverage selector."""

    selected: Thresholds | None  # None ⇔ reject-all (no feasible threshold)
    accepted_count: int  # m_{hat tau} on the calibration set
    accepted_errors: int  # k_{hat tau} on the calibration set
    risk_upper_bound: float  # U_{delta/|T|}(k, m); always reported
    epsilon: float
    delta: float
    grid_size: int
    per_threshold_alpha: float  # delta / |T| used in CP bound

    @property
    def reject_all(self) -> bool:
        return self.selected is None


def make_grid(typ: Iterable[float], prf: Iterable[float], margin: Iterable[float]) -> list[Thresholds]:
    """Cartesian product of the three threshold axes; matches Eq. (8) in the appendix."""
    typ_l, prf_l, margin_l = sorted(set(typ)), sorted(set(prf)), sorted(set(margin))
    return [Thresholds(t, p, m) for t, p, m in product(typ_l, prf_l, margin_l)]


def select_thresholds(
    grid: list[Thresholds],
    accepted: dict[Thresholds, tuple[int, int]],
    *,
    epsilon: float,
    delta: float,
) -> CalibrationResult:
    """Eq. (7): argmax over thresholds whose CP upper bound is at most epsilon.

    ``accepted[tau] = (m_tau, k_tau)`` where ``m_tau`` is the number of accepted calibration
    examples and ``k_tau`` is the number of accepted *errors*.

    Tie-breaking among feasible thresholds with the same ``m_tau``:
      1) smaller risk upper bound,
      2) larger total threshold strictness (sum of typ + prf + margin), to prefer the
         tighter rule among ties so the certificate remains as informative as possible.
    """
    if not grid:
        raise ValueError("threshold grid is empty")
    if not (0.0 < epsilon < 1.0):
        raise ValueError(f"epsilon must be in (0,1), got {epsilon!r}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0,1), got {delta!r}")

    per_alpha = delta / len(grid)

    feasible: list[tuple[Thresholds, int, int, float]] = []
    for tau in grid:
        m, k = accepted.get(tau, (0, 0))
        u = clopper_pearson_upper(k, m, per_alpha)
        if u <= epsilon:
            feasible.append((tau, m, k, u))

    if not feasible:
        return CalibrationResult(
            selected=None,
            accepted_count=0,
            accepted_errors=0,
            risk_upper_bound=1.0,
            epsilon=epsilon,
            delta=delta,
            grid_size=len(grid),
            per_threshold_alpha=per_alpha,
        )

    feasible.sort(key=lambda r: (-r[1], r[3], -(r[0].typ + r[0].prf + r[0].margin)))
    tau, m, k, u = feasible[0]
    return CalibrationResult(
        selected=tau,
        accepted_count=m,
        accepted_errors=k,
        risk_upper_bound=u,
        epsilon=epsilon,
        delta=delta,
        grid_size=len(grid),
        per_threshold_alpha=per_alpha,
    )


def select_thresholds_dev_then_cal(
    grid: list[Thresholds],
    dev_accepted: dict[Thresholds, tuple[int, int]],
    cal_accepted: dict[Thresholds, tuple[int, int]],
    *,
    epsilon: float,
    delta: float,
    epsilon_dev: float | None = None,
) -> CalibrationResult:
    """Theorem 1 (dev-then-cal certificate, no union-bound penalty).

    Stage 1 (cell selection on dev):
        Pick tau_hat = argmax m_tau^D over cells with k_tau^D / m_tau^D <= epsilon_dev
        and m_tau^D > 0. Tie-break deterministically by lex order over
        (tau.typ, tau.prf, tau.margin) so tau_hat is a measurable function of the
        dev sample only. If no cell is feasible on dev, return reject-all.

    Stage 2 (certificate on cal):
        Compute (m, k) = cal_accepted[tau_hat] and U = U_delta(k, m). If m == 0 or
        U > epsilon, return reject-all; otherwise return tau_hat with U as the
        certified upper bound.

    The key statistical fact is that because tau_hat depends only on D and D is
    independent of C, the single Clopper-Pearson bound at level alpha = delta is
    valid -- no Bonferroni correction over the |T| grid cells.

    Defaults: epsilon_dev = epsilon (the dev acceptance criterion is "looks below
    epsilon empirically"; the certificate's actual coverage is enforced on cal).
    """
    if not grid:
        raise ValueError("threshold grid is empty")
    if not (0.0 < epsilon < 1.0):
        raise ValueError(f"epsilon must be in (0,1), got {epsilon!r}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0,1), got {delta!r}")
    if epsilon_dev is None:
        epsilon_dev = epsilon
    if not (0.0 < epsilon_dev <= 1.0):
        raise ValueError(f"epsilon_dev must be in (0,1], got {epsilon_dev!r}")

    # Stage 1: select tau_hat on dev. We require m_dev > 0 and k_dev/m_dev <= epsilon_dev.
    # Among candidates, prefer largest m_dev; lex order over (typ, prf, margin) breaks ties.
    candidates: list[tuple[Thresholds, int]] = []
    for tau in grid:
        m_d, k_d = dev_accepted.get(tau, (0, 0))
        if m_d == 0:
            continue
        if k_d / m_d <= epsilon_dev:
            candidates.append((tau, m_d))

    if not candidates:
        return CalibrationResult(
            selected=None,
            accepted_count=0,
            accepted_errors=0,
            risk_upper_bound=1.0,
            epsilon=epsilon,
            delta=delta,
            grid_size=len(grid),
            per_threshold_alpha=delta,
        )

    # argmax m_dev, with deterministic lex tie-break.
    candidates.sort(key=lambda r: (-r[1], r[0].typ, r[0].prf, r[0].margin))
    tau_hat, _ = candidates[0]

    # Stage 2: certify on cal at alpha = delta (single CP bound, no Bonferroni).
    m, k = cal_accepted.get(tau_hat, (0, 0))
    if m == 0:
        return CalibrationResult(
            selected=None,
            accepted_count=0,
            accepted_errors=0,
            risk_upper_bound=1.0,
            epsilon=epsilon,
            delta=delta,
            grid_size=len(grid),
            per_threshold_alpha=delta,
        )
    u = clopper_pearson_upper(k, m, delta)
    if u > epsilon:
        return CalibrationResult(
            selected=None,
            accepted_count=m,
            accepted_errors=k,
            risk_upper_bound=u,
            epsilon=epsilon,
            delta=delta,
            grid_size=len(grid),
            per_threshold_alpha=delta,
        )
    return CalibrationResult(
        selected=tau_hat,
        accepted_count=m,
        accepted_errors=k,
        risk_upper_bound=u,
        epsilon=epsilon,
        delta=delta,
        grid_size=len(grid),
        per_threshold_alpha=delta,
    )
