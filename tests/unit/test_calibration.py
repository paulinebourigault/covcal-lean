"""Tests for covcal.calibration.

Reference values for Clopper--Pearson come from explicit beta-quantile computations.
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import beta

from covcal.calibration import (
    CalibrationResult,
    clopper_pearson_upper,
    make_grid,
    select_thresholds,
)
from covcal.types import Thresholds


class TestClopperPearsonUpper:
    def test_zero_accepted_returns_one(self):
        assert clopper_pearson_upper(0, 0, 0.05) == 1.0

    def test_zero_errors_closed_form(self):
        # k=0 ⇒ U = 1 - alpha^{1/m}
        m, alpha = 20, 0.05
        assert clopper_pearson_upper(0, m, alpha) == pytest.approx(1.0 - alpha ** (1.0 / m))

    def test_full_errors_returns_one(self):
        # k=m ⇒ U = 1 (anything could be true).
        assert clopper_pearson_upper(5, 5, 0.1) == 1.0

    def test_interior_matches_beta_quantile(self):
        k, m, alpha = 3, 50, 0.05
        expected = beta.ppf(1.0 - alpha, k + 1, m - k)
        assert clopper_pearson_upper(k, m, alpha) == pytest.approx(expected)

    def test_monotone_in_k(self):
        m, alpha = 100, 0.05
        prev = -math.inf
        for k in range(0, m + 1):
            u = clopper_pearson_upper(k, m, alpha)
            assert u >= prev - 1e-12  # non-decreasing in k
            prev = u

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            clopper_pearson_upper(0, 10, 0.0)
        with pytest.raises(ValueError):
            clopper_pearson_upper(0, 10, 1.0)
        with pytest.raises(ValueError):
            clopper_pearson_upper(5, 4, 0.1)


class TestMakeGrid:
    def test_cartesian_product(self):
        grid = make_grid([0.0, 0.5], [0.0, 0.25, 0.5], [-0.1, 0.0])
        assert len(grid) == 2 * 3 * 2
        # Deterministic order: sorted by (typ, prf, margin) tuple insertion.
        assert isinstance(grid[0], Thresholds)


class TestSelectThresholds:
    def test_picks_max_accepted_feasible(self):
        # Two thresholds, both feasible; the one accepting more wins.
        t1 = Thresholds(0.5, 0.5, 0.0)
        t2 = Thresholds(0.0, 0.0, 0.0)
        acc = {t1: (20, 1), t2: (100, 5)}  # both <= 10% empirical risk
        res = select_thresholds([t1, t2], acc, epsilon=0.30, delta=0.05)
        assert res.selected == t2
        assert res.accepted_count == 100

    def test_infeasible_returns_reject_all(self):
        t = Thresholds(0.5, 0.5, 0.0)
        # Many errors among accepted - should fail any reasonable epsilon.
        acc = {t: (10, 9)}
        res = select_thresholds([t], acc, epsilon=0.05, delta=0.05)
        assert res.reject_all
        assert res.selected is None

    def test_zero_accepted_treated_as_infeasible_with_low_epsilon(self):
        t = Thresholds(0.9, 0.9, 0.5)
        # m=0 ⇒ U=1, which is > any epsilon in (0,1).
        res = select_thresholds([t], {t: (0, 0)}, epsilon=0.10, delta=0.05)
        assert res.reject_all

    def test_bonferroni_correction_applied(self):
        # Same (m, k) but grid sizes differ; bigger grid ⇒ stricter alpha ⇒ larger UB.
        t = Thresholds(0.0, 0.0, 0.0)
        small = make_grid([0.0], [0.0], [0.0])  # |T| = 1
        large = make_grid([0.0, 0.5], [0.0, 0.5], [0.0, 0.5])  # |T| = 8
        # Populate only `t`; other thresholds default to (0, 0) which fail.
        res_small = select_thresholds(small, {t: (50, 2)}, epsilon=0.20, delta=0.05)
        res_large = select_thresholds(large, {t: (50, 2)}, epsilon=0.20, delta=0.05)
        assert res_small.per_threshold_alpha > res_large.per_threshold_alpha
        # Both should still be feasible at epsilon=0.20, but UB of large is larger.
        assert res_large.risk_upper_bound >= res_small.risk_upper_bound

    def test_returns_calibration_result_dataclass(self):
        t = Thresholds(0.0, 0.0, 0.0)
        res = select_thresholds([t], {t: (10, 0)}, epsilon=0.5, delta=0.05)
        assert isinstance(res, CalibrationResult)
        assert res.epsilon == 0.5
        assert res.grid_size == 1
