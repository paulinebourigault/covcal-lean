"""Tests for covcal.normalization."""

from __future__ import annotations

import pytest

from covcal.normalization import (
    MC_PREFIX,
    UNNORMALIZED_PREFIX,
    extract_boxed,
    is_unnormalized,
    normalize_answer,
)


class TestExtractBoxed:
    def test_simple(self):
        assert extract_boxed(r"foo \boxed{42}") == "42"

    def test_last_box_wins(self):
        assert extract_boxed(r"\boxed{1} bar \boxed{2}") == "2"

    def test_nested_braces(self):
        assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_missing_box(self):
        assert extract_boxed("no box here") is None


class TestRationals:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("1/2", "1/2"),
            ("2/4", "1/2"),
            ("-3/-6", "1/2"),
            ("3/-6", "-1/2"),
            ("5/1", "5"),
            ("0/3", "0"),
            ("42", "42"),
            ("-7", "-7"),
        ],
    )
    def test_canonical(self, raw, expected):
        assert normalize_answer(raw) == expected


class TestDecimals:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("3.50", "7/2"),
            ("1.000", "1"),
            ("0.5", "1/2"),
            ("-0.25", "-1/4"),
        ],
    )
    def test_decimal_to_rational(self, raw, expected):
        assert normalize_answer(raw) == expected


class TestBoxed:
    def test_strip_box(self):
        assert normalize_answer(r"\boxed{1/2}") == "1/2"

    def test_strip_box_dollars(self):
        assert normalize_answer(r"$\boxed{42}$") == "42"

    def test_latex_frac(self):
        assert normalize_answer(r"\boxed{\frac{1}{2}}") == "1/2"


class TestSymbolic:
    def test_sqrt_canonical(self):
        # sympy canonicalizes "sqrt(2)/2" and "1/sqrt(2)" to the same thing.
        a = normalize_answer("sqrt(2)/2")
        b = normalize_answer("1/sqrt(2)")
        assert a == b

    def test_pi_preserved(self):
        out = normalize_answer("2*pi")
        # Either "2*pi" or "2*pi" canonical form is fine; we only require equality with
        # another spelling of the same thing.
        out2 = normalize_answer(r"\boxed{2\pi}")
        assert out == out2


class TestMultipleChoice:
    @pytest.mark.parametrize("raw", ["A", "(B)", "  c ", "(D)"])
    def test_letters(self, raw):
        out = normalize_answer(raw)
        assert out.startswith(MC_PREFIX)
        assert out.split("::", 1)[1] == raw.strip(" ()").upper()


class TestTuples:
    def test_tuple(self):
        assert normalize_answer("(1,2,3)") == "(1,2,3)"

    def test_tuple_whitespace(self):
        assert normalize_answer("[ 1 , 2 , 3 ]") == "(1,2,3)"

    def test_tuple_with_fractions(self):
        # Each element gets canonicalized.
        assert normalize_answer("(2/4, 0.5)") == "(1/2,1/2)"


class TestUnnormalized:
    def test_garbage_marks_unnormalized(self):
        out = normalize_answer("not an answer at all !!")
        assert is_unnormalized(out)
        assert out.startswith(UNNORMALIZED_PREFIX)

    def test_empty_input(self):
        out = normalize_answer("")
        assert is_unnormalized(out)


class TestUnits:
    def test_strip_dollars_unit(self):
        assert normalize_answer("42 dollars") == "42"

    def test_strip_degrees(self):
        assert normalize_answer("30 degrees") == "30"


class TestSympyReturnsTuple:
    """Regression: parse_expr can return a Python tuple for some MATH-500 references.

    Reproduces the AttributeError that crashed the first cpu1 run: `tuple` has no
    `free_symbols`. The normalizer must degrade gracefully (return an UNNORMALIZED tag
    or fall through to a tuple handler) instead of crashing.
    """

    def test_parenthesised_pair_does_not_crash(self):
        # Cleaned by the upstream tuple handler. The test is that no exception escapes.
        out = normalize_answer("(2, 3)")
        assert isinstance(out, str)
        # Either tuple-normalized or UNNORMALIZED — both acceptable; what matters is no crash.
        assert out in {"(2,3)"} or out.startswith("UNNORMALIZED::")

    def test_nested_tuple_does_not_crash(self):
        out = normalize_answer("((1, 2), 3)")
        assert isinstance(out, str)
