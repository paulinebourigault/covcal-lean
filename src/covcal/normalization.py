"""Answer normalization: raw string -> canonical answer-class label.

Implements the normalization rules listed in Sec. 4.3 of the paper, applied in order:

1. extract from `\\boxed{...}` if present (handled upstream in covcal.generation.extract;
   passed as already-extracted text here, but we re-run a defensive extraction in case);
2. strip surrounding whitespace, dollar signs, and trailing punctuation;
3. canonicalize fractions, decimals, signs;
4. try sympy parsing + numeric equivalence for rationals, decimals, radicals, finite tuples;
5. preserve multiple-choice letters when no mathematical value is available;
6. fall back to `UNNORMALIZED::<raw>` when parsing fails (never silently merged).

This module is intentionally deterministic and side-effect-free. It must be frozen before
calibration labels are inspected.
"""

from __future__ import annotations

import logging
import re
from fractions import Fraction

import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

logger = logging.getLogger(__name__)

UNNORMALIZED_PREFIX = "UNNORMALIZED::"
MC_PREFIX = "MC::"

# Match `\boxed{...}` allowing one level of nested braces.
_BOXED_RE = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_MC_RE = re.compile(r"^\s*\(?([A-Ea-e])\)?\s*$")
_SYMPY_TRANSFORMATIONS = (*standard_transformations, implicit_multiplication_application)
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?]+$")


def extract_boxed(text: str) -> str | None:
    """Return the contents of the last `\\boxed{...}` in `text`, or None."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _strip_wrappers(s: str) -> str:
    s = s.strip()
    # Strip LaTeX inline math wrappers and outer parens that don't carry meaning.
    while s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    # Drop trailing punctuation a model might tack on.
    s = _TRAILING_PUNCT_RE.sub("", s)
    return s.strip()


def _strip_units(s: str) -> str:
    # Strip common unit words appended after the number. Conservative on purpose.
    s = re.sub(
        r"\s*(dollars?|cents?|degrees?|radians?|units?|meters?|cm|mm|km|kg|g|s|hours?|minutes?)\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def _try_multiple_choice(s: str) -> str | None:
    m = _MC_RE.match(s)
    if m is None:
        return None
    return f"{MC_PREFIX}{m.group(1).upper()}"


def _canonical_fraction(s: str) -> str | None:
    """Return canonical "p/q" or "p" for plain rationals like 3/4, -2/6, 5, 5/1."""
    s_clean = s.replace(" ", "")
    if re.fullmatch(r"-?\d+/-?\d+", s_clean):
        num, den = s_clean.split("/")
        try:
            f = Fraction(int(num), int(den))
        except ZeroDivisionError:
            return None
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    if re.fullmatch(r"-?\d+", s_clean):
        return str(int(s_clean))
    return None


def _try_decimal(s: str) -> str | None:
    """Canonicalize plain decimals: "3.50" -> "3.5", "1.000" -> "1"."""
    s_clean = s.replace(" ", "")
    if not re.fullmatch(r"-?\d+\.\d+", s_clean):
        return None
    try:
        f = Fraction(s_clean)
    except (ValueError, ZeroDivisionError):
        return None
    return _canonical_rational(f)


def _canonical_rational(f: Fraction) -> str:
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def _normalize_latex_fraction(s: str) -> str:
    """Rewrite `\\frac{a}{b}` and `\\dfrac{a}{b}` to `(a)/(b)`."""
    pattern = re.compile(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    while True:
        new = pattern.sub(r"(\1)/(\2)", s)
        if new == s:
            return new
        s = new


def _normalize_latex_symbols(s: str) -> str:
    s = _normalize_latex_fraction(s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("\\pi", "pi").replace("\\sqrt", "sqrt")
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    s = re.sub(r"\\left|\\right", "", s)
    return s


def _try_sympy(s: str) -> str | None:
    """Best-effort sympy parse → canonical SymPy string.

    Returns None if parsing fails, the parsed expression contains any free symbols
    (which would only happen on a parse miss for an answer), or simplification errors.
    Uses implicit-multiplication transformations so spellings like ``2pi`` are accepted.
    """
    s_pre = _normalize_latex_symbols(s)
    try:
        expr = parse_expr(s_pre, transformations=_SYMPY_TRANSFORMATIONS, evaluate=True)
    except Exception:
        # Sympy raises a zoo of types here (SympifyError, SyntaxError, TokenError from
        # tokenize, ValueError, AttributeError, ...). The normalizer must never crash —
        # an unparseable input becomes UNNORMALIZED::... upstream, which is fine.
        return None
    # `parse_expr` can return a plain Python tuple (e.g. for input "(2,3)") instead of a
    # sympy expression, which has no `.free_symbols`. Treat that as "not a scalar answer":
    # the tuple/interval handlers earlier in the pipeline are the right path for those.
    free = getattr(expr, "free_symbols", None)
    if free is None:
        return None
    if free:
        # Answers should be concrete: any leftover free symbol is a normalization miss.
        return None
    try:
        simplified = sp.nsimplify(expr, rational=True) if expr.is_rational else sp.simplify(expr)
    except Exception:  # pragma: no cover - sympy can raise arbitrary errors
        simplified = expr
    try:
        return sp.sstr(simplified, full_prec=False)
    except Exception:  # pragma: no cover
        return None


def _try_tuple(s: str) -> str | None:
    """Canonicalize ordered tuples/sequences like "(1,2,3)" or "[1, 2, 3]"."""
    s_clean = s.strip()
    if not ((s_clean.startswith("(") and s_clean.endswith(")")) or
            (s_clean.startswith("[") and s_clean.endswith("]"))):
        return None
    inner = s_clean[1:-1]
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    normalized = []
    for p in parts:
        # Recurse on each element but bail if any element fails to normalize.
        n = normalize_answer(p)
        if n.startswith(UNNORMALIZED_PREFIX):
            return None
        normalized.append(n)
    return "(" + ",".join(normalized) + ")"


def _try_interval(s: str) -> str | None:
    """Canonicalize open/closed intervals like "(0,1]" or "[-1,1)".

    An interval requires *at least one* square bracket; otherwise the input is treated
    as an ordered tuple by :func:`_try_tuple`. This disambiguates "(a,b)" (tuple) from
    "[a,b]", "(a,b]", "[a,b)" (intervals).
    """
    s_clean = s.strip()
    if len(s_clean) < 5:
        return None
    if s_clean[0] not in "([" or s_clean[-1] not in ")]":
        return None
    if s_clean[0] != "[" and s_clean[-1] != "]":
        return None  # both ends round → defer to tuple handler
    inner = s_clean[1:-1]
    if "," not in inner:
        return None
    parts = [p.strip() for p in inner.split(",")]
    if len(parts) != 2:
        return None
    a = normalize_answer(parts[0])
    b = normalize_answer(parts[1])
    if a.startswith(UNNORMALIZED_PREFIX) or b.startswith(UNNORMALIZED_PREFIX):
        return None
    return f"INTERVAL::{s_clean[0]}{a},{b}{s_clean[-1]}"


def normalize_answer(text: str) -> str:
    """Map a raw answer string to a canonical class label.

    Never raises. Returns either a canonical label or `UNNORMALIZED::<raw>` so that
    parse failures contribute to the coverage diagnostics rather than being silently merged.
    """
    if text is None:
        return f"{UNNORMALIZED_PREFIX}<None>"

    raw = text
    boxed = extract_boxed(text)
    if boxed is not None:
        text = boxed

    text = _strip_wrappers(text)
    text = _strip_units(text)
    text = _WHITESPACE_RE.sub("", text)
    if not text:
        return f"{UNNORMALIZED_PREFIX}{raw}"

    mc = _try_multiple_choice(text)
    if mc is not None:
        return mc

    # Try cheap canonicalizations before pulling in sympy.
    rational = _canonical_fraction(text)
    if rational is not None:
        return rational

    decimal = _try_decimal(text)
    if decimal is not None:
        return decimal

    # Intervals tried before tuples because both can start with "(".
    interval = _try_interval(text)
    if interval is not None:
        return interval

    tup = _try_tuple(text)
    if tup is not None:
        return tup

    sym = _try_sympy(text)
    if sym is not None:
        return sym

    return f"{UNNORMALIZED_PREFIX}{raw.strip()}"


def is_unnormalized(label: str) -> bool:
    return label.startswith(UNNORMALIZED_PREFIX)
