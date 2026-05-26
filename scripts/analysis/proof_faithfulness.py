"""Classify each proved Lean artifact by faithfulness, to expose the
ℕ-truncation false-positive hazard in autoformalized answer-checking.

Categories (per proved artifact):
  genuine      arithmetic equality that is TRUE over ℚ (exact rational eval)
  spurious     arithmetic equality that is FALSE over ℚ but Lean "proved" it
               anyway -- i.e. the literals defaulted to ℕ and floor-division
               collapsed the goal to a vacuous truth (e.g. 0 = 0)
  structural   statement uses inherently-ℕ operators (gcd, divisors.card,
               %, quantifiers, MOD, ...) where ℕ typing is correct; not an
               arithmetic-truncation hazard (treated as genuine-structural)
  trivial      `X = X` self-equality (the template/LLM echoed the answer)
  unparsed     could not be parsed for an arithmetic check

The spurious set is a measured failure mode; the "faithful proved coverage"
excludes spurious + trivial artifacts.

Usage:
    python scripts/analysis/proof_faithfulness.py --run-dir runs/<dir> [--show 12]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

# Operators / tokens that make a statement inherently ℕ/structural (not an
# arithmetic-truncation hazard). gcd, divisor counts, modular arithmetic, etc.
_STRUCT_MARKERS = [
    "gcd", "lcm", "divisors", "card", "Nat.", "Int.", "%", "Finset", "∀", "∃",
    "≡", "MOD", "sqrt", "log", "floor", "ceil", "∑", "∏", "choose", "factorial",
    "∣", "!", "min", "max", "Real.", "π", "Prime", "Fin ", "Matrix", "∈", "≤",
    "≥", "<", ">", "≠",
]


def _get_prop(stmt: str) -> tuple[str | None, str]:
    """Split `theorem NAME [binders] : PROP` into (PROP, kind)."""
    m = re.match(r"\s*theorem\s+\w+\s*(.*)", stmt or "", re.DOTALL)
    if not m:
        return None, "no_theorem"
    rest = m.group(1).strip()
    if rest.startswith(":"):
        return rest[1:].strip(), "bare"
    return rest, "binders"  # explicit binders / quantifier before the colon


def _strip_ascription(s: str) -> str:
    return re.sub(r":\s*[ℚℝℤℕ]", " ", s)


def _ev(n: ast.AST) -> Fraction | None:
    """Exact rational evaluation of a parsed arithmetic expression."""
    if isinstance(n, ast.BinOp):
        l, r = _ev(n.left), _ev(n.right)
        if l is None or r is None:
            return None
        if isinstance(n.op, ast.Add):
            return l + r
        if isinstance(n.op, ast.Sub):
            return l - r
        if isinstance(n.op, ast.Mult):
            return l * r
        if isinstance(n.op, ast.Div):
            return l / r if r != 0 else None
        if isinstance(n.op, ast.Pow):
            e = int(r) if isinstance(r, Fraction) and r.denominator == 1 else r
            try:
                return l ** e
            except Exception:
                return None
        return None
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        v = _ev(n.operand)
        return None if v is None else -v
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return Fraction(n.value)
    return None


def _qeval(s: str) -> Fraction | None:
    s = _strip_ascription(s).replace("^", "**").strip()
    try:
        return _ev(ast.parse(s, mode="eval").body)
    except Exception:
        return None


def classify(stmt: str) -> str:
    prop, kind = _get_prop(stmt)
    if prop is None:
        return "unparsed"
    if kind == "binders":
        return "structural"
    if any(mk in prop for mk in _STRUCT_MARKERS):
        return "structural"
    if "=" not in prop:
        return "structural"
    parts = prop.split("=")
    if len(parts) != 2:
        return "structural"
    lhs, rhs = parts[0].strip(), parts[1].strip()
    if lhs == rhs:
        return "trivial"
    lv, rv = _qeval(lhs), _qeval(rhs)
    if lv is None or rv is None:
        return "unparsed"
    return "genuine" if lv == rv else "spurious"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--show", type=int, default=10, help="examples to print per category")
    args = ap.parse_args()

    obs_path = args.run_dir / "observations.jsonl"
    cats: Counter = Counter()
    examples: dict[str, list[str]] = {}
    n_problems = 0
    with obs_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n_problems += 1
            for _cls, arts in r["metadata"].get("artifacts_detail", {}).items():
                for a in arts:
                    if a.get("status") != "proved":
                        continue
                    stmt = a.get("statement", "")
                    cat = classify(stmt)
                    cats[cat] += 1
                    examples.setdefault(cat, [])
                    if len(examples[cat]) < args.show:
                        body = stmt.split(" : ", 1)[-1].replace("\n", " ")[:90]
                        examples[cat].append(f"[{a.get('tactic_used')}] {body}")

    total = sum(cats.values())
    print(f"=== proof faithfulness: {args.run_dir.name} ({n_problems} problems) ===")
    print(f"proved artifacts: {total}")
    for cat in ["genuine", "structural", "spurious", "trivial", "unparsed"]:
        n = cats.get(cat, 0)
        pct = 100 * n / total if total else 0
        print(f"  {cat:11s} {n:4d}  ({pct:4.1f}%)")
    faithful = cats.get("genuine", 0) + cats.get("structural", 0)
    print(f"\nfaithful (genuine+structural): {faithful}/{total} "
          f"({100*faithful/total:.1f}%)" if total else "no proved artifacts")
    print(f"hazardous (spurious):          {cats.get('spurious',0)} "
          f"({100*cats.get('spurious',0)/total:.1f}%)" if total else "")
    for cat in ["genuine", "structural", "spurious", "trivial", "unparsed"]:
        if examples.get(cat):
            print(f"\n-- {cat} --")
            for e in examples[cat]:
                print(f"  {e}")


if __name__ == "__main__":
    main()
