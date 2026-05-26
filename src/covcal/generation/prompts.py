"""Prompt templates for candidate generation, Lean autoformalization, and repair.
"""

from __future__ import annotations

CANDIDATE_GENERATION_TEMPLATE = (
    "Solve the following math problem. Give a concise derivation and put the final answer "
    "in \\boxed{{...}}.\n\nProblem: {problem}"
)

FORMALIZATION_TEMPLATE = (
    "You are formalizing a math contest answer in Lean 4 with Mathlib. Given the problem "
    "and proposed final answer, write a Lean theorem whose proof would certify that the "
    "proposed answer is correct for the problem.\n"
    "Requirements:\n"
    "- State the problem's actual mathematical claim with the proposed answer substituted in. "
    "Do NOT write a trivial identity such as `answer = answer`; the theorem must be false if "
    "the answer is wrong.\n"
    "- Use `ℚ` or `ℝ` (never `ℕ`) for any division, fraction, or non-integer "
    "arithmetic so that division is exact rather than floor division; annotate numeric "
    "literals with their type when needed, e.g. `(125 : ℚ) / 9`.\n"
    "- Prefer simple statements and standard Mathlib notation. Return only Lean code.\n\n"
    "Problem: {problem}\n"
    "Proposed answer: {answer}"
)

REPAIR_TEMPLATE = (
    "The Lean code below failed with the following error. Repair the theorem statement or "
    "proof while preserving the intended meaning of the problem and answer. Return only Lean "
    "code.\n\n"
    "Code:\n```lean\n{code}\n```\n\n"
    "Error:\n```\n{error}\n```"
)


def candidate_generation_prompt(problem: str) -> str:
    return CANDIDATE_GENERATION_TEMPLATE.format(problem=problem)


def formalization_prompt(problem: str, answer: str) -> str:
    return FORMALIZATION_TEMPLATE.format(problem=problem, answer=answer)


def repair_prompt(code: str, error: str) -> str:
    return REPAIR_TEMPLATE.format(code=code, error=error)
