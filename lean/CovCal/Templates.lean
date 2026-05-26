import Mathlib

/-!
# CovCal deterministic Lean templates

These templates correspond to the "deterministic templates" tier of the formalization route
in Sec. 4.4 of the paper. The Python side (`covcal.lean.templates`) decides which template
applies to a given (problem, candidate answer) pair and emits a `theorem` statement that the
runner attempts to close with a fixed tactic.

This file intentionally has no `theorem` declarations of its own: each runtime task is a
freshly elaborated theorem with a unique name. The templates here are documentation +
shared helper notation.

## Coverage of templates (extended in lockstep with the Python side):

* arithmetic equality between concrete rationals (`norm_num`)
* rational equality with explicit fractions (`ring_nf; norm_num`)
* integer arithmetic, divisibility, modular arithmetic (`omega`, `decide`)
* algebraic simplification of polynomial expressions (`ring_nf; ring`)
* boolean propositions and finite-search statements (`decide`)
* finite case analysis on bounded naturals (`decide` / `Finset.decideMem`)
* simple linear inequalities (`linarith`, `nlinarith`)
* multiple-choice equivalence (`decide` over a concrete enumeration)

The runner imports Mathlib, so all of the above is available.
-/

namespace CovCal.Templates

-- Reserved namespace for runtime-generated theorem names. The runner emits names of the
-- form `CovCal.Templates.Generated.<sanitized-id>` so accidental collisions across problems
-- are impossible within a single Runner invocation.
namespace Generated
end Generated

end CovCal.Templates
