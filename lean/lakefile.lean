import Lake
open Lake DSL

package CovCal where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩,
    ⟨`linter.unusedVariables, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.21.0"

require batteries from git
  "https://github.com/leanprover-community/batteries" @ "v4.21.0"

require aesop from git
  "https://github.com/leanprover-community/aesop" @ "v4.21.0"

require Qq from git
  "https://github.com/leanprover-community/quote4" @ "v4.21.0"

@[default_target]
lean_lib CovCal where
  globs := #[.submodules `CovCal]

lean_exe CovCalRunner where
  root := `CovCal.Runner
  supportInterpreter := true
