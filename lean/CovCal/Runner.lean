-- CovCalRunner: persistent Lean worker that classifies formal attempts into the
-- 5-status taxonomy. See README + Python wrapper for the full protocol.

import Lean
import Mathlib

/-!
# CovCalRunner

A persistent Lean process that classifies formal attempts into the 5-status taxonomy.

* loads `Mathlib` once at startup (via a bootstrap file passed on argv);
* prints a single `{"ready": true}` line and then enters a request/response loop:
  reads one JSON task per line from stdin, emits one JSON outcome per line to stdout;
* per-attempt budgets are enforced via Lean's `maxHeartbeats` (deterministic);
* wall-clock backstop is the Python wrapper's subprocess timeout.

## Task schema (stdin, one JSON object per line)

  {
    "name": "task_unique_id",
    "statement": "theorem t : 1 + 1 = 2",
    "tactics": ["norm_num", "decide"],
    "max_heartbeats_per_tactic": 2000000
  }

* `statement` is a Lean theorem signature without `:= ...`. The runner appends `:= by <tac>`.
* `name` must be a valid Lean identifier suffix; the runner salts the theorem with the
  attempt index so multiple tactics for the same task do not collide in the environment.

## Outcome schema (stdout, one JSON object per line)

  {
    "name": "task_unique_id",
    "status": "proved" | "typechecked" | "timeout" | "illtyped",
    "tactic_used": "norm_num" | null,
    "elapsed_seconds": 0.143,
    "log": "..."
  }
-/

open Lean Elab Command Parser System IO

namespace CovCal.Runner

structure Task where
  name : String
  statement : String
  tactics : Array String
  maxHeartbeatsPerTactic : Option Nat := none
  deriving Inhabited, FromJson, ToJson

structure Outcome where
  name : String
  status : String
  tacticUsed : Option String
  elapsedSeconds : Float
  log : String
  deriving Inhabited, ToJson

/-- Heuristic: does any error message indicate a deterministic timeout from `maxHeartbeats`? -/
private def looksLikeTimeout (s : String) : Bool :=
  s.containsSubstr "deterministic timeout" ||
  s.containsSubstr "maximum number of heartbeats" ||
  s.containsSubstr "(deterministic) timeout"

/-- Elaborate every command in `src` against `importEnv` and return the resulting messages.
The environment changes are dropped: each call is independent of every other call. -/
private def elabSnippet (importEnv : Environment) (opts : Options) (src : String) :
    IO MessageLog := do
  let inputCtx := Parser.mkInputContext src "<covcal-attempt>"
  let mut cmdState : Command.State := Command.mkState importEnv {} opts
  let mut ps : ModuleParserState := {}
  let mut hitCrash := false
  while !hitCrash do
    let pmctx : ParserModuleContext := {
      env := cmdState.env, options := cmdState.scopes.head!.opts,
      currNamespace := cmdState.scopes.head!.currNamespace,
      openDecls := cmdState.scopes.head!.openDecls
    }
    let (cmd, ps', msgs) := Parser.parseCommand inputCtx pmctx ps cmdState.messages
    ps := ps'
    cmdState := { cmdState with messages := msgs }
    if cmd.isOfKind ``Parser.Command.eoi then break
    let cmdCtx : Command.Context := {
      fileName := "<covcal-attempt>", fileMap := inputCtx.fileMap, currRecDepth := 0,
      cmdPos := ps.pos, macroStack := [], currMacroScope := firstFrontendMacroScope,
      ref := cmd, snap? := none, cancelTk? := none, suppressElabErrors := false
    }
    let eio := Command.elabCommand cmd |>.run cmdCtx |>.run cmdState
    match ← eio.toBaseIO with
    | .ok ((), s') => cmdState := s'
    | .error _ => hitCrash := true
  pure cmdState.messages

/-- True iff `log` contains any error-severity message. -/
private def hasError (log : MessageLog) : Bool :=
  log.toList.any (·.severity == .error)

/-- Convert one severity to a printable string. -/
private def sevString : MessageSeverity → String
  | MessageSeverity.error => "error"
  | MessageSeverity.warning => "warning"
  | MessageSeverity.information => "info"

/-- Convert all error/warning messages to a concatenated string. -/
private def renderLog (log : MessageLog) : IO String := do
  let mut out := ""
  for msg in log.toList do
    let body ← msg.data.toString
    let sev := sevString msg.severity
    out := out ++ s!"[{sev}] {msg.pos.line}:{msg.pos.column}: {body}\n"
  pure out

/-- Wrap an attempt in a unique namespace so the theorem name in `statement` can repeat
across tasks without colliding in the shared environment. -/
private def wrapInNamespace (suffix : String) (heartbeats : Nat) (body : String) : String :=
  let ns := "CovCal_" ++ suffix
  s!"namespace {ns}\nset_option maxHeartbeats {heartbeats} in\n{body}\nend {ns}\n"

/-- Probe whether the statement *type* elaborates by attempting `<statement> := by sorry`. -/
private def probeElab (importEnv : Environment) (statement suffix : String) :
    IO (Bool × String) := do
  let src := wrapInNamespace (suffix ++ "_elab") 200000 (statement ++ " := by sorry")
  let log ← elabSnippet importEnv {} src
  let err := hasError log
  let txt ← renderLog log
  pure (!err, txt)

/-- Attempt to close the statement with a single tactic. -/
private def attemptTactic (importEnv : Environment) (statement tactic suffix : String)
    (heartbeats : Nat) : IO (Bool × Bool × String) := do
  -- (proved, sawTimeout, log)
  let src := wrapInNamespace suffix heartbeats (statement ++ " := by\n  " ++ tactic)
  let log ← elabSnippet importEnv {} src
  let txt ← renderLog log
  if !hasError log then
    pure (true, false, txt)
  else
    pure (false, looksLikeTimeout txt, txt)

/-- Sanitize a task name into a Lean-identifier-safe suffix. -/
private def sanitize (s : String) : String :=
  s.map (fun c => if c.isAlphanum then c else '_')

/-- Run a single task end-to-end. -/
def runTask (importEnv : Environment) (task : Task) : IO Outcome := do
  let t0 ← IO.monoNanosNow
  let hb := task.maxHeartbeatsPerTactic.getD 2_000_000
  let suffix := sanitize task.name
  let (elabOk, elabLog) ← probeElab importEnv task.statement suffix
  if !elabOk then
    let t1 ← IO.monoNanosNow
    return {
      name := task.name,
      status := "illtyped",
      tacticUsed := none,
      elapsedSeconds := Float.ofNat (t1 - t0) / 1e9,
      log := elabLog
    }
  let mut sawTimeout := false
  let mut combined := ""
  for h : i in [0 : task.tactics.size] do
    let tac := task.tactics[i]
    let (proved, timedOut, tacLog) ← attemptTactic importEnv task.statement tac
      s!"{suffix}_t{i}" hb
    combined := combined ++ s!"--- tactic {i}: {tac} ---\n" ++ tacLog
    if proved then
      let t1 ← IO.monoNanosNow
      return {
        name := task.name,
        status := "proved",
        tacticUsed := some tac,
        elapsedSeconds := Float.ofNat (t1 - t0) / 1e9,
        log := combined
      }
    if timedOut then
      sawTimeout := true
  let t1 ← IO.monoNanosNow
  return {
    name := task.name,
    status := if sawTimeout then "timeout" else "typechecked",
    tacticUsed := none,
    elapsedSeconds := Float.ofNat (t1 - t0) / 1e9,
    log := combined
  }

/-- Read the bootstrap file's header to get a Mathlib-loaded `Environment`. -/
private def loadBootstrap (bootstrapPath : String) : IO Environment := do
  Lean.initSearchPath (← Lean.findSysroot)
  let src ← IO.FS.readFile bootstrapPath
  let inputCtx := Parser.mkInputContext src bootstrapPath
  let (header, _, msgs) ← Parser.parseHeader inputCtx
  let (env, importMsgs) ← Lean.Elab.processHeader header Options.empty msgs inputCtx
  if importMsgs.hasErrors then
    throw <| IO.userError "[CovCalRunner] failed to load bootstrap Mathlib environment"
  pure env

end CovCal.Runner

open CovCal.Runner

/-- Build a synthetic error outcome with a single tag and message. -/
private def errorOutcome (name kind msg : String) : Outcome :=
  Outcome.mk name "illtyped" none 0.0 (kind ++ ": " ++ msg)

/-- Persistent worker. Usage: `CovCalRunner <bootstrap.lean>` then stdin-driven JSONL loop. -/
def main (args : List String) : IO UInt32 := do
  match args with
  | [bootstrapPath] =>
    let t0 ← IO.monoMsNow
    let importEnv ← loadBootstrap bootstrapPath
    let t1 ← IO.monoMsNow
    IO.eprintln s!"[CovCalRunner] Mathlib loaded in {t1 - t0}ms from {bootstrapPath}"
    IO.println "{\"ready\":true}"
    (← IO.getStdout).flush
    let stdin ← IO.getStdin
    let stdout ← IO.getStdout
    let mut processed : Nat := 0
    while true do
      let line ← stdin.getLine
      if line.isEmpty then break
      let trimmed := line.trim
      if trimmed.isEmpty then continue
      try
        match Json.parse trimmed with
        | .error e =>
          stdout.putStrLn (toJson (errorOutcome "<parse-error>" "json parse" e)).compress
        | .ok j =>
          match (fromJson? j : Except String Task) with
          | .error e =>
            stdout.putStrLn (toJson (errorOutcome "<schema-error>" "task schema" e)).compress
          | .ok task =>
            let o ← runTask importEnv task
            stdout.putStrLn (toJson o).compress
            processed := processed + 1
      catch e =>
        stdout.putStrLn (toJson (errorOutcome "<runtime-error>" "runtime" (toString e))).compress
      stdout.flush
    IO.eprintln s!"[CovCalRunner] processed {processed} tasks; exiting."
    (← IO.getStdout).flush
    (← IO.getStderr).flush
    IO.Process.exit 0
  | _ =>
    IO.eprintln "Usage: CovCalRunner <path/to/_mathlib_bootstrap.lean>"
    return 1
