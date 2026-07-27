---
schema: v1
---

## T-01KYJQ3PGEEFSAKZXXRNP42W5Q Broaden PS1001 — it misses per-element AtF64/SetF64 outside Numel/Unravel loops
kind: task
state: draft
created: 2026-07-27

DETECTOR BUG, verified by running the tool. PS1001 is titled "per-element AtF64/SetF64 dispatch in a Numel/Unravel loop with no typed fast path", and that Numel/Unravel qualifier is doing real damage: the anti-pattern is per-element accessor dispatch in ANY hot loop, but the detector only matches the Numel/Unravel shape.

CONFIRMED FALSE NEGATIVES (go run ./internal/perfscan -checks PS1001 ./llamagpu/... ./rl/... reports "no candidate anti-patterns found"):
- llamagpu/t5_decoder.go:226 — emb[j] = float32(d.shared.AtF64(token, j)) inside a plain indexed loop, on the PER-TOKEN decode path, in a file whose sibling three files over (llamagpu/decoder.go:3224) already has the bulk-slice embedRow helper that is the fix.
- rl/continuous.go:245-254 contMat — t.SetF64(v, i, j) in a nested loop, 3 calls x BatchSize x actDim dispatches per SAC.learn, which runs every env step. Its sibling forward (rl/rl.go:143-149) already received exactly this fix.
Both are stragglers of a pattern the rule exists to catch, and both were found by human reading rather than by the tool — which is the failure mode the perfscan currency contract is meant to prevent.

FIX: widen the loop predicate from "Numel/Unravel-driven" to any ForStmt/RangeStmt whose body contains a call to a per-element accessor (AtF64/SetF64/AtF32/SetF32 or the configured accessor set) on a receiver that is loop-invariant, where the index arguments derive from the loop induction variables. Keep the existing typed-fast-path suppression. The Numel/Unravel case stays a strictly narrower sub-case, so existing positives must keep firing.

VALIDATION GATE: this one is not a benchmark task — it is a detector-correctness task, so the gate is fixtures plus a repo sweep. (1) Add both confirmed sites to internal/perfscan/perfscan_test.go as POSITIVE fixtures and keep a negative fixture where the accessor is hoisted or the receiver varies per iteration, so the rule is proven non-vacuous in both directions. (2) After widening, run go run ./internal/perfscan -checks PS1001 ./... and record the full new finding set in the task's closing note — every newly flagged site is either a real straggler worth its own task or a false positive that must be suppressed by construction, and which one it is has to be stated per site rather than assumed. (3) Confirm the pre-existing positives still fire (no regression in detection).

WHY THIS RANKS HIGH DESPITE SHIPPING NO SPEED: the standing requirement is that every generalizable optimization becomes a perfscan rule so instances are found mechanically across the tree. A rule with silent false negatives is worse than no rule, because it produces a clean scan that is read as "no instances" — exactly the false assurance the fuzz-target lesson in FMT-004 already records for a different tool. Fixing the detector multiplies across every future sweep.

SCOPE NOTE: do NOT fix the two sites as part of this task. Widening the detector and fixing what it finds are separate changes with separate risks; the sites get their own tasks once the sweep enumerates them.
