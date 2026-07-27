---
schema: v1
---

## T-01KYJQ3PGEEFSAKZXXRNP42W5Q Broaden PS1001 — it misses per-element AtF64/SetF64 outside Numel/Unravel loops
kind: task
state: draft
created: 2026-07-27

DETECTOR BUG, verified by running the tool, with the root cause now isolated precisely.

ROOT CAUSE: PS1001 keys on Numel/Unravel-driven loops. It therefore misses per-element AtF64/SetF64 nests whose bounds are SHAPE-DERIVED SCALARS — for j := range d, for i := range sinks — which is the far more common shape in this codebase.

FIVE CONFIRMED FALSE NEGATIVES (go run ./internal/perfscan -checks PS1001 reports "no candidate anti-patterns found" on all of them):
- nlp/streaming.go:42-51 keepSinkRecent — nested AtF64/SetF64 over (sinks+window) x d, run twice per layer per token on the StreamingLLM path. At sinks=4 / window=1020 / kvWidth=1024 / 32 layers this is roughly 67 MILLION per-element dispatches per token. The single highest-impact miss.
- nlp/quant_llama_decode.go:25-27 QuantLlama.embedOne — the sole surviving per-element embedding lookup; 28 of the package's 29 embedOne implementations already call the bulk embedRow.
- nlp/kvevict.go:118-122 GatherRows — row gather where a per-row copy is available, reached from EvictStreaming.
- llamagpu/t5_decoder.go:226 — emb[j] = float32(d.shared.AtF64(token, j)) on the per-token decode path, in a package whose decoder.go:3224 already has the bulk embedRow helper.
- rl/continuous.go:245-254 contMat — t.SetF64(v, i, j) nested, 3 calls x BatchSize x actDim per SAC.learn, which runs every env step. Its sibling rl/rl.go:143-149 already received exactly this fix.

FIX: widen the loop predicate from Numel/Unravel-driven to any ForStmt/RangeStmt nest whose innermost body is an ExprStmt of the form X.SetF64(Y.AtF64(i...), j...) where the index expressions are affine in the loop variables AND the innermost loop bound is a shape-derived expression (Shape()[k], a trailing-dim variable, or a struct field holding one). Keep the existing typed-fast-path suppression. The Numel/Unravel case remains a strictly narrower sub-case, so every existing positive must keep firing.

WORTH ADDING AT THE SAME TIME, a majority-pattern-deviation signal: the embedOne case is detectable more sharply as "a bulk sibling with this exact signature exists in-package and 28 other call sites use it, this one does not". That heuristic would have caught the holdout mechanically rather than by reading.

VALIDATION GATE: this is a detector-correctness task, not a speed task, so the gate is fixtures plus a repo sweep, NOT a benchmark. (1) Add all five confirmed sites to internal/perfscan/perfscan_test.go as POSITIVE fixtures, plus a negative fixture where the accessor is hoisted or the receiver varies per iteration, so the rule is proven non-vacuous in both directions. (2) After widening, run go run ./internal/perfscan -checks PS1001 ./... and record the FULL new finding set in the closing note — each newly flagged site is either a real straggler deserving its own task or a false positive that must be suppressed by construction, and which one it is must be stated per site rather than assumed. (3) Confirm the pre-existing positives still fire.

WHY THIS RANKS HIGH DESPITE SHIPPING NO SPEED: the standing requirement is that every generalizable optimization becomes a perfscan rule so instances are found mechanically across the tree. A rule with silent false negatives is worse than no rule, because a clean scan is read as "no instances" — the same false-assurance failure FMT-004 already records for a fuzz target that never reached its parser. Five misses in two sweeps, all of the same shape, means the rule has been giving false assurance since it was written. Fixing the detector multiplies across every future sweep.

SCOPE NOTE: do NOT fix the five sites here. Widening the detector and fixing what it finds are separate changes with separate risks; the sites have their own tasks.
