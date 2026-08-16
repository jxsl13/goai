---
name: self-policing-guard-pattern
description: "A guard that reads the table it checks only proves the table equals itself — derive ground truth independently, and prove the guard can fail before shipping it."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 89975edc-f6ec-4912-922c-d5efd862e3d7
  modified: 2026-08-16T14:19:29.088Z
---

When a mapping/registry must stay in sync with code (op→Attrs type, optimizer→LR reachability, arch→convention), the test that protects it MUST derive ground truth from a source **independent of the thing it validates**, and MUST be proven to fail before shipping.

**Why:** a guard that reads the table it checks only proves the table equals itself. This is the same failure that made §B67 survive for so long (a symmetric round-trip proves nothing about a foreign convention) and that §T842 hit as the "vacuous probe" (a constant-gradient resume test passes even un-checkpointed). A green guard is worthless evidence unless you have seen it go red.

**How to apply:**
- Derive the expectation from the *implementation*, not the declaration. T849's best example: the op→Attrs guard recovers each reference kernel, resolves the function via its program counter, walks the ref-package AST for the assertions the kernel *actually performs* — following the attrs parameter into package-level helpers, which was required since pooling asserts `PoolAttrs` inside `poolDims` — then cross-examines `Execute`'s public behaviour. It never reads `opAttrsSpec`.
- **Always break it on purpose and confirm red, then restore.** Do this for the *hardest* case, not the easiest: T849 was re-proven by removing `OpMaxPool2D` specifically because its assertion lives in a helper, so it proves helper-following works rather than just body-matching.
- Add a floor assertion (`if checked < N { t.Fatal }`) so the sweep itself cannot silently go vacuous — an AST walk that matches nothing passes cheerfully.
- Watch for reflection-based reach: it is NOT compiler-checked, so a renamed/removed field does not break the build. Make the binding call **error rather than no-op** (T848: a scheduler that silently never moves the LR is indistinguishable from a working one in a green suite), and add an accounting guard for newly added types.
- Check test NAMES against actual coverage. T848 shipped `TestBindLRReachesEveryOptimizerFamily` covering 4 of 25 — the name asserted a guarantee the body did not provide.

**A guard written as a CONDITION is not a guard (§B77).** `if t, ok := m[k]; ok && t.Ndim() == 2 { validate(t) }` reads as defensive and behaves as its opposite: a rank-1 tensor makes the condition false, *skips the validation meant to reject it*, and falls through to the unguarded code — a real panic in `nlp/llama_gguf.go`. A check placed in the presence condition only runs on inputs that already pass it. Put the test **inside the body** and return an error. When auditing, grep for `ok && <predicate>` and ask what happens when the predicate is false.

**§B77 also has an OMISSION variant: a guard set that leaves out an input it later dereferences.** `KimiDeltaAttention` (B110, 2026-07-20) looped `for _, t := range []{q,k,v,a}` checking `Ndim()==2` but LEFT OUT beta, then read `beta.AtF64(t,0)` (a rank-2 access) — so a 1-D beta [seq] (which the doc itself wrongly prescribed) sailed past every guard and panicked at that access, and a [seq,W>1] beta silently used only column 0. When a shape/type guard enumerates its inputs, cross-check that set against every tensor the body actually indexes: an input dereferenced but not guarded is the bug. Fix = add it to the guard set + a width check, and fix the doc.

**`backend.*Attrs.WithDefaults()` rewrites zero-valued fields — footgun XOR correct, and telling them apart is where the false positive lives.** AXPYAttrs.WithDefaults() rewrites Alpha 0→1: a REAL footgun (B106) when a caller feeds a `0=off` hyperparameter straight into Alpha — EWCPenalty(λ=0)/FocalLoss(α=0)/RDropLoss(α=0) all silently computed at full scale; guard the off case before the op. But GSPOAttrs.WithDefaults() defaults Epsilon 0→3e-4, which is exactly what GSPO wants, so `gspoConfig{}` (ε=0) is CORRECT — an audit flagged it as a bug and I nearly "fixed" it, a pure false positive. LESSON: a "zero-value config is wrong" finding is only real if you READ the specific WithDefaults and confirm the rewrite is semantically wrong for THIS op. Verify the KERNEL default, not the caller-side struct. Ties to [[subagent-output-is-untrusted]] — this was a subagent finding that reproduced-as-stated but was still wrong.

**Numerical guards degenerate at boundary parameters too.** FSQ (B109) centered even-L channels with `shift = atanh(offset/half)`; at L=2, `offset==half` so `atanh(1)=+Inf` saturated tanh → a dead channel (one constant level, zero gradient) that passed the `l≥2` guard silently. `atanh(x)≈tan(x)` to 3rd order so L≥4 was fine — only the `offset==half` boundary broke. When a formula divides/inverts a ratio of params, check the endpoints where the ratio hits 1 or 0. Similarly WSD (B112): `0.5^((step−decayStart)/halfLife)` → 0/0=NaN at halfLife=0, step==decayStart, poisoning training; guard the degenerate param.

**Two guards on the same data must share one predicate.** Where a quantized loader validated geometry its float twin did not (§B76/§B77, 8+ pairs), the fix was to CALL the twin's predicate, never to restate it — a second independently-worded check drifts. Corollary that is easy to miss: do not make one side *stricter* either. An agent correctly **removed** a rank check it had added because the quant twin lacked one, which would have reopened the asymmetry from the other direction.

**A guard can stop measuring LONG AFTER it was written, without anyone touching it.** 2026-08-16, five found in one session once CI actually ran tests ([[green-ci-is-evidence-about-ci]]):
- **Intercepted upstream.** `TestQ4KMatrixUnitHasNoCrossover` toggled two paths and compared them — but a THIRD path (f16 short-prompt) is checked first in the dispatch, and when the weight cache went on by default its M cap became `1<<20`, so it served both arms. The tell was in the output for months: `1.00x, 1.01x, 0.99x` where the recorded table is `0.36x-0.77x` apart. It only ever failed on thermal noise, and then "concluded" a crossover about a comparison that never ran. **When a guard toggles between paths, assert the arms actually DIFFER** — and anchor that check to the shape where the true gap is widest: a majority-of-shapes rule did NOT catch the real case, because one shape cleared 2% on noise alone.
- **Mislabelled arm.** Same test at M=16: `q4k_dq_gemm_eligible` gates on `M >= 24`, so the arm called "dqgemm" was the fallback kernel. Invisible while the fallback happened to cost about the same; it jumped out only when the fallback changed.
- **Printed, not asserted.** `TestBPEThroughputGuard` printed the token count (the one correctness signal that belongs in CI) and asserted only throughput. Grep your own guards for `fmt.Printf` of a value that should be a comparison.
- **Measuring the instrumentation.** An allocation-budget guard counted the race detector's shadow memory (16 MiB against a 12 MiB budget). **Measuring the toolchain, not the program.**
- **Comparing bytes git rewrote.** A golden byte comparison failed only on Windows because `text: unspecified` let git convert LF→CRLF on checkout. Mark byte-compared goldens `-text` in `.gitattributes`; the failure is invisible on Linux and macOS.

**Corollary — thresholds calibrated on your machine INVERT on shared CI, they do not merely drift.** The Q4_K crossover reports the opposite ordering on GitHub's macOS runners; BPE floors set at 1/3 of an M2 Pro measured 1.3 MB/s there, 36× under. Put wall-clock/throughput/allocation assertions behind `testing.Short()` and keep correctness assertions in the same test unconditional.

Related: [[integration-audit-method]], [[t650-topic-discovery-round]], [[tokenizer-trust-boundary]], [[goai-autonomous-loop]], [[green-ci-is-evidence-about-ci]], [[verify-amd64-on-apple-silicon]].
