---
name: correctness-suites-miss-performance-cliffs
description: "A suite that checks what a model predicts is blind to a change that destroys how fast it gets there — and for iterative solvers, tiny numeric error IS a performance cliff."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c9844d26-52eb-4368-8358-6d744bfde57b
  modified: 2026-08-15T13:50:18.465Z
---

A passing test suite does not mean a change is safe. Correctness tests check **what** a computation produces; they say nothing about **how long** it takes, and the two can diverge by orders of magnitude.

**Why:** measured 2026-08-15 on goai's `classic`. Replacing `math.Exp` with a 1.6e-7-accurate approximation in the RBF kernel left **every test passing** — the fitted models were still correct — while the SVC fit went from 5.8 ms to **9452 ms**. SMO stopped converging in 78 steps and ran to `maxIter`.

The reasoning that justified the swap was the trap: I derived the accuracy budget from the solver's *stopping tolerance* (1e-3 on kernel values in [0,1], so 1e-7 looked four orders spare). But second-order working-set selection compares **objective decreases** computed from kernel entries, and inconsistencies far below `tol` are enough to keep choosing pairs that make no progress. For an iterative solver, kernel accuracy is load-bearing — it is not a speed/precision dial.

**How to apply:**
- Before approximating anything inside an iterative solver, measure ITERATION COUNT, not just output accuracy. A converged-but-slower result and a not-converging result look identical to an accuracy assertion.
- Add an order-of-magnitude fit/step-time ceiling next to the correctness tests (goai: `TestClassicFitTimeGuard`, ~10x measured, loose enough for slow CI, tight enough to catch 100x+). It is a tripwire, not a benchmark — do not tighten it into one or it will flake.
- **Mutation-test the guard.** Inject the failure it exists for and confirm it goes red; a guard never seen failing is not a guard ([[self-policing-guard-pattern]]). Injecting `math.Round(exp(x)*1e7)/1e7` made the guard fail at 10.997s.
- Beware `timeout` on macOS — it does not exist, so the command silently runs nothing and an empty result reads as success. Use `go test -timeout`.
- Related: [[benchmark-comparisons-must-be-in-session]], [[components-not-summing-means-wrong-parameter]].
