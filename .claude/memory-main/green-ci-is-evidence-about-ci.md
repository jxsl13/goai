---
name: green-ci-is-evidence-about-ci
description: "goai's selective CI runner silently ran almost no tests; treat green CI as unverified until the runner is observed failing on known-bad input."
metadata: 
  node_type: memory
  type: project
  originSessionId: c9844d26-52eb-4368-8358-6d744bfde57b
  modified: 2026-08-16T14:19:43.910Z
---

On 2026-08-15 I found that goai's selective test runner had been reporting success **without running the tests it selected**. Fixed in PR #1072; the three `nn` failures it uncovered are PR #1071.

**RESOLVED 2026-08-16: `main` is green on every lane, verified with CI actually running tests.** Merging #1072 uncovered a large backlog, fixed across #1074 (a second selector defect — a package whose files are all behind build tags the lane does not set is a `[setup failed]`, not a no-op, and it failed the whole invocation), #1075 (33 `*BitIdentical` tests red on amd64) and #1076 (spec rules). Green CI on this repo now means something; treat the pre-#1072 history as unverified.

**The backlog was mostly guards that had stopped measuring, not new breakage** — five distinct mechanisms, catalogued in [[self-policing-guard-pattern]]. Expect this whenever a silent gate is repaired: the bugs are not in what changed, they are in what was never checked.

**Cause:** CI runs `cichange -run $BASE HEAD -- -short …`. Go's `flag.Parse` stops at the first *positional* (`base`), so the `--` is never consumed as a terminator — it was forwarded into `go test -- -short <pkgs>`. Everything after `--` goes to the test binary, **the package list is silently discarded**, go test falls back to the working-directory package, and **exits 0**. `-impact` correctly listed 20 packages; `-run` tested one. The selector — the sophisticated part that gets attention — was never the problem.

**Why it matters:** commit `cd41bf8d` (PR #897) merged 15/15 green while leaving `./nn/` panicking on every run. Three failures accumulated on main behind green checks.

**A second defect of the same shape, 2026-08-15.** Neither `make preflight` nor `make preflight-full` could see the cgo+metal lane: both run `CGO_ENABLED=0`, so they never even COMPILE tests under `//go:build darwin && cgo`. Three broken Metal tests reached CI while every local gate reported green — and one was a *reference arm silently measuring the wrong implementation* (a test toggled `SetQ4KDequantGemm(false)` to select the "cooperative" arm, but the newer f16 path is gated independently, so the arm measured cached-f16 against itself and read 6x too fast). A wrong NUMBER, not a build error, which no pure-Go checking can surface. Fixed by adding `make preflight-metal` (runs `-tags metal -short` over backend/metal + llamagpu, no-op off darwin) and wiring it into preflight-full.

**Rules that follow:**
- A local gate mirroring CI must mirror its BUILD CONFIGURATION, not just its test list. `CGO_ENABLED=0` silently drops whole files, and dropped files cannot fail.
- Any test that selects between implementations by toggling a flag needs revisiting when a NEW implementation lands — the flag stops meaning what the test assumes.
- A flaky diagnostic must not fail a gate. `TestMeasurementNoiseFloor` measures the HOST (is this machine quiet enough to A/B?) and failed ~1 run in 3 under thermal load; a gate that cries wolf that often gets bypassed, which is how the lane went unwatched. It now logs under `-short` and still fails a full run.

**How to apply:**
- Don't infer "the code is fine" from green checks. Confirm the runner *ran* — check that the job output names the packages.
- A test harness needs a test that exercises it **as actually invoked**. `TestRunPropagatesFailure` existed and was thorough, but passed args bare, never CI's real argument vector — that gap is where this lived.
- Prefer failure modes that error over ones that no-op. A bogus package list would have errored loudly; `go test --` succeeds while doing nothing.
- A panic aborts the package test binary and hides every later failure in it — after fixing a panicking test, re-run the whole package before believing it's green (fixing the Muon panic exposed two more).
- Ties to [[pre-push-gofmt-gate]] (local gates catch what CI misses) and [[self-policing-guard-pattern]] (a guard that never fails on bad input isn't a guard).
