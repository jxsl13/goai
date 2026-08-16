---
name: verify-amd64-on-apple-silicon
description: "On this M2, GOARCH=amd64 go test RUNS under Rosetta and reproduces CI's amd64 digests EXACTLY — and the goai bit-identity family fails there for two proven causes, not one."
metadata:
  node_type: memory
  type: reference
  originSessionId: c9844d26-52eb-4368-8358-6d744bfde57b
  modified: 2026-08-16T12:41:03.305Z
---

`GOARCH=amd64 CGO_ENABLED=0 go test ./pkg/` **executes** on this machine — Rosetta 2 runs the cross-built test binary. So "I can't verify a linux/amd64 CI failure from here" is usually wrong for pure-Go packages; only cgo lanes genuinely need another host.

**Rosetta REPRODUCES CI's amd64 FAILURES, but NOT its FP RESULTS. Do not generate float goldens from it.** 2026-08-16: `TestQRVJPIsBitIdentical` reports the same digests under Rosetta as on ubuntu-latest and windows-latest — so I generalized "Rosetta reproduces CI bit-for-bit" from one test and was WRONG. `TestMLAVJPIsBitIdentical` gives `10503053519604685430` under Rosetta (at BOTH `GOAMD64=v1` and `v2`) and `2081554234887433254` on real x86; ubuntu and windows agree with each other, so the odd one out is Rosetta. The likely mechanism is that Rosetta translates SSE multiply-add onto ARM FMA, fusing where real amd64 does not — the same class of difference the goldens exist to detect.

**Consequence:** real amd64 float goldens must be harvested from CI logs, not from a local Rosetta run. Rosetta is still the right tool for *reproducing* an amd64 failure and for non-FP logic; it is not an oracle for FP bit patterns. Windows and Linux amd64 DO agree with each other, so one golden per arch class still holds — the arch class just cannot be sampled locally.

`GOAMD64=v1|v2` isolates instruction-set effects. **v3 does NOT work under Rosetta** — no AVX2, so a v3 binary dies before running and produces NO output. Reading that silence as "the test passed" is exactly the mistake to avoid.

**The `*BitIdentical` family: cause now PROVEN, and it is TWO causes.** (Previously recorded here as "FMA, plausible and unproven" — that was half right and I had not measured it.) 33 tests across 8 packages fail on amd64 and pass on arm64:

1. **Non-portable fixtures.** The tests build inputs from `math.Sin`/`math.Cos`. Go's implementations are not bit-identical across GOARCH — 41 of 2048 sampled values differ by 1 ULP (`math.Cos(84)` = `...e523` on arm64, `...e522` on amd64). The *inputs* differ, so every downstream digest differs. Individually spot-checked values agreeing (`Sin(129)`, `Cos(451)` matched) is NOT evidence the sweep agrees — dump all values and diff.
2. **FP contraction in the kernel.** arm64 fuses `v -= a*qd[j]` to FNMSUB, amd64 v1 does not. The clincher: with exact dyadic fixtures, `m=3,n=3` — the ONE shape where the jammed loop never runs — is the ONE shape whose digest matches across arches; every larger shape still differs. Forcing rounding via `float64(a*b)` on arm64 changes the digest but does NOT reproduce amd64's, which is what proves cause 1 is also present.

**How to apply:**
- Before saying an amd64/linux failure can't be reproduced locally, try `GOARCH=amd64 go test`. Add `GOOS=linux` only if the failure is OS-specific — that combination does *not* run (exec format error).
- **A frozen golden digest cannot be portable while the kernel fuses.** Either key the golden on `runtime.GOARCH` (cheap, keeps full bit-exactness per arch, needs a skip for unknown arches), or assert the real invariant in-binary — compare the optimized path against the naive one via a test hook — which is portable by construction and what these tests actually claim.
- Never build a bit-exactness fixture from transcendental functions. Dyadic rationals (`float64(k)/8`) are exact and portable.
- Related: [[green-ci-is-evidence-about-ci]] (this was invisible for a long time because CI tested nothing), [[benchmark-comparisons-must-be-in-session]] (verify the mechanism, don't infer it).
