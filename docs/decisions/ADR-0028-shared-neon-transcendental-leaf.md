# ADR-0028 — one shared NEON transcendental kernel (the "vexp leaf") for every f32 activation

- Status: Accepted (implemented T660-T667 on M2 Pro; every f32 transcendental op routes through it)
- Date: 2026-07-15
- Task: §T660-T667 (follows ADR-0026 NEON GEMM, ADR-0027 Apple AMX) — close the CPU f32 gap to
  torch on the elementwise transcendental ops after the GEMM was matched/beaten
- Related: ADR-0021 (f32-native tolerant policy — the numerics contract this rides), ADR-0026
  (NEON GEMM + the Plan9 WORD-encoding methodology), ADR-0027 (AMX — the GEMM ceiling this sits
  next to), §C26 (recursive-decompose → bottom-up-verified assembly — the design principle),
  §C25 (global-scale last-percent perf), §V22 (A/B discipline)

## Context

After the F32 GEMM reached the Apple AMX ceiling (ADR-0026/0027), a `GOAI_TIME_OPS` op-profile of
the real f32 CPU GPT forward showed the next costs were the **elementwise transcendentals**: the
softmax `exp`, GELU's `erf`, and — on the training step — GELU-backward, cross-entropy, and (for
the Llama/SwiGLU path) SiLU. Every one of these evaluated a transcendental with a scalar `math.Exp`
/`math.Erf`/`math.Log` per element, in f64, and Go 1.26 arm64 does **not** autovectorize f32 (the
ADR-0026 objdump finding, reconfirmed three more times). Several — GELU-backward, cross-entropy
forward+backward, SiLU-backward, add-bias-backward — did not even have a CPU kernel: they fell back
to the serial scalar reference backend (a silent-ref-fallback class caught with `GOAI_LOG_FALLBACK`).

The naive path would be one hand-tuned NEON kernel per op (a softmax kernel, a GELU kernel, a SiLU
kernel, ...). That is a lot of near-duplicate range-reduction asm, each independently error-prone.

## Decision

Build **one** verified NEON transcendental primitive — the Cephes `expf` range reduction
(`vexpQuadsNeonF32` in `backend/cpu/vexp_arm64.s`) plus an Abramowitz-Stegun `erf` and a Cephes
`logf` that share its structure — and have **every** f32 transcendental op *compose on it* rather
than carry its own. This is §C26 (bottom-up assembly on a verified leaf) applied at the kernel level:

- **The leaf**: a 4-wide (two-quads-per-pass, software-pipelined) NEON `e^x` with the standard
  `n = round(x·log2e)`, `r = x − n·ln2`, degree-5 Horner, `2^n` by exponent-bit insertion. The
  `erf` reuses its `e^(−u²)`; `log` is a sibling primitive (frexp-style exponent extraction + a
  Cephes log polynomial — not derivable from exp, so a distinct kernel, but the same methodology).
- **The composers** (each verified in isolation, then assembled): softmax row + fused MHA (T660),
  standalone `OpSoftmax` (T661), GELU forward (T663) and GELU-backward — whose `Φ(x)+x·φ(x)`
  derivative reuses the erf's `e^(−x²/2)` so **one exp feeds both erf and pdf** — plus cross-entropy
  (T664), SiLU forward+backward and sigmoid for SwiGLU (T665), and the standalone `OpExp`/`OpTanh`/
  `OpLog` (T666). The non-transcendental reductions on the same paths (cross-entropy sum,
  add-bias-backward, T667) are typed+parallel kernels registered alongside.
- **Gating**: all fast paths sit behind a compile-time `vexpNeon` const (`goexperiment.simd &&
  arm64`). The default `CGO_ENABLED=0` build, the amd64 SIMD front, and every F64 path keep the
  scalar/reference code **byte-for-byte** — verified by `TestCPUCrossReferenceExact` asserting the
  default-build F32 output equals the reference bit-exactly. The fast paths ride ADR-0021's
  tolerant-f32 contract (rtol ≈2e-3); measured accuracy is 1e-7..3e-5, far inside it.

Methodology, applied per rung: profile the *real* workload to pick the target (`GOAI_TIME_OPS` +
`GOAI_LOG_FALLBACK`, never standalone-op timing); WORD-encode any instruction the Go assembler
lacks and **objdump-verify** each; interleaved §V22 A/B, keep only ≥1.2×; parity vs the f64
reference + `autograd`/`nn` gradcheck for anything on a gradient.

## Outcome (implemented 2026-07-15, measured on the M2 Pro)

The vexp leaf now serves **softmax, MHA, GELU (fwd+bwd), cross-entropy (fwd+bwd), SiLU (fwd+bwd),
sigmoid, exp, tanh, log** — the full f32 transcendental surface, fused and standalone. Per-op wins
ran 2.5×–5.1× (forward) and 25×–45× on the backward ops that had been serial ref-fallbacks. End to
end, the f32 CPU GPT forward went ≈1250 → ≈13,600 tok/s (≈10.9×) and the training step ≈1325 →
≈2220 tok/s (≈1.5×). A final `GOAI_LOG_FALLBACK` audit is clean: the GPT/Llama forward and training
paths are fully cpu-native (only intentional tiny embed host-gathers remain on the reference), and
both are now **matmul-bound at the AMX ceiling** — every compute-bound op is vectorized and every
reduction is typed+parallel. Numbers and method are in `docs/benchmarking.md` (§T656-667).

## Consequences

- **Extending it is cheap and safe.** A new f32 activation reuses the leaf (compute `e^x`/`erf`/
  `log` from it, apply the op's algebra) instead of writing fresh range-reduction asm — the pattern
  T663-T666 repeated five times. The accuracy budget (ADR-0021 rtol 2e-3) and the gating const are
  inherited.
- **One place to trust.** The range reduction, the overflow/underflow masks (e.g. `OpExp`'s split
  `2^n` scale + `+Inf`/`0` boundary masks, bisection-matched to the reference's overflow set), and
  the WORD encodings are verified once and shared, instead of re-derived per op.
- **Bounded scope, honestly.** The leaf is arm64+simd only (opt-in); the default build stays exact.
  It does **not** help bandwidth-bound ops with no transcendental (layernorm apply, embed gather) —
  those are at their memory floor and deliberately not chased (§C25 discipline: optimize
  compute-bound ops, not bandwidth-saturated ones). f64 stays the exact reference regime.
- **The remaining CPU gap is silicon, not code.** With transcendentals vectorized and GEMM at the
  AMX ceiling, the forward/training are matmul-bound on Apple's own coprocessor — the honest floor.
