---
schema: v1
---

## T-01KYJPYBM5E7YAG33QW56DWEVW Build the f64 NEON transcendental leaf so nine ops stop falling to scalar math.Exp on arm64
kind: task
state: active
created: 2026-07-27
targets: go:simd.ExpSumF64, go:simd.ExpScaledF64, go:simd.SigmoidF64, go:simd.SoftplusNegLLSumF64, go:cpu.vsiluPairsNeonF64, asm:cpu.vsiluPairsNeonF64, go:cpu.vexpF64Fast~2

CURRENT GAP: on arm64 with goexperiment.simd, internal/simd.ExpSumF64, ExpScaledF64, SigmoidF64, and SoftplusNegLLSumF64 still compile from the scalar math.Exp fallback. amd64 has a degree-13 F64 vector polynomial. backend/cpu already owns a separate two-lane NEON SiLU polynomial, proving the leaf is implementable, but vexpF64Fast remains false for other backend composites.

FIRST LANDING: add an arm64 SIMD implementation in internal/simd with a two-lane NEON exp leaf using the existing Cody-Waite reduction and degree-13 FMA chain. Preserve the public APIs and exclude only arm64 plus goexperiment.simd from the portable definitions. Compose the four internal/simd consumers. Keep the len and not-one vector body with a scalar math.Exp or stable softplus tail so body/tail selection is length-determined. Preserve in-place ExpSumF64 operation.

SCOPE BOUNDARY: do not flip backend/cpu.vexpF64Fast merely because the leaf exists. Backend sigmoid, softplus, softcap, tanh, and gradients require separate composed-kernel correctness and performance gates. Existing arm64 SiLU remains enabled and unchanged. WKVScanF64 and SSMScanF64 are follow-on consumers after the leaf lands.

BENCHMARKS: add direct no-allocation benchmarks for ExpSumF64 at 4096 and 32768 elements, ExpScaledF64 at 128, SigmoidF64 at 65536, and SoftplusNegLLSumF64 at 65536. Use paired Apple M2 Pro control and candidate binaries, alternating order, count seven, and benchstat. Report isolated leaf throughput plus public consumers.

NUMERICS: retain the existing 1e-13 relative-error contracts, exact zero for deep-underflow and negative-infinity exp lanes, correct signed infinities and NaNs where the scalar contract defines them, in-place safety, input immutability for distinct buffers, and the scalar tail behavior. Non-arm64-SIMD builds and bit-exact arithmetic APIs remain unchanged.

ATTRIBUTION: keep the benchmark-harness commit, scalar control commit, first NEON leaf commit, and final composed commit identifiable. Reject and fully revert any stage that misses its predeclared gate.
