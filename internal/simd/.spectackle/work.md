---
schema: v1
---

## P-01M0J0BWPDE4P97Q05AZV0PSJS Apple arm64 fused F64 WKV recurrence
kind: proposal
state: active
created: 2026-08-21
grilled: 2026-08-21 open=0
targets: go:simd.WKVScanF64~2, go:simd.WKVScanStateF64~2, go:simd.WKVScanRangeF64~2, go:simd.wkvScanStateScalar, go:simd.benchWKVScan, go:cpu.wkvParallelScanF64

Merged main 24360555396d1b694cbd5bcfec979c0416332497 leaves all Apple arm64 F64 WKV fresh, continuing-state, and range entry points on wkvScanStateScalar. Physical M2 Pro baseline at GOEXPERIMENT=simd is 14.6-15.6 ms for the SIMD-labelled 512x1024 benchmark versus 15.3-16.8 ms scalar, both zero allocations; a 2 s CPU profile attributes 46.97% flat to math.archExp and 98.56% cumulative to the scalar recurrence. Implement a native Apple arm64 NEON recurrence, using the existing degree-13 negative-domain exponential strategy where numerically safe, while preserving scalar behavior for unsupported inputs. The implementation must cover fresh and persistent AA/BB/PP state, keep whole/range and whole/chunk execution equivalent under the documented tolerance or bit-exact contract, retain portable and non-arm64 fallbacks, prove vector opcodes, and produce statistically significant physical M2 Pro internal plus backend/cpu gains with zero internal allocations. Pin exact control and candidate commits and disclose scheduling or thermal variance.
