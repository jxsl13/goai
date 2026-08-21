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

## T-01M0J17HEEEF7VTW4RGEJ4MNWF Mask underflow and fuse the F64 NEON WKV recurrence
kind: task
state: done
created: 2026-08-21
parent: P-01M0J0BWPDE4P97Q05AZV0PSJS
grilled: 2026-08-21 open=11
targets: internal/simd/exp_f64_arm64.s, internal/simd/simd_arm64.go, internal/simd/wkv_arm64.go, internal/simd/wkv_arm64_test.go, internal/simd/wkv_test.go, internal/simd/wkv_range_test.go, backend/cpu/wkv.go, backend/cpu/wkv_bench_test.go

Base is verified GoAI merge 24360555396d1b694cbd5bcfec979c0416332497. Physical M2 Pro baseline: SIMD-labelled 512x1024 is 14.6-15.6 ms versus scalar 15.3-16.8 ms, zero allocations; CPU profile is 46.97% flat math.archExp and 98.56% cumulative wkvScanStateScalar. Implement a two-channel NEON leaf for fresh, continuing-state, and aligned range scans. Keep w/u and AA/BB/PP resident across seq and evaluate only the negative member of each stabilized exp pair with the degree-13 vector strategy. A no-allocation pair preflight must reject non-finite operands or PP/max evolution before mutation. The leaf must clamp polynomial evaluation to -708 and mask every finite argument below -708, including the mandatory fresh PP=-1e38 sentinel, to exact float64 +0.0. Preserve scalar tails, fixed pair grouping, portable/default/amd64 behavior, and no heap scratch. Accuracy versus scalar must stay within 1e-10; whole versus 2-aligned range and whole versus stateful chunks must be bit-identical for output and final state. Objdump must prove D2 recurrence, FRINTN, exponent-bit construction, cutoff comparison, and zero mask. Retain only if three alternating count-seven physical M2 campaigns at 500 ms show at least 35% lower internal 512x1024 median and 20% lower backend/cpu median, every comparison p below 0.05, with zero internal allocations. Pin exact commits and disclose variance.
