---
schema: v1
---

## P-01M0JE35SGFV0BNHQA5AQC2GDA Fuse ARM64 Q4_K decode unpack and dot on M2
kind: proposal
state: approved
created: 2026-08-21
targets: go:gguf.QMatMul, format/gguf/q4k.go, format/gguf/dequant_q4k_arm64.go

On Apple M2 Pro, Q4_K M1 N=4096 K=1024 takes 677-711 us while Q8_0 takes 283-286 us. Q4_K single-token QMatMul selects scalar dotQ4_KRow even though eager Q4_K unpack has a bit-exact ARM64 NEON kernel. A measured materialize-then-dot control regressed Q4_K from about 701 us to 781 us and added 49 KiB plus ten allocations, so it is rejected. Implement a fused ARM64 NEON Q4_K unpack-and-dot kernel behind the existing dotQ4KRowFn selector. Preserve portable and M>1 paths. Gate numerical behavior explicitly because the public contract accumulates f64, and require repeated M2 leaf plus full QMatMul benchmarks before retention.

## T-01M0JE3NZKEG4VDTHKGZEKYMKA Implement and gate ARM64 Q4_K fused decode GEMV
kind: task
state: done
created: 2026-08-21
parent: P-01M0JE35SGFV0BNHQA5AQC2GDA
targets: format/gguf, BENCHMARKS.md, docs/benchmarking.md, internal/benchcompare/leadership/evidence/m2-arm64-q4k-fused-dot-20260821, .spectackle

Implement an ARM64-only fused Q4_K M1 row-dot kernel, select it through dotQ4KRowFn, keep portable and all M>1 behavior unchanged, add direct scalar-reference and QMatMul numerical coverage, and retain only if repeated Apple M2 leaf and N=4096/K=1024 measurements show a material speedup with no allocation regression. Record the measured materialize-then-dot rejection as a closed alternative.
