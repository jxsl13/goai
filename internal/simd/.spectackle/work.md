---
schema: v1
---

## P-01M0HWDB7BEBY99M96950C5DHE Apple arm64 fused F64 SSM selective scan
kind: proposal
state: draft
created: 2026-08-21
targets: go:simd.SSMScanF64, go:simd.SSMScanRangeF64, go:simd.ExpScaledF64

Apple arm64 SSM selective scan currently routes through scalar math.Exp and scalar state arithmetic under goexperiment.simd. The merged two-lane negative-exp NEON polynomial makes a fused selective-scan leaf feasible. The change targets only the proven finite nonpositive decay domain, preserves pre-mutation scalar fallback for arbitrary inputs, keeps whole/range ordering identical, and must clear paired M2 performance, numerical, allocation, objdump, cross-build, race, and full-CI gates. WKV, generic Exp capability flags, other devices, and backend ownership allocations remain outside this proposal.
