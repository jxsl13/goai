---
schema: v1
prefix: PERF
---

## PERF-LUT-BEATS-BRANCHY-BIT-MATH-001
IF a large lookup table is proposed for removal in favor of recomputing the value, THEN the implementing agent SHALL measure both arms on a varied-input benchmark first, since a 256KiB table beat the arithmetic it replaced by 2.5x.

Rationale: tensor f16ToF32 keeps a 65536-entry f32 table. Pointing it at the equivalent bit-manipulation reference measured +586% scalar, +158% on a varied 512x512 cast and +147% strided, all p=0.002 at n=6. The subnormal normalization loop costs more than the random-ish table access, even though the table exceeds L1.
