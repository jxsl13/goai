---
schema: v1
prefix: PERF
---

## PERF-LUT-BEATS-BRANCHY-BIT-MATH-001
IF a large lookup table is proposed for removal in favor of recomputing the value, THEN the implementing agent SHALL measure both arms on a varied-input benchmark first, since a 256KiB table beat the arithmetic it replaced by 2.5x.

Rationale: tensor f16ToF32 keeps a 65536-entry f32 table. Pointing it at the equivalent bit-manipulation reference measured +586% scalar, +158% on a varied 512x512 cast and +147% strided, all p=0.002 at n=6. The subnormal normalization loop costs more than the random-ish table access, even though the table exceeds L1.

## DO-NOT-COALLOCATE-A-SHARED-OBJECT-001
IF objects created together are merged into one allocation block, THEN the implementing agent SHALL exclude any object that outlives its block through sharing, and measure time rather than allocation count.

Rationale: NewOn allocated five objects per fresh tensor: the Tensor, its Storage, the shape and stride array, the data slice, and the interface box Storage.data pays to hold that slice. Merging the first THREE cut allocations 23.16 percent and made BenchmarkJambaDecode 6.86 percent SLOWER (p=0.020, n=12), because Storage is shared: every view built by Reshape and friends points at it, so a surviving view retained the whole block including the Tensor it came from, raising the live set and the GC scan. Merging only the Tensor with its shape and stride array, leaving Storage separately allocated, keeps 11.58 percent fewer allocations and is 4.35 percent FASTER (p=0.001, n=12) with bytes unchanged. The mechanism was confirmed by the second variant rather than assumed from the first. ALLOCATION COUNT IS A PROXY AND CAN COST TIME: a 23 percent reduction that loses 7 percent of wall clock is not a win, and both figures needed n=12 because at n=6 this benchmark reported +2.94 percent with a 14 percent spread and at another n=6 it reported nothing at all.
