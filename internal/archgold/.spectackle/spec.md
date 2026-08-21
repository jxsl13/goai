---
schema: v1
prefix: ARCH
---

## ARCH-GOLD-SIMD-001
WHEN an experiment-gated SIMD implementation intentionally changes floating-point bits within its numeric contract, the bit-identity tests SHALL freeze separate GOARCH and goexperiment.simd digests while retaining every default-build golden unchanged.
