---
schema: v1
---

## intent
- R-01KYZNRDJVEAAVHVZ3VWKQFGD0 Round T1048: QR reflector interchange -35% at 128x64 and NOTHING at 32x16: Consumed: QR reflector interchange shipped at -35.0 percent at 128x64 with a verbatim-reference bit-identity gate, and the null result at 32x16 became the evidence behind PS6011 cache caveat. Also corrects an earlier record that listed the ref QR as uninstrumented — it was not; the autograd QR VJP is the one without a benchmark, and two kernels with the same name were conflated.
- R-01KYZP9EG8F4KB1WYTYNFGGGPC Round T1049: SVD column-major -43.9% at 256x128, and the cell-sizing rule: Consumed: SVD column-major relayout shipped at -43.9 percent at 256x128 with a verbatim-reference bit-identity gate over all three outputs, and the cell-sizing lesson became rule SIZE-THE-CELL-PAST-L1-BEFORE-JUDGING-LAYOUT-001. Also records a test-harness bug: inferring tensor rank from the column count panics at 1x1, where every output has one column.
