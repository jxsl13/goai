---
schema: v1
prefix: FMT
---

## FMT-005 {applies: go:safetensors.FuzzLoad}
IF a format reader receives hostile or malformed input, THEN the reader SHALL return a clean error, never panicking, over-allocating or returning a wrong tensor with a nil error.

Rationale: Applies to every foreign-format reader: gguf, safetensors, npy, npz and pytorch. A wrong-but-nil-error result is as dangerous as a panic because it silently corrupts downstream numerics. Migrated from cavekit SPEC.md V15 and V29.

## FMT-006
The length, dimension and nesting claim read from a file header SHALL be capped before any allocation, with header size, element count and array depth bounded (depth at most 64) and growth never pre-allocated from a claimed size.

Rationale: A hostile header otherwise turns a size field directly into an out-of-memory abort. Migrated from cavekit SPEC.md V15.

## FMT-007 {applies: go:pytorch_test.loadHostile}
The a shape product computed from untrusted dimensions SHALL use the division-before-multiply overflow guard and reject negative dimensions, since a post-multiply cap wraps and lets a hostile header through.

Rationale: No opcode or field parser may index a stack or buffer without an emptiness check either. Migrated from cavekit SPEC.md V29.

## FMT-008 {applies: go:npy.FuzzLoad}
The fuzz target SHALL reach the parser it names, taking raw inner bytes rather than a container envelope.

Rationale: A fuzz target aimed at the zip container left the pickle interpreter behind it effectively unreachable: 966k executions yielded 6 interesting inputs, while a target taking raw inner bytes reached 5.77M and 104. A target that does not reach its parser gives false assurance. Migrated from cavekit SPEC.md V29.

## NUM-ACCUM-NARROW-001
IF a dot product accumulates in float64 and stores its result as float32, THEN the exactness gate SHALL state that it covers element mapping and scale selection, not summation order, as TestQMatMulFusedDecodeMatchesGeneralPathExactly does.

## PERF-FASTPATH-FAMILY-001
IF an early-return path short-circuits only some members of a variant family a switch enumerates, THEN the uncovered variant SHALL be benchmarked before the asymmetry is assumed intentional; 3 such in QMatMul measured 1.40x-1.52x.

## intent
- T-01KYMVPJW5FEAVQY0AX02QC420 Benchmark Q2_K/Q3_K/Q5_K single-token QMatMul before fusing them: DONE, all three fused. Q2_K 1.67x (959.1-965.4us -> 571.2-586.3us), Q3_K 1.75x (1092.5-1103.6us -> 617.7-646.8us), Q5_K 1.41x (794.0-796.6us -> 560.7-570.2us). Interleaved, 4 alternations per quant, one quant at a time, every arm inside 5%. All seven quant types now sit at 102 allocs per decode step; the span across formats collapsed from 525-1082us to 528-626us.

THE TASK'S CAUTION WAS RIGHT TO DEMAND A MEASUREMENT AND WRONG ABOUT THE DIRECTION. It argued the aggressive quants might not repay fusion the way the deployment formats did, because they do more per-element unpacking. They repaid it MORE (1.41-1.75x against 1.40-1.52x). The alloc signature the task said to check first — 107 vs 102 per decode step — was present on all three and was the correct predictor; the reasoning about unpacking cost was not.

PS6003 FALSE-POSITIVED ON THE FIRST DRAFT AND THE CODE WAS CHANGED, NOT THE RULE. Wrapping a dispatch switch in a single `if m == 1` guard hides the covered set behind a function value the AST check cannot follow, so it reported a gap that no longer existed. Naming the five types in the guard keeps coverage legible to the check and meaningful for whoever adds an eighth type. Teaching the rule dataflow would have cost soundness and bought nothing. Second time this campaign that dogfooding PS6003 on fresh code corrected something — the first was its own || handling.

GATE: the three paths were added to TestQMatMulFusedDecodeMatchesGeneralPathExactly before being written. Four mutations turn it red (sign-flipped Q2_K dmin offset, wrong Q2_K activation group, inverted Q3_K high-bit select, Q5_K fifth bit shifted one position) while every pre-existing 1e-5 test stays green. Three further probes were REJECTED as invalid rather than counted: two anchors matched both a dequant and its fused twin, one did not compile.

LANDED: commit 621f4b68. Learning in PERF-FASTPATH-FAMILY-001 and NUM-ACCUM-NARROW-001; research R-01KYMVGRENEND is now fully consumed. NOT DONE, deliberately: no fusion for m>1 (it would re-dequantize per activation row), and no SIMD or integer-accumulator rewrite of any dot — that trades the bit-identity these gates rest on.
- R-01KYZK54C3EN69NDQJG1SY3KTC Round T1044: gguf data-section presizing -17.7%, Q6_K window -16.5%, PS3030: Consumed: both measured candidates shipped (data-section presizing -17.7 percent on the tensor-heavy shape, Q6_K destination window -16.5 percent), the class became PS3030, and the hostile-input lesson is recorded in the presizing commit and its guards. The largest-first tensor claim is explicitly NOT taken because the synthetic benchmark builds uniform tensors and cannot show it; that needs a ske [body truncated at tombstone retention cap]
- R-01KZ10CKXZFC98G3TXDKMR2T1V Round T1052: ReadRaw views -50%, and a standing instruction to stop asking: Consumed: ReadRaw views shipped at -53.5 percent time and -49.8 percent bytes with the contract documented and the capacity clamp gated; the slab alternative measured at -2.4 percent and recorded as not worth taking; and the decision escalation answered as a standing instruction, now NEVER-ASK-TAKE-THE-MEASURED-WIN-001.
