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
IF an early-return path short-circuits only some members of a variant family a switch enumerates, THEN the each uncovered variant SHALL be benchmarked before the asymmetry is assumed intentional; 3 such in QMatMul measured 1.40x-1.52x.
