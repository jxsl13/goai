---
schema: v1
---

## T-01KYJPYB6NFNV8K116AB2CZWFZ Replace Storage.data any with a dtype-tagged typed-field layout
kind: task
state: draft
created: 2026-07-27

HIGHEST-LEVERAGE L0 ITEM. AtF64/SetF64 have 707 + 551 non-test call sites repo-wide. Measured on this host (M2 Pro, go1.26.5): BenchmarkAtF64Contig2D 2.76 ns/elem, SetF64 3.56 ns/elem, NewSmall 190.2 ns/op 440 B/op 5 allocs/op.

SITES: tensor/storage.go:12 (data any), :81 atF64, :99 setF64, :49/:58/:69 F32()/F64()/U16(); tensor/tensor.go:109 AtF64, :115 SetF64.

DEFECT, four compounding, all confirmed by disassembly and escape analysis rather than inferred:
1. Per-element eface dispatch with pointer chasing. A slice does not fit in an interface word, so data any holds a POINTER to a slice header. go build -gcflags=-S on AtF64 shows LDP of the eface, a nil type check, MOVWU of the type hash (dependent load 2), a type compare, then LDP of the slice header (dependent load 3) — three dependent loads and about six dispatch instructions BEFORE the element load. A switch on a dtype field over typed struct fields is one byte compare and one load.
2. AtF64/SetF64 are not inlinable: -gcflags='-m -m' reports cost 141 and 139 against budget 80 (flatOffset 75, atF64 59). Every access is a real call with a 32-byte frame, and the compiler can never constant-fold len(idx) at the call site.
3. F32()/F64()/U16() miss the budget by exactly 2 points (cost 82), caused solely by the fmt.Sprintf in their panic arm.
4. One wasted heap alloc per Storage: allocator.go:34 reports the make escaping TWICE — the array and the 24-byte eface box.

FIX: dtype tag over typed fields (f32/f64/u16), with badDtype extracted behind //go:noinline to recover the 2 inline points, and the f16/bf16 arm split into a //go:noinline atF64Half. Storage grows 48 -> about 104 bytes (irrelevant, one per tensor) and LOSES the 24-byte eface box, so net allocation drops. Keep the public Allocator interface unchanged, asserting once in newStorageWith; optionally add an extension interface AllocF32(int) []float32 that heapAllocator and *Pool satisfy to kill the box entirely.
SEQUENCE THE CHANGE AND A/B EACH: (a) badDtype extraction alone, (b) typed fields, (c) flatOffset simplification. (c) is a regression risk if (a)+(b) do not get AtF64 under the inline budget, so it is conditional on measuring that they did.

VALIDATION GATE (benchmark only): existing and exactly on point — BenchmarkAtF64Contig2D, BenchmarkSetF64Contig2D, BenchmarkAtF64Strided2D (tensor/core_bench_test.go:9,23,37), plus BenchmarkNewSmall/NewScalar/PoolAllocFree for the alloc-count drop. All three access benchmarks are F32 ONLY; add BenchmarkAtF64Contig2D_F64, _F16 (exercises the noinline half arm) and BenchmarkStorageF32Hoisted. Downstream confirmation, NOT the gate: go test ./backend/ref -bench . -count=6 and go test ./nn -bench 'Apollo|SpinQuant|Embedding' -count=6.

EXPECTED: 2-4x on AtF64/SetF64 (2.76 -> about 0.8-1.3 ns/elem); New 5 -> 4 allocs from the box alone. High confidence on the microbenchmark (disassembly and inline-cost numbers are direct evidence), medium on end-to-end, since L1 already routes compute-bound kernels around this.

BIT-IDENTITY BAR: NONE — no arithmetic changes. Same float64(d[i]) widening and same f32ToF16(float32(v)) narrowing, reached by a different dispatch. Zero change to accumulation order or width. This is the rare hot-path fix with no FP risk, which is why it should go first.

ADR CONFLICT TO RESOLVE AS PART OF THIS WORK: docs/decisions/ADR-0001-type-erased-storage.md claims the any-held slice needs one type assertion per whole-tensor kernel call, amortized over all elements, NOT per element. storage.go:81/99 violate that premise outright, and backend/ref/devirt.go:5-11 documents that L1 built an entire widen-to-[]float64 copy layer specifically to escape this dispatch — i.e. a full materializing copy of the tensor beats it. The ADR's own "Revisit if: profiling shows the any assertion dominates" clause is satisfied. Supersede the ADR through the decide path rather than silently contradicting it.

PERFSCAN RULES REQUIRED, two: (i) type-switch on an any/interface field inside a function doing per-element work — TypeSwitchStmt whose tag is a SelectorExpr on a receiver field of type any, where the body contains an IndexExpr on the bound variable using a parameter as the index. Distinct from PS1001, which fires on the CALLER's loop; this fires on the CALLEE's definition, catching the cost where no loop is statically visible. (ii) hot accessor pushed over the inline budget by a formatted panic — a FuncDecl of at most 3 statements whose only non-trivial node is panic(fmt.Sprintf(...)), on a method of a configured hot-path type. Remedy is mechanical and bit-identical, so this one qualifies for perfscan -fix, and it can self-verify by shelling out to go build -gcflags=-m and grepping for 'cost (8[0-9]|9[0-9]) exceeds budget 80'.

## T-01KYJQ3R68EGDAQCYFT6HJ64MJ Generalize the cache-blocked gather past rank 2
kind: task
state: draft
created: 2026-07-27

SITE: tensor/tensor.go:430 — blk2d := nd == 2 && !rowRuns (in Contiguous) and tensor/tensor.go:347 — blk2d := nd == 2 && t.strides[nd-1] != 1 (in Cast). Both fall through to gatherCast at tensor/tensor.go:153.

WHY HOT: Contiguous() has 343 non-test call sites. Concrete rank-3 chain, once per forward pass per model: nlp/t5.go:76-82 — m.RelBias.Bias(ctx, seq, seq) returns [seq, seq, heads], then bias.Permute(2,0,1) gives [heads, seq, seq] with strides [1, seq*heads, heads], then bias.Contiguous(). Identical code at nlp/t5_decoder.go:73-79 and llamagpu/t5.go:115. nd == 3 and strides[nd-1] == heads != 1, so rowRuns is false AND blk2d is false — naive gatherCast.

DEFECT: gatherCast walks output row-major, advancing the source by strides[nd-1] per element. For the T5 shape that stride is heads (8-32 elements, 32-128 bytes), so consecutive reads land on a different cache line every 1-2 elements while the source plane spans seq*seq*heads*4 bytes — 8MiB at seq=512, heads=8. It sweeps the entire working set once per output row. This is exactly the pattern already fixed for rank 2 and MEASURED at 1.89x (Contiguous) / 1.50x (Cast) on this machine; the fix was simply never generalized past nd == 2.

FIX: generalize the gate to the innermost two axes of ANY rank — when nd >= 2 && strides[nd-1] != 1, loop the outer nd-2 axes with the existing incremental-offset walk and call gatherBlocked2D on each shape[nd-2] x shape[nd-1] plane, passing the plane's base offset and s0 = strides[nd-2], s1 = strides[nd-1]. gatherBlocked2D already takes exactly (rows, cols, s0, s1, off), so there is no signature change — just an outer driver. Apply symmetrically at both :347 and :430. While there, benchmark blk = 32 and blk = 8 against the current blk = 16 (at f32, blk=16 is a 64-byte dst row, exactly one cache line, which is tight and gives only 16 inner iterations per loop setup). ALSO DELETE the stale NOTE(rejected) block at tensor.go:194-198 claiming tiling landed within noise — it contradicts both the shipped gatherBlocked2D and its own commit history.

VALIDATION GATE (benchmark only): NONE of the existing benchmarks cover rank > 2 — all of perf_bench_test.go and core_bench_test.go use rank-2 or contiguous shapes. Write BenchmarkContiguousPermuted3D on New(F32, Shape{256,256,8}).Permute(2,0,1) (the T5 rel-bias shape), BenchmarkContiguousTransposed4D on New(F32, Shape{4,8,128,64}).Transpose(2,3) (the attention shape), BenchmarkCastPermuted3D, and an F64 variant of the 3D case since the tiling win is size-dependent. The EXISTING BenchmarkContiguousStrided / BenchmarkCastStrided must be run as regression guards — the rank-2 path must be untouched.

EXPECTED: 1.5-2.5x on rank-3/4 strided Contiguous/Cast. High confidence that the naive path is taken (the strides were traced by hand and the gate is a literal nd == 2); medium-high on magnitude, anchored on the measured 1.89x/1.50x for the structurally identical rank-2 change on this same machine.

BIT-IDENTITY BAR: none — pure reordering of independent element copies and conversions, the same (i,j) -> dst[i*cols+j] mapping, no accumulation involved. gatherBlocked2D's doc comment at tensor.go:130-134 already makes this argument for rank 2 and it carries over unchanged. Verify with TestContiguousPermuted3DMatchesGeneric asserting exact []float32 equality against gatherCast.

PERFSCAN RULE REQUIRED: a fast path gated on an exact rank or dimension literal where the underlying condition is rank-agnostic. AST shape: an assignment whose RHS is a BinaryExpr{&&} containing X == <int literal> where X is len(<field>) or a variable assigned from it, and where the guarded branch calls a helper ALSO reachable from a default/else branch handling the same dtypes. Report as "dimensionality-gated fast path — check whether the general case is reachable". Related sub-check worth adding at the same time: flag stale NOTE(... rejected) comments whose claim contradicts a currently-live code path.
