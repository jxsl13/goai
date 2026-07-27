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
