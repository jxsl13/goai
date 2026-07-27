---
schema: v1
---

## T-01KYJNDV2VFRFR5RDBSPKJBRVK Cut the residual serial cost in gguf ReadFile
kind: task
state: approved
created: 2026-07-27
targets: format/gguf

---
schema: v1
---







After the parallel decode landed, ReadFile of the 1.1B model is about 323ms, and the residual is the serial 668MB read-into-heap plus parse. The dequantization itself now parallelizes nearly fully and is bandwidth-bound on the 4.4GB f32 output.

Three independent levers, each a dedicated fresh-context piece of work rather than a tail-end add-on, since each is assembly, platform syscall, or architectural:

(1) Zero-copy mmap: ReadFile maps the file (syscall.Mmap on unix, CreateFileMapping on windows, behind build-tagged mmap_unix.go / mmap_windows.go / mmap_other.go) and parse and decode read directly from the mapped bytes rather than an io.ReadAll heap copy. Requires restructuring the parser to reference the mapped data section rather than copy it, and munmap after decode. decodeTensor already copies into its own tensor storage, so no long-lived mapping is needed. Saves the 668MB allocation and copy plus the peak memory.

(2) SIMD Q4_K and Q6_K dequantization: the scalar path is already invariant-hoisted and destination-direct (measured 745 MB/s for Q4_K), but the inner nibble-unpack and FMA loop is NEON-vectorizable using the established word-encoded Plan9 assembly pattern, arch-gated and f32-parity-tested. Would approach llama.cpp's dequantization speed.

(3) Deferred dequantization: return quantized tensors rather than eager f32, avoiding the 4.4GB write and the 4x memory, matching llama.cpp's quantized-inference model. Architectural, touches model load.

Migrated from cavekit SPEC.md T914.

## T-01KYJPSG8XFXVRA5TER8FSAPMH Pre-size and then mmap the gguf data-section read
kind: task
state: draft
created: 2026-07-27

SITE: format/gguf/gguf.go:332 — data, err := io.ReadAll(rd.r) inside parse.

MEASURED BASELINE on this host (M2 Pro, go1.26.5 darwin/arm64): BenchmarkDequantQ8_0 1844 MB/s, Q4_K 787 MB/s, Q6_K 866 MB/s, DecodeF32 13530 MB/s.

WHY HOT: parse is the single chokepoint for BOTH gguf entry points — ReadFile (:432) -> Read (:341) -> parse, and ReadRaw (:408) -> parse. Every nlp/quant_*_gguf.go constructor goes through ReadRaw. Once per model load, but it is 100% of the serial residual in the 323ms figure: the parallel worker pool at gguf.go:352-370 cannot start until this line returns.

DEFECT: Go 1.26's io.ReadAll reads into a chain of make([]byte, next) chunks (next += next/2), then at EOF allocates make([]byte, finalSize) and appends every chunk into it. For a 668MB data section that is about 668MB of chunk zeroing, a second 668MB zeroed allocation, and a full 668MB memcpy, with roughly 1.3GB transient peak — all pure overhead, because the file size is known (ReadFile already holds an *os.File) and the bytes are never mutated. ReadFile opens the file at :433 and immediately discards the size by passing a bare io.Reader to Read.

FIX, two tiers; do tier A first, it is about 10 lines and isolates cleanly.
A (pre-size): add parseSized(r io.Reader, dataCap int64); ReadFile stats the file and passes size down. Replace :332 with: if dataCap >= 0, data = make([]byte, dataCap - rd.n) plus io.ReadFull, else keep io.ReadAll. This mirrors what the sibling package already does — format/safetensors/safetensors.go:290 loadFrom(r, fileSize) solves the identical problem.
B (mmap): add format/gguf/mmap_unix.go (//go:build unix) plus mmap_other.go fallback using syscall.Mmap(fd, 0, int(size), PROT_READ, MAP_SHARED) — pure Go, no cgo, consistent with the pure-Go-first constraint. ReadFile mmaps, wraps the region in a bytes.Reader for the header, and hands the tail []byte straight to parsed.data. Page faults are then absorbed by the 12 decode workers instead of one serial thread. munmap after Read returns (the f32 tensors are copies, so that is safe); for ReadRaw the mapping must outlive the RawFile, so attach runtime.AddCleanup or an explicit Close.

VALIDATION GATE (benchmark only): BenchmarkReadFileModel (format/gguf/bench_test.go:186) covers this exactly but SKIPS, because models/ holds only README.md. Write BenchmarkReadFileSynthetic building a fixture once in b.TempDir() — 200 tensors of [2048,2048] Q4_K via WriteQuantized, about 470MB on disk (256MB is the minimum useful size) — then ReadFile in the loop with the first call warming the page cache. Run -benchtime 5x -count 6 and compare with benchstat. Report -benchmem: the win shows as roughly 1.3GB/op less B/op even before ns/op moves. Add a ReadRaw variant on the same fixture to isolate this from dequant cost.

EXPECTED: tier A 60-100ms off a 323ms ReadFile (about 1.25x) and -1.3GB transient peak, high confidence because the mechanism is visible in the stdlib source; tier B a further 40-80ms plus elimination of 668MB live heap, medium-high.

CORRECTNESS BAR: does NOT touch a bounds check — the per-tensor offset/need guards at :419-423 and :449-453 use the overflow-safe subtraction form against len(p.data) and are unchanged. Two NEW hazards to gate explicitly: (i) os.Stat size is meaningless for FIFOs and devices, so require fi.Mode().IsRegular() before pre-sizing or mmapping and fall back to io.ReadAll otherwise; (ii) a truncated file must not leave data short-but-uninitialized — use io.ReadFull and propagate the error, never io.ReadAtLeast. For tier B, mmap makes parsed.data alias a file another process can truncate, which is SIGBUS: document ReadFile as being for trusted-length local files and keep the streaming Read(io.Reader) path byte-identical, since the FuzzRead/FuzzReadRaw corpora exercise the reader form.

PERFSCAN RULE REQUIRED: io.ReadAll on a source whose size is known. AST shape: a CallExpr to io.ReadAll whose argument's dataflow root is an *os.File (or io.ReaderAt / *io.SectionReader / *zip.File) reachable in the same function or from a same-package caller where os.Stat / f.Stat() / fi.Size() / UncompressedSize64 is available. A weaker zero-false-positive variant: flag io.ReadAll(f) where f is an *os.File created by os.Open in the same function.

## T-01KYJPSGQMFGER0YD90EG3PGRK Eliminate the bounds checks and unhoisted scale products in the Q4_K and Q6_K dequant loops
kind: task
state: draft
created: 2026-07-27

Both defects were confirmed in EMITTED ASSEMBLY, not inferred. Do the Q4_K half first — it is four lines and its benchmarks already exist, so it calibrates the harness noise floor before the larger Q6_K change.

SITES: format/gguf/q4k.go:74-77 (the dst[base+l] / dst[base+l+32] stores) and :61 (blk := raw[sb*q4kBlockSize:]); format/gguf/q6k.go:37-49, especially :44-47.

WHY HOT: two chains each. Load: ReadFile -> Read -> decodeTensor (gguf.go:493 / :495) -> dequant*Into, once per tensor. Inference: QMatMul (format/gguf/quant_matmul.go) once per weight row per matmul PER TOKEN, i.e. millions of times per generation. Q4_K is the package's own documented dominant modern GGUF weight format (q4k.go:9); in a Q4_K_M mix every attn_v/output tensor is Q6_K. Measured 787 MB/s and 866 MB/s input on a machine with over 100 GB/s of bandwidth — compute-bound, not bandwidth-bound.

DEFECT, Q4_K: go build -gcflags='...gguf=-d=ssa/check_bce/debug=1' reports IsInBounds at q4k.go:75:8 and :76:8 — one bounds check per output element, with 2 CMP, BLS, BHI and 2 CALLs to panicIndex for 2 stores, plus branch pressure blocking unrolling and future vectorization. blk := raw[sb*q4kBlockSize:] at :61 is open-ended so the four derived subslices at :62-65 each carry an IsSliceInBounds. The multiplies here are ALREADY correctly hoisted (:71-72), so this is purely the bounds-check half.
DEFECT, Q6_K: is := l / 16 at :39 takes exactly two values across the 32-iteration l loop, yet d * float32(int8(sc[sco+is+0])) is recomputed every iteration — 8 distinct scale products per 128-element group, computed 128 times. The emitted code shows 8 FMULS for 4 outputs plus 4 MOVB and 4 SCVTFS re-loading and re-converting the int8 scales. Plus 4 un-eliminated bounds checks per iteration (q6k.go:45:8 through 48:8).

FIX: the idiom is ALREADY PROVEN IN THIS PACKAGE — q5k.go:44 (y := dst[base:base+64]) and q2k.go:47 (y := dst[yi:yi+16]) both compile bounds-check-free. Q4_K and Q6_K are the two that were missed. Reslice to fixed lengths: blk := raw[o : o+blockSize] and y := dst[base : base+64] (Q4_K) or dst[yo : yo+128] (Q6_K). For Q6_K additionally split the l loop at the is boundary and hoist the four scale products per half. That is a pure reassociation of (d*sc)*(q-32) with the multiply order unchanged.

VALIDATION GATE (benchmark only): existing and sufficient — BenchmarkDequantQ4_K (bench_test.go:43), BenchmarkDequantQ6_K (:44), BenchmarkQMatMulQ4_K_M1 and _M16 (:166-168). Add BenchmarkQMatMulQ6_K_M1/_M16 via the same benchQMatMul helper (n=64, k=1024), since load and inference paths have different cache behaviour. Run -count 10 through benchstat. MECHANISM CHECK, mandatory: -d=ssa/check_bce/debug=1 must emit ZERO q4k.go:7x and q6k.go:4x IsInBounds lines afterwards. IMPORTANT HARNESS NOTE: BenchmarkDecodeF32 spends about 60% of its time in GC of the 1MB/op allocation (77.5us -> 31.5us with GOGC=off), a harness artifact rather than a ReadFile cost — compare these A/Bs at fixed GOGC or it will mislead.

EXPECTED: Q4_K 1.10-1.25x (medium-high — bounds checks confirmed present, but the loop is already FMA-dense so branches may be partly hidden by the M2's predictor); Q6_K 1.35-1.7x, 866 -> 1200-1450 MB/s (high — confirmed in assembly and the fix pattern is proven in-package). Both are also the correct precursor to any NEON kernel: with scales hoisted the Q6_K inner loop becomes a pure vmul over 16 lanes.

CORRECTNESS BAR: THIS TOUCHES BOUNDS CHECKS, flag it explicitly. It replaces implicit per-element IsInBounds with one IsSliceInBounds. The safety argument is that the caller already guarantees length — decodeTensor validates need via byteSize (gguf.go:513/521, which enforces n % qkK == 0) and the offset range at :449-453 before reaching here. But the invariant len(dst) % qkK == 0 is CURRENTLY IMPLICIT: add an explicit guard at the top of each dequant*Into so a short raw produces a deterministic single-site error rather than a mid-loop panic. Re-run format/gguf/hostile_test.go, q4k_test.go, and the FuzzRead corpus; round-trip parity (quant_write_test.go) must remain exact. Output is bit-identical in both cases — no arithmetic changes.

PERFSCAN RULE REQUIRED, two new classes: (i) open-ended block subslice in a decode loop — inside a for whose body indexes a slice at base+i, a SliceExpr x[a*C:] with no High where C is a package-level constant named *BlockSize/*blockSize, or a store dst[expr] where dst is a function parameter never resliced to a constant length; cross-check against -d=ssa/check_bce output to eliminate false positives. This rule would have caught q4k.go and q6k.go while correctly passing q2k.go and q5k.go. (ii) loop-invariant-per-stride expression not hoisted — an inner for over l containing an index expression E[f(l)] where f is l/K or l>>k with K a constant at least half the trip count, whose value feeds a BinaryExpr otherwise invariant in the loop; suggest loop fission at the K boundary.

## T-01KYJPSH4HE33TTJBJM602RMBG Make gguf ReadRaw subslice instead of copying every tensor
kind: task
state: draft
created: 2026-07-27

SITE: format/gguf/gguf.go:424-425 — raw := make([]byte, need); copy(raw, p.data[ti.offset:ti.offset+uint64(need)]).

NOTE, verified: deferred dequantization ALREADY EXISTS (ReadRaw / QuantTensor, gguf.go:408) and is used by about 40 nlp/quant_*_gguf.go loaders. Do not re-propose it as a lever — but it carries this copy defect.

WHY HOT: ReadRaw is THE load path for quantized inference — entry point of about 40 constructors (nlp/quant_llama_gguf.go:22, quant_qwen_gguf.go:6, quant_mixtral_gguf.go:12, ...) and of internal/benchcompare/prod_decode_external_test.go:30, the harness behind the TinyLlama-1.1B benchmark row. Once per model load, over the entire 668MB weight payload.

DEFECT: the loop allocates and memcpys a fresh buffer for every tensor, reproducing the whole quantized model a second time. Since tensor extents tile the data section, total copied is about len(p.data) — a full 668MB serial memcpy plus roughly 200 separate large allocations, and 2x peak RSS (1.3GB) at exactly the moment the caller is about to upload weights to a device. p.data is discarded immediately after, so the copy buys nothing. Compounded with the io.ReadAll defect, a ReadRaw of a 669MB model currently touches about 2.7GB of memory to deliver 669MB of bytes.

FIX: raw := p.data[ti.offset : ti.offset+uint64(need) : ti.offset+uint64(need)] — the THREE-INDEX form is mandatory, capping cap so a caller's append cannot scribble into the neighbouring tensor. All tensors then share one backing array, which is correct because their union IS that array, so nothing extra is retained. Document on QuantTensor.Data that the slice aliases the parsed file and is read-only. Composes with the mmap tier to make ReadRaw genuinely zero-copy, at which point the mapping must be owned by RawFile via an explicit Close or runtime.AddCleanup.

VALIDATION GATE (benchmark only): BenchmarkReadRawSynthetic on the same 200-tensor Q4_K fixture (about 470MB), opening and reading in the loop. The primary signal is -benchmem: B/op should drop by roughly 470MB and allocs/op by about 200. Measure INDEPENDENTLY of the io.ReadAll change (A/B on top of an unchanged parse) so the two wins are separable. Add a runtime.ReadMemStats peak-RSS reading if the memory claim is to go into BENCHMARKS.md.

EXPECTED: 30-60ms off a ReadRaw of a 669MB model and -669MB peak RSS. High confidence on the allocation and RSS win (arithmetic, not a guess), medium on the wall-clock share since the memcpy is bandwidth-bound and partially overlapped with page-cache reads.

CORRECTNESS BAR: does NOT weaken a bounds check — the guard at :419-423 (the overflow-safe subtraction form) runs BEFORE the slice and is untouched, so the subslice is provably in range. The real risk is ALIASING SEMANTICS: QuantTensor.Data becomes mutation-visible across tensors and keeps the whole data section alive even if the caller retains one tensor. Both are acceptable for the actual callers (weight upload, read-only), but the three-index cap is mandatory and the QuantTensor doc comment must state the aliasing. Re-run format/gguf/hostile_test.go and the FuzzReadRaw corpus unchanged.
