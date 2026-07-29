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
state: done
created: 2026-07-27
targets: format/gguf/q4k.go, format/gguf/q6k.go, docs/benchmarking.md

PROGRESS: the Q4_K half is LANDED and measured; the Q6_K half is NOT DONE and is why this item stays open.

LANDED (Q4_K). dequantQ4_KInto now reslices to fixed lengths — raw[o:o+q4kBlockSize] and dst[base:base+64] — instead of the open-ended raw[sb*q4kBlockSize:]. MEASURED 177,736 -> 154,195 ns on BenchmarkDequantQ4_K, 1.15x, via an interleaved three-round A/B with a file-copy toggle on the same host in one session; inside the predicted 1.10-1.25x. MECHANISM VERIFIED both before and after with -gcflags=-d=ssa/check_bce/debug=1: the per-element IsInBounds at q4k.go:75:8 and :76:8 are gone, replaced by one IsSliceInBounds per sub-block for the two reslices — N per-element checks traded for one per-run check. Bit-identity is exact: no arithmetic expression changed, y[l] addresses the same element as dst[base+l] by construction, so operand order, accumulation and rounding are untouched. Golden, round-trip, hostile-input and fuzz suites pass. The fixed-length reslices also concentrate the length check at one named site, so a short raw or dst panics there rather than mid-loop — the defense-in-depth the task asked for, without a separate guard clause. Recorded in docs/benchmarking.md.

CORRECTION to the original brief, worth carrying forward: it claimed q5k.go and q2k.go compile bounds-check-free and could serve as the clean comparison. They do NOT — a check_bce dump over the package reports 29 findings across those two files. The idiom they use is still the right model for the INNER STORE loop, but the blanket claim that they are BCE-clean is wrong and should not be repeated.

STILL TO DO (Q6_K, the larger win at 1.35-1.7x): q6k.go:37-49. Two parts. (a) The same fixed-length reslicing for blk and for dst[yo:yo+128]. (b) The real win: is := l / 16 at :39 takes exactly two values across the 32-iteration l loop, yet d * float32(int8(sc[sco+is+0])) is recomputed every iteration — 8 distinct scale products per 128-element group, computed 128 times. Split the l loop at the is boundary and hoist the four scale products per half. That is a pure reassociation of (d*sc)*(q-32) with the multiply order unchanged, so it stays bit-identical — but unlike the Q4_K half it DOES touch arithmetic grouping, so verify with a golden comparison rather than assuming. Existing benchmark BenchmarkDequantQ6_K covers it; add BenchmarkQMatMulQ6_K_M1/_M16 via the benchQMatMul helper since load and inference paths have different cache behavior.

HARNESS NOTE that applies to the Q6_K A/B: BenchmarkDecodeF32 spends about 60 percent of its time in GC of its 1 MB/op allocation (77.5us vs 31.5us with GOGC=off), a harness artifact rather than a ReadFile cost. Compare at fixed GOGC or it will mislead.

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

## R-01KYMVGRENENDS71F7VFKRAD5H Fused single-token QMatMul completed for Q4_0/Q4_K/Q6_K — 1.40x, 1.40x, 1.52x measured
kind: research
state: draft
created: 2026-07-28

QMatMul carried a fused m==1 (decode) path for Q8_0 only. Every other quant type materialized each weight row into scratch and read it straight back, once per output row.

SYMPTOM THAT EXPOSED IT: Q4_0 decode benchmarked SLOWER than Q8_0 (747us vs 534us per QuantMamba2 DecodeStep) despite half the memory traffic. That ordering is backwards for the smaller format. The K-quants carried the same signature: 107 allocs against the fused paths' 102.

MEASURED, interleaved A/B in one session, alternating in-tree:
  Q4_0  735.8-746.2us -> 521.2-548.1us   1.40x
  Q4_K  745.8-751.6us -> 534.3-536.5us   1.40x
  Q6_K  855.2-861.7us -> 558.6-570.2us   1.52x
Allocs 107 -> 102 for all three. Q4_0 now edges out Q8_0 (515us vs 526us), restoring the expected ordering. Numbers are whole DecodeStep, not the matmul in isolation.

ONE INTERLEAVE WAS DISCARDED, not published: running Q4_K and Q6_K together drifted the OLD arm 25% within the set (735 -> 921us) while the NEW arm held. A ratio taken from it would have reported machine thermals as a speedup. Re-run one quant at a time at lighter load, both arms came in under 2%.

Q6_K deliberately does NOT accumulate in ascending k: its dequant writes four interleaved streams (l, l+32, l+64, l+96) and following that traversal is what keeps the per-element float32 weights identical. Verified, not assumed.

GATE BUILT FIRST, AND IT FOUND A PRE-EXISTING HOLE: every QMatMul test compared against a float reference at 1e-5, so the EXISTING Q8_0 fused path's bit-for-bit claim had nothing holding it. The new gate runs one activation row as m==1 (fused) and as row 0 of an m==2 call (general), demanding exact equality — production as its own oracle, so it cannot drift like a frozen copy. Mutation-probed: unsigned-instead-of-int8, off-by-one activation index, and a scale read from the wrong block each turn it red while every pre-existing test stays green.

LIMIT OF THAT GATE, recorded so it is not over-trusted: reassociating the accumulation is INVISIBLE here, because a float64 accumulator narrows to a float32 output and discards the difference. A deliberate block-order reversal stayed green. The gate covers element mapping, sign and scale selection — the failure modes this path actually has — not summation order.

GENERALIZED as perfscan PS6003 (partial-fast-path-coverage). REMAINING: Q2_K, Q3_K, Q5_K are still uncovered and still unmeasured — the aggressive quants, lower deployment share, each needs its own benchmark before anyone fuses it.

## R-01KYMWGGNMER3B16KBXB8JY18H Row-parallel QMatMul measures 1.70x on decode, bit-identical — blocked on a threading-policy decision
kind: research
state: draft
created: 2026-07-28

After the fusion campaign closed, a re-profile of QuantMamba2 DecodeStep shows gguf.dotQ4_KRow at 80.95% flat / 82.31% cumulative. The fused dot IS the decode step now; nothing else is close.

The remaining structural win is that the n output rows are INDEPENDENT dots. Each writes its own index and shares no mutable state, so splitting the row range changes no accumulation — bit-identical by construction, not by measurement.

PROTOTYPED AND MEASURED, interleaved, 3 alternations:
  SERIAL   536.3-547.6us
  PARALLEL 315.3-318.7us
  1.70x on the whole decode step. Arms 2.2% and 1.1%.

Only 1.70x on a 12-core host, not 12x, and the reasons are understood: a threshold keeps small matmuls serial (the tied Head at n=64,k=256 is 16384 units, under the crossover), the SSD scan and norms stay serial, and each barrier costs a park.

THRESHOLD taken from backend/cpu's parThreshold, 1<<15 = 32768 units of rows x k — the measured M-series crossover below which pool dispatch exceeds the compute saved. In the benchmark model InProj (n=552, k=256 = 141312) and OutProj (65536) cross it; the Head does not.

NOT SHIPPED, and the reason is not the measurement. This changes threading behavior for every caller of QMatMul, library-wide. A server already running requests concurrently would go from N goroutines to N x GOMAXPROCS, which is a deployment-visible regression that no benchmark on this host would show. That is a standing policy question, not a local optimization, so it goes to a decision rather than being assumed.

PRECEDENT CUTS BOTH WAYS: format/gguf ALREADY spawns a bounded worker pool in ReadFile to decode tensors concurrently, so the package spawning goroutines is not new. But ReadFile is a one-shot load-time call, while QMatMul is the innermost hot path called several times per token — the nesting exposure is different in kind. backend/cpu solved the same problem with a bounded pool plus a guard against calling parallelWork from inside a worker, and its comments record that naive wg.Wait parking cost a full M-stop/restart per barrier across the ~16k barriers a 500-token decode issues.

OPTIONS for the decision: (a) land as prototyped with the threshold, accepting nested oversubscription; (b) route through a bounded pool with an in-worker guard, as backend/cpu does, which means either lifting that pool somewhere both packages can use or duplicating it; (c) leave serial and let the caller parallelize, which forfeits the 1.70x for single-stream decode — the latency-sensitive case.

Prototype is reproducible: fusedRows helper over the existing per-row dot functions, guarded by workers>1 and n*k >= 1<<15.

## ADR-01KYMWJ76AFA2BJ9R8ZE403KB1 May format/gguf QMatMul parallelize across output rows, and under what pooling policy?
kind: adr
state: done
created: 2026-07-28
context: Row-parallel QMatMul measures 1.70x on QuantMamba2 decode (536.3-547.6us serial vs 315.3-318.7us parallel, interleaved, 3 alternations) and is bit-identical by construction: each output row is an independent dot writing its own index, sharing no mutable state. After the fusion campaign, gguf.dotQ4_KRow is 82% of the decode step, so this is where the remaining leverage is. The blocker is threading policy, not correctness or speed. QMatMul runs several times per token on the innermost path, so a caller already serving requests concurrently would go from N goroutines to N x GOMAXPROCS — a regression no benchmark on this host would surface. Precedent cuts both ways: format/gguf already spawns a bounded pool in ReadFile, but that is a one-shot load-time call. backend/cpu solved the same problem with a bounded pool plus an in-worker guard, and records that naive wg.Wait parking cost a full M-stop per barrier. The tree is currently SERIAL; nothing was shipped pending this.
decision: Route through a bounded pool with an in-worker guard, as backend/cpu already does. Correct under nesting, but needs that pool lifted somewhere both packages can import, or duplicated.
status: accepted

kind: radio
option: Land as prototyped: goroutines per call above a 1<<15 rows-x-k threshold, matching backend/cpu's measured M-series crossover. Simplest, gets the 1.70x, accepts nested oversubscription.
option: Route through a bounded pool with an in-worker guard, as backend/cpu already does. Correct under nesting, but needs that pool lifted somewhere both packages can import, or duplicated.
option: Stay serial and let callers parallelize. Forfeits the 1.70x for single-stream decode, which is the latency-sensitive interactive case.
blocks: R-01KYMWGGNMER3B16KBXB8JY18H
choice: Route through a bounded pool with an in-worker guard, as backend/cpu already does. Correct under nesting, but needs that pool lifted somewhere both packages can import, or duplicated.

## R-01KYMZTEPHF67RZTY2T1679NTY Quantized decode: 525-1082us -> 182-322us across seven types, and what each layer contributed
kind: research
state: draft
created: 2026-07-28

Consolidated result of the QMatMul campaign, so the next agent does not re-derive which layer paid what. All figures are whole QuantMamba2 DecodeStep on M2 Pro darwin/arm64 go1.26.5, interleaved per PROC-INTERLEAVE-001.

THREE INDEPENDENT LAYERS, applied in this order:
1. FUSE the per-block dequant into the dot, removing a materialize-and-reread of every weight row. Q4_0 1.40x, Q4_K 1.40x, Q6_K 1.52x, Q2_K 1.67x, Q3_K 1.75x, Q5_K 1.41x. Q8_0 already had it.
2. PARALLELIZE the output-row loop across a bounded pool (ADR-01KYMWJ76AFA2). Q4_K 1.66x, Q8_0 1.19x.
3. REGISTER-BLOCK the output loop by 4, amortizing the shared activation row across 4 accumulators. Q8_0 2.26x (arrived from main), Q4_0 1.55x.

LAYER 3 IS THE LARGEST SINGLE FACTOR and was the last one noticed, because it arrived on main attached to one of seven sibling paths and nothing flagged the asymmetry. Five K-quant dots remain unblocked; T-01KYMZC07EFT6 carries that work.

ORDERING MATTERS FOR ATTRIBUTION, not for the total: measuring layer 2 after layer 3 had landed on Q8_0 is why Q8_0's parallel gain reads 1.19x against Q4_K's 1.66x. The smaller multiplier is a smaller remainder, not a weaker optimization.

COST: allocs per decode step 102 -> 111, the pool's per-call barrier escaping to heap on the matmuls that clear the 1<<15 work threshold. Stated because a caller measuring bytes/op will see it.

GATE FOR ALL OF IT: TestQMatMulFusedDecodeMatchesGeneralPathExactly, which runs one activation row as m==1 and as row 0 of m==2 and demands exact equality — production as its own oracle. It replaced a suite that compared only against a float reference at 1e-5, under which a sign bug in the fused path passed everything. Its limit is recorded in NUM-ACCUM-NARROW-001: a float64 accumulator narrowing to float32 makes reassociation unobservable, so the gate covers element mapping, sign and scale selection, NOT summation order.

GENERALIZED: PS6003 (partial-fast-path-coverage) for layer 1, PS6005 (output-invariant-operand-reload) for layer 3. Layer 2 was NOT generalized into a rule — proving that a loop's iterations are independent is a dataflow question this AST-only scanner cannot answer soundly, and a rule that guesses at it would advise races. Recorded as a deliberate non-action rather than an oversight.

## T-01KYQ65MEEEN69YZJ0V231F7H5 Compose Q4_K's SIMD row kernel with the 4-row scalar blocking behind an override flag
kind: task
state: draft
created: 2026-07-29

Q4_K is the one quantization type whose fused matmul row kernel has BOTH a SIMD
implementation and a 4-row register-blocked scalar variant, and the two are currently not
composed — the merge took main's SIMD-gated dotQ4KRowFn and dropped dot4 for Q4_K.

WHY IT WAS NOT HAND-MERGED: dotQ4KRowFn defaults to the scalar dotQ4_KRow and is overridden
to dotQ4_KRowASM by an init in format/gguf/dot_q4k_asm_amd64.go. On a host where the
override is NOT active (arm64, or amd64 without the SIMD build), dot4 = dotQ4_K4Rows is the
better path and is what every other K-quant uses. Where the asm kernel IS active, it should
almost certainly win over 4-row scalar blocking — but that is an assumption, not a
measurement, and it cannot be measured on the darwin/arm64 host this campaign runs on.

APPROACH: export whether the override took effect (a package-level bool set alongside the
existing init in dot_q4k_asm_amd64.go, false by default), then in QMatMul's kernel switch:
    case Q4_K:
        dot = dotQ4KRowFn
        if !dotQ4KSIMD { dot4 = dotQ4_K4Rows }
Do NOT compare function values to detect the override; Go does not permit it.

VERIFY:
- On arm64: the existing Q4_K fused-path tests stay green, and BenchmarkQMatMul for Q4_K
  recovers the 4-row blocking gain the other K-quants already show.
- On amd64 WITH the SIMD build: benchmark asm-only against asm-plus-blocking to confirm the
  asm kernel really is the faster arm. If blocking wins there too, the flag is unnecessary
  and dot4 should simply be set unconditionally.
- Bit-exactness: dotQ4_K4Rows must produce results identical to four dotQ4_KRow calls; the
  4-row variants are blocked over OUTPUT rows, so each row's accumulation order is unchanged.

This is amd64/SIMD territory, so it belongs to whoever owns that lane rather than the
darwin/arm64 perf loop that found it.
