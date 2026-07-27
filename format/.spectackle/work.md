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
