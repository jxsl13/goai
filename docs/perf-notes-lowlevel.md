# Low-level perf notes: internal/simd, internal/npy, format/*

Measured techniques from the last-percentile pass over the SIMD primitives and
the untrusted-input parsers (§V22: every change A/B'd with
`go test -bench -benchmem -count=3`; non-wins reverted per §C3). Machine:
Apple M2 Pro, Go 1.26, `CGO_ENABLED=0`. Benchmarks live next to the code as
guards (`internal/simd/simd_*_bench_test.go`, `internal/npy/npy_bench_test.go`,
`format/safetensors/bench_test.go`, `format/gguf/bench_test.go`).

## Techniques that won

### 1. Bounds-check elimination by re-slicing to a shared length
`internal/simd/simd.go` — the portable elementwise loops carried two bounds
checks per iteration (`a[i]`, `b[i]`). Re-slicing the inputs up front

```go
a = a[:len(dst)]
b = b[:len(dst)]
for i := range dst {
	dst[i] = a[i] + b[i]
}
```

lets the compiler prove all three accesses in range. AddF64 4K: 1799→1237
ns/op; 256K: 115.1→83.0 µs (~1.45×). The same file-local trick — indexing a
subslice whose length the compiler knows (`y := dst[base : base+8]`,
`blk[b*34 : b*34+34]`) — paid off across the gguf dequant loops (Q8_0 1.35×,
Q4_0 1.22×, Q2_K 1.09×, MXFP4 1.16×, IQ4_NL 1.17×, IQ1_S 1.05×). Canonical
references: go101's BCE chapter (go101.org/optimizations/5-bce.html) and
Ardan Labs "Bounds Check Elimination In Go". Verify with
`go build -gcflags="-d=ssa/check_bce/debug=1"`.

Counterexample (§C3 revert): `dequantQ3_K` resisted both the subslice form
and a branchless rewrite — its `h := 4; if hm[l+g]&m != 0 { h = 0 }` select
already compiles to a CSEL on arm64, and both "optimizations" measured ~4%
slower in a same-binary A/B. The original loop stands, with a NOTE.

### 2. Branchless sign application for data-dependent bits
`format/gguf/iq2*.go, iq3*.go, q5k.go` — the i-quant sign bits and Q5_K
high bits are effectively random per element, so `if bit { v = -v }` /
`if bit { lo += 16 }` mispredicts ~50% of the time. IEEE negation is exactly
a sign-bit flip, so

```go
sbit := (uint32(signs>>k) & 1) << 31
y[k] = math.Float32frombits(math.Float32bits(db*gridRow[k]) ^ sbit)
```

is branch-free and bit-exact (parity verified old-vs-new on random blocks).
Effect: IQ2_XXS 970→227 µs (4.3×), IQ2_XS 4.3×, IQ2_S 4.5×, IQ3_XXS 4.9×,
IQ3_S 5.0×, Q5_K 223→183 µs (1.2×, via `|` of the shifted qh bit).
The lesson from Q3_K (above): only replace a conditional the compiler did
NOT already turn into a conditional-select — always A/B.

### 3. Lookup tables for small input domains
- `format/gguf/gguf.go`: f16→f32 has a 65536-value domain; a 256 KiB
  `[1<<16]float32` table (same approach as ggml's `ggml_table_f32_f16`),
  built at init FROM the reference bit-manipulation converter so they cannot
  drift. F16 tensor decode 561→347 µs (1.6×); every per-block scale load in
  the quant decoders also benefits.
- `format/safetensors/fp8.go`: FP8 has a 256-value domain; two
  `[256]float32` tables built at init from the reference decoders replaced
  per-element float64 `Ldexp` math. Whole-file F8 load: 4.0→0.62 ms (6.4×).

### 4. Bulk storage access instead of per-element AtF64/Unravel dispatch
The known §base-perf anti-pattern, found twice more:
- `format/gguf/quant.go` Quantize built its f32 input with
  `t.AtF64(tensor.Unravel(i, shape)...)` per element → `copy` from
  contiguous F32 storage (or a single F64 widening loop). Whole Quantize:
  Q8_0 1.87→0.78 ms, Q4_K 1.88→0.76 ms (2.4–2.5×).
- `format/gguf/quant_matmul.go` QMatMul read x through `AtF64(mi, ki)` in
  the K-loop → hoisted contiguous `[]float32`/`[]float64` row slices.
  M=1: 339→129 µs (2.6×); M=16: 4.23→1.27 ms (3.3×).
- `format/gguf/writer.go` f32Data non-F32 branch: same fix, F64 tensor
  write 1.30→0.30 ms (4.3×).
- `format/gguf/gguf.go` `decodeTensor` **read** path was the last reader still
  decoding verbatim F32 per element
  (`math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))`) — safetensors,
  npy and pytorch already had the verbatim `rawCopyLE` bulk copy; gguf missed it.
  On a little-endian host the GGUF data section *is* the F32 bit layout, so the
  decode is one `copy` (added a local `rawCopyLE`, big-endian element-wise
  fallback). F32 2890→9757 MB/s (**3.4×**, now above safetensors' 8.4 GB/s and
  within ~1.3× of gguf-py's 12.2). F16 cannot memmove (it widens F16→F32), but
  the *read* was still per-element: bulk-copy the halfwords into an aligned
  `[]uint16` then table-convert sequentially → 1.53× (the `f16ToF32` table
  lookup itself is the remaining, inherently scattered, per-element cost).
  Class-audit lesson: when a bulk-decode fix lands, apply it to **every** sibling
  reader — one straggler carried the slow path for a year.

### 5. Bulk encode instead of per-element Buffer.Write
- `format/safetensors/safetensors.go` Save paid a `bytes.Buffer.Write` call
  per 2–8-byte element AND materialized the whole data section in memory.
  Now: offsets are computed in a first pass, the header is emitted, then each
  tensor is encoded through a fixed 256 KiB scratch chunk written directly to
  w. SaveF32 1M: 3.60→1.44 ms (2.5×), 12.6→4.5 MB and 33→15 allocs per call.
- `format/npy/npy.go` writeData: same per-element `bufio.Write` pattern,
  same chunked fix. SaveF32 1M: 3.38→1.28 ms (2.6×).

### 6. Chunked streaming instead of full-size intermediate buffers
`internal/npy/npy.go` Read/Write moved from "allocate payload-sized []byte,
convert in a second pass" to streaming through a fixed 256 KiB chunk.
Time is a wash to slightly better (WriteF64 1.85→1.61 ms measured
back-to-back; ReadF32/F64 within noise), but transient memory halves
(Read 16.8→8.7 MB per 1M-f64 call; at the §V15 numel cap the old code
allocated payload×2 ≈ 512 MB, now payload + 256 KiB). Chunk-size probes:
64 KiB was worse for f64 (more ReadFull round-trips than the decode
amortizes); 256 KiB is the sweet spot here.

### 7. Decode-loop form: the multiplied-index form is already optimal
Micro-benchmarked four forms of the `[]byte`→`[]float64` loop (1M elems):
open `buf[i*8:]` 322 µs ≈ two-index `buf[i*8:i*8+8]` ≈ capped-hint
`p := buf[:8*n:8*n]` — but the pointer-walk form `p = p[8:]` per element was
1.7× SLOWER (563 µs): it serializes the loop on the slice-header update.
Keep `binary.LittleEndian.UintNN(buf[i*k:])`; on little-endian arm64/amd64 it
compiles to a bare load. **Scope:** this is the optimal form only when the loop
must actually *decode* — i.e. the on-disk type differs from the storage type
(F16→F32 widening, quant unpacking). For a **verbatim-bit** dtype (F32→F32,
F64→F64) the whole loop is unnecessary: the LE bytes already are the layout, so
`rawCopyLE` (§4) is one memmove and wins ~3–4× outright. Reach for the
per-element form only after ruling out the bulk copy.

## Bench-hygiene note
Two sessions of contradictory chunking numbers (npy §6) traced to thermal /
concurrent-load drift between runs, NOT to the code: an "A" measured minutes
before its "B" on a loaded machine can lie by 10–30%. For close calls, put
old and new implementations in the SAME bench binary (temporary
`zz_ab_test.go` with a copied old function + parity test) and interleave.

## Security fixes found during the pass (§V15 class)
FOUR parsers capped a hostile shape/dim product with a POST-multiply check —
which itself wraps the accumulator (2^24·2^40 = 2^64 ≡ 0 in uint64;
2^30·2^34 = 2^64 ≡ 0 in int64; 2^40·2^40 = 2^80 ≡ 0), so a crafted header
passed the cap and the loader returned a nil-error tensor whose shape claims
2^64+ elements over EMPTY storage (downstream indexing via shape.Numel()
would then read out of bounds). Fixed with the division-form guard
`if d != 0 && numel > cap/d { reject }` BEFORE multiplying — the same
discipline as the §B47 subtraction-form offset checks:
- `internal/npy/npy.go` (regression: `TestHostileNumelOverflow`)
- `format/npy/npy.go` (regression: `TestHostileNumelOverflow`)
- `format/safetensors/safetensors.go` (regression: `TestHostileNumelOverflow`)
- `format/gguf/gguf.go` (regression: `TestHostileDimProductOverflow`)
Class-audit note (§integration-audit): the pattern recurred across every
sibling parser — any future size/count cap must use the division (or
subtraction) form, never check after the arithmetic.

## Session: per-element odometers and row-slice matrices (perfscan PS4005/PS4006)

Eighteen measured wins, all bit-identical by construction (index arithmetic or
traversal reordering only — no operand or ordering change), each an interleaved
A/B with a file-copy toggle and an unaffected control benchmark in the same run.

| site | before → after | factor |
|---|---|---|
| `backend/cpu` general broadcast | 1303174 → 246575 ns | 5.29× |
| `backend/einsum` byte-keyed maps → `[256]` arrays | 4608760 → 1002231 ns | 4.60× |
| `backend/ref` IPO loss devirtualized | 37183 → 5026 ns | 7.40× |
| `backend/ref` solvespd flat + devirt | 354615 → 165190 ns | 2.15× |
| `tensor` strided gather (`gatherCast`) | 644021 → 205342 ns | 3.14× |
| `backend/ref` argmax-axis | 1309029 → 733543 ns | 1.78× |
| `backend/ref` DPO/KTO devirtualized | 163676 → 104032 ns | 1.57× |
| `backend/ref` cholesky flat factor | 496002 → 332019 ns | 1.50× |
| `backend/ref` GRPO / CPO | 209776 → 136247 / 106298 → 71461 ns | 1.54× / 1.49× |
| `backend/ref` qr flat factor | 22358 → 16607 ns | 1.35× |
| `backend/ref` PPO-clip devirtualized | 138116 → 112036 ns | 1.23× |
| `linalg` Pinv devirtualized | 1369432 → 1131386 ns | 1.21× |
| `internal/linalg` SymEig flat Jacobi | 3207032 → 2661654 ns | 1.20× |
| `backend/ref` svd flat Jacobi | 711358 → 603454 ns | 1.18× |
| `linalg` LU.Solve shared scratch | — | 17× fewer allocs (135→8) |
| `classic` GMM PredictProba scratch | — | allocs halved (4098→2051) |

### The two patterns, and where each stops paying

**Per-element odometer (PS4005).** An N-D coordinate walk ticked once per element.
The innermost axis has a constant stride across a run, so the run is one straight
walk — a copy at stride 1, a fill at stride 0 — and the odometer need only tick per
run. Bit-identical because the innermost axis contributes
`inner*stride − stride*inner = 0` to the offset over a full run.

**Row-slice matrix (PS4006).** A `[][]T` pays one allocation per row, and a column
walk (`m[k][p]`, k varying) then dereferences one heap row per step. Flattening to
`[rows*cols]` makes it a constant stride through one allocation.

PS4006 went five-for-five and then **lost**: flattening `classic/linalg.go`
cholSolve measured 0.93×, because an OLS fit is dominated by the O(N·d²) Gram-matrix
build and the O(d³) factorization it flags is a small share of it. The flatten pays
when the flagged loop IS the enclosing operation's work. Cheaper mitigation when it
is not: hoist `row := m[i]` above the inner loop — `classic/naivebayes.go` and
`gmm.go` already do this, which is why they were declined.

### Bench-hygiene, reinforcing the note above
The same baseline measured 206765 ns standalone, 133412 ns in one interleaved round
and 163676 ns in another. Only the A-vs-B comparison WITHIN a round means anything.
Controls drifted up to 9% and twice moved opposite to the effect — which strengthens
rather than weakens a win measured under worse conditions. One early result was
discarded entirely when an untouched control moved 46.4 → 37.0 µs on its own.
