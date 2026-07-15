//go:build goexperiment.simd

package cpu

// f32-NATIVE NEON GEMM for arm64 under GOEXPERIMENT=simd (ADR-0026, the arm64
// twin of ADR-0021). The Go 1.26 arm64 backend does NOT auto-vectorize the
// scalar SAXPY loop (verified on the objdump: scalar FMADDS), and the archsimd
// package is amd64-only, so the dense inner tile is a small Plan9 NEON kernel
// (gemm_neon_arm64.s): a 4×16 register tile of sixteen 4-lane FMLA accumulator
// chains. Like the amd64 kernel it accumulates in f32 (vendor-SGEMM
// convention) — a scoped §V10 amendment gated by the same build tag and the
// same ADR-0021 tolerance parity (gemm_f32policy_arm64simd_test.go): each C
// element still sums its k products in ascending p order, in ONE fused-FMA
// chain, so the error carries the usual K·u_f32 bound. The default arm64 build
// is untouched and stays bit-exact (gemm_f32default.go). F64 keeps the shared
// scalar band kernel (gemm_nosimd.go) in both builds — bit-exact everywhere.
//
// Measured on Apple M2 Pro (8P+4E, §V22 A/B -count=6), 256³/512³/1024³:
// 46/52/61 GFLOP/s scalar baseline → ~430/~705/~795 GFLOP/s = 9.4×/13.6×/13×;
// LLM shapes 512×2048×2048 69→~690, 512×2048×8192 60→~578. A pure-Go
// f32-native scalar kernel (no asm) was measured first per the prefer-Go rule
// and reached only ~76 GFLOP/s @1024³ (scalar FMADDS ceiling) — kept as the
// tail/fallback path below, superseded by the NEON tile for the dense interior.

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// gemmF32Tile4x16Neon computes the dense 4×16 store-once tile; see the .s file.
// k must be ≥ 1.
//
//go:noescape
func gemmF32Tile4x16Neon(a, b, c *float32, k, lda, ldb, ldc int)

// The Apple AMX fast-path hooks (ADR-0027). Either may be nil depending on
// the build/host; gemmF32 dispatches per-shape to the measured winner and
// falls back to the NEON path below when both are nil (e.g. CGO_ENABLED=0 on
// a non-Apple arm64 host) — fully functional either way. Both fast paths
// accumulate f32-native like the NEON kernel, so the same ADR-0021 tolerance
// contract applies; the default (non-simd) build never sees any of this.
var (
	// gemmF32Accel: Accelerate cblas_sgemm (AMX via Apple's BLAS), set by
	// gemm_accel_darwin.go under darwin && cgo. Handles the whole matmul
	// (self-threads) when invoked.
	gemmF32Accel func(a, b, c []float32, m, k, n int) bool
	// gemmF32AMX: raw AMX via Plan9 asm (pure Go, no cgo), set by
	// gemm_amx_arm64.go under darwin when the chip is M1/M2/M3.
	gemmF32AMX func(a, b, c []float32, m, k, n int) bool
	// gemmF32GemvAccel: Accelerate cblas_sgemv y[n] = x[k]·w[k,n] for the
	// decode (m<3) shapes, set alongside gemmF32Accel. GEMV is bandwidth-
	// bound; the per-row BLAS2 call ties cblas_sgemm at m=1 and beats its
	// small-m hole 1.40× at m=2 (§V22 A/B in gemm_gemv_bench_test.go).
	gemmF32GemvAccel func(w, x, y []float32, k, n int)
)

// gemmF32 computes C[m,n] = A·B, f32-native (ADR-0026 tolerance, not
// bit-exact). C is written once (store semantics, matching the other gemmF32
// implementations). Parallel grain mirrors gemm_simd.go: 4-row tiles, with a
// column split when there are fewer tiles than workers so medium-m shapes
// (chunked prefill, batched decode) still fill every core.
func gemmF32(A, B, C []float32, m, k, n int) {
	if k == 0 { // degenerate: A[m,0]·B[0,n] = 0; the asm k-loop is do-while
		clear(C[:m*n])
		return
	}
	if m < 3 && m > 0 && n > 0 && gemmF32GemvAccel != nil {
		// Decode shapes: cblas_sgemv per output row. §V22 A/B vs cblas_sgemm
		// at the same shape (gemm_gemv_bench_test.go, M2 Pro medians): m=1 is
		// a tie (~40 GFLOP/s @ k=n=2048 — sgemm's m=1 path IS Apple's GEMV,
		// both bandwidth-bound), but sgemm has a hole at m=2 (27.6 vs 38.7 =
		// 1.40× for per-row sgemv); from m=3 up sgemm wins again (41.8 vs
		// 39.0) — hence the m<3 cutoff. The BLAS2 call is also the honest
		// primitive for m=1, immune to future sgemm small-m regressions.
		for i := 0; i < m; i++ {
			gemmF32GemvAccel(B, A[i*k:(i+1)*k], C[i*n:(i+1)*n], k, n)
		}
		return
	}
	if m > 0 && n > 0 {
		// ADR-0027 per-shape dispatch (M2 Pro head-to-head, §V22 medians in
		// the ADR): raw AMX wins once the k×n B panel outgrows the shared L2
		// and there are enough row tiles to amortize its packing passes —
		// Accelerate falls off there (512×2048×4096: 1606 vs 1300 GFLOP/s)
		// while the prepacked supertiled AMX kernel keeps streaming. On
		// everything smaller Accelerate wins (squares ≤1024³, small/decode
		// shapes). Without cgo, raw AMX takes every shape it supports
		// (m,n ≥ 32 — it beats NEON from 64³ up) and NEON serves the rest.
		if gemmF32AMX != nil && ((k*n >= 4<<20 && m >= 256) || gemmF32Accel == nil) &&
			gemmF32AMX(A, B, C, m, k, n) {
			return
		}
		if gemmF32Accel != nil && gemmF32Accel(A, B, C, m, k, n) {
			return
		}
	}
	if m < 4 {
		// Decode/GEMV shapes: too few rows for a 4-row tile — parallelize over
		// COLUMN blocks so every core streams its own slice of B (bandwidth-
		// bound; block count ≈ workers, same rationale as gemm_simd.go).
		workers := max(runtime.GOMAXPROCS(0), 1)
		jblk := (n + workers - 1) / workers
		jblk = (jblk + 15) &^ 15
		jblk = min(max(jblk, 64), 512)
		blocks := (n + jblk - 1) / jblk
		parallelWork(blocks, m*k*jblk, func(loB, hiB int) {
			gemmF32RowsScalar(A, B, C, 0, m, k, n, loB*jblk, min(hiB*jblk, n))
		})
		return
	}
	// Pack B's full 16-column panels once, panel-major [n/16][k][16], so the
	// tile kernel streams B CONTIGUOUSLY (ldb=16). Unpacked, the tile walks B
	// with an n-element row stride; at the power-of-two n of every LLM shape
	// (1024/2048/8192) that stride lands on 2–4 L1 sets and the whole k-deep
	// column of lines conflict-misses on every tile pass. Packing costs one
	// parallel pass over B (2·k·n moves), amortized over m/4 tile passes —
	// skipped for tiny m where the GEMM itself is one pass, and for B under
	// ~1 MiB where the panel is cache-resident anyway and the pack pass is the
	// measured net loss (256³: 305 → 262 packed; 512³ neutral; 1024³ +25%).
	panels := n / 16
	var packed []float32
	var packP *[]float32
	if panels > 0 && m >= 8 && k*n*4 > 1<<20 {
		packP = getF32Pack(panels * k * 16)
		packed = *packP
		parallelWork(panels, 2*k*16, func(loP, hiP int) {
			packBPanels(B, packed, loP, hiP, k, n)
		})
	}
	rowTiles := (m + 3) / 4
	workers := max(runtime.GOMAXPROCS(0), 1)
	colBlocks := 1
	if rowTiles < workers {
		colBlocks = (workers + rowTiles - 1) / rowTiles
	}
	if colBlocks == 1 {
		// Dynamic tile scheduling: static equal chunks leave the wall clock to
		// the slowest core — on asymmetric M-series (8P+4E) the E-core chunks
		// finish last while P-cores idle. Run one pool task per worker, each
		// pulling BATCHES of row tiles off an atomic counter, so fast cores
		// naturally take more tiles.
		var next atomic.Int64
		const batch = 2
		parallelWork(workers, max(rowTiles/workers, 1)*4*k*n, func(_, _ int) {
			for {
				t := int(next.Add(batch)) - batch
				if t >= rowTiles {
					return
				}
				loRow := t * 4
				hiRow := min((t+batch)*4, m)
				gemmF32BandNeon(A, B, packed, C, loRow, hiRow, k, n)
			}
		})
	} else {
		jblk := (n + colBlocks - 1) / colBlocks
		jblk = max((jblk+15)&^15, 16)
		colBlocks = (n + jblk - 1) / jblk
		grains := rowTiles * colBlocks
		parallelWork(grains, 4*k*jblk, func(loG, hiG int) {
			for g := loG; g < hiG; g++ {
				rt, cb := g/colBlocks, g%colBlocks
				jLo := cb * jblk
				gemmF32BandNeonCols(A, B, packed, C, rt*4, min(rt*4+4, m), k, n, jLo, min(jLo+jblk, n))
			}
		})
	}
	if packP != nil {
		putF32Pack(packP)
	}
}

// f32PackScratch pools the packed-B panel buffers. NOT zeroed on get — every
// element is overwritten by packBPanels before use (contrast getF64, whose
// callers accumulate).
var f32PackScratch = sync.Pool{New: func() any { b := make([]float32, 0); return &b }}

func getF32Pack(n int) *[]float32 {
	bp := f32PackScratch.Get().(*[]float32)
	if cap(*bp) < n {
		*bp = make([]float32, n)
	} else {
		*bp = (*bp)[:n]
	}
	return bp
}

func putF32Pack(bp *[]float32) { f32PackScratch.Put(bp) }

// packBPanels copies B's 16-column panels [loP,hiP) into pack, panel-major:
// pack[(panel*k+p)*16+j] = B[p*n+panel*16+j]. Strided reads, contiguous writes.
func packBPanels(B, pack []float32, loP, hiP, k, n int) {
	for panel := loP; panel < hiP; panel++ {
		src := panel * 16
		dst := pack[panel*k*16 : (panel+1)*k*16]
		for p := 0; p < k; p++ {
			copy(dst[p*16:p*16+16], B[p*n+src:p*n+src+16])
		}
	}
}

// gemmF32BandNeon computes rows [loRow,hiRow) with L2 column blocking: when
// the k×n B panel outgrows the per-core L2 budget, split the columns so the
// k×NC block (packed: its panels) stays resident across all the band's 4-row
// tiles instead of being re-streamed from memory once per tile. j-splits only —
// accumulation order per element is unchanged. Same 256 KiB budget as the
// amd64 kernel; measured on M2 Pro (pre-packing A/B): 512×2048×8192 82 → 104
// GFLOP/s, 512×2048×2048 156 → 160, squares neutral — kept for the big-panel
// shapes.
func gemmF32BandNeon(A, B, packed, C []float32, loRow, hiRow, k, n int) {
	const l2Target = 256 << 10
	nc := n
	if k*n*4 > l2Target {
		nc = max(l2Target/(4*k)/16*16, 64)
	}
	for jb := 0; jb < n; jb += nc {
		gemmF32BandNeonCols(A, B, packed, C, loRow, hiRow, k, n, jb, min(jb+nc, n))
	}
}

// gemmF32BandNeonCols computes columns [jLo,jHi) of rows [loRow,hiRow): the
// NEON tile over the 16-aligned interior — streaming the packed panel
// (ldb=16) when packed is non-nil, the raw strided B rows otherwise — and
// scalar f32-native tails for the n%16 columns and m%4 remainder rows.
// jLo is always 16-aligned (column splits are multiples of 16).
func gemmF32BandNeonCols(A, B, packed, C []float32, loRow, hiRow, k, n, jLo, jHi int) {
	span := jHi - jLo
	nv16 := jLo + span - span%16
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		if packed != nil {
			for j := jLo; j < nv16; j += 16 {
				gemmF32Tile4x16Neon(&A[i*k], &packed[(j/16)*k*16], &C[i*n+j], k, k, 16, n)
			}
		} else {
			for j := jLo; j < nv16; j += 16 {
				gemmF32Tile4x16Neon(&A[i*k], &B[j], &C[i*n+j], k, k, n, n)
			}
		}
		if nv16 < jHi { // scalar column tail, 4 rows
			a0 := A[(i+0)*k : (i+1)*k]
			a1 := A[(i+1)*k : (i+2)*k]
			a2 := A[(i+2)*k : (i+3)*k]
			a3 := A[(i+3)*k : (i+4)*k]
			for j := nv16; j < jHi; j++ {
				var t0, t1, t2, t3 float32
				bo := j
				for p := 0; p < k; p++ {
					bv := B[bo]
					t0 += a0[p] * bv
					t1 += a1[p] * bv
					t2 += a2[p] * bv
					t3 += a3[p] * bv
					bo += n
				}
				C[(i+0)*n+j], C[(i+1)*n+j], C[(i+2)*n+j], C[(i+3)*n+j] = t0, t1, t2, t3
			}
		}
	}
	if i < hiRow { // <4 remainder rows
		gemmF32RowsScalar(A, B, C, i, hiRow, k, n, jLo, jHi)
	}
}

// gemmF32RowsScalar computes C[i][jLo:jHi] for rows [loRow,hiRow), f32-native
// scalar (the pure-Go fallback the asm tile superseded): zero the span, then
// SAXPY the k rank-1 updates over the contiguous B row slice — ascending p,
// store semantics preserved via the explicit clear.
func gemmF32RowsScalar(A, B, C []float32, loRow, hiRow, k, n, jLo, jHi int) {
	for i := loRow; i < hiRow; i++ {
		ci := C[i*n+jLo : i*n+jHi]
		clear(ci)
		ai := A[i*k : (i+1)*k]
		for p := 0; p < k; p++ {
			ap := ai[p]
			bp := B[p*n+jLo : p*n+jHi]
			for j, bv := range bp {
				ci[j] += ap * bv
			}
		}
	}
}
