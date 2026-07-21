//go:build amd64 && goexperiment.simd

package cpu

// archsimd (AVX) GEMM band kernels (§T11b/§T74). These replace the scalar twins
// in gemm_nosimd.go under GOEXPERIMENT=simd on amd64. They vectorize the FREE
// dimension j in 4-wide f64 lanes (Float64x4) while the reduction over p stays
// scalar-ordered ascending — so every C[i][j] sums its k-products in the exact
// same order as the scalar kernel. Products use Mul THEN Add (two roundings),
// never a fused MulAdd (one rounding), so the result is BIT-IDENTICAL to the
// scalar/reference GEMM (§V3, §V11 tol 0), not merely within tolerance.
//
// The accumulator is loaded from C before the p-loop and stored after, so the
// += contract of the scalar kernel is preserved verbatim (conv im2col scatter
// depends on it, §T597). F32 has its own f32-NATIVE direct kernel below
// (ADR-0021 tolerance, FMA-gated); F64 keeps f64 accumulation (§V10) and
// bit-exactness.
//
// gemmHasAVX runtime-gates the intrinsics (§I4): built with the experiment but
// run on a pre-AVX CPU, gemmF64Band falls back to a correct scalar accumulate
// instead of faulting.

import (
	"runtime"

	"simd/archsimd"
)

var gemmHasAVX = archsimd.X86.AVX()

func gemmF64Band(A, B, C []float64, loRow, hiRow, k, n int) {
	if !gemmHasAVX {
		gemmF64BandScalar(A, B, C, loRow, hiRow, k, n)
		return
	}
	// Column blocking, same rationale as gemmF32BandDirect: keep the k×NC B
	// block L2-resident across the band's i-tiles so B is streamed from memory
	// once per band, not once per 4-row tile. Blocks are j-splits only — every
	// C[i][j] still accumulates its k products in one ascending-p pass, so the
	// result stays bit-identical (§V3, §V11 tol 0) and the += contract holds.
	const l2Target = 256 << 10
	nc := n
	if k*n*8 > l2Target {
		nc = l2Target / (8 * k) / 8 * 8 // multiple of the 8-wide j-tile
		if nc < 32 {
			nc = 32
		}
	}
	for jb := 0; jb < n; jb += nc {
		gemmF64BandCols(A, B, C, loRow, hiRow, k, n, jb, min(jb+nc, n))
	}
}

// gemmF64BandCols computes columns [jLo,jHi) of rows [loRow,hiRow). jLo is
// 8-aligned (blocks are multiples of 8); jHi==n on the last block carries the
// 4-wide and scalar tails.
func gemmF64BandCols(A, B, C []float64, loRow, hiRow, k, n, jLo, jHi int) {
	span := jHi - jLo
	nv := jLo + span - span%4  // 4-wide vector boundary
	nv8 := jLo + span - span%8 // 8-wide boundary (two Float64x4 per row)
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		// Hoisted k-length A row slices (p < k == len(ar0) eliminates the four A
		// bounds checks) and a running bo offset (strength-reduces p*n+j): same
		// address-computation-only rewrite as gemmF32BandDirect, measured there
		// to empty the loop of GPR spill/reload bookkeeping. Arithmetic order is
		// untouched, so the result stays bit-identical (§V3, §V11 tol 0).
		ar0 := A[(i+0)*k : (i+1)*k]
		ar1 := A[(i+1)*k : (i+2)*k]
		ar2 := A[(i+2)*k : (i+3)*k]
		ar3 := A[(i+3)*k : (i+4)*k]
		c0 := C[(i+0)*n : (i+0)*n+n]
		c1 := C[(i+1)*n : (i+1)*n+n]
		c2 := C[(i+2)*n : (i+2)*n+n]
		c3 := C[(i+3)*n : (i+3)*n+n]
		j := jLo
		// 8-wide body: two column groups per row = 8 independent accumulator
		// chains, enough in flight to hide the add latency (nr=4 left only 4).
		for ; j < nv8; j += 8 {
			a0 := archsimd.LoadFloat64x4Slice(c0[j:])
			a0h := archsimd.LoadFloat64x4Slice(c0[j+4:])
			a1 := archsimd.LoadFloat64x4Slice(c1[j:])
			a1h := archsimd.LoadFloat64x4Slice(c1[j+4:])
			a2 := archsimd.LoadFloat64x4Slice(c2[j:])
			a2h := archsimd.LoadFloat64x4Slice(c2[j+4:])
			a3 := archsimd.LoadFloat64x4Slice(c3[j:])
			a3h := archsimd.LoadFloat64x4Slice(c3[j+4:])
			bo := j
			for p := 0; p < k; p++ {
				lo := archsimd.LoadFloat64x4Slice(B[bo:])
				hi := archsimd.LoadFloat64x4Slice(B[bo+4:])
				b0 := archsimd.BroadcastFloat64x4(ar0[p])
				b1 := archsimd.BroadcastFloat64x4(ar1[p])
				b2 := archsimd.BroadcastFloat64x4(ar2[p])
				b3 := archsimd.BroadcastFloat64x4(ar3[p])
				a0 = a0.Add(b0.Mul(lo))
				a0h = a0h.Add(b0.Mul(hi))
				a1 = a1.Add(b1.Mul(lo))
				a1h = a1h.Add(b1.Mul(hi))
				a2 = a2.Add(b2.Mul(lo))
				a2h = a2h.Add(b2.Mul(hi))
				a3 = a3.Add(b3.Mul(lo))
				a3h = a3h.Add(b3.Mul(hi))
				bo += n
			}
			a0.StoreSlice(c0[j:])
			a0h.StoreSlice(c0[j+4:])
			a1.StoreSlice(c1[j:])
			a1h.StoreSlice(c1[j+4:])
			a2.StoreSlice(c2[j:])
			a2h.StoreSlice(c2[j+4:])
			a3.StoreSlice(c3[j:])
			a3h.StoreSlice(c3[j+4:])
		}
		for ; j < nv; j += 4 { // 4-wide cleanup (n%8 in [4,7])
			acc0 := archsimd.LoadFloat64x4Slice(c0[j:])
			acc1 := archsimd.LoadFloat64x4Slice(c1[j:])
			acc2 := archsimd.LoadFloat64x4Slice(c2[j:])
			acc3 := archsimd.LoadFloat64x4Slice(c3[j:])
			bo := j
			for p := 0; p < k; p++ {
				bv := archsimd.LoadFloat64x4Slice(B[bo:])
				acc0 = acc0.Add(archsimd.BroadcastFloat64x4(ar0[p]).Mul(bv))
				acc1 = acc1.Add(archsimd.BroadcastFloat64x4(ar1[p]).Mul(bv))
				acc2 = acc2.Add(archsimd.BroadcastFloat64x4(ar2[p]).Mul(bv))
				acc3 = acc3.Add(archsimd.BroadcastFloat64x4(ar3[p]).Mul(bv))
				bo += n
			}
			acc0.StoreSlice(c0[j:])
			acc1.StoreSlice(c1[j:])
			acc2.StoreSlice(c2[j:])
			acc3.StoreSlice(c3[j:])
		}
		for ; j < jHi; j++ { // scalar column tail (span%4)
			s0, s1, s2, s3 := c0[j], c1[j], c2[j], c3[j]
			bo := j
			for p := 0; p < k; p++ {
				bv := B[bo]
				s0 += ar0[p] * bv
				s1 += ar1[p] * bv
				s2 += ar2[p] * bv
				s3 += ar3[p] * bv
				bo += n
			}
			c0[j], c1[j], c2[j], c3[j] = s0, s1, s2, s3
		}
	}
	for ; i < hiRow; i++ { // remainder rows
		ci := C[i*n : i*n+n]
		ai := A[i*k : (i+1)*k]
		j := jLo
		for ; j < nv; j += 4 {
			acc := archsimd.LoadFloat64x4Slice(ci[j:])
			bo := j
			for p := 0; p < k; p++ {
				acc = acc.Add(archsimd.BroadcastFloat64x4(ai[p]).Mul(archsimd.LoadFloat64x4Slice(B[bo:])))
				bo += n
			}
			acc.StoreSlice(ci[j:])
		}
		for ; j < jHi; j++ {
			s := ci[j]
			bo := j
			for p := 0; p < k; p++ {
				s += ai[p] * B[bo]
				bo += n
			}
			ci[j] = s
		}
	}
}

// gemmF32Band stays the 4-row-blocked SCALAR kernel under the experiment too.
// The obvious f64-accumulating SIMD twin (load f32 lane, ConvertToFloat64,
// f64x4 Mul/Add) was BUILT and MEASURED here and REGRESSED ~25× (512³/1024³:
// ~43 -> ~1.7 GFLOP/s): the per-iteration 128-bit LoadFloat32x4Slice +
// ConvertToFloat64 widen in the inner loop is pathological on this path.
// DISCARDED per §C3/V-CGO (never ship a non-winning opt). Moreover f64
// accumulation caps f32 at the f64 throughput (~63 GFLOP/s, cf gemmF64Band),
// nowhere near vendor-BLAS SGEMM — the real f32 win needs f32-NATIVE 8-wide
// accumulation (Float32x8 + MulAdd), which changes the §V10 f64-accum policy
// and so is deferred to its own task with an ADR + tolerance parity. Keeping
// the blocked scalar here means F32 GEMM is unchanged (bit-exact, no regress).
// gemmHasFMA gates the f32-NATIVE kernel: MulAdd is an FMA, and f32-native
// accumulation only pays off with it. Without FMA (pre-Haswell), fall back to
// the bit-exact f64-accumulating scalar.
var gemmHasFMA = archsimd.X86.FMA()

// gemmF32Band: f32-NATIVE 8-wide accumulation (Float32x8 + MulAdd). Unlike the
// f64-accumulating kernels this is NOT bit-exact to the f64 reference — it
// accumulates products in f32 (the vendor-BLAS SGEMM convention), which trades
// §V10's f64 accumulation for a tolerance in exchange for ~2× f32 lane density
// and the FMA. The store widens each Float32x8 to the f64 carrier ONCE per tile
// (amortized over k), so there is no per-iteration convert (contrast the
// discarded f64-accumulating twin). acc is the zeroed scratch from
// matmulKernel; each row-band writes disjoint rows once, so results are stored,
// not accumulated. §V16 accuracy: f32 dot-product error ~ K·eps_f32 worst case,
// gated by the tolerance parity test.
// gemmF32 computes C[m,n]=A·B, f32. With FMA it runs the f32-native SIMD kernel
// writing C DIRECTLY — no f64 accumulation scratch and no narrowing pass, which
// were ~pure overhead for f32-native (doubled store traffic + a full extra pass).
// Without FMA it falls back to the bit-exact f64-accumulating scalar. The
// f32-native result is within the ADR-0021 tolerance of ref, not bit-exact.
func gemmF32(A, B, C []float32, m, k, n int) {
	if !gemmHasFMA {
		accP := getF64(m * n) // pooled zeroed f64 accumulation scratch (§V10, §T463)
		acc := *accP
		parallelWork(m, k*n, func(loRow, hiRow int) {
			gemmF32BandScalarF64(A, B, acc, loRow, hiRow, k, n)
		})
		for i := range C {
			C[i] = float32(acc[i])
		}
		putF64(accP)
		return
	}
	if m < 4 {
		// Small-m (decode GEMV shapes, m ≤ 3): row-parallel gives ≤3 workers and
		// only the slow 1-row remainder kernel. Parallelize over COLUMN blocks
		// instead — each worker owns a j-range of C, so writes stay disjoint and
		// every core streams its own slice of B.
		//
		// A GEMV is memory-bandwidth-bound (each B element is read once, no reuse),
		// so throughput scales with the number of CORES streaming B — the block size
		// must yield ≈one block PER WORKER, not a fixed width. A fixed 512 gave only
		// ceil(n/512) blocks (e.g. 4 for the common n=2048), leaving most cores idle;
		// sizing jblk to ≈n/workers puts every core to work. Floored to keep the
		// 32-wide tiles whole + the per-worker B slice sensible, capped so a large n
		// still streams an L2/L3-friendly k×jblk×4B slice per worker. Measured
		// (n=2048, 16 cores): 14 → 27 GFLOP/s = +87%, no change to the fits-in-tile
		// GEMM path. Block ranges stay disjoint, so the result is bit-identical.
		workers := runtime.GOMAXPROCS(0)
		if workers < 1 {
			workers = 1
		}
		jblk := (n + workers - 1) / workers
		jblk = (jblk + 31) &^ 31 // whole 32-wide j-tiles
		if jblk < 64 {
			jblk = 64
		} else if jblk > 512 {
			jblk = 512
		}
		blocks := (n + jblk - 1) / jblk
		parallelWork(blocks, m*k*jblk, func(loB, hiB int) {
			gemmF32SmallM(A, B, C, loB*jblk, min(hiB*jblk, n), m, k, n)
		})
		return
	}
	// m ≥ 6, n ≥ 16: the hand-asm 6×16 AVX2 tile (12 YMM accumulators — past the archsimd
	// allocator ceiling, golang/go#76969) roughly doubles the register tile's arithmetic
	// intensity; the m%6 / n%16 edges fall back to the 4×16 band kernel below.
	if gemmAsmF32Ok(m, n) {
		gemmF32Asm6x16(A, B, C, m, k, n)
		return
	}
	// m ≥ 4: the fast path is the 4-row register tile (gemmF32BandDirect reuses each
	// B load across 4 rows). Parallelising over ROWS alone starves it — a worker handed
	// 1–3 leftover rows can't form a 4-row tile and falls to the no-B-reuse single-row
	// remainder, and when m < 4·workers most cores stay idle. So grain over 4-ROW TILES,
	// and — when there are fewer tiles than workers (small m) — also split COLUMNS, so
	// every core runs a full 4-row tile with B-reuse. This lifts the medium-m regime
	// (chunked prefill / batched + speculative decode) 2–3× (e.g. m=32 35→66, m=48 38→117
	// GFLOP/s @2048²) with no change to the large-m GEMM (m=512 stays ≈230). Each
	// (tile,col-block) range writes a disjoint C region and still accumulates each C[i][j]
	// in one ascending-p pass, so the result is bit-identical.
	rowTiles := (m + 3) / 4
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	colBlocks := 1
	if rowTiles < workers {
		colBlocks = (workers + rowTiles - 1) / rowTiles
	}
	if colBlocks == 1 {
		// Enough row tiles to fill every core — grain over tiles only.
		parallelWork(rowTiles, 4*k*n, func(loT, hiT int) {
			gemmF32BandDirect(A, B, C, loT*4, min(hiT*4, m), k, n)
		})
		return
	}
	jblk := (n + colBlocks - 1) / colBlocks
	jblk = (jblk + 15) &^ 15 // whole 16-wide j-tiles
	if jblk < 16 {
		jblk = 16
	}
	colBlocks = (n + jblk - 1) / jblk
	grains := rowTiles * colBlocks
	parallelWork(grains, 4*k*jblk, func(loG, hiG int) {
		for g := loG; g < hiG; g++ {
			rt, cb := g/colBlocks, g%colBlocks
			jLo := cb * jblk
			gemmF32BandDirectCols(A, B, C, rt*4, min(rt*4+4, m), k, n, jLo, min(jLo+jblk, n))
		}
	})
}

// gemmF32SmallM computes C[i][jLo:jHi] for ALL rows i < m (m ≤ 3), f32-native.
// Per row: 32-wide j-tiles = 4 independent FMA chains to hide latency. B is
// walked with stride n between p steps, but a 32-col tile covers two whole 64B
// lines per step, so every fetched line is fully consumed and the k×128B
// per-tile footprint streams through L2. Same ADR-0021 f32 accumulation as
// gemmF32BandDirect.
func gemmF32SmallM(A, B, C []float32, jLo, jHi, m, k, n int) {
	span := jHi - jLo
	nv32 := jLo + span - span%32
	nv := jLo + span - span%8
	for i := 0; i < m; i++ {
		ai := A[i*k : (i+1)*k]
		ci := C[i*n : (i+1)*n]
		j := jLo
		for ; j < nv32; j += 32 {
			s0 := archsimd.BroadcastFloat32x8(0)
			s1 := archsimd.BroadcastFloat32x8(0)
			s2 := archsimd.BroadcastFloat32x8(0)
			s3 := archsimd.BroadcastFloat32x8(0)
			bo := j
			for p := 0; p < k; p++ {
				b := archsimd.BroadcastFloat32x8(ai[p])
				s0 = b.MulAdd(archsimd.LoadFloat32x8Slice(B[bo:]), s0)
				s1 = b.MulAdd(archsimd.LoadFloat32x8Slice(B[bo+8:]), s1)
				s2 = b.MulAdd(archsimd.LoadFloat32x8Slice(B[bo+16:]), s2)
				s3 = b.MulAdd(archsimd.LoadFloat32x8Slice(B[bo+24:]), s3)
				bo += n
			}
			s0.StoreSlice(ci[j:])
			s1.StoreSlice(ci[j+8:])
			s2.StoreSlice(ci[j+16:])
			s3.StoreSlice(ci[j+24:])
		}
		for ; j < nv; j += 8 {
			s := archsimd.BroadcastFloat32x8(0)
			bo := j
			for p := 0; p < k; p++ {
				s = archsimd.BroadcastFloat32x8(ai[p]).MulAdd(archsimd.LoadFloat32x8Slice(B[bo:]), s)
				bo += n
			}
			s.StoreSlice(ci[j:])
		}
		for ; j < jHi; j++ {
			var t float32
			bo := j
			for p := 0; p < k; p++ {
				t += ai[p] * B[bo]
				bo += n
			}
			ci[j] = t
		}
	}
}

// gemmF32BandDirect: f32-NATIVE 8-wide accumulation (Float32x8 + MulAdd) writing
// rows [loRow,hiRow) of C in f32 DIRECTLY (no f64 carrier). Each C[i][j] sums its
// k products in f32 (ADR-0021 tolerance). mr=4 × nr=16 (8 chains); C rows written
// once (disjoint per row-band).
//
// Cache blocking: when the k×n B panel outgrows L2, each 4-row i-tile streaming
// all of B evicts it for the next tile — B then comes from L3/memory (hiRow-
// loRow)/4 times per band. Splitting the columns into NC-wide blocks sized so
// k×NC×4B ≈ half of L2 (512 KiB on Zen 3) keeps the block resident across all
// i-tiles of the band: B is read from memory once per band instead of once per
// i-tile. Column blocking does not touch the per-element accumulation order,
// and every (i,j) tile is still computed in one pass (store-once), so numerics
// are unchanged.
func gemmF32BandDirect(A, B, C []float32, loRow, hiRow, k, n int) {
	const l2Target = 256 << 10 // bytes of L2 to budget for the B block
	nc := n
	if k*n*4 > l2Target {
		nc = l2Target / (4 * k) / 16 * 16 // multiple of the 16-wide j-tile
		if nc < 64 {
			nc = 64
		}
	}
	for jb := 0; jb < n; jb += nc {
		gemmF32BandDirectCols(A, B, C, loRow, hiRow, k, n, jb, min(jb+nc, n))
	}
}

// gemmF32BandDirectCols computes columns [jLo,jHi) of rows [loRow,hiRow).
// jLo is 16-aligned (blocks are multiples of 16); jHi==n on the last block
// carries the 8-wide and scalar tails.
//
// mr=4 × nr=16 (8 accumulators) is the register-blocking ceiling for the
// current Go SIMD compiler: its allocator only uses Y0–Y14 (golang/go#76969,
// closed not-planned), and an mr=6 × nr=16 variant (12 accumulators + 2 B
// vectors + rotating broadcast) was BUILT and MEASURED here — the inner loop
// picked up 33 SP-relative spill ops and 512³ dropped 198→102 GFLOP/s.
// REVERTED per §C3. Re-try only after the upstream allocator can keep ≥15
// vectors live.
func gemmF32BandDirectCols(A, B, C []float32, loRow, hiRow, k, n, jLo, jHi int) {
	span := jHi - jLo
	nv16 := jLo + span - span%16
	nv := jLo + span - span%8
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		// Hoisted k-length A row slices: the p-loop condition p < k == len(a0)
		// lets the compiler eliminate all four A bounds checks, and the running
		// bo offset strength-reduces p*n+j — together they empty the inner loop
		// of GPR spill/reload bookkeeping (measured on the SSA dump: r0..r3 and
		// the B header were reloaded from SP every iteration before).
		a0 := A[(i+0)*k : (i+1)*k]
		a1 := A[(i+1)*k : (i+2)*k]
		a2 := A[(i+2)*k : (i+3)*k]
		a3 := A[(i+3)*k : (i+4)*k]
		c0 := C[(i+0)*n : (i+0)*n+n]
		c1 := C[(i+1)*n : (i+1)*n+n]
		c2 := C[(i+2)*n : (i+2)*n+n]
		c3 := C[(i+3)*n : (i+3)*n+n]
		j := jLo
		for ; j < nv16; j += 16 {
			s0 := archsimd.BroadcastFloat32x8(0)
			s0h := archsimd.BroadcastFloat32x8(0)
			s1 := archsimd.BroadcastFloat32x8(0)
			s1h := archsimd.BroadcastFloat32x8(0)
			s2 := archsimd.BroadcastFloat32x8(0)
			s2h := archsimd.BroadcastFloat32x8(0)
			s3 := archsimd.BroadcastFloat32x8(0)
			s3h := archsimd.BroadcastFloat32x8(0)
			bo := j
			for p := 0; p < k; p++ {
				lo := archsimd.LoadFloat32x8Slice(B[bo:])
				hi := archsimd.LoadFloat32x8Slice(B[bo+8:])
				b0 := archsimd.BroadcastFloat32x8(a0[p])
				b1 := archsimd.BroadcastFloat32x8(a1[p])
				b2 := archsimd.BroadcastFloat32x8(a2[p])
				b3 := archsimd.BroadcastFloat32x8(a3[p])
				s0 = b0.MulAdd(lo, s0)
				s0h = b0.MulAdd(hi, s0h)
				s1 = b1.MulAdd(lo, s1)
				s1h = b1.MulAdd(hi, s1h)
				s2 = b2.MulAdd(lo, s2)
				s2h = b2.MulAdd(hi, s2h)
				s3 = b3.MulAdd(lo, s3)
				s3h = b3.MulAdd(hi, s3h)
				bo += n
			}
			s0.StoreSlice(c0[j:])
			s0h.StoreSlice(c0[j+8:])
			s1.StoreSlice(c1[j:])
			s1h.StoreSlice(c1[j+8:])
			s2.StoreSlice(c2[j:])
			s2h.StoreSlice(c2[j+8:])
			s3.StoreSlice(c3[j:])
			s3h.StoreSlice(c3[j+8:])
		}
		for ; j < nv; j += 8 {
			s0 := archsimd.BroadcastFloat32x8(0)
			s1 := archsimd.BroadcastFloat32x8(0)
			s2 := archsimd.BroadcastFloat32x8(0)
			s3 := archsimd.BroadcastFloat32x8(0)
			bo := j
			for p := 0; p < k; p++ {
				bv := archsimd.LoadFloat32x8Slice(B[bo:])
				s0 = archsimd.BroadcastFloat32x8(a0[p]).MulAdd(bv, s0)
				s1 = archsimd.BroadcastFloat32x8(a1[p]).MulAdd(bv, s1)
				s2 = archsimd.BroadcastFloat32x8(a2[p]).MulAdd(bv, s2)
				s3 = archsimd.BroadcastFloat32x8(a3[p]).MulAdd(bv, s3)
				bo += n
			}
			s0.StoreSlice(c0[j:])
			s1.StoreSlice(c1[j:])
			s2.StoreSlice(c2[j:])
			s3.StoreSlice(c3[j:])
		}
		for ; j < jHi; j++ { // f32-native scalar tail (span%8)
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
			c0[j], c1[j], c2[j], c3[j] = t0, t1, t2, t3
		}
	}
	for ; i < hiRow; i++ {
		ci := C[i*n : i*n+n]
		ai := A[i*k : (i+1)*k]
		j := jLo
		for ; j < nv; j += 8 {
			s := archsimd.BroadcastFloat32x8(0)
			bo := j
			for p := 0; p < k; p++ {
				s = archsimd.BroadcastFloat32x8(ai[p]).MulAdd(archsimd.LoadFloat32x8Slice(B[bo:]), s)
				bo += n
			}
			s.StoreSlice(ci[j:])
		}
		for ; j < jHi; j++ {
			var t float32
			bo := j
			for p := 0; p < k; p++ {
				t += ai[p] * B[bo]
				bo += n
			}
			ci[j] = t
		}
	}
}

// gemmF32BandScalarF64 is the bit-exact f64-accumulating fallback (no FMA).
func gemmF32BandScalarF64(A, B []float32, acc []float64, loRow, hiRow, k, n int) {
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		c0 := acc[(i+0)*n : (i+1)*n]
		c1 := acc[(i+1)*n : (i+2)*n]
		c2 := acc[(i+2)*n : (i+3)*n]
		c3 := acc[(i+3)*n : (i+4)*n]
		for p := range k {
			bp := B[p*n : (p+1)*n]
			a0 := float64(A[(i+0)*k+p])
			a1 := float64(A[(i+1)*k+p])
			a2 := float64(A[(i+2)*k+p])
			a3 := float64(A[(i+3)*k+p])
			for j, bv := range bp {
				bf := float64(bv)
				c0[j] += a0 * bf
				c1[j] += a1 * bf
				c2[j] += a2 * bf
				c3[j] += a3 * bf
			}
		}
	}
	for ; i < hiRow; i++ {
		ci := acc[i*n : (i+1)*n]
		for p := range k {
			aip := float64(A[i*k+p])
			bp := B[p*n : (p+1)*n]
			for j, bv := range bp {
				ci[j] += aip * float64(bv)
			}
		}
	}
}

// gemmF64BandScalar is the fallback for the (vanishingly rare) pre-AVX amd64
// CPU under the simd experiment. Unblocked; correctness only, identical +=.
func gemmF64BandScalar(A, B, C []float64, loRow, hiRow, k, n int) {
	for i := loRow; i < hiRow; i++ {
		ci := C[i*n : i*n+n]
		for p := 0; p < k; p++ {
			aip := A[i*k+p]
			bp := B[p*n : p*n+n]
			for j, bv := range bp {
				ci[j] += aip * bv
			}
		}
	}
}
