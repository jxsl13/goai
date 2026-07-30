//go:build !(amd64 && goexperiment.simd)

package cpu

import (
	"runtime"
	"sync"
)

// Scalar GEMM band kernels — the portable default. On amd64+GOEXPERIMENT=simd
// gemm_simd.go replaces these with archsimd (AVX) twins that hold the identical
// ascending-p accumulation order, so exactly one definition compiles per build
// and the default pure-Go result is unchanged (§V3, §V11 tol 0). Contract for
// both: C is ACCUMULATED into (+=), so callers pass a zeroed buffer for a plain
// product and a live buffer to add onto (conv im2col scatter relies on this).
//
// The gemmF32 entry point lives in gemm_f32default.go (default builds) /
// gemm_simd.go (amd64+simd) / gemm_neon_arm64.go (arm64+simd); the band kernels
// below stay in this file because the f64 path — and the f32 f64-accumulating
// default — are shared by every non-amd64-simd build, including arm64+simd
// (whose experiment only replaces the F32 matmul entry point, ADR-0026).

// gemmPackTileBlocksF32 / gemmPackTileBlocksF64 are the number of 4-row TILE BLOCKS a single band
// must get before packing B pays. The gate is expressed in blocks per band, not in rows, because
// that is the quantity the mechanism actually turns on — and stating it in rows hardcodes this
// machine's core count into the kernel.
//
// parallelWork hands each worker ceil(m/workers) rows, and the packed kernel's tile consumes them
// four at a time, so a band gets ceil(m/workers)/4 blocks. Below one block the tile never executes
// at all and every row falls to the single-row remainder, which reads B directly — the pack is then
// pure cost. Measured at k=n=512 on a 12-way host, forcing each arm, with blocks computed that way:
//
//	blocks 0 → f32 +13.59% (m=32)     blocks 1 → f32 +0.16% (m=48), +0.14% (m=64)
//	blocks 2 → f32  -8.01% (m=96)     blocks 4 → f32 -17.69% (m=192)
//	blocks 4 → f64  +6.76% (m=192)    blocks 5 → f64  -5.47% (m=256), -8.16% (m=512)
//
// So f32 needs 2 blocks and f64 5 — f64's pack moves twice the bytes for the same element count, so
// it takes more reuse to amortize. On this host those work out to m>=96 and m>=240, which is what
// the previous fixed constants encoded; expressed in blocks they now follow GOMAXPROCS instead of
// assuming twelve of them.
//
// DO NOT REFINE THIS FURTHER ON TWO AXES — that was attempted and the measurement refuses it.
//
// The open question was whether blocks-per-band and pack size together predict the outcome, since
// the f32 square case at n=64 gets 1 block yet measured -9.29% packed while a 1MB pack at 1 block
// is neutral. Sweeping the grid (BenchmarkGemmF32PackGrid) came back non-monotonic: at k*n=16384,
// one block measured -20.85% while two measured -2.64%.
//
// The reason is a third variable neither axis carries. BenchmarkGemmF32PackSameBlocks holds the
// block count at exactly 1 and the pack size at 16KB and varies ONLY the rows left over past the
// tile:
//
//	1 block + 0 remainder rows  +30.37%
//	1 block + 1 remainder row   +71.02%
//	1 block + 2 remainder rows  -28.09%
//	1 block + 3 remainder rows  -27.53%
//
// A hundred-point swing with both modelled axes pinned. Those remainder rows run the single-row
// loop, which reads B directly and gets nothing from the pack, but that alone does not explain the
// sign flip at two — the mechanism is not established. What IS established is that a gate fit to
// (blocks, pack size) would be fit to noise. The current gate is deliberately conservative: it
// declines in the region where the outcome is unpredictable, forgoing the -9.29% at n=64 rather
// than risking the +71%.
// Vars, not consts, so the sweeps in gemm_portable_bench_test.go and the parity tests can force
// either arm without a rebuild.
var (
	gemmPackTileBlocksF32 = 2
	gemmPackTileBlocksF64 = 5
)

// gemmPackBands reports whether each band will get at least minBlocks 4-row tile blocks, mirroring
// parallelWork's own partitioning — including its serial branch, where the single band gets all m
// rows and the block count is m/4 regardless of how many cores the machine has.
func gemmPackBands(m, k, n, minBlocks int) bool {
	return gemmPackBandCount(m, k, n) >= minBlocks
}

// gemmPackBandCount is the number of 4-row tile blocks a single band receives.
func gemmPackBandCount(m, k, n int) int {
	w := runtime.GOMAXPROCS(0)
	if w <= 1 || m*k*n < parThreshold {
		w = 1
	}
	return ((m + w - 1) / w) / 4
}

// gemmPackMinWorkF32 / gemmPackMinWorkF64 are the B element counts above which packing pays.
// BOTH dtypes need a gate and they need DIFFERENT ones, because packing is a cache fix and the two
// kernels stop being resident at different sizes. Swept by forcing each arm inside one binary; see
// BenchmarkGemmF32Portable / BenchmarkGemmF64Portable, square k=n, packed vs unpacked:
//
//	f32  n=32 +42.00%  n=48 +18.09%  n=64 -9.29%  n=96 -22.53%  n=128 -5.77%
//	     n=256 -17.17%  n=384 -19.00%  n=512 -17.20%  n=1024 -24.52%
//	f64  n=64 +26.16%  n=128 +1.48%  n=256 -1.71%  n=512 -7.46%  n=1024 -28.06%
//
// THE F32 GATE MOVED, and why is the point. It was first set at 1<<19 against a packed band that
// still widened both operands inside its innermost loop; at that time packing measured +2.78% at
// n=256 and break-even at 512, so the threshold sat above 1MB of B. Hoisting those widenings made
// the packed band roughly 18% faster, which moved the crossover down by two orders of magnitude —
// the same n=256 that cost 2.78% now saves 17.17%. A threshold calibrated against an older, slower
// arm is stale by construction, so re-sweep this whenever either band changes.
//
// f32 now turns over between 2304 and 4096 elements; f64, whose band has no conversions to hoist
// and is unchanged, still turns over between 16384 and 65536. Vars rather than consts so the
// parity tests and the benchmarks can force either arm.
var (
	gemmPackMinWorkF32 = 1 << 12 // 4096 elements = 16KB of f32
	gemmPackMinWorkF64 = 1 << 16 // 65536 elements = 512KB of f64
)

// gemmF64Band computes rows [loRow,hiRow) of C with 4-row register blocking
// (§T12b): each B row is loaded once and reused for four C rows, quartering
// B-traffic. Every C element still accumulates its k-products in ascending p
// order, so results stay bit-identical to the reference (§V3, §V11 tol 0).
func gemmF64Band(A, B, C []float64, loRow, hiRow, k, n int) {
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		c0 := C[(i+0)*n : (i+1)*n]
		c1 := C[(i+1)*n : (i+2)*n]
		c2 := C[(i+2)*n : (i+3)*n]
		c3 := C[(i+3)*n : (i+4)*n]
		a0r := A[(i+0)*k : (i+1)*k]
		a1r := A[(i+1)*k : (i+2)*k]
		a2r := A[(i+2)*k : (i+3)*k]
		a3r := A[(i+3)*k : (i+4)*k]
		// 4x4 register tile — see the F32 twin below for why: the p-outer form streamed C,
		// reading and writing four cells per j to issue four FMAs. Same bit-identity argument,
		// each cell accumulating over ascending p from the value already in C.
		j := 0
		for ; j+3 < n; j += 4 {
			v00, v01, v02, v03 := c0[j], c0[j+1], c0[j+2], c0[j+3]
			v10, v11, v12, v13 := c1[j], c1[j+1], c1[j+2], c1[j+3]
			v20, v21, v22, v23 := c2[j], c2[j+1], c2[j+2], c2[j+3]
			v30, v31, v32, v33 := c3[j], c3[j+1], c3[j+2], c3[j+3]
			for p := range k {
				bp := B[p*n+j : p*n+j+4]
				b0, b1, b2, b3 := bp[0], bp[1], bp[2], bp[3]
				a0 := a0r[p]
				v00 += a0 * b0
				v01 += a0 * b1
				v02 += a0 * b2
				v03 += a0 * b3
				a1 := a1r[p]
				v10 += a1 * b0
				v11 += a1 * b1
				v12 += a1 * b2
				v13 += a1 * b3
				a2 := a2r[p]
				v20 += a2 * b0
				v21 += a2 * b1
				v22 += a2 * b2
				v23 += a2 * b3
				a3 := a3r[p]
				v30 += a3 * b0
				v31 += a3 * b1
				v32 += a3 * b2
				v33 += a3 * b3
			}
			c0[j], c0[j+1], c0[j+2], c0[j+3] = v00, v01, v02, v03
			c1[j], c1[j+1], c1[j+2], c1[j+3] = v10, v11, v12, v13
			c2[j], c2[j+1], c2[j+2], c2[j+3] = v20, v21, v22, v23
			c3[j], c3[j+1], c3[j+2], c3[j+3] = v30, v31, v32, v33
		}
		if j < n { // column remainder: the original p-outer form
			for p := range k {
				bp := B[p*n : (p+1)*n]
				a0, a1 := a0r[p], a1r[p]
				a2, a3 := a2r[p], a3r[p]
				for jj := j; jj < n; jj++ {
					bv := bp[jj]
					c0[jj] += a0 * bv
					c1[jj] += a1 * bv
					c2[jj] += a2 * bv
					c3[jj] += a3 * bv
				}
			}
		}
	}
	for ; i < hiRow; i++ { // remainder rows
		ci := C[i*n : (i+1)*n]
		for p := range k {
			aip := A[i*k+p]
			bp := B[p*n : (p+1)*n]
			for j, bv := range bp {
				ci[j] += aip * bv
			}
		}
	}
}

// gemmF32Band is the F32 twin accumulating into an f64 scratch (§V10).
func gemmF32Band(A, B []float32, acc []float64, loRow, hiRow, k, n int) {
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		c0 := acc[(i+0)*n : (i+1)*n]
		c1 := acc[(i+1)*n : (i+2)*n]
		c2 := acc[(i+2)*n : (i+3)*n]
		c3 := acc[(i+3)*n : (i+4)*n]
		a0r := A[(i+0)*k : (i+1)*k]
		a1r := A[(i+1)*k : (i+2)*k]
		a2r := A[(i+2)*k : (i+3)*k]
		a3r := A[(i+3)*k : (i+4)*k]
		// 4x4 REGISTER TILE. The p-outer form below streamed the accumulator: each j step read
		// and wrote four f64 cells to perform four FMAs, so C traffic was ~16 bytes per FLOP-pair
		// and the kernel ran load/store bound rather than FMA bound. Holding a 4x4 block of C in
		// locals across the whole k pass removes that traffic entirely — per p it loads 4 A and 4
		// B values to issue 16 FMAs.
		//
		// Bit-identical, and for the same reason the p-outer form was: each C cell still
		// accumulates its k products in ASCENDING p, starting from the value already in acc (this
		// kernel accumulates into a live buffer, which the conv im2col scatter relies on).
		j := 0
		for ; j+3 < n; j += 4 {
			v00, v01, v02, v03 := c0[j], c0[j+1], c0[j+2], c0[j+3]
			v10, v11, v12, v13 := c1[j], c1[j+1], c1[j+2], c1[j+3]
			v20, v21, v22, v23 := c2[j], c2[j+1], c2[j+2], c2[j+3]
			v30, v31, v32, v33 := c3[j], c3[j+1], c3[j+2], c3[j+3]
			for p := range k {
				bp := B[p*n+j : p*n+j+4]
				b0, b1 := float64(bp[0]), float64(bp[1])
				b2, b3 := float64(bp[2]), float64(bp[3])
				a0 := float64(a0r[p])
				v00 += a0 * b0
				v01 += a0 * b1
				v02 += a0 * b2
				v03 += a0 * b3
				a1 := float64(a1r[p])
				v10 += a1 * b0
				v11 += a1 * b1
				v12 += a1 * b2
				v13 += a1 * b3
				a2 := float64(a2r[p])
				v20 += a2 * b0
				v21 += a2 * b1
				v22 += a2 * b2
				v23 += a2 * b3
				a3 := float64(a3r[p])
				v30 += a3 * b0
				v31 += a3 * b1
				v32 += a3 * b2
				v33 += a3 * b3
			}
			c0[j], c0[j+1], c0[j+2], c0[j+3] = v00, v01, v02, v03
			c1[j], c1[j+1], c1[j+2], c1[j+3] = v10, v11, v12, v13
			c2[j], c2[j+1], c2[j+2], c2[j+3] = v20, v21, v22, v23
			c3[j], c3[j+1], c3[j+2], c3[j+3] = v30, v31, v32, v33
		}
		// column remainder: the original p-outer form over the last n%4 columns.
		if j < n {
			for p := range k {
				bp := B[p*n : (p+1)*n]
				a0 := float64(a0r[p])
				a1 := float64(a1r[p])
				a2 := float64(a2r[p])
				a3 := float64(a3r[p])
				for jj := j; jj < n; jj++ {
					bf := float64(bp[jj])
					c0[jj] += a0 * bf
					c1[jj] += a1 * bf
					c2[jj] += a2 * bf
					c3[jj] += a3 * bf
				}
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

// gemmF32BandPacked is gemmF32Band reading B from a 4-column PACKED panel:
// pack[(t*k+p)*4+c] == B[p*n+t*4+c]. Identical arithmetic, identical order.
//
// Why: the tiled loop reads B[p*n+j .. j+4] for a fixed j-tile across all p, so consecutive p
// are n floats apart. Every one of those pulls a 64-byte line to use 16 bytes, and each i-block
// re-walks the whole matrix that way. Packed, a tile's k values are contiguous, so the line is
// fully used and the panel stays resident across the p sweep. gemmF32 packs once and every row
// band shares it, which is what makes the copy worth paying.
func gemmF32BandPacked(A, B []float32, pack []float64, acc []float64, loRow, hiRow, k, n int) {
	nt := n >> 2 // full 4-column tiles
	var aw []float64
	if nt > 0 && loRow+3 < hiRow {
		awP := getAWiden(4 * k)
		aw = *awP
		defer putAWiden(awP)
	}
	// pack already holds B WIDENED to f64. That conversion used to sit in the innermost loop,
	// where it ran once per 4 FMAs and was repeated for every row block and every column tile —
	// m*k*n/4 of them; packed, it costs k*n once for the whole matmul. float64(float32) is exact,
	// so moving it earlier cannot change a value.
	//
	// A's four rows are widened once per row block into aw, from a DEDICATED pool. Measured
	// separately this is the LARGER half of the two hoists — B alone -4.93%, both -16.37% — so
	// it is not the afterthought it looks like. It needs its own pool because a 4*k buffer drawn
	// from the shared f64 scratch fights the k*n pack buffer for the same slots, and that churn
	// measured +19.6% allocs/op.

	i := loRow
	for ; i+3 < hiRow; i += 4 {
		c0 := acc[(i+0)*n : (i+1)*n]
		c1 := acc[(i+1)*n : (i+2)*n]
		c2 := acc[(i+2)*n : (i+3)*n]
		c3 := acc[(i+3)*n : (i+4)*n]
		a0r := A[(i+0)*k : (i+1)*k]
		a1r := A[(i+1)*k : (i+2)*k]
		a2r := A[(i+2)*k : (i+3)*k]
		a3r := A[(i+3)*k : (i+4)*k]
		aw0, aw1 := aw[0:k], aw[k:2*k]
		aw2, aw3 := aw[2*k:3*k], aw[3*k:4*k]
		for x, v := range a0r {
			aw0[x] = float64(v)
		}
		for x, v := range a1r {
			aw1[x] = float64(v)
		}
		for x, v := range a2r {
			aw2[x] = float64(v)
		}
		for x, v := range a3r {
			aw3[x] = float64(v)
		}
		for t := range nt {
			j := t * 4
			bcol := pack[t*k*4 : (t+1)*k*4]
			v00, v01, v02, v03 := c0[j], c0[j+1], c0[j+2], c0[j+3]
			v10, v11, v12, v13 := c1[j], c1[j+1], c1[j+2], c1[j+3]
			v20, v21, v22, v23 := c2[j], c2[j+1], c2[j+2], c2[j+3]
			v30, v31, v32, v33 := c3[j], c3[j+1], c3[j+2], c3[j+3]
			for p := range k {
				bp := bcol[p*4 : p*4+4]
				b0, b1, b2, b3 := bp[0], bp[1], bp[2], bp[3]
				a0 := aw0[p]
				v00 += a0 * b0
				v01 += a0 * b1
				v02 += a0 * b2
				v03 += a0 * b3
				a1 := aw1[p]
				v10 += a1 * b0
				v11 += a1 * b1
				v12 += a1 * b2
				v13 += a1 * b3
				a2 := aw2[p]
				v20 += a2 * b0
				v21 += a2 * b1
				v22 += a2 * b2
				v23 += a2 * b3
				a3 := aw3[p]
				v30 += a3 * b0
				v31 += a3 * b1
				v32 += a3 * b2
				v33 += a3 * b3
			}
			c0[j], c0[j+1], c0[j+2], c0[j+3] = v00, v01, v02, v03
			c1[j], c1[j+1], c1[j+2], c1[j+3] = v10, v11, v12, v13
			c2[j], c2[j+1], c2[j+2], c2[j+3] = v20, v21, v22, v23
			c3[j], c3[j+1], c3[j+2], c3[j+3] = v30, v31, v32, v33
		}
		if rem := nt * 4; rem < n { // unpacked column remainder
			for p := range k {
				bp := B[p*n : (p+1)*n]
				a0 := float64(a0r[p])
				a1 := float64(a1r[p])
				a2 := float64(a2r[p])
				a3 := float64(a3r[p])
				for jj := rem; jj < n; jj++ {
					bf := float64(bp[jj])
					c0[jj] += a0 * bf
					c1[jj] += a1 * bf
					c2[jj] += a2 * bf
					c3[jj] += a3 * bf
				}
			}
		}
	}
	// Single-row remainder. It reads the PACKED panel for the full tiles, like the 4-row body
	// above, rather than streaming B directly: the pack has already been built and paid for by
	// this point, so a remainder row that ignored it would re-walk B strided for nothing. Only the
	// columns past the last full tile fall back to B.
	//
	// This matters more than "remainder" suggests. A band gets ceil(m/workers) rows and the tile
	// takes them four at a time, so ANY m that is not a multiple of 4*workers leaves rows here —
	// which is most shapes.
	for ; i < hiRow; i++ {
		ci := acc[i*n : (i+1)*n]
		ar := A[i*k : (i+1)*k]
		for t := range nt {
			j := t * 4
			bcol := pack[t*k*4 : (t+1)*k*4]
			v0, v1, v2, v3 := ci[j], ci[j+1], ci[j+2], ci[j+3]
			for p := range k {
				bp := bcol[p*4 : p*4+4]
				a := float64(ar[p])
				v0 += a * bp[0]
				v1 += a * bp[1]
				v2 += a * bp[2]
				v3 += a * bp[3]
			}
			ci[j], ci[j+1], ci[j+2], ci[j+3] = v0, v1, v2, v3
		}
		if rem := nt * 4; rem < n {
			for p := range k {
				aip := float64(ar[p])
				bp := B[p*n : (p+1)*n]
				for jj := rem; jj < n; jj++ {
					ci[jj] += aip * float64(bp[jj])
				}
			}
		}
	}
}

// packBTiles4 fills pack[(t*k+p)*4+c] = float64(B[p*n+t*4+c]) for the n/4 full tiles. The widening
// happens HERE, once per matmul, instead of once per 4 FMAs in the band's innermost loop.
func packBTiles4(B []float32, pack []float64, k, n int) {
	nt := n >> 2
	for t := range nt {
		src := t * 4
		dst := pack[t*k*4 : (t+1)*k*4]
		for p := range k {
			r := B[p*n+src : p*n+src+4]
			d := dst[p*4 : p*4+4]
			d[0], d[1], d[2], d[3] = float64(r[0]), float64(r[1]), float64(r[2]), float64(r[3])
		}
	}
}

// gemmF64Rows runs the F64 row-band fan-out, packing B into 4-column panels first when that pays.
// The packed band reads a tile's k values contiguously instead of striding by n; see
// gemmF32BandPacked for the full argument and the measurements behind the two gates.
//
// gemm.go calls this for the large-matmul path. The conv im2col callers keep calling gemmF64Band
// directly: they pass a B whose shape is the kernel matrix, not a large operand, so they sit below
// the gate anyway and there is no reason to route them through an extra decision.
func gemmF64Rows(A, B, C []float64, m, k, n int) {
	if n >= 4 && k*n >= gemmPackMinWorkF64 && gemmPackBands(m, k, n, gemmPackTileBlocksF64) {
		packP := getF64Raw((n >> 2) * k * 4)
		pack := *packP
		packBTiles4F64(B, pack, k, n)
		parallelWork(m, k*n, func(loRow, hiRow int) {
			gemmF64BandPacked(A, B, pack, C, loRow, hiRow, k, n)
		})
		putF64(packP)
		return
	}
	parallelWork(m, k*n, func(loRow, hiRow int) {
		gemmF64Band(A, B, C, loRow, hiRow, k, n)
	})
}

// packBTiles4F64 fills pack[(t*k+p)*4+c] = B[p*n+t*4+c] for the n/4 full tiles.
func packBTiles4F64(B, pack []float64, k, n int) {
	nt := n >> 2
	for t := range nt {
		src := t * 4
		dst := pack[t*k*4 : (t+1)*k*4]
		for p := range k {
			copy(dst[p*4:p*4+4], B[p*n+src:p*n+src+4])
		}
	}
}

// gemmF64BandPacked is gemmF64Band reading B from the 4-column packed panel. Identical
// arithmetic in identical order; each C cell still sums ascending p from its incoming value.
func gemmF64BandPacked(A, B, pack, C []float64, loRow, hiRow, k, n int) {
	nt := n >> 2
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		c0 := C[(i+0)*n : (i+1)*n]
		c1 := C[(i+1)*n : (i+2)*n]
		c2 := C[(i+2)*n : (i+3)*n]
		c3 := C[(i+3)*n : (i+4)*n]
		a0r := A[(i+0)*k : (i+1)*k]
		a1r := A[(i+1)*k : (i+2)*k]
		a2r := A[(i+2)*k : (i+3)*k]
		a3r := A[(i+3)*k : (i+4)*k]
		for t := range nt {
			j := t * 4
			bcol := pack[t*k*4 : (t+1)*k*4]
			v00, v01, v02, v03 := c0[j], c0[j+1], c0[j+2], c0[j+3]
			v10, v11, v12, v13 := c1[j], c1[j+1], c1[j+2], c1[j+3]
			v20, v21, v22, v23 := c2[j], c2[j+1], c2[j+2], c2[j+3]
			v30, v31, v32, v33 := c3[j], c3[j+1], c3[j+2], c3[j+3]
			for p := range k {
				bp := bcol[p*4 : p*4+4]
				b0, b1, b2, b3 := bp[0], bp[1], bp[2], bp[3]
				a0 := a0r[p]
				v00 += a0 * b0
				v01 += a0 * b1
				v02 += a0 * b2
				v03 += a0 * b3
				a1 := a1r[p]
				v10 += a1 * b0
				v11 += a1 * b1
				v12 += a1 * b2
				v13 += a1 * b3
				a2 := a2r[p]
				v20 += a2 * b0
				v21 += a2 * b1
				v22 += a2 * b2
				v23 += a2 * b3
				a3 := a3r[p]
				v30 += a3 * b0
				v31 += a3 * b1
				v32 += a3 * b2
				v33 += a3 * b3
			}
			c0[j], c0[j+1], c0[j+2], c0[j+3] = v00, v01, v02, v03
			c1[j], c1[j+1], c1[j+2], c1[j+3] = v10, v11, v12, v13
			c2[j], c2[j+1], c2[j+2], c2[j+3] = v20, v21, v22, v23
			c3[j], c3[j+1], c3[j+2], c3[j+3] = v30, v31, v32, v33
		}
		if rem := nt * 4; rem < n {
			for p := range k {
				bp := B[p*n : (p+1)*n]
				a0, a1 := a0r[p], a1r[p]
				a2, a3 := a2r[p], a3r[p]
				for jj := rem; jj < n; jj++ {
					bv := bp[jj]
					c0[jj] += a0 * bv
					c1[jj] += a1 * bv
					c2[jj] += a2 * bv
					c3[jj] += a3 * bv
				}
			}
		}
	}
	// Single-row remainder, reading the PACKED panel for the full tiles — same reasoning as the
	// f32 twin: the pack is already built by this point, so a remainder row that streamed B
	// instead would re-walk it strided for nothing. Any m that is not a multiple of 4*workers
	// leaves rows here, which is most shapes.
	for ; i < hiRow; i++ {
		ci := C[i*n : (i+1)*n]
		ar := A[i*k : (i+1)*k]
		for t := range nt {
			j := t * 4
			bcol := pack[t*k*4 : (t+1)*k*4]
			v0, v1, v2, v3 := ci[j], ci[j+1], ci[j+2], ci[j+3]
			for p := range k {
				bp := bcol[p*4 : p*4+4]
				a := ar[p]
				v0 += a * bp[0]
				v1 += a * bp[1]
				v2 += a * bp[2]
				v3 += a * bp[3]
			}
			ci[j], ci[j+1], ci[j+2], ci[j+3] = v0, v1, v2, v3
		}
		if rem := nt * 4; rem < n {
			for p := range k {
				aip := ar[p]
				bp := B[p*n : (p+1)*n]
				for jj := rem; jj < n; jj++ {
					ci[jj] += aip * bp[jj]
				}
			}
		}
	}
}

// awScratch pools the per-row-block f64 widening of A used by gemmF32BandPacked. A POOL OF ITS
// OWN, not the shared f64 scratch: these buffers are 4*k long while the packed-B panel taken from
// that pool is k*n, and mixing the two sizes measured +19.6% allocs/op from the churn. Not zeroed
// on get — every element is overwritten before use.
var awScratch = sync.Pool{New: func() any { b := make([]float64, 0); return &b }}

func getAWiden(n int) *[]float64 {
	bp := awScratch.Get().(*[]float64)
	if cap(*bp) < n {
		*bp = make([]float64, n)
	} else {
		*bp = (*bp)[:n]
	}
	return bp
}

func putAWiden(bp *[]float64) { awScratch.Put(bp) }
