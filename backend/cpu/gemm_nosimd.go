//go:build !(amd64 && goexperiment.simd)

package cpu

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

// gemmPackMinRows is the row count above which gemmF32 packs B: the pack costs one k*n copy and is
// reused by m/4 row blocks, so its overhead is ~2/m, and 32 rows puts that near 6%.
const gemmPackMinRows = 32

// gemmPackMinWorkF32 / gemmPackMinWorkF64 are the B element counts above which packing pays.
// BOTH dtypes need a gate and they need DIFFERENT ones — packing is a cache fix, so it only wins
// once B is too big to stay resident while the tiles walk it, and the two kernels reach that point
// at different sizes. Measured by sweeping the gate inside one binary (see
// BenchmarkGemmF32Portable / BenchmarkGemmF64Portable), square k=n:
//
//	f32   n=256 (B 256KB) +2.78%    n=512 (1MB) ~        n=1024 (4MB) -13.16%
//	f64   n=64  (B  32KB) +24.50%   n=128 (128KB) +1.56%  n=256 (512KB) -2.80%
//	                                n=512 (2MB) -5.67%    n=1024 (8MB) -26.77%
//
// So f32 turns over between 1MB and 4MB of B and f64 between 128KB and 512KB — a single shared
// threshold would either leave the f64 wins on the floor or pack f32 where it costs 2.78%. The
// thresholds sit inside each measured bracket. Vars rather than consts so a parity test and the
// benchmarks can force either arm.
var (
	gemmPackMinWorkF32 = 1 << 19 // 524288 elements = 2MB of f32
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
func gemmF32BandPacked(A, B, pack []float32, acc []float64, loRow, hiRow, k, n int) {
	nt := n >> 2 // full 4-column tiles
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
		for t := range nt {
			j := t * 4
			bcol := pack[t*k*4 : (t+1)*k*4]
			v00, v01, v02, v03 := c0[j], c0[j+1], c0[j+2], c0[j+3]
			v10, v11, v12, v13 := c1[j], c1[j+1], c1[j+2], c1[j+3]
			v20, v21, v22, v23 := c2[j], c2[j+1], c2[j+2], c2[j+3]
			v30, v31, v32, v33 := c3[j], c3[j+1], c3[j+2], c3[j+3]
			for p := range k {
				bp := bcol[p*4 : p*4+4]
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
	for ; i < hiRow; i++ { // single-row remainder: unchanged, reads B directly
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

// packBTiles4 fills pack[(t*k+p)*4+c] = B[p*n+t*4+c] for the n/4 full tiles.
func packBTiles4(B, pack []float32, k, n int) {
	nt := n >> 2
	for t := range nt {
		src := t * 4
		dst := pack[t*k*4 : (t+1)*k*4]
		for p := range k {
			copy(dst[p*4:p*4+4], B[p*n+src:p*n+src+4])
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
	if m >= gemmPackMinRows && n >= 4 && k*n >= gemmPackMinWorkF64 {
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
	for ; i < hiRow; i++ {
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
