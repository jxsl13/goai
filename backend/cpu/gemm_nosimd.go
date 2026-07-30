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
		for p := range k {
			bp := B[p*n : (p+1)*n]
			a0 := A[(i+0)*k+p]
			a1 := A[(i+1)*k+p]
			a2 := A[(i+2)*k+p]
			a3 := A[(i+3)*k+p]
			for j, bv := range bp {
				c0[j] += a0 * bv
				c1[j] += a1 * bv
				c2[j] += a2 * bv
				c3[j] += a3 * bv
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
