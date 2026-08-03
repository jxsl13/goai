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
		// Two B rows per pass over the four C rows — see gemmF32Band for the reasoning and the
		// register-pressure limit. Bit-identical: p then p+1, two separate roundings, ascending.
		p := 0
		for ; p+1 < k; p += 2 {
			bp0 := B[p*n : (p+1)*n]
			bp1 := B[(p+1)*n : (p+2)*n]
			a00, a01 := A[(i+0)*k+p], A[(i+0)*k+p+1]
			a10, a11 := A[(i+1)*k+p], A[(i+1)*k+p+1]
			a20, a21 := A[(i+2)*k+p], A[(i+2)*k+p+1]
			a30, a31 := A[(i+3)*k+p], A[(i+3)*k+p+1]
			for j, b0 := range bp0 {
				b1 := bp1[j]
				v0 := c0[j]
				v0 += a00 * b0
				v0 += a01 * b1
				c0[j] = v0
				v1 := c1[j]
				v1 += a10 * b0
				v1 += a11 * b1
				c1[j] = v1
				v2 := c2[j]
				v2 += a20 * b0
				v2 += a21 * b1
				c2[j] = v2
				v3 := c3[j]
				v3 += a30 * b0
				v3 += a31 * b1
				c3[j] = v3
			}
		}
		for ; p < k; p++ {
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
	for ; i < hiRow; i++ { // remainder rows: a gemv, four B rows per pass
		ci := C[i*n : (i+1)*n]
		p := 0
		for ; p+3 < k; p += 4 {
			a0, a1 := A[i*k+p], A[i*k+p+1]
			a2, a3 := A[i*k+p+2], A[i*k+p+3]
			bp0 := B[p*n : (p+1)*n]
			bp1 := B[(p+1)*n : (p+2)*n]
			bp2 := B[(p+2)*n : (p+3)*n]
			bp3 := B[(p+3)*n : (p+4)*n]
			for j, b0 := range bp0 {
				v := ci[j]
				v += a0 * b0
				v += a1 * bp1[j]
				v += a2 * bp2[j]
				v += a3 * bp3[j]
				ci[j] = v
			}
		}
		for ; p < k; p++ {
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
		// TWO B ROWS PER PASS OVER THE FOUR C ROWS. Each c element was loaded and stored once per
		// p; holding it across two p steps halves that traffic. Two and not four: four C rows times
		// four B rows needs sixteen live values plus their scalars, which spills — the same limit
		// the decode kernel's 1/4/8 sweep found at eight.
		//
		// Bit-identical: every element still adds its p then its p+1 contribution as two separate
		// roundings in ascending order, never as a summed pair.
		p := 0
		for ; p+1 < k; p += 2 {
			bp0 := B[p*n : (p+1)*n]
			bp1 := B[(p+1)*n : (p+2)*n]
			a00, a01 := float64(A[(i+0)*k+p]), float64(A[(i+0)*k+p+1])
			a10, a11 := float64(A[(i+1)*k+p]), float64(A[(i+1)*k+p+1])
			a20, a21 := float64(A[(i+2)*k+p]), float64(A[(i+2)*k+p+1])
			a30, a31 := float64(A[(i+3)*k+p]), float64(A[(i+3)*k+p+1])
			for j, bv0 := range bp0 {
				b0, b1 := float64(bv0), float64(bp1[j])
				v0 := c0[j]
				v0 += a00 * b0
				v0 += a01 * b1
				c0[j] = v0
				v1 := c1[j]
				v1 += a10 * b0
				v1 += a11 * b1
				c1[j] = v1
				v2 := c2[j]
				v2 += a20 * b0
				v2 += a21 * b1
				c2[j] = v2
				v3 := c3[j]
				v3 += a30 * b0
				v3 += a31 * b1
				c3[j] = v3
			}
		}
		for ; p < k; p++ {
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
		// The single-row tail is a gemv: four B rows per pass, the transform measured on the
		// decode kernel.
		p := 0
		for ; p+3 < k; p += 4 {
			a0 := float64(A[i*k+p])
			a1 := float64(A[i*k+p+1])
			a2 := float64(A[i*k+p+2])
			a3 := float64(A[i*k+p+3])
			bp0 := B[p*n : (p+1)*n]
			bp1 := B[(p+1)*n : (p+2)*n]
			bp2 := B[(p+2)*n : (p+3)*n]
			bp3 := B[(p+3)*n : (p+4)*n]
			for j, bv := range bp0 {
				v := ci[j]
				v += a0 * float64(bv)
				v += a1 * float64(bp1[j])
				v += a2 * float64(bp2[j])
				v += a3 * float64(bp3[j])
				ci[j] = v
			}
		}
		for ; p < k; p++ {
			aip := float64(A[i*k+p])
			bp := B[p*n : (p+1)*n]
			for j, bv := range bp {
				ci[j] += aip * float64(bv)
			}
		}
	}
}
