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
// gemmATF64Band (portable build) is just the scalar kernel; the archsimd twin
// lives in gemm_simd.go.
func gemmATF64Band(A, B, C []float64, loRow, hiRow, m, k, n int) {
	gemmATF64BandScalar(A, B, C, loRow, hiRow, m, k, n)
}

func gemmF64Band(A, B, C []float64, loRow, hiRow, k, n int) {
	// SINGLE-COLUMN FAST PATH. With n == 1 every C element is a SCALAR, so four of them fit in
	// registers for the whole k loop and the destination leaves memory entirely — the blocked
	// form still walks C once per p because its inner loop is written over j, which here runs
	// once. Not a corner case: a conv2d with one output filter reaches the GEMM as n == 1, and
	// the multi-token-attention head convolution issues thirty-two of those per forward.
	//
	// Bit-identical: each C element still takes its k products in ascending p into the value it
	// already held.
	if n == 1 {
		i := loRow
		for ; i+3 < hiRow; i += 4 {
			a0 := A[(i+0)*k : (i+0)*k+k : (i+0)*k+k]
			a1 := A[(i+1)*k : (i+1)*k+k : (i+1)*k+k]
			a2 := A[(i+2)*k : (i+2)*k+k : (i+2)*k+k]
			a3 := A[(i+3)*k : (i+3)*k+k : (i+3)*k+k]
			v0, v1, v2, v3 := C[i+0], C[i+1], C[i+2], C[i+3]
			//perfscan:ignore PS3010 nosimd fallback build; already ikj 4-accumulator blocked
			for p, bv := range B[:k] {
				v0 += a0[p] * bv
				v1 += a1[p] * bv
				v2 += a2[p] * bv
				v3 += a3[p] * bv
			}
			C[i+0], C[i+1], C[i+2], C[i+3] = v0, v1, v2, v3
		}
		for ; i < hiRow; i++ {
			ar := A[i*k : i*k+k : i*k+k]
			v := C[i]
			//perfscan:ignore PS3010 gemmF32Band nosimd fallback, already ikj-blocked
			for p, bv := range B[:k] {
				v += ar[p] * bv
			}
			C[i] = v
		}
		return
	}
	i := loRow
	//perfscan:ignore PS3066 nosimd fallback kernel inner axpy, already optimal shape
	for ; i+3 < hiRow; i += 4 {
		c0 := C[(i+0)*n : (i+1)*n]
		c1 := C[(i+1)*n : (i+2)*n]
		c2 := C[(i+2)*n : (i+3)*n]
		c3 := C[(i+3)*n : (i+4)*n]
		// SIX B rows per pass over the four C rows — see gemmF32Band. Bit-identical: p through
		// p+5, six separate roundings, ascending.
		p := 0
		for ; p+5 < k; p += 6 {
			bp0 := B[p*n : (p+1)*n]
			bp1 := B[(p+1)*n : (p+2)*n]
			bp2 := B[(p+2)*n : (p+3)*n]
			bp3 := B[(p+3)*n : (p+4)*n]
			bp4 := B[(p+4)*n : (p+5)*n]
			bp5 := B[(p+5)*n : (p+6)*n]
			a00, a01, a02, a03, a04, a05 := A[(i+0)*k+p], A[(i+0)*k+p+1], A[(i+0)*k+p+2], A[(i+0)*k+p+3], A[(i+0)*k+p+4], A[(i+0)*k+p+5]
			a10, a11, a12, a13, a14, a15 := A[(i+1)*k+p], A[(i+1)*k+p+1], A[(i+1)*k+p+2], A[(i+1)*k+p+3], A[(i+1)*k+p+4], A[(i+1)*k+p+5]
			a20, a21, a22, a23, a24, a25 := A[(i+2)*k+p], A[(i+2)*k+p+1], A[(i+2)*k+p+2], A[(i+2)*k+p+3], A[(i+2)*k+p+4], A[(i+2)*k+p+5]
			a30, a31, a32, a33, a34, a35 := A[(i+3)*k+p], A[(i+3)*k+p+1], A[(i+3)*k+p+2], A[(i+3)*k+p+3], A[(i+3)*k+p+4], A[(i+3)*k+p+5]
			for j, b0 := range bp0 {
				b1, b2, b3, b4, b5 := bp1[j], bp2[j], bp3[j], bp4[j], bp5[j]
				v0 := c0[j]
				v0 += a00 * b0
				v0 += a01 * b1
				v0 += a02 * b2
				v0 += a03 * b3
				v0 += a04 * b4
				v0 += a05 * b5
				c0[j] = v0
				v1 := c1[j]
				v1 += a10 * b0
				v1 += a11 * b1
				v1 += a12 * b2
				v1 += a13 * b3
				v1 += a14 * b4
				v1 += a15 * b5
				c1[j] = v1
				v2 := c2[j]
				v2 += a20 * b0
				v2 += a21 * b1
				v2 += a22 * b2
				v2 += a23 * b3
				v2 += a24 * b4
				v2 += a25 * b5
				c2[j] = v2
				v3 := c3[j]
				v3 += a30 * b0
				v3 += a31 * b1
				v3 += a32 * b2
				v3 += a33 * b3
				v3 += a34 * b4
				v3 += a35 * b5
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
	if n == 1 { // see gemmF64Band: with one column the destination lives in registers
		i := loRow
		for ; i+3 < hiRow; i += 4 {
			a0 := A[(i+0)*k : (i+0)*k+k : (i+0)*k+k]
			a1 := A[(i+1)*k : (i+1)*k+k : (i+1)*k+k]
			a2 := A[(i+2)*k : (i+2)*k+k : (i+2)*k+k]
			a3 := A[(i+3)*k : (i+3)*k+k : (i+3)*k+k]
			v0, v1, v2, v3 := acc[i+0], acc[i+1], acc[i+2], acc[i+3]
			//perfscan:ignore PS3010 same nosimd ikj kernel (stale line)
			for p, bv := range B[:k] {
				bf := float64(bv)
				v0 += float64(a0[p]) * bf
				v1 += float64(a1[p]) * bf
				v2 += float64(a2[p]) * bf
				v3 += float64(a3[p]) * bf
			}
			acc[i+0], acc[i+1], acc[i+2], acc[i+3] = v0, v1, v2, v3
		}
		for ; i < hiRow; i++ {
			ar := A[i*k : i*k+k : i*k+k]
			v := acc[i]
			//perfscan:ignore PS3010 nosimd fallback ikj kernel, not production SIMD path
			for p, bv := range B[:k] {
				v += float64(ar[p]) * float64(bv)
			}
			acc[i] = v
		}
		return
	}
	i := loRow
	//perfscan:ignore PS3066 nosimd fallback inner axpy, already optimal
	for ; i+3 < hiRow; i += 4 {
		c0 := acc[(i+0)*n : (i+1)*n]
		c1 := acc[(i+1)*n : (i+2)*n]
		c2 := acc[(i+2)*n : (i+3)*n]
		c3 := acc[(i+3)*n : (i+4)*n]
		// SIX B ROWS PER PASS OVER THE FOUR C ROWS. Each c element was loaded and stored once per
		// p; holding it across six p steps cuts that traffic to a sixth.
		//
		// SIX BECAUSE IT WAS SWEPT, NOT BECAUSE IT WAS ARGUED. This was two for a long time, on
		// the reasoning that four C rows times four B rows needs sixteen live values plus their
		// scalars and would spill. The argument was directionally right and numerically wrong:
		// swept at 3, 4, 6 and 8 the f32 band gave BenchmarkMTAForward_ch16 260.8, 252.3, 250.3
		// and 277.4 ms against 277.2 at two, so the spill boundary is real and sits at eight.
		// The f64 band moved with it: GemmDirF64_1024 19.10 to 15.74 ms, -17.6%, and the
		// 512x2048x2048 cell -11.7%.
		//
		// Bit-identical at any factor, which is what makes the sweep cheap: every element still
		// adds its p, p+1 ... p+5 contributions as separate roundings in ascending order, never
		// as a summed pair, so TestGemmBandUnrollIsBitExact gates every arm of the sweep.
		p := 0
		for ; p+5 < k; p += 6 {
			bp0 := B[p*n : (p+1)*n]
			bp1 := B[(p+1)*n : (p+2)*n]
			bp2 := B[(p+2)*n : (p+3)*n]
			bp3 := B[(p+3)*n : (p+4)*n]
			bp4 := B[(p+4)*n : (p+5)*n]
			bp5 := B[(p+5)*n : (p+6)*n]
			a00, a01, a02, a03, a04, a05 := float64(A[(i+0)*k+p]), float64(A[(i+0)*k+p+1]), float64(A[(i+0)*k+p+2]), float64(A[(i+0)*k+p+3]), float64(A[(i+0)*k+p+4]), float64(A[(i+0)*k+p+5])
			a10, a11, a12, a13, a14, a15 := float64(A[(i+1)*k+p]), float64(A[(i+1)*k+p+1]), float64(A[(i+1)*k+p+2]), float64(A[(i+1)*k+p+3]), float64(A[(i+1)*k+p+4]), float64(A[(i+1)*k+p+5])
			a20, a21, a22, a23, a24, a25 := float64(A[(i+2)*k+p]), float64(A[(i+2)*k+p+1]), float64(A[(i+2)*k+p+2]), float64(A[(i+2)*k+p+3]), float64(A[(i+2)*k+p+4]), float64(A[(i+2)*k+p+5])
			a30, a31, a32, a33, a34, a35 := float64(A[(i+3)*k+p]), float64(A[(i+3)*k+p+1]), float64(A[(i+3)*k+p+2]), float64(A[(i+3)*k+p+3]), float64(A[(i+3)*k+p+4]), float64(A[(i+3)*k+p+5])
			for j, bv0 := range bp0 {
				b0, b1, b2, b3, b4, b5 := float64(bv0), float64(bp1[j]), float64(bp2[j]), float64(bp3[j]), float64(bp4[j]), float64(bp5[j])
				v0 := c0[j]
				v0 += a00 * b0
				v0 += a01 * b1
				v0 += a02 * b2
				v0 += a03 * b3
				v0 += a04 * b4
				v0 += a05 * b5
				c0[j] = v0
				v1 := c1[j]
				v1 += a10 * b0
				v1 += a11 * b1
				v1 += a12 * b2
				v1 += a13 * b3
				v1 += a14 * b4
				v1 += a15 * b5
				c1[j] = v1
				v2 := c2[j]
				v2 += a20 * b0
				v2 += a21 * b1
				v2 += a22 * b2
				v2 += a23 * b3
				v2 += a24 * b4
				v2 += a25 * b5
				c2[j] = v2
				v3 := c3[j]
				v3 += a30 * b0
				v3 += a31 * b1
				v3 += a32 * b2
				v3 += a33 * b3
				v3 += a34 * b4
				v3 += a35 * b5
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
