//go:build !(amd64 && goexperiment.simd)

package cpu

// Scalar GEMM band kernels — the portable default. On amd64+GOEXPERIMENT=simd
// gemm_simd.go replaces these with archsimd (AVX) twins that hold the identical
// ascending-p accumulation order, so exactly one definition compiles per build
// and the default pure-Go result is unchanged (§V3, §V11 tol 0). Contract for
// both: C is ACCUMULATED into (+=), so callers pass a zeroed buffer for a plain
// product and a live buffer to add onto (conv im2col scatter relies on this).

// gemmF32 computes C[m,n] = A·B, f32. Default build: accumulate in an f64 scratch
// (§V10) then narrow — bit-exact. The amd64+simd build overrides this with an
// f32-native kernel that writes C directly (gemm_simd.go).
func gemmF32(A, B, C []float32, m, k, n int) {
	accP := getF64(m * n) // pooled zeroed f64 accumulation scratch (§V10, §T463)
	acc := *accP
	parallelWork(m, k*n, func(loRow, hiRow int) {
		gemmF32Band(A, B, acc, loRow, hiRow, k, n)
	})
	for i := range C {
		C[i] = float32(acc[i])
	}
	putF64(accP)
}

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
