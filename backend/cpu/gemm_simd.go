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
// depends on it, §T597). Only gemmF64Band is vectorized: the F32 SIMD twin
// regressed and stays scalar (see gemmF32Band's note); f64 accumulation (§V10)
// is unchanged.
//
// gemmHasAVX runtime-gates the intrinsics (§I4): built with the experiment but
// run on a pre-AVX CPU, gemmF64Band falls back to a correct scalar accumulate
// instead of faulting.

import "simd/archsimd"

var gemmHasAVX = archsimd.X86.AVX()

func gemmF64Band(A, B, C []float64, loRow, hiRow, k, n int) {
	if !gemmHasAVX {
		gemmF64BandScalar(A, B, C, loRow, hiRow, k, n)
		return
	}
	nv := n - n%4 // columns covered by full 4-wide vectors
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		r0, r1, r2, r3 := (i+0)*k, (i+1)*k, (i+2)*k, (i+3)*k
		c0 := C[(i+0)*n : (i+0)*n+n]
		c1 := C[(i+1)*n : (i+1)*n+n]
		c2 := C[(i+2)*n : (i+2)*n+n]
		c3 := C[(i+3)*n : (i+3)*n+n]
		j := 0
		for ; j < nv; j += 4 {
			acc0 := archsimd.LoadFloat64x4Slice(c0[j:])
			acc1 := archsimd.LoadFloat64x4Slice(c1[j:])
			acc2 := archsimd.LoadFloat64x4Slice(c2[j:])
			acc3 := archsimd.LoadFloat64x4Slice(c3[j:])
			for p := 0; p < k; p++ {
				bv := archsimd.LoadFloat64x4Slice(B[p*n+j:])
				acc0 = acc0.Add(archsimd.BroadcastFloat64x4(A[r0+p]).Mul(bv))
				acc1 = acc1.Add(archsimd.BroadcastFloat64x4(A[r1+p]).Mul(bv))
				acc2 = acc2.Add(archsimd.BroadcastFloat64x4(A[r2+p]).Mul(bv))
				acc3 = acc3.Add(archsimd.BroadcastFloat64x4(A[r3+p]).Mul(bv))
			}
			acc0.StoreSlice(c0[j:])
			acc1.StoreSlice(c1[j:])
			acc2.StoreSlice(c2[j:])
			acc3.StoreSlice(c3[j:])
		}
		for ; j < n; j++ { // column tail
			s0, s1, s2, s3 := c0[j], c1[j], c2[j], c3[j]
			for p := 0; p < k; p++ {
				bv := B[p*n+j]
				s0 += A[r0+p] * bv
				s1 += A[r1+p] * bv
				s2 += A[r2+p] * bv
				s3 += A[r3+p] * bv
			}
			c0[j], c1[j], c2[j], c3[j] = s0, s1, s2, s3
		}
	}
	for ; i < hiRow; i++ { // remainder rows
		ci := C[i*n : i*n+n]
		ri := i * k
		j := 0
		for ; j < nv; j += 4 {
			acc := archsimd.LoadFloat64x4Slice(ci[j:])
			for p := 0; p < k; p++ {
				acc = acc.Add(archsimd.BroadcastFloat64x4(A[ri+p]).Mul(archsimd.LoadFloat64x4Slice(B[p*n+j:])))
			}
			acc.StoreSlice(ci[j:])
		}
		for ; j < n; j++ {
			s := ci[j]
			for p := 0; p < k; p++ {
				s += A[ri+p] * B[p*n+j]
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
