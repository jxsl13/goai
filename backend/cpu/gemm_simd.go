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
	nv := n - n%4  // 4-wide vector boundary
	nv8 := n - n%8 // 8-wide boundary (two Float64x4 per row)
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		r0, r1, r2, r3 := (i+0)*k, (i+1)*k, (i+2)*k, (i+3)*k
		c0 := C[(i+0)*n : (i+0)*n+n]
		c1 := C[(i+1)*n : (i+1)*n+n]
		c2 := C[(i+2)*n : (i+2)*n+n]
		c3 := C[(i+3)*n : (i+3)*n+n]
		j := 0
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
			for p := 0; p < k; p++ {
				lo := archsimd.LoadFloat64x4Slice(B[p*n+j:])
				hi := archsimd.LoadFloat64x4Slice(B[p*n+j+4:])
				b0 := archsimd.BroadcastFloat64x4(A[r0+p])
				b1 := archsimd.BroadcastFloat64x4(A[r1+p])
				b2 := archsimd.BroadcastFloat64x4(A[r2+p])
				b3 := archsimd.BroadcastFloat64x4(A[r3+p])
				a0 = a0.Add(b0.Mul(lo))
				a0h = a0h.Add(b0.Mul(hi))
				a1 = a1.Add(b1.Mul(lo))
				a1h = a1h.Add(b1.Mul(hi))
				a2 = a2.Add(b2.Mul(lo))
				a2h = a2h.Add(b2.Mul(hi))
				a3 = a3.Add(b3.Mul(lo))
				a3h = a3h.Add(b3.Mul(hi))
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
		for ; j < n; j++ { // scalar column tail (n%4)
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
func gemmF32Band(A, B []float32, acc []float64, loRow, hiRow, k, n int) {
	if !gemmHasFMA {
		gemmF32BandScalarF64(A, B, acc, loRow, hiRow, k, n)
		return
	}
	nv16 := n - n%16 // 16-wide: two Float32x8 per row = 8 accumulator chains
	nv := n - n%8    // 8-wide cleanup boundary
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		r0, r1, r2, r3 := (i+0)*k, (i+1)*k, (i+2)*k, (i+3)*k
		c0 := acc[(i+0)*n : (i+0)*n+n]
		c1 := acc[(i+1)*n : (i+1)*n+n]
		c2 := acc[(i+2)*n : (i+2)*n+n]
		c3 := acc[(i+3)*n : (i+3)*n+n]
		j := 0
		// 8 independent MulAdd chains — enough in flight to hide the FMA latency
		// on the 2 FMA units (mr=4 × nr=8 left only 4, ≈half-saturated).
		for ; j < nv16; j += 16 {
			s0 := archsimd.BroadcastFloat32x8(0)
			s0h := archsimd.BroadcastFloat32x8(0)
			s1 := archsimd.BroadcastFloat32x8(0)
			s1h := archsimd.BroadcastFloat32x8(0)
			s2 := archsimd.BroadcastFloat32x8(0)
			s2h := archsimd.BroadcastFloat32x8(0)
			s3 := archsimd.BroadcastFloat32x8(0)
			s3h := archsimd.BroadcastFloat32x8(0)
			for p := 0; p < k; p++ {
				lo := archsimd.LoadFloat32x8Slice(B[p*n+j:])
				hi := archsimd.LoadFloat32x8Slice(B[p*n+j+8:])
				b0 := archsimd.BroadcastFloat32x8(A[r0+p])
				b1 := archsimd.BroadcastFloat32x8(A[r1+p])
				b2 := archsimd.BroadcastFloat32x8(A[r2+p])
				b3 := archsimd.BroadcastFloat32x8(A[r3+p])
				s0 = b0.MulAdd(lo, s0)
				s0h = b0.MulAdd(hi, s0h)
				s1 = b1.MulAdd(lo, s1)
				s1h = b1.MulAdd(hi, s1h)
				s2 = b2.MulAdd(lo, s2)
				s2h = b2.MulAdd(hi, s2h)
				s3 = b3.MulAdd(lo, s3)
				s3h = b3.MulAdd(hi, s3h)
			}
			storeF32x8(s0, c0[j:])
			storeF32x8(s0h, c0[j+8:])
			storeF32x8(s1, c1[j:])
			storeF32x8(s1h, c1[j+8:])
			storeF32x8(s2, c2[j:])
			storeF32x8(s2h, c2[j+8:])
			storeF32x8(s3, c3[j:])
			storeF32x8(s3h, c3[j+8:])
		}
		for ; j < nv; j += 8 { // 8-wide cleanup (n%16 in [8,15])
			s0 := archsimd.BroadcastFloat32x8(0)
			s1 := archsimd.BroadcastFloat32x8(0)
			s2 := archsimd.BroadcastFloat32x8(0)
			s3 := archsimd.BroadcastFloat32x8(0)
			for p := 0; p < k; p++ {
				bv := archsimd.LoadFloat32x8Slice(B[p*n+j:])
				s0 = archsimd.BroadcastFloat32x8(A[r0+p]).MulAdd(bv, s0)
				s1 = archsimd.BroadcastFloat32x8(A[r1+p]).MulAdd(bv, s1)
				s2 = archsimd.BroadcastFloat32x8(A[r2+p]).MulAdd(bv, s2)
				s3 = archsimd.BroadcastFloat32x8(A[r3+p]).MulAdd(bv, s3)
			}
			storeF32x8(s0, c0[j:])
			storeF32x8(s1, c1[j:])
			storeF32x8(s2, c2[j:])
			storeF32x8(s3, c3[j:])
		}
		for ; j < n; j++ { // f32-native scalar tail (n%8)
			var t0, t1, t2, t3 float32
			for p := 0; p < k; p++ {
				bv := B[p*n+j]
				t0 += A[r0+p] * bv
				t1 += A[r1+p] * bv
				t2 += A[r2+p] * bv
				t3 += A[r3+p] * bv
			}
			c0[j], c1[j], c2[j], c3[j] = float64(t0), float64(t1), float64(t2), float64(t3)
		}
	}
	for ; i < hiRow; i++ {
		ci := acc[i*n : i*n+n]
		ri := i * k
		j := 0
		for ; j < nv; j += 8 {
			s := archsimd.BroadcastFloat32x8(0)
			for p := 0; p < k; p++ {
				s = archsimd.BroadcastFloat32x8(A[ri+p]).MulAdd(archsimd.LoadFloat32x8Slice(B[p*n+j:]), s)
			}
			storeF32x8(s, ci[j:])
		}
		for ; j < n; j++ {
			var t float32
			for p := 0; p < k; p++ {
				t += A[ri+p] * B[p*n+j]
			}
			ci[j] = float64(t)
		}
	}
}

// storeF32x8 widens an 8-lane f32 vector into the f64 accumulation carrier.
func storeF32x8(v archsimd.Float32x8, dst []float64) {
	var t [8]float32
	v.Store(&t)
	dst[0] = float64(t[0])
	dst[1] = float64(t[1])
	dst[2] = float64(t[2])
	dst[3] = float64(t[3])
	dst[4] = float64(t[4])
	dst[5] = float64(t[5])
	dst[6] = float64(t[6])
	dst[7] = float64(t[7])
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
