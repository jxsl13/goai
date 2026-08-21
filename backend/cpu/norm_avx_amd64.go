//go:build goexperiment.simd && amd64

package cpu

import "simd/archsimd"

// amd64 SIMD perf build: the f32 normalize passes of layernorm/rmsnorm run through AVX2 8-wide FMA
// (the twins of the vexp exp kernels in vexp_amd64.go). normF32Fast gates them; the mean/variance
// reductions stay f64 (sumF32/varSumF32/sumSqF32) so only the elementwise normalize — the profiled
// ~45% scalar-f64 hotspot — is vectorized. Numerics ride the ADR-0021 tolerant f32 budget the
// surrounding f32-native kernels already use (norm_test rtol 5e-5): mean/inv are computed in f64 and
// rounded to f32 once here, the per-element (x−mean)·inv·γ+β is evaluated in f32.
const (
	normF32ForwardFast  = true
	normF32BackwardFast = true
)

// layerNormNormalizeF32 writes out[j] = (x[j]−mean)·inv·gamma[j] + beta[j] 8-wide via AVX2 FMA.
// mean/inv are the f64-computed row statistics rounded to f32 by the caller. 8-lane body + scalar
// tail. (x−mean)·inv is kept as sub-then-mul — NOT the algebraically-equal x·inv−mean·inv — to avoid
// amplifying the x≈mean cancellation.
func layerNormNormalizeF32(x, gamma, beta, out []float32, mean, inv float32) {
	if !vexpHasAVX {
		for j, g := range gamma {
			out[j] = (x[j]-mean)*inv*g + beta[j]
		}
		return
	}
	mv := archsimd.BroadcastFloat32x8(mean)
	iv := archsimd.BroadcastFloat32x8(inv)
	n8 := len(gamma) &^ 7
	for i := 0; i < n8; i += 8 {
		xv := archsimd.LoadFloat32x8Slice(x[i:])
		gv := archsimd.LoadFloat32x8Slice(gamma[i:])
		bv := archsimd.LoadFloat32x8Slice(beta[i:])
		t := xv.Sub(mv).Mul(iv)              // (x−mean)·inv
		t.MulAdd(gv, bv).StoreSlice(out[i:]) // t·gamma + beta
	}
	for i := n8; i < len(gamma); i++ {
		out[i] = (x[i]-mean)*inv*gamma[i] + beta[i]
	}
}

// rmsNormNormalizeF32 writes out[j] = x[j]·inv·gamma[j] 8-wide via AVX2 FMA, inv the f64-computed
// 1/rms rounded to f32. 8-lane body + scalar tail.
func rmsNormNormalizeF32(x, gamma, out []float32, inv float32) {
	if !vexpHasAVX {
		for j, g := range gamma {
			out[j] = x[j] * inv * g
		}
		return
	}
	iv := archsimd.BroadcastFloat32x8(inv)
	n8 := len(gamma) &^ 7
	for i := 0; i < n8; i += 8 {
		xv := archsimd.LoadFloat32x8Slice(x[i:])
		gv := archsimd.LoadFloat32x8Slice(gamma[i:])
		xv.Mul(iv).Mul(gv).StoreSlice(out[i:]) // x·inv·gamma
	}
	for i := n8; i < len(gamma); i++ {
		out[i] = x[i] * inv * gamma[i]
	}
}

// rmsNormDxF32 writes the RMSNorm backward dx row 8-wide: dx[j] = inv·(u[j]·g[j] −
// x[j]·c), the f32 twin of the scalar-f64 loop in rmsNormBwd (c = inv²·s/d hoisted by
// the caller). Same associativity as the scalar form — the subtract is folded as one
// FMA (x·(−c)+a) — evaluated in f32, so it rides the f32 norm budget (rtol 5e-5, ADR-0021)
// exactly like the forward normalize. 8-lane body + scalar tail; scalar fallback when the
// CPU lacks AVX/FMA.
func rmsNormDxF32(u, g, x, dx []float32, inv, c float32) {
	if !vexpHasAVX {
		for j := range dx {
			dx[j] = inv * (u[j]*g[j] - x[j]*c)
		}
		return
	}
	ivv := archsimd.BroadcastFloat32x8(inv)
	ncv := archsimd.BroadcastFloat32x8(-c)
	n8 := len(dx) &^ 7
	for i := 0; i < n8; i += 8 {
		uv := archsimd.LoadFloat32x8Slice(u[i:])
		gv := archsimd.LoadFloat32x8Slice(g[i:])
		xv := archsimd.LoadFloat32x8Slice(x[i:])
		a := uv.Mul(gv)               // u·g
		t := xv.MulAdd(ncv, a)        // x·(−c) + a = a − x·c
		t.Mul(ivv).StoreSlice(dx[i:]) // inv·(a − x·c)
	}
	for i := n8; i < len(dx); i++ {
		dx[i] = inv * (u[i]*g[i] - x[i]*c)
	}
}

// layerNormDxF32 writes the LayerNorm backward dx row 8-wide: dx[j] = inv·(u[j]·g[j] −
// meanA − x̂·meanAX) with x̂ = (x[j]−mu)·inv, the f32 twin of the scalar-f64 loop in
// layerNormBwd. Same (a−meanA)−x̂·meanAX association (the last term folded as one FMA),
// evaluated in f32 on the f32 norm budget. 8-lane body + scalar tail; scalar fallback
// without AVX/FMA.
func layerNormDxF32(u, g, x, dx []float32, mu, inv, meanA, meanAX float32) {
	if !vexpHasAVX {
		for j := range dx {
			xhat := (x[j] - mu) * inv
			dx[j] = inv * (u[j]*g[j] - meanA - xhat*meanAX)
		}
		return
	}
	muv := archsimd.BroadcastFloat32x8(mu)
	ivv := archsimd.BroadcastFloat32x8(inv)
	mAv := archsimd.BroadcastFloat32x8(meanA)
	nmAXv := archsimd.BroadcastFloat32x8(-meanAX)
	n8 := len(dx) &^ 7
	for i := 0; i < n8; i += 8 {
		uv := archsimd.LoadFloat32x8Slice(u[i:])
		gv := archsimd.LoadFloat32x8Slice(g[i:])
		xv := archsimd.LoadFloat32x8Slice(x[i:])
		a := uv.Mul(gv)              // u·g
		xhat := xv.Sub(muv).Mul(ivv) // (x−mu)·inv
		t := a.Sub(mAv)              // a − meanA
		r := xhat.MulAdd(nmAXv, t)   // x̂·(−meanAX) + t = t − x̂·meanAX
		r.Mul(ivv).StoreSlice(dx[i:])
	}
	for i := n8; i < len(dx); i++ {
		xhat := (x[i] - mu) * inv
		dx[i] = inv * (u[i]*g[i] - meanA - xhat*meanAX)
	}
}

// The f32 norm reductions, AVX2 8-wide with f64-WIDENED accumulation: each 8-lane f32 load splits
// into two f32x4 halves, widens to f64x4 (ConvertToFloat64) and accumulates into f64x4 lanes — so the
// sum stays f64-accurate (8 independent f64 partials, ~1e-15 rel), matching the scalar 4-accumulator
// versions in norm_avx_other.go to well within the §V9 budget. Only the accumulator GROUPING differs
// from the scalar order (both f64), a ≤ulp perturbation the f32 norm parity tolerates.

// sumF32 returns Σx in f64.
func sumF32(x []float32) float64 {
	if !vexpHasAVX {
		var s float64
		for _, v := range x {
			s += float64(v)
		}
		return s
	}
	lo, hi := archsimd.BroadcastFloat64x4(0), archsimd.BroadcastFloat64x4(0)
	n8 := len(x) &^ 7
	for i := 0; i < n8; i += 8 {
		v := archsimd.LoadFloat32x8Slice(x[i:])
		lo = lo.Add(v.GetLo().ConvertToFloat64())
		hi = hi.Add(v.GetHi().ConvertToFloat64())
	}
	s := horizF64x4(lo) + horizF64x4(hi)
	//perfscan:ignore PS3010 scalar remainder tail of already-AVX-vectorized sumF32
	for i := n8; i < len(x); i++ {
		s += float64(x[i])
	}
	return s
}

// sumSqF32 returns Σx² in f64.
func sumSqF32(x []float32) float64 {
	if !vexpHasAVX {
		var s float64
		for _, v := range x {
			f := float64(v)
			s += f * f
		}
		return s
	}
	lo, hi := archsimd.BroadcastFloat64x4(0), archsimd.BroadcastFloat64x4(0)
	n8 := len(x) &^ 7
	for i := 0; i < n8; i += 8 {
		v := archsimd.LoadFloat32x8Slice(x[i:])
		vl, vh := v.GetLo().ConvertToFloat64(), v.GetHi().ConvertToFloat64()
		lo = vl.MulAdd(vl, lo) // lo += vl·vl
		hi = vh.MulAdd(vh, hi)
	}
	s := horizF64x4(lo) + horizF64x4(hi)
	for i := n8; i < len(x); i++ {
		f := float64(x[i])
		s += f * f
	}
	return s
}

// varSumF32 returns Σ(x−mu)² in f64.
func varSumF32(x []float32, mu float64) float64 {
	if !vexpHasAVX {
		var s float64
		for _, v := range x {
			d := float64(v) - mu
			s += d * d
		}
		return s
	}
	muv := archsimd.BroadcastFloat64x4(mu)
	lo, hi := archsimd.BroadcastFloat64x4(0), archsimd.BroadcastFloat64x4(0)
	n8 := len(x) &^ 7
	for i := 0; i < n8; i += 8 {
		v := archsimd.LoadFloat32x8Slice(x[i:])
		dl := v.GetLo().ConvertToFloat64().Sub(muv)
		dh := v.GetHi().ConvertToFloat64().Sub(muv)
		lo = dl.MulAdd(dl, lo) // lo += dl·dl
		hi = dh.MulAdd(dh, hi)
	}
	s := horizF64x4(lo) + horizF64x4(hi)
	for i := n8; i < len(x); i++ {
		d := float64(x[i]) - mu
		s += d * d
	}
	return s
}

// horizF64x4 returns the sum of the four f64 lanes (tree order, matching the scalar (s0+s1)+(s2+s3)).
func horizF64x4(v archsimd.Float64x4) float64 {
	var a [4]float64
	v.Store(&a)
	return (a[0] + a[1]) + (a[2] + a[3])
}
