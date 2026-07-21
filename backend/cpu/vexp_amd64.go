//go:build goexperiment.simd

package cpu

import (
	"math"
	"simd/archsimd"
)

// amd64 perf build: the perf-critical f32 transcendentals run through AVX2 8-wide kernels (this
// file), the twins of the arm64 NEON kernels (vexp_arm64.s). vexpF32Fast gates the vectorized fast
// paths that are wired up on amd64 — the MHA/softmax-family exp+sum (mhaSoftmaxBandF32) AND the GELU
// forward/backward (§T657/§T664) AND SiLU forward/backward + sigmoid (§T665, the Llama/Qwen/Mistral
// FFN activation). vexpNeon stays false so only the standalone exp/tanh/log kernels keep their scalar
// paths on amd64 (that campaign is separate); their v* funcs below are scalar (dead here).
const (
	vexpNeon    = false
	vexpF32Fast = true
	vexpF64Fast = true // amd64 SIMD build: vsiluF64 (F64 SwiGLU FFN activation, §T667)
)

var vexpHasAVX = archsimd.X86.AVX() && archsimd.X86.FMA()

// exp poly / erf constants as broadcast vectors (built once at init; Broadcast is a cheap splat).
var (
	vLo     = archsimd.BroadcastFloat32x8(expLoClamp)
	vHi     = archsimd.BroadcastFloat32x8(expHiClamp)
	vLog2e  = archsimd.BroadcastFloat32x8(expLog2e)
	vMagic  = archsimd.BroadcastFloat32x8(expMagic)
	vNHi    = archsimd.BroadcastFloat32x8(-expLn2Hi)
	vNLo    = archsimd.BroadcastFloat32x8(-expLn2Lo)
	vE0     = archsimd.BroadcastFloat32x8(1.9875691500e-4)
	vE1     = archsimd.BroadcastFloat32x8(1.3981999507e-3)
	vE2     = archsimd.BroadcastFloat32x8(8.3334519073e-3)
	vE3     = archsimd.BroadcastFloat32x8(4.1665795894e-2)
	vE4     = archsimd.BroadcastFloat32x8(1.6666665459e-1)
	vHalf   = archsimd.BroadcastFloat32x8(0.5)
	vOne    = archsimd.BroadcastFloat32x8(1.0)
	vZero   = archsimd.BroadcastFloat32x8(0.0)
	vBias   = archsimd.BroadcastInt32x8(127)
	vAbs    = archsimd.BroadcastUint32x8(0x7fffffff)
	vSign   = archsimd.BroadcastUint32x8(0x80000000)
	vGInvS2 = archsimd.BroadcastFloat32x8(geluInvSqrt2)
	vGP     = archsimd.BroadcastFloat32x8(geluP)
	vGA1    = archsimd.BroadcastFloat32x8(geluA1)
	vGA2    = archsimd.BroadcastFloat32x8(geluA2)
	vGA3    = archsimd.BroadcastFloat32x8(geluA3)
	vGA4    = archsimd.BroadcastFloat32x8(geluA4)
	vGA5    = archsimd.BroadcastFloat32x8(geluA5)
	vGPdf   = archsimd.BroadcastFloat32x8(geluInvSqrt2Pi)
	vGPdfMn = archsimd.BroadcastFloat32x8(geluGradPdfMin)
	// full-domain exp (standalone OpExp): wider hi clamp + split 2ⁿ scale + overflow/underflow pins.
	vFullHi = archsimd.BroadcastFloat32x8(expFullHiClamp)
	vInfCut = archsimd.BroadcastFloat32x8(expFullInfCut)
	vMaxF32 = archsimd.BroadcastFloat32x8(math.MaxFloat32)
	vInf    = archsimd.BroadcastFloat32x8(float32(math.Inf(1)))
	vNegInf = archsimd.BroadcastFloat32x8(float32(math.Inf(-1)))
	vNaN    = archsimd.BroadcastFloat32x8(float32(math.NaN()))
	// standalone OpLog (Cephes logf): exponent extract + mantissa fold + degree-8 poly.
	vLogMin  = archsimd.BroadcastFloat32x8(logMinNormal)
	vLog2P25 = archsimd.BroadcastFloat32x8(logTwoP25)
	v25i     = archsimd.BroadcastInt32x8(25)
	vMant    = archsimd.BroadcastUint32x8(0x007FFFFF)
	vExp1    = archsimd.BroadcastUint32x8(0x3F800000)
	vSqrt2   = archsimd.BroadcastFloat32x8(logSqrtHalfx2)
	vLLn2Hi  = archsimd.BroadcastFloat32x8(logLn2Hi)
	vLLn2Lo  = archsimd.BroadcastFloat32x8(logLn2Lo)
	vLL0     = archsimd.BroadcastFloat32x8(logL0)
	vLL1     = archsimd.BroadcastFloat32x8(logL1)
	vLL2     = archsimd.BroadcastFloat32x8(logL2)
	vLL3     = archsimd.BroadcastFloat32x8(logL3)
	vLL4     = archsimd.BroadcastFloat32x8(logL4)
	vLL5     = archsimd.BroadcastFloat32x8(logL5)
	vLL6     = archsimd.BroadcastFloat32x8(logL6)
	vLL7     = archsimd.BroadcastFloat32x8(logL7)
	vLL8     = archsimd.BroadcastFloat32x8(logL8)
)

// expF32x8 evaluates exp(x) 8-wide via the Cephes range-reduced degree-5 polynomial — exp(x) =
// 2ⁿ·exp(r), n = rint(x·log2e), r = x − n·ln2 — the register-level primitive shared by vexpF32 and
// the GELU kernels. Same math as the scalar expF32 (clamps → magic-round n → reduced r → Horner →
// 2ⁿ scale), so ~3e-7 max rel err vs math.Exp; MulAdd fuses where the scalar mul-then-adds (~1 ulp).
func expF32x8(x archsimd.Float32x8) archsimd.Float32x8 {
	x = x.Max(vLo).Min(vHi)
	n := x.MulAdd(vLog2e, vMagic).Sub(vMagic) // rint(x·log2e), round-half-even via the 1.5·2²³ magic
	r := n.MulAdd(vNHi, x)                    // x − n·ln2Hi
	r = n.MulAdd(vNLo, r)                     // − n·ln2Lo
	pp := vE0.MulAdd(r, vE1)
	pp = pp.MulAdd(r, vE2)
	pp = pp.MulAdd(r, vE3)
	pp = pp.MulAdd(r, vE4)
	pp = pp.MulAdd(r, vHalf)
	res := pp.MulAdd(r.Mul(r), r.Add(vOne)) // p·r² + r + 1
	scale := n.ConvertToInt32().Add(vBias).AsUint32x8().ShiftAllLeft(23).AsFloat32x8()
	return res.Mul(scale)
}

// vexpF32 computes p[i] = exp(p[i]-m) in place and returns Σ p[i], 8-wide via AVX2 FMA — the amd64
// twin of the NEON vexpF32 (vexp_arm64.s). len(p) is a multiple of 4 (the shared vexpRowF32
// contract): an 8-lane body plus a scalar tail for the final 0 or 4 elements. Accuracy matches the
// scalar expF32 (~3e-7 rel), riding the ADR-0021 f32 tolerance the surrounding f32-native gemms use.
func vexpF32(p []float32, m float32) float32 {
	if !vexpHasAVX {
		var sum float32
		for i, v := range p {
			e := expF32(v - m)
			p[i] = e
			sum += e
		}
		return sum
	}
	mv := archsimd.BroadcastFloat32x8(m)
	sumv := vZero
	n8 := len(p) &^ 7
	for i := 0; i < n8; i += 8 {
		res := expF32x8(archsimd.LoadFloat32x8Slice(p[i:]).Sub(mv))
		res.StoreSlice(p[i:])
		sumv = sumv.Add(res)
	}
	var lanes [8]float32
	sumv.Store(&lanes)
	sum := lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7]
	for i := n8; i < len(p); i++ {
		e := expF32(p[i] - m)
		p[i] = e
		sum += e
	}
	return sum
}

// vgeluF32 computes dst[i] = gelu(src[i]) 8-wide via AVX2 — the amd64 twin of the NEON vgelu
// pipeline. Same AS-7.1.26 erf assembled on expF32x8 as the scalar geluF32 (erf(x) ≈ sign(x)·(1 −
// t·poly(t)·e^(−x²)), t = 1/(1+p·|x|)), so it matches within the ADR-0021 f32 tolerance
// (TestGeluF32Accuracy). 8-lane body + scalar geluF32 tail.
func vgeluF32(dst, src []float32) {
	if !vexpHasAVX {
		for i, v := range src {
			dst[i] = geluF32(v)
		}
		return
	}
	n8 := len(src) &^ 7
	for i := 0; i < n8; i += 8 {
		x := archsimd.LoadFloat32x8Slice(src[i:])
		a := x.Mul(vGInvS2).AsUint32x8().And(vAbs).AsFloat32x8() // |x/√2|
		t := vOne.Div(a.MulAdd(vGP, vOne))                       // 1/(1+p·a)
		s := vGA5.MulAdd(t, vGA4)
		s = s.MulAdd(t, vGA3)
		s = s.MulAdd(t, vGA2)
		s = s.MulAdd(t, vGA1)
		e := expF32x8(vZero.Sub(a.Mul(a)))                   // e^(−a²)
		E := vOne.Sub(s.Mul(t).Mul(e))                       // erf magnitude
		erf := E.AsUint32x8().Or(x.AsUint32x8().And(vSign))  // re-apply sign(x)
		res := x.Mul(vHalf.MulAdd(erf.AsFloat32x8(), vHalf)) // x·(0.5 + 0.5·erf)
		res.StoreSlice(dst[i:])
	}
	for i := n8; i < len(src); i++ {
		dst[i] = geluF32(src[i])
	}
}

// vgeluGradF32 computes dst[i] = g[i]·gelu'(x[i]) 8-wide via AVX2 — the amd64 twin of the NEON
// vgeluGrad pipeline: dx = g·(Φ(x) + x·φ(x)), Φ = 0.5·(1+erf), φ = e^(−x²/2)/√(2π), with ONE exp
// feeding both erf and pdf. The pdf-underflow mask (zero e at/below the clamp floor) restores exact
// ±Inf/NaN semantics, matching geluGradF32. 8-lane body + scalar geluGradF32 tail.
func vgeluGradF32(dst, x, g []float32) {
	if !vexpHasAVX {
		for i, v := range x {
			dst[i] = geluGradF32(v, g[i])
		}
		return
	}
	n8 := len(x) &^ 7
	for i := 0; i < n8; i += 8 {
		xv := archsimd.LoadFloat32x8Slice(x[i:])
		gv := archsimd.LoadFloat32x8Slice(g[i:])
		a := xv.Mul(vGInvS2).AsUint32x8().And(vAbs).AsFloat32x8() // |x/√2|
		t := vOne.Div(a.MulAdd(vGP, vOne))
		s := vGA5.MulAdd(t, vGA4)
		s = s.MulAdd(t, vGA3)
		s = s.MulAdd(t, vGA2)
		s = s.MulAdd(t, vGA1)
		e := expF32x8(vZero.Sub(a.Mul(a))) // e^(−x²/2) (a² = x²/2)
		e = e.Masked(e.Greater(vGPdfMn))   // zero sub-clamp/NaN e (NaN-safe: Greater→false→0)
		E := vOne.Sub(s.Mul(t).Mul(e))     // erf magnitude
		erf := E.AsUint32x8().Or(xv.AsUint32x8().And(vSign)).AsFloat32x8()
		phi := vHalf.MulAdd(erf, vHalf)    // Φ(x) = 0.5 + 0.5·erf
		pdf := vGPdf.Mul(e)                // φ(x)
		res := gv.Mul(xv.MulAdd(pdf, phi)) // g·(Φ + x·φ)
		res.StoreSlice(dst[i:])
	}
	for i := n8; i < len(x); i++ {
		dst[i] = geluGradF32(x[i], g[i])
	}
}

// vsiluF32 computes dst[i] = silu(src[i]) = x·σ(x) 8-wide via AVX2 — the amd64 twin of the NEON
// vsilu pipeline. Numerically-stable split on one exp: z = e^(−|x|), num = (x≥0 ? 1 : z), silu =
// x·num/(1+z) (no positive magnitude is ever exponentiated). The pdf-underflow mask on z (zero at/
// below the clamp floor) keeps ±Inf·0 = NaN exact, matching siluF32. 8-lane body + scalar tail.
func vsiluF32(dst, src []float32) {
	if !vexpHasAVX {
		for i, v := range src {
			dst[i] = siluF32(v)
		}
		return
	}
	n8 := len(src) &^ 7
	for i := 0; i < n8; i += 8 {
		x := archsimd.LoadFloat32x8Slice(src[i:])
		z := expF32x8(vZero.Sub(x.AsUint32x8().And(vAbs).AsFloat32x8())) // e^(−|x|)
		z = z.Masked(z.Greater(vGPdfMn))
		num := vOne.Merge(z, x.GreaterEqual(vZero)) // x≥0 ? 1 : z
		x.Mul(num).Div(vOne.Add(z)).StoreSlice(dst[i:])
	}
	for i := n8; i < len(src); i++ {
		dst[i] = siluF32(src[i])
	}
}

// vsiluGradF32 computes dst[i] = g[i]·silu'(x[i]) 8-wide via AVX2 — the amd64 twin of the NEON
// vsiluGrad pipeline. silu'(x) = (num·(1+z) + x·z)/(1+z)² with z = e^(−|x|), num = (x≥0 ? 1 : z) —
// the single-division form that is algebraically identical to σ·(1+x·(1−σ)). 8-lane body + tail.
func vsiluGradF32(dst, x, g []float32) {
	if !vexpHasAVX {
		for i, v := range x {
			dst[i] = siluGradF32(v, g[i])
		}
		return
	}
	n8 := len(x) &^ 7
	for i := 0; i < n8; i += 8 {
		xv := archsimd.LoadFloat32x8Slice(x[i:])
		gv := archsimd.LoadFloat32x8Slice(g[i:])
		z := expF32x8(vZero.Sub(xv.AsUint32x8().And(vAbs).AsFloat32x8()))
		z = z.Masked(z.Greater(vGPdfMn))
		num := vOne.Merge(z, xv.GreaterEqual(vZero))
		den := vOne.Add(z)
		d := num.MulAdd(den, xv.Mul(z)).Div(den.Mul(den)) // (num·den + x·z)/den²
		gv.Mul(d).StoreSlice(dst[i:])
	}
	for i := n8; i < len(x); i++ {
		dst[i] = siluGradF32(x[i], g[i])
	}
}

// vsigmoidF32 computes dst[i] = σ(src[i]) 8-wide via AVX2 — the amd64 twin of the NEON vsigmoid
// pipeline: z = e^(−|x|), num = (x≥0 ? 1 : z), σ = num/(1+z). No pdf mask (NaN propagates through z),
// matching sigmoidF32. 8-lane body + scalar tail.
func vsigmoidF32(dst, src []float32) {
	if !vexpHasAVX {
		for i, v := range src {
			dst[i] = sigmoidF32(v)
		}
		return
	}
	n8 := len(src) &^ 7
	for i := 0; i < n8; i += 8 {
		x := archsimd.LoadFloat32x8Slice(src[i:])
		z := expF32x8(vZero.Sub(x.AsUint32x8().And(vAbs).AsFloat32x8()))
		num := vOne.Merge(z, x.GreaterEqual(vZero))
		num.Div(vOne.Add(z)).StoreSlice(dst[i:])
	}
	for i := n8; i < len(src); i++ {
		dst[i] = sigmoidF32(src[i])
	}
}

// The standalone exp/tanh/log vexp fast paths are NOT vectorized on amd64 yet (vexpNeon is false, so
// elementwise.go keeps their scalar-f64 kernels bit-for-bit). These scalar instantiations exist only
// so the driver type-checks — dead code at run time here, exactly as in vexp_default.go.

func vexpFullF32(dst, src []float32) {
	if !vexpHasAVX {
		for i, v := range src {
			dst[i] = expFullF32(v)
		}
		return
	}
	n8 := len(src) &^ 7
	for i := 0; i < n8; i += 8 {
		x := archsimd.LoadFloat32x8Slice(src[i:])
		xc := x.Max(vLo).Min(vFullHi)
		n := xc.MulAdd(vLog2e, vMagic).Sub(vMagic)
		r := n.MulAdd(vNHi, xc)
		r = n.MulAdd(vNLo, r)
		pp := vE0.MulAdd(r, vE1)
		pp = pp.MulAdd(r, vE2)
		pp = pp.MulAdd(r, vE3)
		pp = pp.MulAdd(r, vE4)
		pp = pp.MulAdd(r, vHalf)
		res := pp.MulAdd(r.Mul(r), r.Add(vOne))
		// split the 2ⁿ scale as 2^floor(n/2)·2^ceil(n/2) so neither exponent field overflows for n≤129.
		n1f := n.Mul(vHalf).Floor()
		s1 := n1f.ConvertToInt32().Add(vBias).AsUint32x8().ShiftAllLeft(23).AsFloat32x8()
		s2 := n.Sub(n1f).ConvertToInt32().Add(vBias).AsUint32x8().ShiftAllLeft(23).AsFloat32x8()
		res = res.Mul(s1).Mul(s2).Min(vMaxF32)
		res = vZero.Merge(res, x.Less(vLo))       // underflow → exact 0 (NaN keeps res)
		res = vInf.Merge(res, x.Greater(vInfCut)) // overflow → +Inf (NaN keeps res)
		res.StoreSlice(dst[i:])
	}
	for i := n8; i < len(src); i++ {
		dst[i] = expFullF32(src[i])
	}
}

func vtanhF32(dst, src []float32) {
	if !vexpHasAVX {
		for i, v := range src {
			dst[i] = tanhF32(v)
		}
		return
	}
	n8 := len(src) &^ 7
	for i := 0; i < n8; i += 8 {
		x := archsimd.LoadFloat32x8Slice(src[i:])
		a := x.AsUint32x8().And(vAbs).AsFloat32x8()                       // |x|
		z := expF32x8(vZero.Sub(a.Add(a)))                                // e^(−2|x|), only the lo clamp is live
		t := vOne.Sub(z).Div(vOne.Add(z))                                 // (1−z)/(1+z)
		res := t.AsUint32x8().Or(x.AsUint32x8().And(vSign)).AsFloat32x8() // re-apply sign(x)
		res.StoreSlice(dst[i:])
	}
	for i := n8; i < len(src); i++ {
		dst[i] = tanhF32(src[i])
	}
}

func vlogF32(dst, src []float32) {
	if !vexpHasAVX {
		for i, v := range src {
			dst[i] = logF32(v)
		}
		return
	}
	n8 := len(src) &^ 7
	for i := 0; i < n8; i += 8 {
		x := archsimd.LoadFloat32x8Slice(src[i:])
		sub := x.Less(vLogMin)              // subnormal (also 0/neg; fixed by the specials below)
		xs := x.Mul(vLog2P25).Merge(x, sub) // pre-scale subnormals by 2²⁵
		bx := xs.AsUint32x8()
		e := bx.ShiftAllRight(23).AsInt32x8().Sub(vBias) // unbiased exponent
		e = e.Sub(sub.ToInt32x8().And(v25i))             // undo the 2²⁵ pre-scale where it was applied
		m := bx.And(vMant).Or(vExp1).AsFloat32x8()       // mantissa ∈ [1,2)
		fold := m.GreaterEqual(vSqrt2)
		m = m.Mul(vHalf).Merge(m, fold) // fold m ≥ √2 into [√½,√2)
		e = e.Sub(fold.ToInt32x8())     // and bump the exponent (subtract −1)
		ef := e.ConvertToFloat32()
		f := m.Sub(vOne) // Sterbenz-exact
		z := f.Mul(f)
		p := vLL0.MulAdd(f, vLL1)
		p = p.MulAdd(f, vLL2)
		p = p.MulAdd(f, vLL3)
		p = p.MulAdd(f, vLL4)
		p = p.MulAdd(f, vLL5)
		p = p.MulAdd(f, vLL6)
		p = p.MulAdd(f, vLL7)
		p = p.MulAdd(f, vLL8)
		y := f.Mul(z.Mul(p))
		y = ef.MulAdd(vLLn2Lo, y)
		y = y.Sub(vHalf.Mul(z))
		r := f.Add(y)
		r = ef.MulAdd(vLLn2Hi, r)
		// specials in ref's order (each later select wins): NaN→x, 0→−Inf, <0→NaN, +Inf→+Inf.
		r = x.Merge(r, x.IsNaN())
		r = vNegInf.Merge(r, x.Equal(vZero))
		r = vNaN.Merge(r, x.Less(vZero))
		r = vInf.Merge(r, x.Equal(vInf))
		r.StoreSlice(dst[i:])
	}
	for i := n8; i < len(src); i++ {
		dst[i] = logF32(src[i])
	}
}

// ---- F64 SiLU vectorization (§T667: the F64 SwiGLU FFN activation) ----------
//
// The F32 FFN activation runs the 8-wide vexp pipeline (vsiluF32); the F64 path
// was scalar `x/(1+math.Exp(-x))` — and end-to-end Llama F64 prefill profiles as
// ~40% math.Exp inside siluKernelCPU (the SwiGLU gate over the large hidden dim).
// vsiluF64 vectorizes it 4-wide on an accurate AVX2 f64 exp (expF64x4), gated by
// vexpF64Fast (compile-time; the non-SIMD build keeps the scalar path). F64 SiLU
// is NOT under the CPU==Ref bit-exact invariant — TestCPUCrossReferenceExact
// SKIPS OpSiLU/F64 (cpu uses x/(1+e⁻ˣ), ref uses x·σ(x), already ulp-split), so a
// ~1-ulp vectorized exp is admissible and rides the model f64 tolerance (the
// Llama golden is 1e-12·max(1,|ref|); expF64x4 is sub-ulp — see TestExpF64x4Accuracy).
//
// expF64x4 evaluates eˣ for x ≤ 0 (vsiluF64 only ever feeds it −|x|): Cody-Waite
// range reduction x = k·ln2 + r with the full-f64 ln2 split (k·ln2Hi exact, ln2Hi
// has 21 trailing zero mantissa bits), then a degree-13 Taylor of eʳ over
// |r|≤ln2/2 (remainder ≈4e-18, sub-ulp), scaled by 2ᵏ built from the exponent
// field. The arg is clamped to ≥ −708 so k+1023 stays a valid biased exponent.
var (
	vF64ExpLo  = archsimd.BroadcastFloat64x4(-708.0)                      // min arg before 2ᵏ underflows
	vF64Log2e  = archsimd.BroadcastFloat64x4(1.4426950408889634)          // 1/ln2
	vF64NLn2Hi = archsimd.BroadcastFloat64x4(-6.93147180369123816490e-01) // −ln2 high (exact·k)
	vF64NLn2Lo = archsimd.BroadcastFloat64x4(-1.90821492927058770002e-10) // −ln2 low
	vF64Bias   = archsimd.BroadcastInt32x4(1023)
	vF64Abs    = archsimd.BroadcastUint64x4(0x7fffffffffffffff)
	vF64Zero   = archsimd.BroadcastFloat64x4(0.0)
	vF64One    = archsimd.BroadcastFloat64x4(1.0)
	// eʳ Taylor coefficients 1/k! (k = 13 … 0), Horner high→low.
	vF64E13 = archsimd.BroadcastFloat64x4(1.6059043836821613e-10)
	vF64E12 = archsimd.BroadcastFloat64x4(2.08767569878681e-09)
	vF64E11 = archsimd.BroadcastFloat64x4(2.505210838544172e-08)
	vF64E10 = archsimd.BroadcastFloat64x4(2.755731922398589e-07)
	vF64E9  = archsimd.BroadcastFloat64x4(2.7557319223985893e-06)
	vF64E8  = archsimd.BroadcastFloat64x4(2.48015873015873e-05)
	vF64E7  = archsimd.BroadcastFloat64x4(1.984126984126984e-04)
	vF64E6  = archsimd.BroadcastFloat64x4(1.388888888888889e-03)
	vF64E5  = archsimd.BroadcastFloat64x4(8.333333333333333e-03)
	vF64E4  = archsimd.BroadcastFloat64x4(4.166666666666666e-02)
	vF64E3  = archsimd.BroadcastFloat64x4(1.6666666666666666e-01)
	vF64E2  = archsimd.BroadcastFloat64x4(0.5)
)

// expF64x4 returns eˣ (accurate to ~1 ulp) for each lane, valid for x ≤ 0.
func expF64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	x = x.Max(vF64ExpLo)
	kf := x.Mul(vF64Log2e).RoundToEven()
	r := kf.MulAdd(vF64NLn2Hi, x) // x − k·ln2Hi
	r = kf.MulAdd(vF64NLn2Lo, r)  // − k·ln2Lo
	// degree-13 Taylor of eʳ: p = Σ rᵏ/k!, Horner from 1/13! down to 1 + r.
	p := vF64E13
	p = p.MulAdd(r, vF64E12)
	p = p.MulAdd(r, vF64E11)
	p = p.MulAdd(r, vF64E10)
	p = p.MulAdd(r, vF64E9)
	p = p.MulAdd(r, vF64E8)
	p = p.MulAdd(r, vF64E7)
	p = p.MulAdd(r, vF64E6)
	p = p.MulAdd(r, vF64E5)
	p = p.MulAdd(r, vF64E4)
	p = p.MulAdd(r, vF64E3)
	p = p.MulAdd(r, vF64E2)
	p = p.MulAdd(r, vF64One) // + r
	p = p.MulAdd(r, vF64One) // + 1
	// scale = 2ᵏ via the biased exponent field: (k+1023)<<52. Built through int32
	// (VCVTPD2DQ) + sign-extend to int64 (VPMOVSXDQ) + VPSLLQ — all AVX2, avoiding
	// the AVX-512-only f64→i64 convert (VCVTTPD2QQ).
	scale := kf.ConvertToInt32().Add(vF64Bias).ExtendToInt64().ShiftAllLeft(52).AsFloat64x4()
	return p.Mul(scale)
}

// expF64poly is the SCALAR twin of expF64x4 — bit-for-bit identical (same clamp,
// Cody-Waite reduction, degree-13 Horner and 2ᵏ construction, every step a fused
// math.FMA to match archsimd's MulAdd). vsiluF64's tail uses it so a given input
// value yields the SAME silu whether it lands in the 4-lane body or the scalar
// tail — the invariant TestQuantDeepSeekV2DecodeMatchesForward relies on (absorbed
// decode == batched forward, byte-exact, regardless of array length/alignment).
func expF64poly(x float64) float64 {
	if x < -708.0 {
		x = -708.0
	}
	kf := math.RoundToEven(x * 1.4426950408889634)
	r := math.FMA(kf, -6.93147180369123816490e-01, x)
	r = math.FMA(kf, -1.90821492927058770002e-10, r)
	p := 1.6059043836821613e-10
	p = math.FMA(p, r, 2.08767569878681e-09)
	p = math.FMA(p, r, 2.505210838544172e-08)
	p = math.FMA(p, r, 2.755731922398589e-07)
	p = math.FMA(p, r, 2.7557319223985893e-06)
	p = math.FMA(p, r, 2.48015873015873e-05)
	p = math.FMA(p, r, 1.984126984126984e-04)
	p = math.FMA(p, r, 1.388888888888889e-03)
	p = math.FMA(p, r, 8.333333333333333e-03)
	p = math.FMA(p, r, 4.166666666666666e-02)
	p = math.FMA(p, r, 1.6666666666666666e-01)
	p = math.FMA(p, r, 0.5)
	p = math.FMA(p, r, 1.0)
	p = math.FMA(p, r, 1.0)
	scale := math.Float64frombits(uint64(int64(kf)+1023) << 52)
	return p * scale
}

// siluF64poly = silu via the expF64poly primitive, the exact scalar mirror of the
// vsiluF64 SIMD lane (overflow-safe z = e^(−|x|), num = (x≥0 ? 1 : z), x·num/(1+z)).
func siluF64poly(x float64) float64 {
	z := expF64poly(-math.Abs(x))
	num := z
	if x >= 0 {
		num = 1
	}
	return x * num / (1 + z)
}

// vsiluF64 computes dst[i] = silu(src[i]) = x·σ(x) 4-wide via AVX2 — the f64 twin
// of vsiluF32. Uses the overflow-safe form z = e^(−|x|), num = (x≥0 ? 1 : z),
// silu = x·num/(1+z). 4-lane body + scalar tail via siluF64poly (bit-identical to
// the body, not math.Exp — see expF64poly).
func vsiluF64(dst, src []float64) {
	if !vexpHasAVX {
		for i, v := range src {
			dst[i] = siluF64poly(v)
		}
		return
	}
	n4 := len(src) &^ 3
	for i := 0; i < n4; i += 4 {
		x := archsimd.LoadFloat64x4Slice(src[i:])
		z := expF64x4(vF64Zero.Sub(x.AsUint64x4().And(vF64Abs).AsFloat64x4())) // e^(−|x|)
		num := vF64One.Merge(z, x.GreaterEqual(vF64Zero))                      // x≥0 ? 1 : z
		x.Mul(num).Div(vF64One.Add(z)).StoreSlice(dst[i:])
	}
	for i := n4; i < len(src); i++ {
		dst[i] = siluF64poly(src[i])
	}
}

// F64 softplus (Mamba/Jamba Δ discretization) constants. softplus was scalar
// math.Exp+math.Log1p on the ref backend (~32% of Mamba f64 prefill, §T——).
// log1p(u) is computed by the Cephes double log rational on v=1+u: because the
// stable form feeds u = e^(−|x|) ∈ (0,1], v ∈ (1,2] so the frexp exponent is only
// 0 or 1 — a mask-selected float, NO bit-level exponent extract and NO AVX-512-only
// i64→f64 convert (which VCVTQQ2PD would need). log(m) = f − 0.5f² + f³·P(f)/Q(f),
// 1/√2 ≤ m < √2, f = m−1.
var (
	vF64Half  = archsimd.BroadcastFloat64x4(0.5)
	vLogSqrt2 = archsimd.BroadcastFloat64x4(1.4142135623730951)          // √2: the v=1+u fold threshold
	vLogLn2Hi = archsimd.BroadcastFloat64x4(6.93359375e-1)               // ln2 high (exact)
	vLogLn2Lo = archsimd.BroadcastFloat64x4(-2.121944400546905827679e-4) // ln2 low
	// Cephes double log: polevl(f,P,5)/p1evl(f,Q,5) (Q monic).
	vLogP0 = archsimd.BroadcastFloat64x4(1.01875663804580931796e-4)
	vLogP1 = archsimd.BroadcastFloat64x4(4.97494994976747001425e-1)
	vLogP2 = archsimd.BroadcastFloat64x4(4.70579119878881725854e0)
	vLogP3 = archsimd.BroadcastFloat64x4(1.44989225341610930846e1)
	vLogP4 = archsimd.BroadcastFloat64x4(1.79368678507819816313e1)
	vLogP5 = archsimd.BroadcastFloat64x4(7.70838733755885391666e0)
	vLogQ0 = archsimd.BroadcastFloat64x4(1.12873587189167450590e1)
	vLogQ1 = archsimd.BroadcastFloat64x4(4.52279145837532221105e1)
	vLogQ2 = archsimd.BroadcastFloat64x4(8.29875266912776603211e1)
	vLogQ3 = archsimd.BroadcastFloat64x4(7.11544750618563894466e1)
	vLogQ4 = archsimd.BroadcastFloat64x4(2.31251620126765340583e1)
)

// softplusF64x4 returns softplus(x) = log(1+eˣ) 4-wide via AVX2, in the overflow-
// safe form max(x,0) + log1p(e^(−|x|)). The log1p rides the expF64x4 primitive plus
// the Cephes double log rational; ~1 ulp, well inside the model f64 tolerance (the
// Mamba goldens gate at 1e-9).
func softplusF64x4(x archsimd.Float64x4) archsimd.Float64x4 {
	ax := x.AsUint64x4().And(vF64Abs).AsFloat64x4()
	u := expF64x4(vF64Zero.Sub(ax)) // e^(−|x|) ∈ (0,1]
	v := vF64One.Add(u)             // 1+u ∈ (1,2]
	// frexp+fold, specialized to v ∈ (1,2]: exponent e is 0 (v<√2) or 1 (v≥√2).
	fold := v.GreaterEqual(vLogSqrt2)
	f := v.Mul(vF64Half).Sub(vF64One).Merge(v.Sub(vF64One), fold) // fold ? v/2−1 : v−1
	e := vF64One.Merge(vF64Zero, fold)                            // fold ? 1 : 0
	z := f.Mul(f)
	p := vLogP0
	p = p.MulAdd(f, vLogP1)
	p = p.MulAdd(f, vLogP2)
	p = p.MulAdd(f, vLogP3)
	p = p.MulAdd(f, vLogP4)
	p = p.MulAdd(f, vLogP5)
	q := f.Add(vLogQ0)
	q = q.MulAdd(f, vLogQ1)
	q = q.MulAdd(f, vLogQ2)
	q = q.MulAdd(f, vLogQ3)
	q = q.MulAdd(f, vLogQ4)
	y := f.Mul(z.Mul(p).Div(q)) // f·(z·P/Q)
	y = e.MulAdd(vLogLn2Lo, y)  // − e·2.1219e-4
	y = y.Sub(z.Mul(vF64Half))  // − 0.5·z
	r := f.Add(y)               // log1p(u) = f + y + e·ln2Hi
	r = e.MulAdd(vLogLn2Hi, r)
	return x.Max(vF64Zero).Add(r) // + max(x,0)
}

// softplusF64poly is the SCALAR twin of softplusF64x4 — bit-for-bit identical (same
// abs mask, expF64poly, Cody-Waite reduction, Cephes Horner and mask-select, each
// step a fused math.FMA to match archsimd MulAdd) so a value yields the same
// softplus in the 4-lane body or the scalar tail.
func softplusF64poly(x float64) float64 {
	ax := math.Float64frombits(math.Float64bits(x) & 0x7fffffffffffffff)
	v := 1.0 + expF64poly(-ax)
	var f, e float64
	if v >= 1.4142135623730951 {
		f, e = v*0.5-1, 1
	} else {
		f, e = v-1, 0
	}
	z := f * f
	p := 1.01875663804580931796e-4
	p = math.FMA(p, f, 4.97494994976747001425e-1)
	p = math.FMA(p, f, 4.70579119878881725854e0)
	p = math.FMA(p, f, 1.44989225341610930846e1)
	p = math.FMA(p, f, 1.79368678507819816313e1)
	p = math.FMA(p, f, 7.70838733755885391666e0)
	q := f + 1.12873587189167450590e1
	q = math.FMA(q, f, 4.52279145837532221105e1)
	q = math.FMA(q, f, 8.29875266912776603211e1)
	q = math.FMA(q, f, 7.11544750618563894466e1)
	q = math.FMA(q, f, 2.31251620126765340583e1)
	y := f * (z * p / q)
	y = math.FMA(e, -2.121944400546905827679e-4, y)
	y = y - z*0.5
	r := f + y
	r = math.FMA(e, 6.93359375e-1, r)
	mx := x
	if mx < 0 {
		mx = 0
	}
	return mx + r
}

// vsoftplusF64 computes dst[i] = softplus(src[i]) 4-wide via AVX2 — 4-lane body +
// scalar tail via softplusF64poly (bit-identical to the body, not math.Log1p).
func vsoftplusF64(dst, src []float64) {
	if !vexpHasAVX {
		for i, v := range src {
			dst[i] = softplusF64poly(v)
		}
		return
	}
	n4 := len(src) &^ 3
	for i := 0; i < n4; i += 4 {
		softplusF64x4(archsimd.LoadFloat64x4Slice(src[i:])).StoreSlice(dst[i:])
	}
	for i := n4; i < len(src); i++ {
		dst[i] = softplusF64poly(src[i])
	}
}
