//go:build amd64 && goexperiment.simd

package simd

// AMD64 archsimd overrides of the portable elementwise primitives (§T11b,
// ADR-0005). Compiled only under GOEXPERIMENT=simd on amd64; the scalar
// simd.go carries the complementary tag, so exactly one definition of each
// symbol exists per build and the default pure-Go build is untouched (§V7).
//
// f32 uses 256-bit Float32x8 (8 lanes), f64 uses Float64x4 (4 lanes) — the
// widest AVX vectors. A whole-vector body strides by the lane count; a scalar
// tail finishes lengths not divisible by it. hasAVX gates the intrinsics at
// runtime (§I4): a binary built with the experiment but run on a pre-AVX CPU
// falls back to the scalar loop instead of executing an illegal instruction.

import (
	"math"

	"simd/archsimd"
)

// 256-bit float arithmetic is AVX; that is the only feature these kernels need.
var hasAVX = archsimd.X86.AVX()

func AddF64(dst, a, b []float64) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+4 <= n; i += 4 {
		archsimd.LoadFloat64x4Slice(a[i:]).Add(archsimd.LoadFloat64x4Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

func SubF64(dst, a, b []float64) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] - b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+4 <= n; i += 4 {
		archsimd.LoadFloat64x4Slice(a[i:]).Sub(archsimd.LoadFloat64x4Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] - b[i]
	}
}

func MulF64(dst, a, b []float64) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] * b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+4 <= n; i += 4 {
		archsimd.LoadFloat64x4Slice(a[i:]).Mul(archsimd.LoadFloat64x4Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func DivF64(dst, a, b []float64) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] / b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+4 <= n; i += 4 {
		archsimd.LoadFloat64x4Slice(a[i:]).Div(archsimd.LoadFloat64x4Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] / b[i]
	}
}

func AddF32(dst, a, b []float32) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] + b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+8 <= n; i += 8 {
		archsimd.LoadFloat32x8Slice(a[i:]).Add(archsimd.LoadFloat32x8Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

func SubF32(dst, a, b []float32) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] - b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+8 <= n; i += 8 {
		archsimd.LoadFloat32x8Slice(a[i:]).Sub(archsimd.LoadFloat32x8Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] - b[i]
	}
}

func MulF32(dst, a, b []float32) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] * b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+8 <= n; i += 8 {
		archsimd.LoadFloat32x8Slice(a[i:]).Mul(archsimd.LoadFloat32x8Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func DivF32(dst, a, b []float32) {
	if !hasAVX {
		for i := range dst {
			dst[i] = a[i] / b[i]
		}
		return
	}
	i, n := 0, len(dst)
	for ; i+8 <= n; i += 8 {
		archsimd.LoadFloat32x8Slice(a[i:]).Div(archsimd.LoadFloat32x8Slice(b[i:])).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] / b[i]
	}
}

var hasFMA = archsimd.X86.FMA()

// exp constants for ExpSumF64: full-f64 Cody-Waite ln2 split + degree-13 Taylor
// of eʳ over |r|≤ln2/2 (remainder ≈4e-18, ~1 ulp), scaled by 2ᵏ built through
// int32→sign-extend→VPSLLQ (all AVX2, avoiding the AVX-512-only f64→i64 convert).
var (
	eLo    = archsimd.BroadcastFloat64x4(-708.0)
	eLog2e = archsimd.BroadcastFloat64x4(1.4426950408889634)
	eNHi   = archsimd.BroadcastFloat64x4(-6.93147180369123816490e-01)
	eNLo   = archsimd.BroadcastFloat64x4(-1.90821492927058770002e-10)
	eBias  = archsimd.BroadcastInt32x4(1023)
	eOne   = archsimd.BroadcastFloat64x4(1.0)
	eZero  = archsimd.BroadcastFloat64x4(0.0)
	eC13   = archsimd.BroadcastFloat64x4(1.6059043836821613e-10)
	eC12   = archsimd.BroadcastFloat64x4(2.08767569878681e-09)
	eC11   = archsimd.BroadcastFloat64x4(2.505210838544172e-08)
	eC10   = archsimd.BroadcastFloat64x4(2.755731922398589e-07)
	eC9    = archsimd.BroadcastFloat64x4(2.7557319223985893e-06)
	eC8    = archsimd.BroadcastFloat64x4(2.48015873015873e-05)
	eC7    = archsimd.BroadcastFloat64x4(1.984126984126984e-04)
	eC6    = archsimd.BroadcastFloat64x4(1.388888888888889e-03)
	eC5    = archsimd.BroadcastFloat64x4(8.333333333333333e-03)
	eC4    = archsimd.BroadcastFloat64x4(4.166666666666666e-02)
	eC3    = archsimd.BroadcastFloat64x4(1.6666666666666666e-01)
	eC2    = archsimd.BroadcastFloat64x4(0.5)
	eAbs   = archsimd.BroadcastUint64x4(0x7fffffffffffffff) // clear sign bit → |x|
)

// Cephes double-log rational constants for softplus's log1p (folded on v=1+u∈(1,2],
// so no bit exponent extract) — the same recipe as backend/cpu's softplusF64x4.
var (
	spSqrt2 = archsimd.BroadcastFloat64x4(1.4142135623730951)
	spLn2Hi = archsimd.BroadcastFloat64x4(6.93359375e-1)
	spLn2Lo = archsimd.BroadcastFloat64x4(-2.121944400546905827679e-4)
	spTwo   = archsimd.BroadcastFloat64x4(2.0)
	spP0    = archsimd.BroadcastFloat64x4(1.01875663804580931796e-4)
	spP1    = archsimd.BroadcastFloat64x4(4.97494994976747001425e-1)
	spP2    = archsimd.BroadcastFloat64x4(4.70579119878881725854e0)
	spP3    = archsimd.BroadcastFloat64x4(1.44989225341610930846e1)
	spP4    = archsimd.BroadcastFloat64x4(1.79368678507819816313e1)
	spP5    = archsimd.BroadcastFloat64x4(7.70838733755885391666e0)
	spQ0    = archsimd.BroadcastFloat64x4(1.12873587189167450590e1)
	spQ1    = archsimd.BroadcastFloat64x4(4.52279145837532221105e1)
	spQ2    = archsimd.BroadcastFloat64x4(8.29875266912776603211e1)
	spQ3    = archsimd.BroadcastFloat64x4(7.11544750618563894466e1)
	spQ4    = archsimd.BroadcastFloat64x4(2.31251620126765340583e1)
)

// softplusF64x4v returns softplus(x) = max(x,0)+log1p(e^(−|x|)) per lane, ~1 ulp,
// any sign — the log1p rides expF64x4v + the Cephes double-log rational.
func softplusF64x4v(x archsimd.Float64x4) archsimd.Float64x4 {
	ax := x.AsUint64x4().And(eAbs).AsFloat64x4()
	u := expF64x4v(eZero.Sub(ax)) // e^(−|x|) ∈ (0,1]
	v := eOne.Add(u)              // 1+u ∈ (1,2]
	fold := v.GreaterEqual(spSqrt2)
	f := v.Mul(eC2).Sub(eOne).Merge(v.Sub(eOne), fold) // fold ? v/2−1 : v−1
	e := eOne.Merge(eZero, fold)                       // fold ? 1 : 0
	z := f.Mul(f)
	p := spP0
	p = p.MulAdd(f, spP1)
	p = p.MulAdd(f, spP2)
	p = p.MulAdd(f, spP3)
	p = p.MulAdd(f, spP4)
	p = p.MulAdd(f, spP5)
	q := f.Add(spQ0)
	q = q.MulAdd(f, spQ1)
	q = q.MulAdd(f, spQ2)
	q = q.MulAdd(f, spQ3)
	q = q.MulAdd(f, spQ4)
	y := f.Mul(z.Mul(p).Div(q))
	y = e.MulAdd(spLn2Lo, y)
	y = y.Sub(z.Mul(eC2))
	r := f.Add(y)
	r = e.MulAdd(spLn2Hi, r)
	return x.Max(eZero).Add(r)
}

func softplusScalar(x float64) float64 {
	if x > 0 {
		return x + math.Log1p(math.Exp(-x))
	}
	return math.Log1p(math.Exp(x))
}

// SoftplusNegLLSumF64 returns Σ softplus((1−2·y[i])·f[i]) — the binary cross-entropy
// SUM for labels y∈{0,1} (y=1 → softplus(−f), y=0 → softplus(f)), computed via
// softplus so there is no eps clamp and no separate sigmoid+log. 4-wide AVX2+FMA;
// scalar tail + non-AVX fall back to the scalar softplus. ~1 ulp — the caller (GBM
// trainLoss) is monitoring-only (goldens gate on the loss curve's monotonicity, not
// its exact value; predictions come from the trees).
func SoftplusNegLLSumF64(f, y []float64) float64 {
	if !hasAVX || !hasFMA {
		var s float64
		for i := range f {
			s += softplusScalar((1 - 2*y[i]) * f[i])
		}
		return s
	}
	acc := eZero
	i, n := 0, len(f)
	for ; i+4 <= n; i += 4 {
		fv := archsimd.LoadFloat64x4Slice(f[i:])
		yv := archsimd.LoadFloat64x4Slice(y[i:])
		acc = acc.Add(softplusF64x4v(eOne.Sub(yv.Mul(spTwo)).Mul(fv))) // softplus((1−2y)·f)
	}
	var lanes [4]float64
	acc.StoreSlice(lanes[:])
	s := lanes[0] + lanes[1] + lanes[2] + lanes[3]
	for ; i < n; i++ {
		s += softplusScalar((1 - 2*y[i]) * f[i])
	}
	return s
}

// expF64x4v returns eˣ (~1 ulp) per lane for x ≤ 0 (the softmax numerator feeds
// it z−max ≤ 0). Masked/−Inf lanes clamp to eLo → ~3e-308 (≈0 after normalize).
func expF64x4v(x archsimd.Float64x4) archsimd.Float64x4 {
	x = x.Max(eLo)
	kf := x.Mul(eLog2e).RoundToEven()
	r := kf.MulAdd(eNHi, x)
	r = kf.MulAdd(eNLo, r)
	p := eC13
	p = p.MulAdd(r, eC12)
	p = p.MulAdd(r, eC11)
	p = p.MulAdd(r, eC10)
	p = p.MulAdd(r, eC9)
	p = p.MulAdd(r, eC8)
	p = p.MulAdd(r, eC7)
	p = p.MulAdd(r, eC6)
	p = p.MulAdd(r, eC5)
	p = p.MulAdd(r, eC4)
	p = p.MulAdd(r, eC3)
	p = p.MulAdd(r, eC2)
	p = p.MulAdd(r, eOne)
	p = p.MulAdd(r, eOne)
	scale := kf.ConvertToInt32().Add(eBias).ExtendToInt64().ShiftAllLeft(52).AsFloat64x4()
	return p.Mul(scale)
}

// ExpSumF64 sets dst[i] = exp(src[i]-bias) and returns Σ dst[i], 4-wide AVX2+FMA.
// The exp is ~1 ulp; the caller (nlp large-vocab softmax) rides an f64 tolerance
// (Dist golden 1e-12). Scalar tail + a pre-AVX/pre-FMA CPU fall back to math.Exp.
func ExpSumF64(dst, src []float64, bias float64) float64 {
	if !hasAVX || !hasFMA {
		var sum float64
		for i, v := range src {
			e := math.Exp(v - bias)
			dst[i] = e
			sum += e
		}
		return sum
	}
	vb := archsimd.BroadcastFloat64x4(bias)
	acc := eZero
	i, n := 0, len(src)
	for ; i+4 <= n; i += 4 {
		e := expF64x4v(archsimd.LoadFloat64x4Slice(src[i:]).Sub(vb))
		e.StoreSlice(dst[i:])
		acc = acc.Add(e)
	}
	var lanes [4]float64
	acc.StoreSlice(lanes[:])
	sum := lanes[0] + lanes[1] + lanes[2] + lanes[3]
	for ; i < n; i++ {
		e := math.Exp(src[i] - bias)
		dst[i] = e
		sum += e
	}
	return sum
}

// ExpScaledF64 writes dst[i] = exp(scale·src[i]) 4-wide via AVX2 — the SSM/Mamba
// state-decay kernel (abar = exp(Δ·A), A<0 so the argument is ≤ 0). The 4-lane
// body runs the expF64x4v poly; the len%4 tail uses math.Exp, exactly as ExpSumF64
// does. Callers that need cross-path bit-parity (Mamba prefill vs decode) get it for
// free by passing the same-length slice on both paths — the body/tail split, and so
// every element's value, is identical. The non-SIMD build below is math.Exp for all.
func ExpScaledF64(dst, src []float64, scale float64) {
	if !hasAVX || !hasFMA {
		for i, v := range src {
			dst[i] = math.Exp(scale * v)
		}
		return
	}
	vs := archsimd.BroadcastFloat64x4(scale)
	i, n := 0, len(src)
	for ; i+4 <= n; i += 4 {
		expF64x4v(archsimd.LoadFloat64x4Slice(src[i:]).Mul(vs)).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = math.Exp(scale * src[i])
	}
}

// SigmoidF64 sets dst[i] = 1/(1+e^(−src[i])), 4-wide AVX2+FMA, in the overflow-safe
// form z = e^(−|x|), sigmoid = (x≥0 ? 1 : z)/(1+z) — the logistic gradient the GBM
// residual (yᵢ − σ(fᵢ)) recomputes every boosting round. ~1 ulp; the scalar tail is
// the exact overflow-safe form. Rides the caller's tolerance (GBM goldens gate at
// 1e-2 R² vs sklearn). Scalar fallback off the AVX/FMA path.
func SigmoidF64(dst, src []float64) {
	if !hasAVX || !hasFMA {
		for i, x := range src {
			if x >= 0 {
				dst[i] = 1 / (1 + math.Exp(-x))
			} else {
				z := math.Exp(x)
				dst[i] = z / (1 + z)
			}
		}
		return
	}
	i, n := 0, len(src)
	for ; i+4 <= n; i += 4 {
		x := archsimd.LoadFloat64x4Slice(src[i:])
		z := expF64x4v(eZero.Sub(x.AsUint64x4().And(eAbs).AsFloat64x4())) // e^(−|x|)
		num := eOne.Merge(z, x.GreaterEqual(eZero))                       // x≥0 ? 1 : z
		num.Div(eOne.Add(z)).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		x := src[i]
		if x >= 0 {
			dst[i] = 1 / (1 + math.Exp(-x))
		} else {
			z := math.Exp(x)
			dst[i] = z / (1 + z)
		}
	}
}

// WKVScanF64 runs the RWKV-4 WKV time-mixing recurrence (Peng et al. 2023,
// arXiv:2305.13048): k,v are [seq,d] row-major; w,u are [d]; out is [seq,d]. The d
// CHANNELS are independent (each carries its own aa/bb/pp state), so this
// vectorizes the scan over d 4-wide. Every exp argument is ≤0 (max-subtracted, the
// log-space stabilization), so expF64x4v is valid. Same operations, per channel, as
// wkvScanScalar; the len%4 channel tail runs the scalar form. ~1 ulp per exp vs
// libm — the RWKV caller rides the model f64 tolerance; the non-AVX build is the
// bit-exact scalar path.
func WKVScanF64(k, v, w, u, out []float64, seq, d int) {
	WKVScanStateF64(k, v, w, u, out, nil, nil, nil, seq, d)
}

// WKVScanStateF64 is WKVScanF64 generalized to a CONTINUING scan against persistent
// per-channel state. When aa0/bb0/pp0 (each [d]) are non-nil, every channel resumes
// from its carried (aa0[c], bb0[c], pp0[c]) instead of the fresh (0, 0, -1e38) start,
// and the final state is written back so the caller can absorb the next token chunk —
// this is the RWKV decode/prefill recurrence, which rides persistent AA/BB/PP. Passing
// nil for all three is exactly WKVScanF64 (the fresh forward scan): the nil branch sits
// outside the per-token loop, so the fresh path keeps WKVScanF64's throughput and stays
// bit-identical to it. Because StoreSlice/Load round-trips the f64 state losslessly,
// chunking a sequence through this (decode) is bit-exact with one whole-sequence scan
// (forward) — the property RWKV's forward/decode parity depends on.
func WKVScanStateF64(k, v, w, u, out, aa0, bb0, pp0 []float64, seq, d int) {
	if !hasAVX || !hasFMA {
		wkvScanStateScalar(k, v, w, u, out, aa0, bb0, pp0, seq, d, 0, d)
		return
	}
	pInit := archsimd.BroadcastFloat64x4(-1e38)
	c := 0
	for ; c+4 <= d; c += 4 {
		wc := archsimd.LoadFloat64x4Slice(w[c:])
		uc := archsimd.LoadFloat64x4Slice(u[c:])
		aa, bb, pp := eZero, eZero, pInit
		if aa0 != nil {
			aa = archsimd.LoadFloat64x4Slice(aa0[c:])
			bb = archsimd.LoadFloat64x4Slice(bb0[c:])
			pp = archsimd.LoadFloat64x4Slice(pp0[c:])
		}
		for t := 0; t < seq; t++ {
			base := t*d + c
			kk := archsimd.LoadFloat64x4Slice(k[base:])
			vv := archsimd.LoadFloat64x4Slice(v[base:])
			ww := uc.Add(kk)
			q := pp.Max(ww)
			e1 := expF64x4v(pp.Sub(q))
			e2 := expF64x4v(ww.Sub(q))
			e1.Mul(aa).Add(e2.Mul(vv)).Div(e1.Mul(bb).Add(e2)).StoreSlice(out[base:])
			ppw := pp.Sub(wc)
			q2 := ppw.Max(kk)
			e3 := expF64x4v(ppw.Sub(q2))
			e4 := expF64x4v(kk.Sub(q2))
			aa = e3.Mul(aa).Add(e4.Mul(vv))
			bb = e3.Mul(bb).Add(e4)
			pp = q2
		}
		if aa0 != nil {
			aa.StoreSlice(aa0[c:])
			bb.StoreSlice(bb0[c:])
			pp.StoreSlice(pp0[c:])
		}
	}
	if c < d {
		wkvScanStateScalar(k, v, w, u, out, aa0, bb0, pp0, seq, d, c, d)
	}
}

// SSMScanF64 runs the Mamba/S6 selective-scan recurrence (Gu & Dao 2023), vectorizing
// the reduction over the state dim N 4-wide: per (t,d) the decay abar = exp(Δ·A[d,n])
// (A<0 ⟹ argument ≤ 0, so expF64x4v is valid), the state update h = abar·h + Δ·B·u,
// and the output y = Σ_n C·h — the exp is the kernel's dominant cost (~65% scalar).
// Same operations and iteration order as ssmScanScalar; the len%4 state-dim tail runs
// the scalar form (N is a multiple of 4 for every real Mamba — 16/128/256 — so the
// tail is dead in practice). h (D·N) is the scan state, allocated by the caller and
// zero at entry. ~1 ulp per exp vs libm; the caller rides the model f64 tolerance.
//
//	u,delta [L,D]; as (A) [D,N]; bs (B), cs (C) [L,N]; dsk (D-skip) [D] or nil; out [L,D].
func SSMScanF64(u, delta, as, bs, cs, dsk, out, h []float64, L, D, N int) {
	if !hasAVX || !hasFMA {
		ssmScanScalar(u, delta, as, bs, cs, dsk, out, h, L, D, N, 0, N)
		return
	}
	nMain := N - N%4
	for t := 0; t < L; t++ {
		for d := 0; d < D; d++ {
			dt := delta[t*D+d]
			ut := u[t*D+d]
			base := d * N
			tn := t * N
			dtVec := archsimd.BroadcastFloat64x4(dt)
			dtut := archsimd.BroadcastFloat64x4(dt * ut)
			yacc := eZero
			for n := 0; n < nMain; n += 4 {
				abar := expF64x4v(dtVec.Mul(archsimd.LoadFloat64x4Slice(as[base+n:])))
				hv := abar.Mul(archsimd.LoadFloat64x4Slice(h[base+n:])).
					Add(dtut.Mul(archsimd.LoadFloat64x4Slice(bs[tn+n:])))
				hv.StoreSlice(h[base+n:])
				yacc = yacc.Add(archsimd.LoadFloat64x4Slice(cs[tn+n:]).Mul(hv))
			}
			var lanes [4]float64
			yacc.StoreSlice(lanes[:])
			y := lanes[0] + lanes[1] + lanes[2] + lanes[3]
			if dsk != nil { // AVX main owns the per-d skip (the tail pass must not re-add it)
				y += dsk[d] * ut
			}
			out[t*D+d] = y
		}
	}
	if nMain < N {
		ssmScanScalar(u, delta, as, bs, cs, dsk, out, h, L, D, N, nMain, N)
	}
}
