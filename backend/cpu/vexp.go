package cpu

import (
	"math"
	"runtime"
)

// Vectorized f32 exp for the gemm-routed MHA softmax (§T657 follow-up): after
// the NEON GEMM landed, scalar math.Exp was ~35% of the MHA forward. The vexp
// path replaces the per-element f64 math.Exp (plus the f64 scratch row and the
// serial f64 sum around it) with an f32-native range-reduced polynomial —
// exp(x) = 2^n·exp(r), n = rint(x·log2e), r = x − n·ln2 — evaluated 4-wide in
// NEON on the arm64 perf build (vexp_arm64.s). Everything here is gated by the
// vexpNeon const (vexp_arm64.go / vexp_default.go): the default build and the
// amd64 perf build never take this path, so their numerics are bit-identical
// to before. Accuracy: the poly's max relative error vs math.Exp is ~3e-7
// over the full f32 exp domain (TestExpF32Accuracy) — two orders below the
// f32 MHA parity budget (assertCloseUlps rtol 5e-5) that already covers the
// f32-native score/output gemms feeding and consuming these weights.
//
// Cephes expf constants: ln2 is split hi/lo so n·ln2Hi is exact (ln2Hi has a
// 12-bit mantissa; |n| ≤ 127 adds ≤ 7 bits), keeping the reduced r accurate to
// f32 ulps; the degree-5 poly then covers r ∈ [−ln2/2, ln2/2].
const (
	expLog2e = 1.44269504088896341 // 1/ln2
	expLn2Hi = 0.693359375
	expLn2Lo = -2.12194440e-4
	// Clamp bounds: below expLoClamp exp underflows f32 normals (the clamped
	// result ~1e-38 is an exact-enough zero for a softmax weight against a row
	// sum ≥ 1); above expHiClamp the 2^n exponent trick would overflow the
	// exponent field (n > 127). The softmax feeds x−max ≤ 0, so the hi clamp
	// is defensive only.
	expLoClamp = -87.33654
	//perfscan:ignore PS6023 const declaration (expHiClamp), not a loop; false positive
	expHiClamp = 88.0
	// expMagic = 1.5·2^23: adding then subtracting it rounds an f32 in
	// (−2^22, 2^22) to the nearest integer (round-half-even, matching the NEON
	// kernel's FRINTN) without a float64 round trip.
	expMagic = 12582912.0
)

// expF32 is the scalar instantiation of the vexp polynomial — the tail lanes
// of the NEON kernel (len%4) and the type-check fallback build use it. Branch-
// free except for the clamps; ~ulp-level agreement with the NEON lanes (both
// evaluate the same Horner chain; only FMA contraction may differ).
func expF32(x float32) float32 {
	if x < expLoClamp {
		x = expLoClamp
	}
	if x > expHiClamp {
		x = expHiClamp
	}
	zm := x*expLog2e + expMagic
	n := zm - expMagic
	r := x - n*expLn2Hi
	r -= n * expLn2Lo
	p := float32(1.9875691500e-4)
	p = p*r + 1.3981999507e-3
	p = p*r + 8.3334519073e-3
	p = p*r + 4.1665795894e-2
	p = p*r + 1.6666665459e-1
	p = p*r + 0.5
	return (p*r*r + r + 1) * math.Float32frombits(uint32(int32(n)+127)<<23)
}

// GELU on the vexp leaf (§C26 assemble-on-a-verified-leaf): the f32 GELU fast
// path evaluates erf via Abramowitz-Stegun 7.1.26 — erf(x) ≈ sign(x)·(1 −
// t·(a1 + t·(a2 + t·(a3 + t·(a4 + t·a5))))·e^(−x²)), t = 1/(1+p·|x|) — whose
// only transcendental is e^(−x²), i.e. the vexp exp primitive above. So
// gelu(x) = x·(0.5 + 0.5·erf(x/√2)) becomes a NEON pipeline (vgeluQuadsNeonF32
// in vexp_arm64.s) with no f64 math.Erf. Gated exactly like vexp: only the
// arm64 perf build (vexpNeon) routes the OpGELU F32 kernel here; every other
// build keeps the scalar f64 math.Erf path bit-for-bit. Accuracy vs the exact
// erf GELU (TestGeluF32Accuracy): max abs err ~5e-7, rel err ≤ ~2e-4 wherever
// |gelu| > 1e-6 — far inside the ADR-0021 f32 tolerance (rtol 2e-3). The
// AS-7.1.26 absolute erf error (~1.5e-7) makes the deep-negative tail
// saturate to −0 slightly early (|gelu| < ~1.5e-6 there), which the combined
// atol+rtol budget covers.
const (
	geluInvSqrt2 = float32(0.7071067811865476)
	geluP        = float32(0.3275911)
	geluA1       = float32(0.254829592)
	geluA2       = float32(-0.284496736)
	geluA3       = float32(1.421413741)
	geluA4       = float32(-1.453152027)
	geluA5       = float32(1.061405429)
)

// geluF32 is the scalar instantiation of the vgelu pipeline — the tail lanes
// (len%4) and the type-check fallback build use it. Same operations per
// element as the NEON lanes (only FMA contraction may differ): AS-7.1.26 erf
// magnitude on expF32, sign re-applied bitwise from x (E ∈ [0,1], so the OR
// is exact; NaN payloads stay NaN), then x·(0.5 + 0.5·erf).
func geluF32(x float32) float32 {
	a := math.Float32frombits(math.Float32bits(x*geluInvSqrt2) &^ (1 << 31)) // |x/√2|
	t := 1 / (1 + geluP*a)
	s := geluA4 + geluA5*t
	s = geluA3 + s*t
	s = geluA2 + s*t
	s = geluA1 + s*t
	e := expF32(-(a * a))
	E := 1 - s*t*e
	erf := math.Float32frombits(math.Float32bits(E) | math.Float32bits(x)&(1<<31))
	return x * (0.5 + 0.5*erf)
}

// GELU BACKWARD on the same leaf (§T664): the exact-erf GELU derivative is
// d/dx[0.5·x·(1+erf(x/√2))] = Φ(x) + x·φ(x), Φ(x) = 0.5·(1+erf(x/√2)),
// φ(x) = e^(−x²/2)/√(2π) — the SAME two primitives the vgelu forward computes:
// with a = |x|/√2, the AS-7.1.26 erf magnitude already needs e^(−a²) = e^(−x²/2),
// so ONE exp evaluation feeds both erf and the pdf. The VJP kernel then does
// dx = g·(Φ + x·φ) per element (formula identical to ref's geluGrad, §T353 —
// only the evaluation is f32/NEON). Gated exactly like vexp/vgelu: only the
// arm64 perf build (vexpNeon) registers the OpGELUBackward F32 kernel; every
// other build keeps the ref scalar f64 path bit-for-bit.
//
// geluGradPdfMin guards the exp underflow clamp: expF32 pins deep-negative
// inputs at ~1.1755e-38 instead of a true zero, which is harmless inside the
// erf (≤1e-38 on a value near 1) but NOT inside x·φ(x) — a huge |x| would
// resurrect the clamped residue (and x=±Inf must yield Inf·0 = NaN exactly
// like ref's f64 math.Exp path). Zeroing e when it is at-or-below the clamp
// floor (any true pdf that small contributes ≤ 6e-39·|x| ≈ 0 anyway, since
// unclamped e < 1.3e-38 implies |x| < 13.3) restores exact ±Inf/NaN semantics.
const (
	geluInvSqrt2Pi = float32(0.3989422804014327) // 1/√(2π), pdf normalizer
	geluGradPdfMin = float32(1.3e-38)            // just above expF32's underflow-clamp output
)

// geluGradF32 is the scalar instantiation of the vgeluGrad pipeline — the tail
// lanes (len%4) and the type-check fallback build use it. Same operations per
// element as the NEON lanes (only FMA contraction may differ). Note the
// !(e > min) spelling: it must zero e for NaN exp results too, matching the
// NEON FCMGT+AND mask (NaN still propagates through t/s and the erf bits).
func geluGradF32(x, g float32) float32 {
	a := math.Float32frombits(math.Float32bits(x*geluInvSqrt2) &^ (1 << 31)) // |x/√2|
	t := 1 / (1 + geluP*a)
	s := geluA4 + geluA5*t
	s = geluA3 + s*t
	s = geluA2 + s*t
	s = geluA1 + s*t
	e := expF32(-(a * a)) // e^(−x²/2): shared by erf and pdf
	//perfscan:ignore PS3070 geluGradF32 scalar tail-lane fallback (NEON prod on arm64); not a loop
	if !(e > geluGradPdfMin) {
		e = 0
	}
	E := 1 - s*t*e
	erf := math.Float32frombits(math.Float32bits(E) | math.Float32bits(x)&(1<<31))
	phi := float32(0.5) + 0.5*erf // Φ(x)
	pdf := geluInvSqrt2Pi * e     // φ(x)
	return g * (phi + x*pdf)
}

// SiLU / sigmoid on the same leaf (§T665, the §C26 pattern's 4th application):
// the LLAMA/Qwen/Mistral FFN activation silu(x) = x·σ(x) and the raw σ(x) are
// one exp away from the vexp primitive. Both use the numerically stable split
// the scalar sigmoidKernelCPU already uses, folded branch-free: with
// z = e^(−|x|) (always ≤ 1, so only the exp LO clamp is live),
//
//	σ(x) = num/(1+z),  num = (x ≥ 0 ? 1 : z)
//
// — for x ≥ 0 that is 1/(1+e^(−x)), for x < 0 it is e^x/(1+e^x); no positive
// magnitude is ever exponentiated. Then silu(x) = x·num/(1+z), and the SiLU
// derivative σ(x)·(1+x·(1−σ(x))) folds to a single division: 1−σ(x) =
// other/(1+z) with num·other = z in BOTH branches, so
//
//	silu'(x) = (num·(1+z) + x·z) / (1+z)²
//
// — algebraically identical to ref's s·(1+x·(1−s)) (§T362); only the
// evaluation is f32/NEON, riding the ADR-0021 tolerant parity budget.
//
// The geluGradPdfMin mask (above) is reused on z for silu/silu' — expF32's
// underflow clamp pins deep-negative z at ~1.18e-38 instead of 0, which a huge
// |x| would resurrect through x·z (and x = ±Inf must give Inf·0 = NaN exactly
// like ref's f64 path: silu(−Inf) = −Inf·σ(−Inf) = −Inf·0 = NaN). Zeroing
// sub-clamp z restores exact ±Inf semantics; any true z that small contributes
// ≤ |x|·1.3e-38 with |x| < 88 ≈ 0 anyway. NaN still propagates: the mask
// zeroes z, but num = z = 0 (x ≥ 0 is false for NaN) meets x·num = NaN·0 =
// NaN in the forward and x·z = NaN·0 = NaN in the derivative. Sigmoid has no
// ·x product to rescue NaN, so it keeps z unmasked — NaN flows through the
// poly into num/(1+z); the clamp residue σ(−Inf) ≈ 1.18e-38 (vs ref's exact 0)
// is inside the tolerant budget.

// siluF32 is the scalar instantiation of the vsilu pipeline — the tail lanes
// (len%4) and the type-check fallback build use it. Same operations per
// element as the NEON lanes (only FMA contraction may differ).
func siluF32(x float32) float32 {
	a := math.Float32frombits(math.Float32bits(x) &^ (1 << 31)) // |x|
	z := expF32(-a)
	if !(z > geluGradPdfMin) { // NaN-safe spelling, matches the NEON FCMGT+AND
		z = 0
	}
	num := z
	if x >= 0 {
		num = 1
	}
	return x * num / (1 + z)
}

// siluGradF32 is the scalar instantiation of the vsiluGrad pipeline — the tail
// lanes (len%4) and the type-check fallback build use it: dx = g·silu'(x) with
// silu' in the single-division form above.
func siluGradF32(x, g float32) float32 {
	a := math.Float32frombits(math.Float32bits(x) &^ (1 << 31)) // |x|
	z := expF32(-a)
	//perfscan:ignore PS3070 siluGradF32 scalar tail-lane fallback; amd64 uses f64 path, not this
	if !(z > geluGradPdfMin) {
		z = 0
	}
	num := z
	if x >= 0 {
		num = 1
	}
	den := 1 + z
	d := (num*den + x*z) / (den * den)
	return g * d
}

// sigmoidF32 is the scalar instantiation of the vsigmoid pipeline — the tail
// lanes (len%4) and the type-check fallback build use it. No pdf mask (see
// the section comment): NaN propagates through z itself.
func sigmoidF32(x float32) float32 {
	a := math.Float32frombits(math.Float32bits(x) &^ (1 << 31)) // |x|
	z := expF32(-a)
	num := z
	if x >= 0 {
		num = 1
	}
	return num / (1 + z)
}

// STANDALONE UNARIES on the same leaf (§T666, the §C26 pattern's 5th
// application): the T660-T665 campaign vectorized every FUSED transcendental
// (softmax/MHA/GELU±/crossentropy/SiLU±/sigmoid) but left the standalone
// OpExp / OpTanh / OpLog F32 kernels on scalar f64 math.Exp/Tanh/Log — a
// caller hitting backend.Execute(OpExp, ...) got the slow path while the same
// exp inside softmax was NEON. These close the gap. Gated exactly like the
// siblings: only the arm64 perf build (vexpNeon) routes the F32 kernels here;
// every other build keeps the scalar f64 paths bit-for-bit.
//
// exp: unlike the softmax's x−max ≤ 0, a standalone exp sees large +x, so the
// softmax kernel's single 2^n exponent-insertion (n ≤ 127) is not enough —
// f32 exp is finite up to x = 88.72283 (n = 128). expFullF32 splits the scale
// into 2^(n>>1)·2^(n−n>>1) (both factors' exponents stay in range for the
// whole clamped domain, n ∈ [−126, 128]) and raises the hi clamp above the
// overflow cutoff. Semantics vs the f64 ref: x > 88.72283172607421875 (the
// largest f32 with a finite float32(math.Exp(x)); bits 0x42B17217) → +Inf
// exactly; x < expLoClamp → 0 exactly (ref returns subnormals down to
// x ≈ −103.28 — the true values there are ≤ 1.18e-38, inside the atol
// budget — and 0 below; this also makes exp(−Inf) = 0 exact); a spuriously
// overflowing poly result just under the cutoff is pinned to MaxFloat32
// (FMIN); NaN propagates. Max rel err vs math.Exp over the full finite
// domain: ~8e-8 (TestExpFullF32Accuracy).
//
// tanh: the numerically stable sign-split form on one exp — for a = |x|,
// z = e^(−2a) ∈ (0, 1], tanh magnitude t = (1−z)/(1+z), sign re-applied
// bitwise from x (t ∈ [0,1], so the OR is exact; NaN payloads stay NaN). No
// positive magnitude is ever exponentiated, and both 1−z and 1+z are exact or
// half-ulp f32 ops, so the only error sources are z itself (~1e-7 abs near
// z=1) and the final divide: max abs err ~9e-8, rel err ≤ ~2e-4 for
// |x| ≥ 1e-5 (TestTanhF32Accuracy; the atol term covers smaller x, where
// tanh(x) = x±1e-7 anyway). Tails are exact: tanh(±Inf) = ±1 (z underflows,
// (1−z)/(1+z) rounds to 1), tanh(±0) = ±0.
//
// log: a NEW NEON primitive (not exp) — the Cephes logf reduction. x is split
// as m·2^e by exponent-field extraction (subnormals pre-scaled by 2^25 with
// e adjusted, so the full positive range is covered), m ∈ [1,2) folded to
// m ∈ [√½, √2) (m ≥ √2 → m/2, e+1), f = m−1 ∈ [−0.293, 0.414), then
// log(x) = e·ln2 + f + f³·P(f) − f²/2 with Cephes' degree-8 P and the same
// hi/lo ln2 split as exp (e·ln2Hi exact: 12-bit ln2Hi mantissa, |e| ≤ 151).
// f = m−1 is exact (Sterbenz), so there is no catastrophic cancellation near
// x = 1 and log(1) = +0 exactly. Max rel err vs math.Log: ~8e-8 across
// [1e-30, 1e30] AND the subnormal range (TestLogF32Accuracy). Specials match
// ref exactly: log(±0) = −Inf, log(x<0) = NaN, log(+Inf) = +Inf, NaN
// propagates.
const (
	//perfscan:ignore PS6023 const declaration (expFullHiClamp); false positive
	expFullHiClamp = 89.0 // > the Inf cutoff; safe with split scaling (n ≤ 129)
	//perfscan:ignore PS6023 const declaration (expFullInfCut); false positive
	expFullInfCut = 88.72283172607421875               // bits 0x42B17217: largest f32 with finite f32(exp(x))
	logSqrtHalfx2 = float32(1.41421353816986083984375) // bits 0x3FB504F3: f32(√2), the m-fold threshold
	logMinNormal  = float32(1.1754943508222875e-38)    // bits 0x00800000: smallest normal f32
	logTwoP25     = float32(1 << 25)                   // subnormal pre-scale
	logLn2Hi      = float32(0.693359375)               // same hi/lo split as exp
	logLn2Lo      = float32(-2.12194440e-4)            //
	logL0         = float32(7.0376836292e-2)
	logL1         = float32(-1.1514610310e-1)
	logL2         = float32(1.1676998740e-1)
	logL3         = float32(-1.2420140846e-1)
	logL4         = float32(1.4249322787e-1)
	logL5         = float32(-1.6668057665e-1)
	logL6         = float32(2.0000714765e-1)
	logL7         = float32(-2.4999993993e-1)
	logL8         = float32(3.3333331174e-1)
)

// expFullF32 is the scalar instantiation of the full-domain vexpFull pipeline —
// the tail lanes (len%4) and the type-check fallback build use it. Same
// operations per element as the NEON lanes (only FMA contraction may differ):
// the vexp Cephes reduction with hi clamp 89, split 2^n scaling, MaxFloat32
// pin, then the NaN-safe underflow-zero and overflow-Inf masks on the
// ORIGINAL x (comparisons with NaN are false, so NaN keeps the poly's NaN).
func expFullF32(x float32) float32 {
	xc := x
	if xc < expLoClamp {
		xc = expLoClamp
	}
	if xc > expFullHiClamp {
		xc = expFullHiClamp
	}
	zm := xc*expLog2e + expMagic
	n := zm - expMagic
	r := xc - n*expLn2Hi
	r -= n * expLn2Lo
	p := float32(1.9875691500e-4)
	p = p*r + 1.3981999507e-3
	p = p*r + 8.3334519073e-3
	p = p*r + 4.1665795894e-2
	p = p*r + 1.6666665459e-1
	p = p*r + 0.5
	res := p*r*r + r + 1
	ni := int32(n)
	n1 := ni >> 1
	res *= math.Float32frombits(uint32(n1+127) << 23)
	res *= math.Float32frombits(uint32(ni-n1+127) << 23)
	if res > math.MaxFloat32 { // FMIN(res, MaxFloat32); NaN falls through
		res = math.MaxFloat32
	}
	if expLoClamp > x { // underflow → exact 0 (NaN-safe spelling)
		res = 0
	}
	if x > expFullInfCut { // overflow → exact +Inf (matches ref's cutoff)
		res = float32(math.Inf(1))
	}
	return res
}

// tanhF32 is the scalar instantiation of the vtanh pipeline — the tail lanes
// (len%4) and the type-check fallback build use it. Same operations per
// element as the NEON lanes (only FMA contraction may differ).
func tanhF32(x float32) float32 {
	a := math.Float32frombits(math.Float32bits(x) &^ (1 << 31)) // |x|
	z := expF32(-(a + a))                                       // e^(−2|x|) ≤ 1: only the lo clamp is live
	t := (1 - z) / (1 + z)
	return math.Float32frombits(math.Float32bits(t) | math.Float32bits(x)&(1<<31))
}

// softcapF32 is the scalar bit-twin of one vsoftcapF32 lane: cap·tanh(x/cap), the
// Gemma-2 logit soft-cap evaluated f32-native through the same expF32 primitive
// (via tanhF32) — the len%8 remainder tail and the no-AVX fallback. tanhF32 is
// tanh at cap=1, so this is that lane on x/cap, re-scaled by cap.
func softcapF32(x, cap float32) float32 { return cap * tanhF32(x/cap) }

// softplusF32 is the scalar bit-twin of one softplusF32x8 lane: the overflow-safe
// softplus(x) = max(x,0) + log(1+e^(−|x|)), evaluated f32-native through the same
// expF32/logF32 primitives. The log argument 1+e^(−|x|) ∈ (1,2] is always well
// conditioned, so no branch on the sign of x is needed (unlike ref's f64 form).
// The len%8 remainder tail and the no-AVX fallback.
func softplusF32(x float32) float32 {
	a := math.Float32frombits(math.Float32bits(x) &^ (1 << 31)) // |x|
	relu := x
	if relu < 0 {
		relu = 0
	}
	return relu + logF32(1+expF32(-a))
}

// logF32 is the scalar instantiation of the vlog pipeline — the tail lanes
// (len%4) and the type-check fallback build use it. Same operations per
// element as the NEON lanes (only FMA contraction may differ); the special-
// case selects mirror the NEON mask order exactly (ordered → zero → negative
// → +Inf, each later select winning).
func logF32(x float32) float32 {
	xs := x
	var eAdj int32
	if logMinNormal > x { // subnormal pre-scale (also 0/neg — fixed by masks below)
		xs = x * logTwoP25
		eAdj = 25
	}
	bx := math.Float32bits(xs)
	e := int32(bx>>23) - 127 - eAdj
	m := math.Float32frombits(bx&0x007FFFFF | 0x3F800000) // m ∈ [1,2)
	if m >= logSqrtHalfx2 {
		m *= 0.5
		e++
	}
	f := m - 1 // exact (Sterbenz)
	z := f * f
	p := logL0
	p = p*f + logL1
	p = p*f + logL2
	p = p*f + logL3
	p = p*f + logL4
	p = p*f + logL5
	p = p*f + logL6
	p = p*f + logL7
	p = p*f + logL8
	ef := float32(e)
	y := f * (z * p)
	y += ef * logLn2Lo
	y -= 0.5 * z
	r := f + y
	r += ef * logLn2Hi
	if !(x == x) { // NaN propagates (BSL on FCMEQ x,x)
		r = x
	}
	if x == 0 { // ±0 → −Inf
		r = float32(math.Inf(-1))
	}
	if x < 0 { // negative → NaN
		r = float32(math.NaN())
	}
	if x == float32(math.Inf(1)) { // +Inf → +Inf
		r = x
	}
	return r
}

// mhaSoftmaxBandVexpF32 is the arm64 perf-build body of mhaSoftmaxBandF32
// (reached only when vexpNeon; see the gate there): the same scale (+ALiBi) →
// max-shift exp → normalize pipeline, but f32-native end to end — no f64
// scratch row, exp+sum through vexpF32 (4-wide NEON), masked spans zeroed as
// before. Numerics: score scaling in f32 perturbs x by ≤ |x|·2⁻²⁴ (|x−max| ≤
// ~90 post-clamp → ≤ 6e-6 relative on the weight), the poly adds ~3e-7, and
// the 4-lane f32 sum of ≤ sk positives adds ≤ sk/4·2⁻²⁴ — all far inside the
// 5e-5 f32 MHA parity budget dominated by the surrounding f32-native gemms.
func mhaSoftmaxBandVexpF32(sb []float32, g mhaGeo, h, i0, iN int) {
	sk := g.sk
	scale := float32(g.scale)
	for r := range iN {
		i := i0 + r
		jmin, jmax := g.bounds(i)
		sr := sb[r*sk : (r+1)*sk : (r+1)*sk]
		span := sr[jmin:jmax]
		m := float32(math.Inf(-1))
		if g.slopes != nil {
			slope := float32(g.slopes[h])
			for jj, v := range span {
				x := v*scale + slope*float32(jmin+jj-(g.off+i))
				span[jj] = x
				if x > m {
					m = x
				}
			}
		} else {
			// No-slope band: scale + row-max + ×1/sum all run 8-wide (rowMaxF32/scaleRowF32 are the
			// AVX2 amd64 primitives, scalar elsewhere) — the exp already goes through vexpRowF32.
			scaleRowF32(span, scale)
			m = rowMaxF32(span)
			sum := vexpRowF32(span, m)
			scaleRowF32(span, 1/sum)
			clear(sr[:jmin])
			clear(sr[jmax:])
			continue
		}
		sum := vexpRowF32(span, m)
		inv := 1 / sum
		for jj := range span {
			span[jj] *= inv
		}
		clear(sr[:jmin])
		clear(sr[jmax:])
	}
}

// vexpRowF32 computes row[j] = exp(row[j]-m) in place over the whole row —
// the 4-wide NEON kernel for the quad-aligned body, the scalar poly for the
// len%4 tail — and returns the f32 sum of the results. This is THE shared
// exp+sum band: the MHA band softmax, the OpSoftmax f32 fast path and the
// FlashAttn online-softmax block all route through it (one asm kernel, one
// tail contract).
func vexpRowF32(row []float32, m float32) float32 {
	nv := len(row) &^ 3
	var sum float32
	if nv > 0 {
		sum = vexpF32(row[:nv], m)
	}
	for j := nv; j < len(row); j++ {
		e := expF32(row[j] - m)
		row[j] = e
		sum += e
	}
	return sum
}

// softmaxVexpF32 is the arm64 perf-build body of the standalone OpSoftmax
// kernel for F32 (softmaxKernelCPU gates on vexpNeon — the default build and
// amd64 keep softmaxTyped's f64 math.Exp path bit-for-bit): the same stable
// max-shift softmax, but f32-native end to end — f32 row max, exp+sum through
// vexpRowF32 (4-wide NEON), ×1/sum normalize in f32. Same rows==1 huge-d
// split as softmaxTyped (§C3). Numerics ride the ADR-0021 tolerant-parity
// budget: the poly adds ~3e-7 relative, the 4-lane f32 sum of ≤ d positives
// adds ≤ d/4·2⁻²⁴ (1.5e-5 at d=1024, observed ~1e-6 — TestSoftmaxVexpF32
// ParityVsF64Ref), all far inside rtol 2e-3 and inside the 5e-5 f32 kernel
// parity budget at the ref-parity test shapes.
func softmaxVexpF32(x, out []float32, rows, d int) {
	if nw := runtime.GOMAXPROCS(0); rows == 1 && d >= wideSoftmaxMinD && nw > 1 {
		softmaxWideVexpF32(x, out, d, nw)
		return
	}
	parallelWork(rows, 4*d, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			base := r * d
			xr := x[base : base+d : base+d]
			or := out[base : base+d : base+d]
			m := rowMaxF32(xr) // AVX2 max (amd64-SIMD) / scalar −Inf-start reduction elsewhere
			copy(or, xr)
			sum := vexpRowF32(or, m)
			scaleRowF32(or, 1/sum) // AVX2 ×1/sum / scalar elsewhere
		}
	})
}

// softmaxWideVexpF32 is softmaxVexpF32's few-rows × huge-d form (LLM logit
// shape, 1×32000), parallel WITHIN the row like softmaxWide: chunked f32 max,
// then per-chunk vexp exp+sum (chunks quad-aligned so only the last chunk has
// a scalar tail), chunk sums combined in f64, parallel ×1/sum. No f64 scratch
// row — exp streams straight into out.
func softmaxWideVexpF32(x, out []float32, d, nw int) {
	chunk := ((d+nw-1)/nw + 3) &^ 3
	nch := (d + chunk - 1) / chunk
	maxs := make([]float32, nch)
	sums := make([]float64, nch)
	parallelWork(nch, 4*chunk, func(lo, hi int) {
		for c := lo; c < hi; c++ {
			maxs[c] = rowMaxF32(x[c*chunk : min((c+1)*chunk, d)]) // AVX2 max / scalar elsewhere
		}
	})
	m := float32(math.Inf(-1))
	for _, cm := range maxs {
		if cm > m {
			m = cm
		}
	}
	parallelWork(nch, 4*chunk, func(lo, hi int) {
		for c := lo; c < hi; c++ {
			j0, j1 := c*chunk, min((c+1)*chunk, d)
			or := out[j0:j1]
			copy(or, x[j0:j1])
			sums[c] = float64(vexpRowF32(or, m))
		}
	})
	var sum float64
	for _, s := range sums {
		sum += s
	}
	inv := float32(1 / sum)
	parallelWork(d, 2, func(lo, hi int) {
		scaleRowF32(out[lo:hi], inv) // AVX2 ×1/sum / scalar elsewhere
	})
}
