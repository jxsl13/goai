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
	if !(e > geluGradPdfMin) {
		e = 0
	}
	E := 1 - s*t*e
	erf := math.Float32frombits(math.Float32bits(E) | math.Float32bits(x)&(1<<31))
	phi := float32(0.5) + 0.5*erf // Φ(x)
	pdf := geluInvSqrt2Pi * e     // φ(x)
	return g * (phi + x*pdf)
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
			for jj, v := range span {
				x := v * scale
				span[jj] = x
				if x > m {
					m = x
				}
			}
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
			m := float32(math.Inf(-1)) // −Inf start keeps ref's NaN/−Inf row semantics
			for _, v := range xr {
				if v > m {
					m = v
				}
			}
			copy(or, xr)
			sum := vexpRowF32(or, m)
			inv := 1 / sum
			for j, v := range or {
				or[j] = v * inv
			}
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
			m := float32(math.Inf(-1))
			for _, v := range x[c*chunk : min((c+1)*chunk, d)] {
				if v > m {
					m = v
				}
			}
			maxs[c] = m
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
		for j := lo; j < hi; j++ {
			out[j] *= inv
		}
	})
}
