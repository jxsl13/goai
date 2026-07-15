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
