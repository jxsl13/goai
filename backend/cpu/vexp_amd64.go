//go:build goexperiment.simd

package cpu

import "simd/archsimd"

// amd64 perf build: the MHA/softmax-family exp+sum runs through an AVX2 8-wide vexp (this file),
// the twin of the arm64 NEON kernel (vexp_arm64.s). vexpF32Fast gates ONLY the exp+sum softmax
// paths (mhaSoftmaxBandF32 etc.); vexpNeon stays false here so the GELU/SiLU/exp/tanh/log activation
// kernels keep their existing scalar-f64 paths bit-for-bit (only the softmax exp changes on amd64).
const (
	vexpNeon    = false
	vexpF32Fast = true
)

var vexpHasAVX = archsimd.X86.AVX() && archsimd.X86.FMA()

// vexpF32 computes p[i] = exp(p[i]-m) in place and returns Σ p[i], 8-wide via AVX2 FMA — the amd64
// twin of the NEON vexpF32 (vexp_arm64.s). It evaluates the SAME Cephes range-reduced degree-5
// polynomial as the scalar expF32 — exp(x) = 2ⁿ·exp(r), n = rint(x·log2e), r = x − n·ln2 — so its
// accuracy matches (~3e-7 max rel err vs math.Exp, TestExpF32Accuracy) and rides the ADR-0021 f32
// tolerance the surrounding f32-native score/output gemms already use. len(p) is a multiple of 4
// (the shared vexpRowF32 contract): an 8-lane body plus a scalar tail for the final 0 or 4 elements.
// Products fuse (MulAdd) where the scalar uses mul-then-add, so results agree to ~1 ulp, exactly as
// the NEON kernel does (the "FMA contraction may differ" note in vexp.go).
func vexpF32(p []float32, m float32) float32 {
	if !vexpHasAVX { // built with the experiment but run on a pre-AVX/FMA CPU: correct scalar poly
		var sum float32
		for i, v := range p {
			e := expF32(v - m)
			p[i] = e
			sum += e
		}
		return sum
	}
	mv := archsimd.BroadcastFloat32x8(m)
	lo := archsimd.BroadcastFloat32x8(expLoClamp)
	hi := archsimd.BroadcastFloat32x8(expHiClamp)
	log2e := archsimd.BroadcastFloat32x8(expLog2e)
	magic := archsimd.BroadcastFloat32x8(expMagic)
	nHi := archsimd.BroadcastFloat32x8(-expLn2Hi)
	nLo := archsimd.BroadcastFloat32x8(-expLn2Lo)
	c0 := archsimd.BroadcastFloat32x8(1.9875691500e-4)
	c1 := archsimd.BroadcastFloat32x8(1.3981999507e-3)
	c2 := archsimd.BroadcastFloat32x8(8.3334519073e-3)
	c3 := archsimd.BroadcastFloat32x8(4.1665795894e-2)
	c4 := archsimd.BroadcastFloat32x8(1.6666665459e-1)
	c5 := archsimd.BroadcastFloat32x8(0.5)
	one := archsimd.BroadcastFloat32x8(1.0)
	bias := archsimd.BroadcastInt32x8(127)
	sumv := archsimd.BroadcastFloat32x8(0.0)

	n8 := len(p) &^ 7
	for i := 0; i < n8; i += 8 {
		x := archsimd.LoadFloat32x8Slice(p[i:]).Sub(mv) // x = p[i] - m
		x = x.Max(lo).Min(hi)                           // clamp to the finite exp domain
		// n = rint(x·log2e) via the 1.5·2²³ magic (round-half-even, matches expF32/FRINTN).
		n := x.MulAdd(log2e, magic).Sub(magic)
		// r = x − n·ln2Hi − n·ln2Lo (FMA with the negated split constants).
		r := n.MulAdd(nHi, x)
		r = n.MulAdd(nLo, r)
		// degree-5 Horner poly on r ∈ [−ln2/2, ln2/2].
		pp := c0.MulAdd(r, c1)
		pp = pp.MulAdd(r, c2)
		pp = pp.MulAdd(r, c3)
		pp = pp.MulAdd(r, c4)
		pp = pp.MulAdd(r, c5)
		// exp(r) ≈ p·r² + r + 1.
		res := pp.MulAdd(r.Mul(r), r.Add(one))
		// scale by 2ⁿ: ((int32(n)+127) << 23) reinterpreted as f32 bits.
		scale := n.ConvertToInt32().Add(bias).AsUint32x8().ShiftAllLeft(23).AsFloat32x8()
		res = res.Mul(scale)
		res.StoreSlice(p[i:])
		sumv = sumv.Add(res)
	}
	var lanes [8]float32
	sumv.Store(&lanes)
	sum := lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7]
	for i := n8; i < len(p); i++ { // scalar tail (len%8 ∈ {0,4})
		e := expF32(p[i] - m)
		p[i] = e
		sum += e
	}
	return sum
}

// The activation vexp fast paths are NOT vectorized on amd64 (vexpNeon is false, so elementwise.go
// keeps the scalar-f64 GELU/SiLU/exp/tanh/log kernels bit-for-bit). These scalar instantiations exist
// only so the driver type-checks — dead code at run time here, exactly as in vexp_default.go.

func vgeluF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = geluF32(v)
	}
}

func vgeluGradF32(dst, x, g []float32) {
	for i, v := range x {
		dst[i] = geluGradF32(v, g[i])
	}
}

func vsiluF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = siluF32(v)
	}
}

func vsiluGradF32(dst, x, g []float32) {
	for i, v := range x {
		dst[i] = siluGradF32(v, g[i])
	}
}

func vsigmoidF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = sigmoidF32(v)
	}
}

func vexpFullF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = expFullF32(v)
	}
}

func vtanhF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = tanhF32(v)
	}
}

func vlogF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = logF32(v)
	}
}
