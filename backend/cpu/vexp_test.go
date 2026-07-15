package cpu

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestExpF32Accuracy sweeps the vexp kernel (through the real build-selected
// vexpF32 entry — NEON on arm64+simd, the scalar poly elsewhere) against
// math.Exp over the full f32 exp domain and the softmax's post-max range.
// Budget: 1e-6 relative (observed ~3e-7), two orders under the 5e-5 f32 MHA
// parity budget. Below the underflow clamp the result must be ≤ 2^-126
// (an exact-enough softmax zero); NaN must propagate.
func TestExpF32Accuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	check := func(xs []float32, label string) {
		t.Helper()
		got := make([]float32, len(xs))
		copy(got, xs)
		vexpF32(got[:len(got)&^3], 0)
		for i := len(got) &^ 3; i < len(got); i++ {
			got[i] = expF32(got[i])
		}
		var maxRel float64
		for i, x := range xs {
			w := math.Exp(float64(x))
			gv := float64(got[i])
			if x < expLoClamp {
				// The clamp pins deep-underflow inputs at exp(expLoClamp),
				// which sits right at the smallest normal (~1.18e-38).
				if gv > 1.3e-38 {
					t.Fatalf("%s: exp(%g) = %g, want underflow ~0", label, x, gv)
				}
				continue
			}
			rel := math.Abs(gv-w) / w
			if rel > maxRel {
				maxRel = rel
			}
			if rel > 1e-6 {
				t.Fatalf("%s: exp(%g) = %g, want %g (rel %.2e)", label, x, gv, w, rel)
			}
		}
		t.Logf("%s: max rel err %.2e over %d points (vexpNeon=%v)", label, maxRel, len(xs), vexpNeon)
	}

	// Dense uniform sweep of the whole valid domain [-87.33, 88].
	n := 1 << 20
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(-87.33 + 175.33*float64(i)/float64(n-1))
	}
	check(xs, "uniform[-87.33,88]")

	// Random softmax-typical range [-30, 0] (scores minus row max).
	for i := range xs {
		xs[i] = -30 * rng.Float32()
	}
	check(xs, "softmax[-30,0]")

	// Random near-zero fine grid (poly center).
	for i := range xs {
		xs[i] = rng.Float32() - 0.5
	}
	check(xs, "center[-0.5,0.5]")

	// Underflow region and edges.
	edge := []float32{-88, -100, -1000, float32(math.Inf(-1)), -87.34, -87.33, 0, 87.9, 88}
	check(edge, "edges")

	// NaN propagates.
	nanRow := []float32{1, float32(math.NaN()), 2, 3}
	vexpF32(nanRow, 0)
	if !math.IsNaN(float64(nanRow[1])) {
		t.Fatalf("exp(NaN) = %g, want NaN", nanRow[1])
	}
}

// TestVexpF32MatchesScalarTail: the vector lanes and the scalar tail poly
// agree to a few ulps, so a row's weights don't jump at the len%4 boundary.
func TestVexpF32MatchesScalarTail(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 5))
	xs := make([]float32, 256)
	for i := range xs {
		xs[i] = -20 * rng.Float32()
	}
	vec := make([]float32, len(xs))
	copy(vec, xs)
	vexpF32(vec, 0)
	for i, x := range xs {
		s := expF32(x)
		diff := math.Abs(float64(vec[i]) - float64(s))
		if diff > 2e-7*float64(s) {
			t.Fatalf("lane %d: vexp %g vs scalar %g", i, vec[i], s)
		}
	}
}

// TestSoftmaxVexpF32ParityVsF64Ref: the f32-native OpSoftmax fast path
// (softmaxVexpF32, taken on the arm64 perf build) matches an f64 math.Exp
// reference within the ADR-0021 tolerant budget (rtol 2e-3; observed ~1e-6 —
// the vexp poly is ~3e-7 and the 4-lane f32 sum dominates). Runs on every
// build (the driver falls back to the scalar poly when !vexpNeon), across
// normal, all-equal, large-magnitude, single-element, negative and huge-d
// (wide intra-row split) rows.
func TestSoftmaxVexpF32ParityVsF64Ref(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 23))
	const rtol = 2e-3
	var maxRel float64
	check := func(x []float32, rows, d int, label string) {
		t.Helper()
		got := make([]float32, len(x))
		softmaxVexpF32(x, got, rows, d)
		for r := 0; r < rows; r++ {
			xr := x[r*d : (r+1)*d]
			m := math.Inf(-1)
			for _, v := range xr {
				if f := float64(v); f > m {
					m = f
				}
			}
			var sum float64
			w := make([]float64, d)
			for j, v := range xr {
				w[j] = math.Exp(float64(v) - m)
				sum += w[j]
			}
			for j := range w {
				want := w[j] / sum
				rel := math.Abs(float64(got[r*d+j])-want) / math.Max(want, 1e-30)
				if rel > maxRel {
					maxRel = rel
				}
				if rel > rtol {
					t.Fatalf("%s row %d col %d: got %g want %g (rel %.2e)", label, r, j, got[r*d+j], want, rel)
				}
			}
		}
	}

	randRows := func(n, d int, scale float32) []float32 {
		x := make([]float32, n*d)
		for i := range x {
			x[i] = scale * (rng.Float32() - 0.5)
		}
		return x
	}
	check(randRows(64, 512, 8), 64, 512, "normal[64x512]")
	check(randRows(7, 33, 8), 7, 33, "tail[7x33]") // d%4 != 0 exercises the scalar tail
	check(randRows(4, 1024, 160), 4, 1024, "largemag[4x1024,±80]")
	neg := randRows(8, 256, 20)
	for i := range neg {
		neg[i] = -1 - float32(math.Abs(float64(neg[i]))) // all-negative rows
	}
	check(neg, 8, 256, "negative[8x256]")
	eq := make([]float32, 3*64)
	for i := range eq {
		eq[i] = 3.25
	}
	check(eq, 3, 64, "allequal[3x64]")
	check([]float32{-42}, 1, 1, "single[1x1]")
	check(randRows(1, 32000, 12), 1, 32000, "wide[1x32000]") // softmaxWideVexpF32 split
	t.Logf("f32-native softmax vs f64 ref: max rel err %.2e (rtol %g, vexpNeon=%v)", maxRel, rtol, vexpNeon)
}

// TestGeluF32Accuracy sweeps the vgelu pipeline (through the build-selected
// vgeluF32 — NEON on arm64+simd, the scalar poly elsewhere) against the exact
// f64 math.Erf GELU. Budget: |got−ref| ≤ 1e-6 + 2e-4·|ref| (observed max abs
// err ~5e-7, rel err ~1e-7 across GELU's active region) — an order under the
// ADR-0021 f32 tolerance (rtol 2e-3). The atol term covers the deep-negative
// tail (x ≲ −5.5, |gelu| < ~1.5e-6) where AS-7.1.26's absolute erf error
// dominates the vanishing reference. Tails must saturate exactly: x for
// large +x, −0 for large −x; ±Inf/NaN mirror the f64 reference.
func TestGeluF32Accuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 17))
	geluRef := func(x float32) float64 {
		xd := float64(x)
		return 0.5 * xd * (1 + math.Erf(xd/math.Sqrt2))
	}
	check := func(xs []float32, label string) {
		t.Helper()
		got := make([]float32, len(xs))
		vgeluF32(got, xs)
		var maxAbs, maxRel float64
		for i, x := range xs {
			w := geluRef(x)
			gv := float64(got[i])
			abs := math.Abs(gv - w)
			if abs > maxAbs {
				maxAbs = abs
			}
			if math.Abs(w) > 1e-6 {
				if rel := abs / math.Abs(w); rel > maxRel {
					maxRel = rel
				}
			}
			if abs > 1e-6+2e-4*math.Abs(w) {
				t.Fatalf("%s: gelu(%g) = %g, want %g (abs %.2e)", label, x, gv, w, abs)
			}
		}
		t.Logf("%s: max abs err %.2e, max rel err %.2e (|ref|>1e-6) over %d points (vexpNeon=%v)",
			label, maxAbs, maxRel, len(xs), vexpNeon)
	}

	// Dense uniform sweep of the active region.
	n := 1 << 20
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(-10 + 20*float64(i)/float64(n-1))
	}
	check(xs, "uniform[-10,10]")

	// Random FFN-typical pre-activations.
	for i := range xs {
		xs[i] = 8 * (rng.Float32() - 0.5)
	}
	check(xs, "random[-4,4]")

	// Fine grid around 0 (erf poly center, sign handoff).
	for i := range xs {
		xs[i] = rng.Float32() - 0.5
	}
	check(xs, "center[-0.5,0.5]")

	// Tail saturation: gelu → x on the right, −0 on the left.
	sat := []float32{8, 10, 100, 1e30}
	out := make([]float32, len(sat))
	vgeluF32(out, sat)
	for i, x := range sat {
		if out[i] != x {
			t.Errorf("gelu(%g) = %g, want exact x", x, out[i])
		}
	}
	neg := []float32{-8, -10, -100, -1e30}
	vgeluF32(out[:len(neg)], neg)
	for i, x := range neg {
		if out[i] != 0 || !math.Signbit(float64(out[i])) {
			t.Errorf("gelu(%g) = %g, want -0", x, out[i])
		}
	}

	// Edge semantics mirror the f64 reference: gelu(+Inf)=+Inf,
	// gelu(-Inf)=NaN (Inf·0), NaN propagates, ±0 keep their sign.
	edge := []float32{float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), 0, float32(math.Copysign(0, -1))}
	eout := make([]float32, len(edge))
	vgeluF32(eout, edge)
	if !math.IsInf(float64(eout[0]), 1) {
		t.Errorf("gelu(+Inf) = %g, want +Inf", eout[0])
	}
	if !math.IsNaN(float64(eout[1])) {
		t.Errorf("gelu(-Inf) = %g, want NaN", eout[1])
	}
	if !math.IsNaN(float64(eout[2])) {
		t.Errorf("gelu(NaN) = %g, want NaN", eout[2])
	}
	if eout[3] != 0 || math.Signbit(float64(eout[3])) {
		t.Errorf("gelu(+0) = %g, want +0", eout[3])
	}
	if eout[4] != 0 || !math.Signbit(float64(eout[4])) {
		t.Errorf("gelu(-0) = %g, want -0", eout[4])
	}
}

// TestGeluGradF32Accuracy sweeps the vgeluGrad pipeline (through the
// build-selected vgeluGradF32 — NEON on arm64+simd, the scalar poly elsewhere)
// against the exact f64 GELU derivative g·(Φ(x)+x·φ(x)) — ref's geluGrad
// formula (§T353), the VERIFY-BEFORE-BENCH parity gate for the OpGELUBackward
// fast path. Budget: |got−ref| ≤ 1e-6·|g| + 2e-4·|ref| (the same envelope as
// the forward, scaled by the upstream gradient; observed max rel err ~1e-7 in
// the active region) — an order under the ADR-0021 f32 tolerance (rtol 2e-3).
// Tails must saturate exactly: gelu'(x)→1 for large +x (dx = g exactly), →0
// for large −x; ±Inf give Inf·0 = NaN exactly like ref's f64 math.Exp path
// (the pdfMin mask, vexp.go); NaN propagates.
func TestGeluGradF32Accuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(29, 31))
	gradRef := func(x, g float32) float64 {
		xd, gd := float64(x), float64(g)
		phi := 0.5 * (1 + math.Erf(xd/math.Sqrt2))
		pdf := 0.3989422804014327 * math.Exp(-0.5*xd*xd)
		return gd * (phi + xd*pdf)
	}
	check := func(xs, gs []float32, label string) {
		t.Helper()
		got := make([]float32, len(xs))
		vgeluGradF32(got, xs, gs)
		var maxAbs, maxRel float64
		for i, x := range xs {
			w := gradRef(x, gs[i])
			gv := float64(got[i])
			abs := math.Abs(gv - w)
			if abs > maxAbs {
				maxAbs = abs
			}
			// Relative error away from the derivative's zero crossing
			// (x ≈ −0.75 and the deep-negative tail, where |ref|→0 and the
			// atol term of the budget governs).
			if math.Abs(w) > 1e-3 {
				if rel := abs / math.Abs(w); rel > maxRel {
					maxRel = rel
				}
			}
			if abs > 1e-6*math.Abs(float64(gs[i]))+1e-7+2e-4*math.Abs(w) {
				t.Fatalf("%s: geluGrad(%g)·%g = %g, want %g (abs %.2e)", label, x, gs[i], gv, w, abs)
			}
		}
		t.Logf("%s: max abs err %.2e, max rel err %.2e (|ref|>1e-3) over %d points (vexpNeon=%v)",
			label, maxAbs, maxRel, len(xs), vexpNeon)
	}

	// Dense uniform sweep of the active region, unit upstream gradient.
	n := 1 << 20
	xs := make([]float32, n)
	gs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(-10 + 20*float64(i)/float64(n-1))
		gs[i] = 1
	}
	check(xs, gs, "uniform[-10,10]·1")

	// Random FFN-typical pre-activations with random upstream gradients.
	for i := range xs {
		xs[i] = 8 * (rng.Float32() - 0.5)
		gs[i] = 4 * (rng.Float32() - 0.5)
	}
	check(xs, gs, "random[-4,4]·[-2,2]")

	// Fine grid around 0 (erf poly center, sign handoff, gelu'(0) = 0.5).
	for i := range xs {
		xs[i] = rng.Float32() - 0.5
		gs[i] = 1
	}
	check(xs, gs, "center[-0.5,0.5]")

	// Tail saturation: gelu'(x) → 1 for large +x (dx = g exactly, the pdf
	// term masked to a true 0), → 0 (+0) for large −x.
	sat := []float32{14, 100, 1e30, 1e38}
	ones := []float32{2, 2, 2, 2}
	out := make([]float32, len(sat))
	vgeluGradF32(out, sat, ones)
	for i, x := range sat {
		if out[i] != 2 {
			t.Errorf("geluGrad(%g)·2 = %g, want exact 2", x, out[i])
		}
	}
	neg := []float32{-14, -100, -1e30, -1e38}
	vgeluGradF32(out, neg, ones)
	for i, x := range neg {
		if out[i] != 0 {
			t.Errorf("geluGrad(%g)·2 = %g, want 0", x, out[i])
		}
	}

	// Edge semantics mirror the f64 reference: ±Inf → NaN (Inf·pdf0 = Inf·0),
	// NaN propagates, x=0 → 0.5·g.
	edge := []float32{float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), 0}
	eg := []float32{1, 1, 1, 2}
	eout := make([]float32, len(edge))
	vgeluGradF32(eout, edge, eg)
	if !math.IsNaN(float64(eout[0])) {
		t.Errorf("geluGrad(+Inf) = %g, want NaN (ref: Inf·0)", eout[0])
	}
	if !math.IsNaN(float64(eout[1])) {
		t.Errorf("geluGrad(-Inf) = %g, want NaN (ref: -Inf·0)", eout[1])
	}
	if !math.IsNaN(float64(eout[2])) {
		t.Errorf("geluGrad(NaN) = %g, want NaN", eout[2])
	}
	if math.Abs(float64(eout[3])-1) > 1e-6 {
		t.Errorf("geluGrad(0)·2 = %g, want 1 (Φ(0)=0.5)", eout[3])
	}
}

// TestVgeluGradF32MatchesScalarTail: the vector lanes and the scalar tail poly
// agree to a few ulps, so a chunk's gradients don't jump at the len%4 boundary
// (the parallel() chunking makes the lane/tail split position vary).
func TestVgeluGradF32MatchesScalarTail(t *testing.T) {
	rng := rand.New(rand.NewPCG(37, 41))
	xs := make([]float32, 256)
	gs := make([]float32, 256)
	for i := range xs {
		xs[i] = 12 * (rng.Float32() - 0.5)
		gs[i] = 2 * (rng.Float32() - 0.5)
	}
	vec := make([]float32, len(xs))
	vgeluGradF32(vec, xs, gs)
	for i, x := range xs {
		s := geluGradF32(x, gs[i])
		diff := math.Abs(float64(vec[i]) - float64(s))
		if diff > 1e-6+2e-6*math.Abs(float64(s)) {
			t.Fatalf("lane %d: vgeluGrad(%g)·%g = %g vs scalar %g", i, x, gs[i], vec[i], s)
		}
	}
}

// TestVgeluF32MatchesScalarTail: the vector lanes and the scalar tail poly
// agree to a few ulps, so a chunk's values don't jump at the len%4 boundary.
func TestVgeluF32MatchesScalarTail(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 23))
	xs := make([]float32, 256)
	for i := range xs {
		xs[i] = 12 * (rng.Float32() - 0.5)
	}
	vec := make([]float32, len(xs))
	vgeluF32(vec, xs)
	for i, x := range xs {
		s := geluF32(x)
		diff := math.Abs(float64(vec[i]) - float64(s))
		if diff > 1e-6+2e-6*math.Abs(float64(s)) {
			t.Fatalf("lane %d: vgelu(%g) = %g vs scalar %g", i, x, vec[i], s)
		}
	}
}

// The exp-kernel microbenchmarks isolate the softmax exp fraction of the MHA
// forward: one 512-wide attention row, exp(x-m)+sum, matching the band
// softmax's inner loop.
func benchExpRow(b *testing.B, f func(row []float32, m float32) float32) {
	rng := rand.New(rand.NewPCG(1, 2))
	src := make([]float32, 512)
	for i := range src {
		src[i] = -25 * rng.Float32()
	}
	row := make([]float32, len(src))
	var sink float32
	for b.Loop() {
		copy(row, src)
		sink += f(row, 0)
	}
	_ = sink
}

func BenchmarkExpRow512(b *testing.B) {
	b.Run("mathExp", func(b *testing.B) {
		benchExpRow(b, func(row []float32, m float32) float32 {
			var sum float64
			for i, v := range row {
				e := math.Exp(float64(v - m))
				row[i] = float32(e)
				sum += e
			}
			return float32(sum)
		})
	})
	b.Run("goPoly", func(b *testing.B) {
		benchExpRow(b, func(row []float32, m float32) float32 {
			var sum float32
			for i, v := range row {
				e := expF32(v - m)
				row[i] = e
				sum += e
			}
			return sum
		})
	})
	b.Run("vexp", func(b *testing.B) {
		benchExpRow(b, vexpF32)
	})
}

// TestSiluF32Accuracy sweeps the vsilu pipeline (through the build-selected
// vsiluF32 — NEON on arm64+simd, the scalar poly elsewhere) against the exact
// f64 x·sigmoid(x) reference (ref's OpSiLU formula, §T665) — the
// VERIFY-BEFORE-BENCH parity gate for the OpSiLU F32 fast path. Budget:
// |got−ref| ≤ 1e-6 + 2e-4·|ref| (observed rel err ~1e-7 in the active
// region) — an order under the ADR-0021 f32 tolerance (rtol 2e-3). Tails must
// saturate: silu(x) → x exactly for large +x, → −0 for x ≤ −88 (the pdfMin
// mask); silu(−Inf) = −Inf·0 = NaN and silu(+Inf) = +Inf exactly like the f64
// reference; NaN propagates; ±0 keep their sign.
func TestSiluF32Accuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(43, 47))
	siluRef := func(x float32) float64 {
		xd := float64(x)
		var s float64
		if xd >= 0 {
			s = 1 / (1 + math.Exp(-xd))
		} else {
			z := math.Exp(xd)
			s = z / (1 + z)
		}
		return xd * s
	}
	check := func(xs []float32, label string) {
		t.Helper()
		got := make([]float32, len(xs))
		vsiluF32(got, xs)
		var maxAbs, maxRel float64
		for i, x := range xs {
			w := siluRef(x)
			gv := float64(got[i])
			abs := math.Abs(gv - w)
			if abs > maxAbs {
				maxAbs = abs
			}
			if math.Abs(w) > 1e-6 {
				if rel := abs / math.Abs(w); rel > maxRel {
					maxRel = rel
				}
			}
			if abs > 1e-6+2e-4*math.Abs(w) {
				t.Fatalf("%s: silu(%g) = %g, want %g (abs %.2e)", label, x, gv, w, abs)
			}
		}
		t.Logf("%s: max abs err %.2e, max rel err %.2e (|ref|>1e-6) over %d points (vexpNeon=%v)",
			label, maxAbs, maxRel, len(xs), vexpNeon)
	}

	// Dense uniform sweep of the task's accuracy domain [−40, 40].
	n := 1 << 20
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(-40 + 80*float64(i)/float64(n-1))
	}
	check(xs, "uniform[-40,40]")

	// Random SwiGLU-typical pre-activations.
	for i := range xs {
		xs[i] = 8 * (rng.Float32() - 0.5)
	}
	check(xs, "random[-4,4]")

	// Fine grid around 0 (poly center, sign handoff, silu(0)=0).
	for i := range xs {
		xs[i] = rng.Float32() - 0.5
	}
	check(xs, "center[-0.5,0.5]")

	// Tail saturation: silu → x on the right, −0 on the left.
	sat := []float32{40, 88, 100, 1e30}
	out := make([]float32, len(sat))
	vsiluF32(out, sat)
	for i, x := range sat {
		if out[i] != x {
			t.Errorf("silu(%g) = %g, want exact x", x, out[i])
		}
	}
	neg := []float32{-88, -100, -1e30, -1e38}
	vsiluF32(out[:len(neg)], neg)
	for i, x := range neg {
		if out[i] != 0 || !math.Signbit(float64(out[i])) {
			t.Errorf("silu(%g) = %g, want -0", x, out[i])
		}
	}

	// Edge semantics mirror the f64 reference: silu(+Inf)=+Inf,
	// silu(-Inf)=NaN (−Inf·0), NaN propagates, ±0 keep their sign.
	edge := []float32{float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), 0, float32(math.Copysign(0, -1))}
	eout := make([]float32, len(edge))
	vsiluF32(eout, edge)
	if !math.IsInf(float64(eout[0]), 1) {
		t.Errorf("silu(+Inf) = %g, want +Inf", eout[0])
	}
	if !math.IsNaN(float64(eout[1])) {
		t.Errorf("silu(-Inf) = %g, want NaN", eout[1])
	}
	if !math.IsNaN(float64(eout[2])) {
		t.Errorf("silu(NaN) = %g, want NaN", eout[2])
	}
	if eout[3] != 0 || math.Signbit(float64(eout[3])) {
		t.Errorf("silu(+0) = %g, want +0", eout[3])
	}
	if eout[4] != 0 || !math.Signbit(float64(eout[4])) {
		t.Errorf("silu(-0) = %g, want -0", eout[4])
	}
}

// TestSiluGradF32Accuracy sweeps the vsiluGrad pipeline (through the
// build-selected vsiluGradF32) against the exact f64 SiLU derivative
// g·σ(x)·(1+x·(1−σ(x))) — ref's siluBackwardKernel formula (§T362), the
// VERIFY-BEFORE-BENCH parity gate for the OpSiLUBackward fast path. Budget:
// |got−ref| ≤ 1e-6·|g| + 1e-7 + 2e-4·|ref| (the geluGrad envelope; observed
// rel err ~1e-7 in the active region, the atol terms govern near the
// derivative's zero crossing at x ≈ −1.278 and in the deep tails). Tails:
// silu'(x) → 1 for large +x (dx = g exactly), → 0 for large −x; ±Inf give
// NaN exactly like ref's f64 path (Inf·0 in the x·(1−σ) term); NaN propagates.
func TestSiluGradF32Accuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(53, 59))
	gradRef := func(x, g float32) float64 {
		xd, gd := float64(x), float64(g)
		var s float64
		if xd >= 0 {
			s = 1 / (1 + math.Exp(-xd))
		} else {
			z := math.Exp(xd)
			s = z / (1 + z)
		}
		return gd * s * (1 + xd*(1-s))
	}
	check := func(xs, gs []float32, label string) {
		t.Helper()
		got := make([]float32, len(xs))
		vsiluGradF32(got, xs, gs)
		var maxAbs, maxRel float64
		for i, x := range xs {
			w := gradRef(x, gs[i])
			gv := float64(got[i])
			abs := math.Abs(gv - w)
			if abs > maxAbs {
				maxAbs = abs
			}
			if math.Abs(w) > 1e-3 {
				if rel := abs / math.Abs(w); rel > maxRel {
					maxRel = rel
				}
			}
			if abs > 1e-6*math.Abs(float64(gs[i]))+1e-7+2e-4*math.Abs(w) {
				t.Fatalf("%s: siluGrad(%g)·%g = %g, want %g (abs %.2e)", label, x, gs[i], gv, w, abs)
			}
		}
		t.Logf("%s: max abs err %.2e, max rel err %.2e (|ref|>1e-3) over %d points (vexpNeon=%v)",
			label, maxAbs, maxRel, len(xs), vexpNeon)
	}

	// Dense uniform sweep of the task's accuracy domain [−40, 40], unit g.
	n := 1 << 20
	xs := make([]float32, n)
	gs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(-40 + 80*float64(i)/float64(n-1))
		gs[i] = 1
	}
	check(xs, gs, "uniform[-40,40]·1")

	// Random SwiGLU-typical pre-activations with random upstream gradients.
	for i := range xs {
		xs[i] = 8 * (rng.Float32() - 0.5)
		gs[i] = 4 * (rng.Float32() - 0.5)
	}
	check(xs, gs, "random[-4,4]·[-2,2]")

	// Fine grid around the derivative's zero crossing (x ≈ −1.278) and 0.
	for i := range xs {
		xs[i] = -2 + 2.5*rng.Float32()
		gs[i] = 1
	}
	check(xs, gs, "crossing[-2,0.5]")

	// Tail saturation: silu'(x) → 1 for large +x (dx = g exactly, z masked to
	// a true 0), → 0 for large −x.
	sat := []float32{88, 100, 1e30, 1e38}
	twos := []float32{2, 2, 2, 2}
	out := make([]float32, len(sat))
	vsiluGradF32(out, sat, twos)
	for i, x := range sat {
		if out[i] != 2 {
			t.Errorf("siluGrad(%g)·2 = %g, want exact 2", x, out[i])
		}
	}
	neg := []float32{-88, -100, -1e30, -1e38}
	vsiluGradF32(out, neg, twos)
	for i, x := range neg {
		if out[i] != 0 {
			t.Errorf("siluGrad(%g)·2 = %g, want 0", x, out[i])
		}
	}

	// Edge semantics mirror the f64 reference: ±Inf → NaN (the Inf·0 in
	// x·(1−σ) / σ·x), NaN propagates, x=0 → 0.5·g (σ(0)=0.5).
	edge := []float32{float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), 0}
	eg := []float32{1, 1, 1, 2}
	eout := make([]float32, len(edge))
	vsiluGradF32(eout, edge, eg)
	if !math.IsNaN(float64(eout[0])) {
		t.Errorf("siluGrad(+Inf) = %g, want NaN (ref: 1+Inf·0)", eout[0])
	}
	if !math.IsNaN(float64(eout[1])) {
		t.Errorf("siluGrad(-Inf) = %g, want NaN (ref: 0·-Inf)", eout[1])
	}
	if !math.IsNaN(float64(eout[2])) {
		t.Errorf("siluGrad(NaN) = %g, want NaN", eout[2])
	}
	if math.Abs(float64(eout[3])-1) > 1e-6 {
		t.Errorf("siluGrad(0)·2 = %g, want 1 (silu'(0)=0.5)", eout[3])
	}

	// ±40 tails against the f64 reference (the task's accuracy bracket edges).
	brk := []float32{40, -40}
	bg := []float32{1, 1}
	bout := make([]float32, len(brk))
	vsiluGradF32(bout, brk, bg)
	for i, x := range brk {
		w := gradRef(x, 1)
		if math.Abs(float64(bout[i])-w) > 1e-6+2e-4*math.Abs(w) {
			t.Errorf("siluGrad(%g) = %g, want %g", x, bout[i], w)
		}
	}
}

// TestSigmoidF32Accuracy sweeps the vsigmoid pipeline (through the
// build-selected vsigmoidF32) against the exact f64 stable-split sigmoid
// (ref's formula) — the parity gate for the OpSigmoid F32 fast path. Budget:
// |got−ref| ≤ 1e-6 + 2e-4·|ref|. σ(−Inf) and the x ≤ −87.34 tail return the
// exp underflow-clamp residue ~1.18e-38 instead of ref's exact 0 (no pdfMin
// mask — see vexp.go) — inside the atol term; σ(+Inf) = 1 and NaN propagation
// are exact.
func TestSigmoidF32Accuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(61, 67))
	sigRef := func(x float32) float64 {
		xd := float64(x)
		if xd >= 0 {
			return 1 / (1 + math.Exp(-xd))
		}
		z := math.Exp(xd)
		return z / (1 + z)
	}
	check := func(xs []float32, label string) {
		t.Helper()
		got := make([]float32, len(xs))
		vsigmoidF32(got, xs)
		var maxAbs, maxRel float64
		for i, x := range xs {
			w := sigRef(x)
			gv := float64(got[i])
			abs := math.Abs(gv - w)
			if abs > maxAbs {
				maxAbs = abs
			}
			if math.Abs(w) > 1e-6 {
				if rel := abs / math.Abs(w); rel > maxRel {
					maxRel = rel
				}
			}
			if abs > 1e-6+2e-4*math.Abs(w) {
				t.Fatalf("%s: sigmoid(%g) = %g, want %g (abs %.2e)", label, x, gv, w, abs)
			}
		}
		t.Logf("%s: max abs err %.2e, max rel err %.2e (|ref|>1e-6) over %d points (vexpNeon=%v)",
			label, maxAbs, maxRel, len(xs), vexpNeon)
	}

	// Dense uniform sweep of the task's accuracy domain [−40, 40].
	n := 1 << 20
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(-40 + 80*float64(i)/float64(n-1))
	}
	check(xs, "uniform[-40,40]")

	// Random gate-typical inputs and the poly center.
	for i := range xs {
		xs[i] = 16 * (rng.Float32() - 0.5)
	}
	check(xs, "random[-8,8]")
	for i := range xs {
		xs[i] = rng.Float32() - 0.5
	}
	check(xs, "center[-0.5,0.5]")

	// Saturation and edges: σ(+Inf)=1 and σ(large +x)=1 exactly; σ(−Inf) and
	// σ(deep −x) ≤ the clamp residue (~1.18e-38, an exact-enough 0); σ(±0)=0.5;
	// NaN propagates.
	sat := []float32{40, 100, 1e30, float32(math.Inf(1))}
	out := make([]float32, len(sat))
	vsigmoidF32(out, sat)
	for i, x := range sat {
		if out[i] != 1 {
			t.Errorf("sigmoid(%g) = %g, want exact 1", x, out[i])
		}
	}
	neg := []float32{-100, -1e30, float32(math.Inf(-1))}
	vsigmoidF32(out[:len(neg)], neg)
	for i, x := range neg {
		if !(out[i] >= 0 && out[i] <= 1.3e-38) {
			t.Errorf("sigmoid(%g) = %g, want ≤ 1.3e-38", x, out[i])
		}
	}
	edge := []float32{float32(math.NaN()), 0, float32(math.Copysign(0, -1))}
	eout := make([]float32, len(edge))
	vsigmoidF32(eout, edge)
	if !math.IsNaN(float64(eout[0])) {
		t.Errorf("sigmoid(NaN) = %g, want NaN", eout[0])
	}
	if eout[1] != 0.5 || eout[2] != 0.5 {
		t.Errorf("sigmoid(±0) = %g/%g, want 0.5", eout[1], eout[2])
	}
}

// TestVsiluF32MatchesScalarTail / TestVsiluGradF32MatchesScalarTail /
// TestVsigmoidF32MatchesScalarTail: the vector lanes and the scalar tail poly
// agree to a few ulps, so a chunk's values don't jump at the len%4 boundary
// (the parallel() chunking makes the lane/tail split position vary).
func TestVsiluF32MatchesScalarTail(t *testing.T) {
	rng := rand.New(rand.NewPCG(71, 73))
	xs := make([]float32, 256)
	for i := range xs {
		xs[i] = 24 * (rng.Float32() - 0.5)
	}
	vec := make([]float32, len(xs))
	vsiluF32(vec, xs)
	for i, x := range xs {
		s := siluF32(x)
		diff := math.Abs(float64(vec[i]) - float64(s))
		if diff > 1e-6+2e-6*math.Abs(float64(s)) {
			t.Fatalf("lane %d: vsilu(%g) = %g vs scalar %g", i, x, vec[i], s)
		}
	}
}

func TestVsiluGradF32MatchesScalarTail(t *testing.T) {
	rng := rand.New(rand.NewPCG(79, 83))
	xs := make([]float32, 256)
	gs := make([]float32, 256)
	for i := range xs {
		xs[i] = 24 * (rng.Float32() - 0.5)
		gs[i] = 2 * (rng.Float32() - 0.5)
	}
	vec := make([]float32, len(xs))
	vsiluGradF32(vec, xs, gs)
	for i, x := range xs {
		s := siluGradF32(x, gs[i])
		diff := math.Abs(float64(vec[i]) - float64(s))
		if diff > 1e-6+2e-6*math.Abs(float64(s)) {
			t.Fatalf("lane %d: vsiluGrad(%g)·%g = %g vs scalar %g", i, x, gs[i], vec[i], s)
		}
	}
}

func TestVsigmoidF32MatchesScalarTail(t *testing.T) {
	rng := rand.New(rand.NewPCG(89, 97))
	xs := make([]float32, 256)
	for i := range xs {
		xs[i] = 24 * (rng.Float32() - 0.5)
	}
	vec := make([]float32, len(xs))
	vsigmoidF32(vec, xs)
	for i, x := range xs {
		s := sigmoidF32(x)
		diff := math.Abs(float64(vec[i]) - float64(s))
		if diff > 1e-6+2e-6*math.Abs(float64(s)) {
			t.Fatalf("lane %d: vsigmoid(%g) = %g vs scalar %g", i, x, vec[i], s)
		}
	}
}

// TestExpFullF32Accuracy sweeps the standalone full-domain exp (through the
// build-selected vexpFullF32 — NEON on arm64+simd, the scalar poly elsewhere)
// against the f64 math.Exp reference — the VERIFY-BEFORE-BENCH parity gate
// for the OpExp F32 fast path (§T666). Budget: 1e-6 relative over the whole
// finite domain INCLUDING the (88, 88.7228] band the softmax kernel's single
// 2^n scale could not represent (observed ~8e-8). Overflow must hit +Inf at
// exactly the ref's cutoff (x > 88.72283172607422), underflow returns an
// exact 0 for x < the lo clamp (ref's subnormals there are ≤ 1.18e-38 — the
// atol side of the ADR-0021 budget), and exp(±Inf)/exp(0)/NaN are exact.
func TestExpFullF32Accuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(101, 103))
	check := func(xs []float32, label string) {
		t.Helper()
		got := make([]float32, len(xs))
		vexpFullF32(got, xs)
		var maxRel float64
		for i, x := range xs {
			w := math.Exp(float64(x))
			gv := float64(got[i])
			if x < expLoClamp {
				if gv != 0 {
					t.Fatalf("%s: exp(%g) = %g, want exact 0", label, x, gv)
				}
				continue
			}
			rel := math.Abs(gv-w) / w
			if rel > maxRel {
				maxRel = rel
			}
			if rel > 1e-6 {
				t.Fatalf("%s: exp(%g) = %g, want %g (rel %.2e)", label, x, gv, w, rel)
			}
		}
		t.Logf("%s: max rel err %.2e over %d points (vexpNeon=%v)", label, maxRel, len(xs), vexpNeon)
	}

	// Dense uniform sweep of the task's accuracy bracket [−80, 80].
	n := 1 << 20
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(-80 + 160*float64(i)/float64(n-1))
	}
	check(xs, "uniform[-80,80]")

	// The full finite domain, including the split-scaling band n = 128.
	for i := range xs {
		xs[i] = float32(-87.33 + (88.7228+87.33)*float64(i)/float64(n-1))
	}
	check(xs, "uniform[-87.33,88.7228]")

	// Random near-zero fine grid (poly center) and RL/reward-typical range.
	for i := range xs {
		xs[i] = rng.Float32() - 0.5
	}
	check(xs, "center[-0.5,0.5]")
	for i := range xs {
		xs[i] = 20 * (rng.Float32() - 0.5)
	}
	check(xs, "random[-10,10]")

	// Overflow cutoff is bit-exact vs the f64 ref: the largest finite-exp f32
	// (88.72283172607422 = 0x42B17217) stays finite, the next ulp is +Inf.
	lastFinite := math.Float32frombits(0x42B17217)
	firstInf := math.Float32frombits(0x42B17218)
	cut := []float32{88, 88.5, lastFinite, firstInf, 88.8, 89, 100, 1e30}
	out := make([]float32, len(cut))
	vexpFullF32(out, cut)
	for i, x := range cut {
		w := float64(float32(math.Exp(float64(x))))
		gv := float64(out[i])
		if math.IsInf(w, 1) != math.IsInf(gv, 1) {
			t.Errorf("exp(%g) = %g, want Inf-ness of %g", x, gv, w)
		}
		if !math.IsInf(w, 1) && math.Abs(gv-w)/w > 1e-6 {
			t.Errorf("exp(%g) = %g, want %g", x, gv, w)
		}
	}

	// Underflow band and edges: exact 0 below the lo clamp (incl. −Inf); the
	// ref's subnormal results there are ≤ 1.18e-38, inside the atol budget.
	edge := []float32{-87.34, -90, -103, -200, float32(math.Inf(-1))}
	vexpFullF32(out[:len(edge)], edge)
	for i, x := range edge {
		if out[i] != 0 {
			t.Errorf("exp(%g) = %g, want exact 0", x, out[i])
		}
	}

	// exp(±0) = 1 exactly, exp(+Inf) = +Inf, NaN propagates.
	sp := []float32{0, float32(math.Copysign(0, -1)), float32(math.Inf(1)), float32(math.NaN())}
	vexpFullF32(out[:len(sp)], sp)
	if out[0] != 1 || out[1] != 1 {
		t.Errorf("exp(±0) = %g/%g, want 1", out[0], out[1])
	}
	if !math.IsInf(float64(out[2]), 1) {
		t.Errorf("exp(+Inf) = %g, want +Inf", out[2])
	}
	if !math.IsNaN(float64(out[3])) {
		t.Errorf("exp(NaN) = %g, want NaN", out[3])
	}
}

// TestTanhF32Accuracy sweeps the vtanh pipeline (through the build-selected
// vtanhF32) against the f64 math.Tanh reference — the VERIFY-BEFORE-BENCH
// parity gate for the OpTanh F32 fast path (§T666). Budget: |got−ref| ≤
// 1e-6 + 2e-4·|ref| (observed max abs err ~9e-8; the atol term governs only
// |x| ≲ 1e-4, where 1−e^(−2|x|) meets the half-ulp-of-1 floor of the exp
// poly). Tails must saturate exactly: tanh(±Inf) = ±1, tanh(x) = ±1 for
// |x| ≥ ~9; tanh(±0) = ±0 with the sign preserved; NaN propagates.
func TestTanhF32Accuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(107, 109))
	check := func(xs []float32, label string) {
		t.Helper()
		got := make([]float32, len(xs))
		vtanhF32(got, xs)
		var maxAbs, maxRel float64
		for i, x := range xs {
			w := math.Tanh(float64(x))
			gv := float64(got[i])
			abs := math.Abs(gv - w)
			if abs > maxAbs {
				maxAbs = abs
			}
			if math.Abs(w) > 1e-3 {
				if rel := abs / math.Abs(w); rel > maxRel {
					maxRel = rel
				}
			}
			if abs > 1e-6+2e-4*math.Abs(w) {
				t.Fatalf("%s: tanh(%g) = %g, want %g (abs %.2e)", label, x, gv, w, abs)
			}
		}
		t.Logf("%s: max abs err %.2e, max rel err %.2e (|ref|>1e-3) over %d points (vexpNeon=%v)",
			label, maxAbs, maxRel, len(xs), vexpNeon)
	}

	// Dense uniform sweep of the task's accuracy domain [−40, 40].
	n := 1 << 20
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(-40 + 80*float64(i)/float64(n-1))
	}
	check(xs, "uniform[-40,40]")

	// Random gate/DyT-typical inputs and the active region.
	for i := range xs {
		xs[i] = 8 * (rng.Float32() - 0.5)
	}
	check(xs, "random[-4,4]")
	for i := range xs {
		xs[i] = rng.Float32() - 0.5
	}
	check(xs, "center[-0.5,0.5]")

	// Saturation: exact ±1 from ~|x| ≥ 9 up through ±Inf.
	sat := []float32{9, 44, 100, 1e30, float32(math.Inf(1))}
	out := make([]float32, len(sat))
	vtanhF32(out, sat)
	for i, x := range sat {
		if out[i] != 1 {
			t.Errorf("tanh(%g) = %g, want exact 1", x, out[i])
		}
	}
	neg := []float32{-9, -44, -100, -1e30, float32(math.Inf(-1))}
	vtanhF32(out[:len(neg)], neg)
	for i, x := range neg {
		if out[i] != -1 {
			t.Errorf("tanh(%g) = %g, want exact -1", x, out[i])
		}
	}

	// ±0 keep their sign; NaN propagates.
	edge := []float32{0, float32(math.Copysign(0, -1)), float32(math.NaN())}
	eout := make([]float32, len(edge))
	vtanhF32(eout, edge)
	if eout[0] != 0 || math.Signbit(float64(eout[0])) {
		t.Errorf("tanh(+0) = %g, want +0", eout[0])
	}
	if eout[1] != 0 || !math.Signbit(float64(eout[1])) {
		t.Errorf("tanh(-0) = %g, want -0", eout[1])
	}
	if !math.IsNaN(float64(eout[2])) {
		t.Errorf("tanh(NaN) = %g, want NaN", eout[2])
	}
}

// TestLogF32Accuracy sweeps the vlog pipeline (through the build-selected
// vlogF32) against the f64 math.Log reference — the VERIFY-BEFORE-BENCH
// parity gate for the OpLog F32 fast path (§T666). The tricky budget is
// RELATIVE accuracy across magnitudes (log's task bracket [1e-30, 1e30], 125
// binades): ≤ 1e-6 relative wherever |ref| ≥ 1e-3 (observed ~8e-8 — the ln2
// hi/lo split keeps e·ln2 exact), plus the abs envelope 1e-6 + 2e-4·|ref|
// near log(1) = 0 where f = m−1 is exact and the result vanishes. Subnormal
// inputs go through the 2^25 pre-scale and must hold the same relative
// budget. Specials must be exact: log(±0) = −Inf, log(x<0) = NaN (incl.
// −Inf), log(+Inf) = +Inf, log(1) = +0, NaN propagates.
func TestLogF32Accuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(113, 127))
	check := func(xs []float32, label string) {
		t.Helper()
		got := make([]float32, len(xs))
		vlogF32(got, xs)
		var maxAbs, maxRel float64
		for i, x := range xs {
			w := math.Log(float64(x))
			gv := float64(got[i])
			abs := math.Abs(gv - w)
			if abs > maxAbs {
				maxAbs = abs
			}
			if math.Abs(w) > 1e-3 {
				rel := abs / math.Abs(w)
				if rel > maxRel {
					maxRel = rel
				}
				if rel > 1e-6 {
					t.Fatalf("%s: log(%g) = %g, want %g (rel %.2e)", label, x, gv, w, rel)
				}
				continue
			}
			if abs > 1e-6+2e-4*math.Abs(w) {
				t.Fatalf("%s: log(%g) = %g, want %g (abs %.2e)", label, x, gv, w, abs)
			}
		}
		t.Logf("%s: max abs err %.2e, max rel err %.2e (|ref|>1e-3) over %d points (vexpNeon=%v)",
			label, maxAbs, maxRel, len(xs), vexpNeon)
	}

	// Log-uniform sweep of the task's accuracy bracket [1e-30, 1e30].
	n := 1 << 20
	xs := make([]float32, n)
	lo, hi := math.Log(1e-30), math.Log(1e30)
	for i := range xs {
		xs[i] = float32(math.Exp(lo + (hi-lo)*float64(i)/float64(n-1)))
	}
	check(xs, "loguniform[1e-30,1e30]")

	// Dense linear sweep around 1 (the f = m−1 cancellation-free zone and the
	// √2 fold boundary) and a probability-typical range.
	for i := range xs {
		xs[i] = float32(0.5 + 1.5*float64(i)/float64(n-1))
	}
	check(xs, "uniform[0.5,2]")
	for i := range xs {
		xs[i] = rng.Float32()*0.999 + 0.001
	}
	check(xs, "random(0.001,1]")

	// Subnormal inputs (2^25 pre-scale path) and the largest finite values.
	sub := make([]float32, 4096)
	for i := range sub {
		sub[i] = math.Float32frombits(uint32(rng.Int64N(0x007FFFFF)) + 1) // (0, minNormal)
	}
	check(sub, "subnormal")
	check([]float32{logMinNormal, math.MaxFloat32, 1e30, 3e38}, "extremes")

	// The task's named probes.
	probes := []float32{1e-30, 1e30}
	pout := make([]float32, len(probes))
	vlogF32(pout, probes)
	for i, x := range probes {
		w := math.Log(float64(x))
		if rel := math.Abs(float64(pout[i])-w) / math.Abs(w); rel > 1e-6 {
			t.Errorf("log(%g) = %g, want %g (rel %.2e)", x, pout[i], w, rel)
		}
	}

	// Specials: log(±0) = −Inf, negatives (incl. −Inf) = NaN, log(+Inf) =
	// +Inf, log(1) = +0 exactly, NaN propagates.
	edge := []float32{0, float32(math.Copysign(0, -1)), -1, -1e-42, float32(math.Inf(-1)), float32(math.Inf(1)), 1, float32(math.NaN())}
	eout := make([]float32, len(edge))
	vlogF32(eout, edge)
	if !math.IsInf(float64(eout[0]), -1) || !math.IsInf(float64(eout[1]), -1) {
		t.Errorf("log(±0) = %g/%g, want -Inf", eout[0], eout[1])
	}
	for i := 2; i <= 4; i++ {
		if !math.IsNaN(float64(eout[i])) {
			t.Errorf("log(%g) = %g, want NaN", edge[i], eout[i])
		}
	}
	if !math.IsInf(float64(eout[5]), 1) {
		t.Errorf("log(+Inf) = %g, want +Inf", eout[5])
	}
	if eout[6] != 0 || math.Signbit(float64(eout[6])) {
		t.Errorf("log(1) = %g, want +0", eout[6])
	}
	if !math.IsNaN(float64(eout[7])) {
		t.Errorf("log(NaN) = %g, want NaN", eout[7])
	}
}

// TestVexpFullF32MatchesScalarTail / TestVtanhF32MatchesScalarTail /
// TestVlogF32MatchesScalarTail: the vector lanes and the scalar tail poly
// agree to a few ulps, so a chunk's values don't jump at the len%4 boundary
// (the parallel() chunking makes the lane/tail split position vary).
func TestVexpFullF32MatchesScalarTail(t *testing.T) {
	rng := rand.New(rand.NewPCG(131, 137))
	xs := make([]float32, 257) // odd length: exercises quads + tail in one call
	for i := range xs {
		xs[i] = 176 * (rng.Float32() - 0.5) // spans under/overflow
	}
	vec := make([]float32, len(xs))
	vexpFullF32(vec, xs)
	for i, x := range xs {
		s := expFullF32(x)
		if math.IsInf(float64(s), 1) || s == 0 {
			if vec[i] != s {
				t.Fatalf("lane %d: vexpFull(%g) = %g vs scalar %g", i, x, vec[i], s)
			}
			continue
		}
		diff := math.Abs(float64(vec[i]) - float64(s))
		if diff > 2e-7*float64(s) {
			t.Fatalf("lane %d: vexpFull(%g) = %g vs scalar %g", i, x, vec[i], s)
		}
	}
}

func TestVtanhF32MatchesScalarTail(t *testing.T) {
	rng := rand.New(rand.NewPCG(139, 149))
	xs := make([]float32, 257)
	for i := range xs {
		xs[i] = 24 * (rng.Float32() - 0.5)
	}
	vec := make([]float32, len(xs))
	vtanhF32(vec, xs)
	for i, x := range xs {
		s := tanhF32(x)
		diff := math.Abs(float64(vec[i]) - float64(s))
		if diff > 1e-6+2e-6*math.Abs(float64(s)) {
			t.Fatalf("lane %d: vtanh(%g) = %g vs scalar %g", i, x, vec[i], s)
		}
	}
}

func TestVlogF32MatchesScalarTail(t *testing.T) {
	rng := rand.New(rand.NewPCG(151, 157))
	xs := make([]float32, 257)
	for i := range xs {
		xs[i] = float32(math.Exp(140 * (rng.Float64() - 0.5))) // log-uniform e^±70
	}
	vec := make([]float32, len(xs))
	vlogF32(vec, xs)
	for i, x := range xs {
		s := logF32(x)
		diff := math.Abs(float64(vec[i]) - float64(s))
		if diff > 1e-6+2e-6*math.Abs(float64(s)) {
			t.Fatalf("lane %d: vlog(%g) = %g vs scalar %g", i, x, vec[i], s)
		}
	}
}
