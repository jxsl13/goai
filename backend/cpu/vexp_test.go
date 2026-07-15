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
