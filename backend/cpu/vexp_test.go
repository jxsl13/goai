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
