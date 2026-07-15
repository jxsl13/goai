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
