package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
)

// §V16 tier-1: the Noam / inverse-sqrt schedule (Vaswani 2017 §5.3) rises linearly through
// warmup, peaks exactly at step == warmup−1 (step number == warmup), then decays with the
// inverse square root of the step — checked against an independent recomputation of the paper's
// lrate formula at 1e-12, with the peak value (dModel·warmup)^(−1/2)·scale verified in closed form.
func TestInverseSqrt(t *testing.T) {
	const dModel, warmup, scale = 4, 10, 1.0
	ref := func(step int) float64 {
		sn := float64(step + 1)
		return scale * math.Pow(dModel, -0.5) *
			math.Min(math.Pow(sn, -0.5), sn*math.Pow(warmup, -1.5))
	}
	for _, step := range []int{0, 3, 8, 9, 10, 50, 99} {
		if got, want := nn.InverseSqrt(step, warmup, dModel, scale), ref(step); math.Abs(got-want) > 1e-12 {
			t.Errorf("step %d: got %.14g want %.14g", step, got, want)
		}
	}
	// peak is at step == warmup−1 (step number == warmup): (dModel·warmup)^(−1/2)·scale
	peak := nn.InverseSqrt(warmup-1, warmup, dModel, scale)
	if want := scale * math.Pow(dModel*warmup, -0.5); math.Abs(peak-want) > 1e-12 {
		t.Errorf("peak: got %.14g want %.14g", peak, want)
	}
	// strictly increasing through warmup, strictly decreasing after
	if nn.InverseSqrt(0, warmup, dModel, scale) >= peak {
		t.Error("warmup must rise to the peak")
	}
	if nn.InverseSqrt(warmup+40, warmup, dModel, scale) >= peak {
		t.Error("post-warmup must decay below the peak")
	}
	// scale is a linear multiplier on the whole schedule
	if got, want := nn.InverseSqrt(5, warmup, dModel, 2.0), 2*nn.InverseSqrt(5, warmup, dModel, 1.0); math.Abs(got-want) > 1e-12 {
		t.Errorf("scale linearity: got %.14g want %.14g", got, want)
	}
}

// The Noam schedule warms the rate up linearly then lets it decay as 1/√step. Here, with a tiny
// model width and a 10-step warmup, the rate at the warmup peak is (dModel·warmup)^(−1/2).
func ExampleInverseSqrt() {
	fmt.Printf("%.6f\n", nn.InverseSqrt(9, 10, 4, 1.0))
	// Output: 0.158114
}

// §V16 tier-1: the WSD schedule (MiniCPM, arXiv:2404.06395 §4.2 Eq.1) — linear warmup s/W to the
// peak, a constant stable plateau, then exponential decay halving every H steps — reproduced
// exactly at the phase boundaries and interior points, with the three-phase continuity (warmup→
// stable at s=W and stable→decay at s=S both equal to the peak) verified.
func TestWSD(t *testing.T) {
	const warmup, decayStart, halfLife, peak = 10, 100, 20, 1.0
	cases := []struct {
		step int
		want float64
	}{
		{0, 0.0},              // warmup start: 0·/W
		{5, 0.5},              // warmup mid: 5/10·η
		{10, 1.0},             // warmup→stable boundary: 10/10·η == η (continuous)
		{50, 1.0},             // stable plateau
		{99, 1.0},             // last stable step
		{100, 1.0},            // stable→decay boundary: 0.5^0 == η (continuous)
		{110, math.Sqrt2 / 2}, // half a half-life: 0.5^(10/20) = 1/√2
		{120, 0.5},            // one half-life: 0.5^1
		{140, 0.25},           // two half-lives: 0.5^2
		{160, 0.125},          // three half-lives: 0.5^3
	}
	for _, c := range cases {
		if got := nn.WSD(c.step, warmup, decayStart, halfLife, peak); math.Abs(got-c.want) > 1e-12 {
			t.Errorf("WSD step %d: got %.14g want %.14g", c.step, got, c.want)
		}
	}
	// KEY property: the stable phase is a flat plateau at the peak (unlike cosine, which is
	// already decaying there) — every step in [W, S) returns exactly peak.
	for s := warmup; s < decayStart; s++ {
		if got := nn.WSD(s, warmup, decayStart, halfLife, peak); got != peak {
			t.Fatalf("stable phase must be flat at peak: step %d got %.14g", s, got)
		}
	}
	// peak scales the whole schedule linearly
	if got, want := nn.WSD(130, warmup, decayStart, halfLife, 2.0), 2*nn.WSD(130, warmup, decayStart, halfLife, 1.0); math.Abs(got-want) > 1e-12 {
		t.Errorf("peak linearity: got %.14g want %.14g", got, want)
	}
}

// WSD holds the learning rate flat on a long "stable" plateau and only decays at the end, so a
// run can branch from any stable checkpoint. Here, one half-life (20 steps) into the decay phase
// that starts at step 100, the rate has halved from its peak of 1.0.
func ExampleWSD() {
	fmt.Printf("%.4f\n", nn.WSD(120, 10, 100, 20, 1.0))
	// Output: 0.5000
}
