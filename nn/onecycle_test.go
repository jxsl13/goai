package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
)

// §V16 tier-1 (arXiv:1708.07120 / PyTorch OneCycleLR): the anchor LRs are max/25 at step 0, max at
// the peak (step_up = round(0.3·total)), and max/25/1e4 at the final step.
func TestOneCycleAnchors(t *testing.T) {
	const maxLR, total = 1.0, 100
	if got := nn.OneCycleLR(0, total, maxLR, 0, 0, 0); math.Abs(got-maxLR/25) > 1e-12 {
		t.Errorf("step 0 LR = %.12g, want %g (max/25)", got, maxLR/25)
	}
	if got := nn.OneCycleLR(30, total, maxLR, 0, 0, 0); math.Abs(got-maxLR) > 1e-12 {
		t.Errorf("peak LR = %.12g, want %g (max)", got, maxLR)
	}
	if got, want := nn.OneCycleLR(total, total, maxLR, 0, 0, 0), maxLR/25/1e4; math.Abs(got-want) > 1e-15 {
		t.Errorf("final LR = %.12g, want %g (min)", got, want)
	}
}

// §V16 tier-1 (up then down): the LR rises through phase 1 and falls through phase 2.
func TestOneCycleMonotonic(t *testing.T) {
	const maxLR, total = 0.5, 100
	const stepUp = 30 // round(0.3·100)
	for i := 0; i < stepUp; i++ {
		if nn.OneCycleLR(i, total, maxLR, 0, 0, 0) >= nn.OneCycleLR(i+1, total, maxLR, 0, 0, 0) {
			t.Fatalf("phase 1 not increasing at step %d", i)
		}
	}
	for i := stepUp; i < total; i++ {
		if nn.OneCycleLR(i, total, maxLR, 0, 0, 0) <= nn.OneCycleLR(i+1, total, maxLR, 0, 0, 0) {
			t.Fatalf("phase 2 not decreasing at step %d", i)
		}
	}
}

// §V16 tier-1 (cosine anneal): hand-computed midpoints of each phase.
func TestOneCycleCosineParity(t *testing.T) {
	const maxLR, total = 1.0, 100
	// phase 1 midpoint (step 15, pct 0.5): anneal_cos(0.04, 1.0, 0.5) = 1.0 + (0.04−1.0)/2·1 = 0.52.
	if got := nn.OneCycleLR(15, total, maxLR, 0, 0, 0); math.Abs(got-0.52) > 1e-12 {
		t.Errorf("phase-1 midpoint LR = %.12g, want 0.52", got)
	}
	// phase 2 midpoint (step 65, pct 0.5): anneal_cos(1.0, 4e-6, 0.5) = 4e-6 + (1.0−4e-6)/2.
	want := 4e-6 + (1.0-4e-6)/2
	if got := nn.OneCycleLR(65, total, maxLR, 0, 0, 0); math.Abs(got-want) > 1e-12 {
		t.Errorf("phase-2 midpoint LR = %.12g, want %.12g", got, want)
	}
}

// §V16 tier-1 (peak is the global max): no step exceeds maxLR, and the peak is attained.
func TestOneCyclePeakIsMax(t *testing.T) {
	const maxLR, total = 2.0, 200
	best := 0.0
	for i := 0; i <= total; i++ {
		if lr := nn.OneCycleLR(i, total, maxLR, 0, 0, 0); lr > best {
			best = lr
		}
	}
	if math.Abs(best-maxLR) > 1e-9 {
		t.Errorf("peak LR over the run = %.12g, want maxLR %g", best, maxLR)
	}
}

// §V2 (defaults + clamp): non-positive args use the PyTorch defaults; out-of-range steps clamp.
func TestOneCycleDefaultsAndClamp(t *testing.T) {
	const maxLR, total = 1.0, 100
	if a, b := nn.OneCycleLR(40, total, maxLR, 0, 0, 0), nn.OneCycleLR(40, total, maxLR, nn.OneCyclePctStart, nn.OneCycleDivFactor, nn.OneCycleFinalDivFactor); a != b {
		t.Errorf("default args %.12g != explicit %.12g", a, b)
	}
	if got := nn.OneCycleLR(-5, total, maxLR, 0, 0, 0); math.Abs(got-maxLR/25) > 1e-12 {
		t.Errorf("negative step LR = %.12g, want initial max/25", got)
	}
	if got, want := nn.OneCycleLR(999, total, maxLR, 0, 0, 0), maxLR/25/1e4; math.Abs(got-want) > 1e-15 {
		t.Errorf("past-end LR = %.12g, want min", got)
	}
}

func ExampleOneCycleLR() {
	// A 10-step run (maxLR 1.0): starts at max/25, peaks at max, ends near-zero.
	for _, s := range []int{0, 3, 10} {
		fmt.Printf("%.4f ", nn.OneCycleLR(s, 10, 1.0, 0, 0, 0))
	}
	fmt.Println()
	// Output:
	// 0.0400 1.0000 0.0000
}

// §V16 tier-1 (§R233, PyTorch OneCycleLR): momentum is maxMomentum (0.95) at the ends and
// baseMomentum (0.85) at the peak — the inverse of the LR.
func TestOneCycleMomentumAnchors(t *testing.T) {
	const total = 100
	if got := nn.OneCycleMomentum(0, total, 0, 0, 0); math.Abs(got-0.95) > 1e-12 {
		t.Errorf("step 0 momentum = %.12g, want 0.95", got)
	}
	if got := nn.OneCycleMomentum(30, total, 0, 0, 0); math.Abs(got-0.85) > 1e-12 {
		t.Errorf("peak momentum = %.12g, want 0.85 (base)", got)
	}
	if got := nn.OneCycleMomentum(total, total, 0, 0, 0); math.Abs(got-0.95) > 1e-12 {
		t.Errorf("final momentum = %.12g, want 0.95", got)
	}
}

// §V16 tier-1 (inverse of LR): momentum FALLS while the LR rises (phase 1) and RISES while the LR
// decays (phase 2) — the defining coupling.
func TestOneCycleMomentumInverse(t *testing.T) {
	const total = 100
	const stepUp = 30
	for i := 0; i < stepUp; i++ {
		mUp := nn.OneCycleMomentum(i, total, 0, 0, 0) > nn.OneCycleMomentum(i+1, total, 0, 0, 0) // momentum ↓
		lrUp := nn.OneCycleLR(i, total, 1, 0, 0, 0) < nn.OneCycleLR(i+1, total, 1, 0, 0, 0)      // LR ↑
		if !mUp || !lrUp {
			t.Fatalf("phase 1 step %d: momentum should fall while LR rises", i)
		}
	}
	for i := stepUp; i < total; i++ {
		mDown := nn.OneCycleMomentum(i, total, 0, 0, 0) < nn.OneCycleMomentum(i+1, total, 0, 0, 0) // momentum ↑
		lrDown := nn.OneCycleLR(i, total, 1, 0, 0, 0) > nn.OneCycleLR(i+1, total, 1, 0, 0, 0)      // LR ↓
		if !mDown || !lrDown {
			t.Fatalf("phase 2 step %d: momentum should rise while LR falls", i)
		}
	}
}

// §V16 tier-1 (cosine anneal): both phase midpoints sit halfway between the extremes at 0.90.
func TestOneCycleMomentumCosineParity(t *testing.T) {
	const total = 100
	if got := nn.OneCycleMomentum(15, total, 0, 0, 0); math.Abs(got-0.90) > 1e-12 { // anneal_cos(0.95,0.85,0.5)
		t.Errorf("phase-1 midpoint momentum = %.12g, want 0.90", got)
	}
	if got := nn.OneCycleMomentum(65, total, 0, 0, 0); math.Abs(got-0.90) > 1e-12 { // anneal_cos(0.85,0.95,0.5)
		t.Errorf("phase-2 midpoint momentum = %.12g, want 0.90", got)
	}
}

// §V2 (defaults + clamp): non-positive args use the PyTorch defaults; out-of-range steps clamp to
// maxMomentum.
func TestOneCycleMomentumDefaultsAndClamp(t *testing.T) {
	const total = 100
	a := nn.OneCycleMomentum(40, total, 0, 0, 0)
	b := nn.OneCycleMomentum(40, total, nn.OneCycleMaxMomentum, nn.OneCycleBaseMomentum, nn.OneCyclePctStart)
	if a != b {
		t.Errorf("default args %.12g != explicit %.12g", a, b)
	}
	if got := nn.OneCycleMomentum(999, total, 0, 0, 0); math.Abs(got-0.95) > 1e-12 {
		t.Errorf("past-end momentum = %.12g, want maxMomentum 0.95", got)
	}
}

func ExampleOneCycleMomentum() {
	// Momentum runs opposite the LR: high at the ends, low (base) at the peak.
	for _, s := range []int{0, 3, 10} {
		fmt.Printf("%.2f ", nn.OneCycleMomentum(s, 10, 0, 0, 0))
	}
	fmt.Println()
	// Output:
	// 0.95 0.85 0.95
}
