package nn_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/nn"
)

// §V16 tier-1 (paper CONFIRMED, arXiv:2408.15664): top-k selection uses the BIASED
// score s_i+b_i, but the returned weight is the ORIGINAL affinity s_i.
func TestLossFreeBiasedSelection(t *testing.T) {
	b := nn.NewLossFreeBalancer(3, 0.001)
	b.Bias = []float64{0, 1.0, 0} // lift expert 1
	aff := []float64{0.9, 0.5, 0.1}
	// affinity favors expert 0, but biased score 0.5+1.0=1.5 makes expert 1 win.
	experts, weights := b.Route(aff, 1)
	if len(experts) != 1 || experts[0] != 1 {
		t.Fatalf("selected %v, want [1] (biased score wins)", experts)
	}
	if weights[0] != 0.5 { // original affinity, NOT the biased 1.5
		t.Errorf("weight=%g, want original affinity 0.5 (not biased score)", weights[0])
	}
	// with zero bias the selection follows affinity.
	b0 := nn.NewLossFreeBalancer(3, 0.001)
	e, w := b0.Route(aff, 2)
	if len(e) != 2 || e[0] != 0 || e[1] != 1 {
		t.Errorf("zero-bias top-2 = %v, want [0 1]", e)
	}
	if w[0] != 0.9 || w[1] != 0.5 {
		t.Errorf("weights=%v, want [0.9 0.5]", w)
	}
}

// §V16 tier-1: the bias update b_i ← b_i + u·sign(c̄ − c_i) — up for under-loaded,
// down for over-loaded, unchanged when perfectly balanced.
func TestLossFreeUpdateRule(t *testing.T) {
	b := nn.NewLossFreeBalancer(3, 0.5)
	b.Update([]int{3, 0, 0}) // mean 1: expert 0 over, 1&2 under
	if b.Bias[0] != -0.5 || b.Bias[1] != 0.5 || b.Bias[2] != 0.5 {
		t.Errorf("bias=%v, want [-0.5 0.5 0.5]", b.Bias)
	}
	// a perfectly balanced load leaves the bias untouched.
	bal := nn.NewLossFreeBalancer(3, 0.5)
	bal.Bias = []float64{0.2, -0.1, 0.4}
	bal.Update([]int{5, 5, 5})
	if bal.Bias[0] != 0.2 || bal.Bias[1] != -0.1 || bal.Bias[2] != 0.4 {
		t.Errorf("balanced update changed bias to %v", bal.Bias)
	}
}

// §V2 (the defining property): with strongly skewed affinities the bias controller
// drives the routing toward balance — every expert ends up selected a fair share,
// even though the raw affinity always favors expert 0.
func TestLossFreeBalancesSkewedRouting(t *testing.T) {
	const N, rounds = 3, 300
	b := nn.NewLossFreeBalancer(N, 0.1)
	aff := []float64{1.0, 0.0, 0.0} // expert 0 always the highest raw affinity
	counts := make([]int, N)
	for r := 0; r < rounds; r++ {
		experts, _ := b.Route(aff, 1) // route one token, top-1
		load := make([]int, N)
		load[experts[0]] = 1
		counts[experts[0]]++
		b.Update(load)
	}
	for i, c := range counts { // without balancing expert 0 would take all 300
		if c < rounds/5 {
			t.Errorf("expert %d selected %d/%d (<20%%) — not balanced: counts=%v", i, c, rounds, counts)
		}
	}
}

func TestLossFreeEdge(t *testing.T) {
	b := nn.NewLossFreeBalancer(4, 0.001)
	if e, _ := b.Route([]float64{1, 2, 3, 4}, 10); len(e) != 4 { // k>N clamps to N
		t.Errorf("k>N returned %d experts, want 4", len(e))
	}
	if e, _ := b.Route([]float64{1, 2}, 0); e != nil { // k≤0 ⇒ nil
		t.Errorf("k=0 returned %v, want nil", e)
	}
	b.Update(nil) // empty loads is a no-op (must not panic)
	if def := nn.NewLossFreeBalancer(2, 0); def.Rate != 0.001 {
		t.Errorf("default rate=%g, want 0.001", def.Rate)
	}
}

func ExampleLossFreeBalancer() {
	b := nn.NewLossFreeBalancer(3, 0.1)
	aff := []float64{1.0, 0.0, 0.0} // expert 0 favored by raw affinity
	// simulate 30 single-token batches; the bias steers tokens off the hot expert.
	counts := make([]int, 3)
	for r := 0; r < 30; r++ {
		e, _ := b.Route(aff, 1)
		load := make([]int, 3)
		load[e[0]] = 1
		counts[e[0]]++
		b.Update(load)
	}
	fmt.Printf("balanced counts: %v\n", counts) // was 30/0/0 without balancing
	// Output:
	// balanced counts: [14 8 8]
}
