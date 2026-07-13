package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/jxsl13/goai/nn"
)

// §V16 tier-1 (arXiv:2309.06657): acceptance probability p_i = exp((r_i − r_max)/β). Hand check:
// rewards [1,2,0], β=1, r_max=2 ⇒ [e⁻¹, 1, e⁻²].
func TestRSOAcceptProbsParity(t *testing.T) {
	p, err := nn.RSOAcceptProbs([]float64{1, 2, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{math.Exp(-1), 1, math.Exp(-2)}
	for i := range want {
		if math.Abs(p[i]-want[i]) > 1e-12 {
			t.Errorf("p[%d] = %.12g, want %.12g", i, p[i], want[i])
		}
	}
}

// §V16 tier-1 (best always kept): the maximum-reward candidate has acceptance probability 1 and
// survives every pass.
func TestRSOMaxAlwaysAccepted(t *testing.T) {
	rewards := []float64{0.5, 3.0, 1.2, 3.0 - 1e-9}
	if p, _ := nn.RSOAcceptProbs(rewards, 0.7); p[1] != 1 {
		t.Errorf("max-reward candidate prob = %g, want 1", p[1])
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for trial := 0; trial < 200; trial++ {
		kept, _ := nn.RSOAccept(rewards, 0.7, rng)
		if !slices.Contains(kept, 1) {
			t.Fatalf("trial %d: max-reward candidate 1 was rejected", trial)
		}
	}
}

// §V16 tier-1 (rejection distribution): the empirical acceptance rate matches exp((r−r_max)/β).
func TestRSOAcceptanceRate(t *testing.T) {
	rewards := []float64{2, 1, 0} // r_max=2, β=1 ⇒ probs [1, e⁻¹≈.368, e⁻²≈.135]
	const beta = 1.0
	want, _ := nn.RSOAcceptProbs(rewards, beta)
	rng := rand.New(rand.NewPCG(7, 11))
	const N = 40000
	counts := make([]int, len(rewards))
	for range N {
		for _, i := range mustAccept(t, rewards, beta, rng) {
			counts[i]++
		}
	}
	for i := range rewards {
		if got := float64(counts[i]) / N; math.Abs(got-want[i]) > 0.02 {
			t.Errorf("candidate %d acceptance rate %.3f, want ≈%.3f", i, got, want[i])
		}
	}
}

func mustAccept(t *testing.T, r []float64, beta float64, rng *rand.Rand) []int {
	t.Helper()
	k, err := nn.RSOAccept(r, beta, rng)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

// §V16 tier-1 (β sharpness): a smaller β makes the target sharper, so sub-maximal candidates are
// accepted LESS often (more aggressive rejection).
func TestRSOBetaEffect(t *testing.T) {
	rewards := []float64{2, 1} // gap 1 below the max
	sharp, _ := nn.RSOAcceptProbs(rewards, 0.5)
	soft, _ := nn.RSOAcceptProbs(rewards, 2.0)
	if !(sharp[1] < soft[1]) {
		t.Errorf("smaller β should lower sub-max acceptance: sharp %.4f !< soft %.4f", sharp[1], soft[1])
	}
	// exact: e^{-1/0.5}=e⁻² and e^{-1/2}=e^{-0.5}.
	if math.Abs(sharp[1]-math.Exp(-2)) > 1e-12 || math.Abs(soft[1]-math.Exp(-0.5)) > 1e-12 {
		t.Errorf("β probs wrong: sharp %.6g (want e⁻²), soft %.6g (want e^-0.5)", sharp[1], soft[1])
	}
}

// §V2 (edges): a single candidate is always kept.
func TestRSOSingle(t *testing.T) {
	p, err := nn.RSOAcceptProbs([]float64{42}, 1)
	if err != nil || len(p) != 1 || p[0] != 1 {
		t.Errorf("single candidate: p=%v err=%v, want [1]", p, err)
	}
}

func TestRSOErrors(t *testing.T) {
	if _, err := nn.RSOAcceptProbs(nil, 1); err == nil {
		t.Error("empty rewards: want error")
	}
	if _, err := nn.RSOAcceptProbs([]float64{1, 2}, 0); err == nil {
		t.Error("beta 0: want error")
	}
}

func ExampleRSOAcceptProbs() {
	// The best candidate is always kept; lower-reward ones survive with exp((r−r_max)/β).
	p, _ := nn.RSOAcceptProbs([]float64{3, 2, 1}, 1)
	fmt.Printf("%.3f %.3f %.3f\n", p[0], p[1], p[2])
	// Output:
	// 1.000 0.368 0.135
}
