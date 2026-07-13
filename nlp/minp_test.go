package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// lnProbs returns logits whose softmax (temperature 1) is exactly probs.
func lnProbs(probs []float64) []float64 {
	z := make([]float64, len(probs))
	for i, p := range probs {
		z[i] = math.Log(p)
	}
	return z
}

func sampleCounts(s *nlp.Sampler, logits []float64, n int) map[int]int {
	c := map[int]int{}
	for range n {
		c[s.Sample(append([]float64(nil), logits...))]++
	}
	return c
}

// §R63: min-p keeps tokens with prob ≥ MinP·max-prob and removes the rest. For
// probs [0.5,0.3,0.15,0.05] and MinP 0.2 (threshold 0.1) token 3 (0.05) is cut and
// never sampled; for MinP 0.5 (threshold 0.25) only tokens 0,1 survive.
func TestMinPThresholdMasks(t *testing.T) {
	logits := lnProbs([]float64{0.5, 0.3, 0.15, 0.05})

	c := sampleCounts(nlp.NewSampler(1, nlp.WithMinP(0.2)), logits, 4000)
	if c[3] != 0 {
		t.Errorf("MinP 0.2: token 3 (prob 0.05 < 0.1) must be masked, sampled %d×", c[3])
	}
	if c[0] == 0 || c[1] == 0 || c[2] == 0 {
		t.Error("MinP 0.2: tokens 0,1,2 should all survive")
	}

	c = sampleCounts(nlp.NewSampler(1, nlp.WithMinP(0.5)), logits, 4000)
	if c[2] != 0 || c[3] != 0 {
		t.Errorf("MinP 0.5: only tokens 0,1 survive, got 2=%d 3=%d", c[2], c[3])
	}
}

// The single most-probable token always survives (P_max ≥ MinP·P_max for MinP≤1).
func TestMinPKeepsTopToken(t *testing.T) {
	logits := lnProbs([]float64{0.4, 0.3, 0.2, 0.1})
	// even a very aggressive MinP keeps the argmax token
	c := sampleCounts(nlp.NewSampler(1, nlp.WithMinP(0.99)), logits, 500)
	if c[0] != 500 {
		t.Errorf("aggressive MinP must keep only the top token, got %v", c)
	}
}

// §R63 adaptive property: at the SAME MinP, a confident (peaked) distribution
// truncates more than a flat one — the threshold scales with the top probability.
func TestMinPAdaptiveWithConfidence(t *testing.T) {
	const minP = 0.2
	// peaked: max 0.7 → threshold 0.14 → only the top token (others 0.1 < 0.14)
	peaked := sampleCounts(nlp.NewSampler(1, nlp.WithMinP(minP)), lnProbs([]float64{0.7, 0.1, 0.1, 0.1}), 2000)
	// flat: max 0.25 → threshold 0.05 → all four tokens survive
	flat := sampleCounts(nlp.NewSampler(2, nlp.WithMinP(minP)), lnProbs([]float64{0.25, 0.25, 0.25, 0.25}), 2000)
	if len(peaked) != 1 {
		t.Errorf("peaked distribution should truncate to 1 token, got %d", len(peaked))
	}
	if len(flat) != 4 {
		t.Errorf("flat distribution should keep all 4 tokens, got %d", len(flat))
	}
}

// MinP 0 disables the filter (all tokens reachable).
func TestMinPDisabled(t *testing.T) {
	logits := lnProbs([]float64{0.5, 0.3, 0.15, 0.05})
	c := sampleCounts(nlp.NewSampler(3), logits, 4000) // no min-p
	if len(c) != 4 {
		t.Errorf("MinP disabled should reach all tokens, got %d", len(c))
	}
}

// ExampleSampler_minP shows min-p sampling via the functional-options constructor.
// A confident distribution (token 0 ≈ 0.83) with min-p 0.2 sets the keep-threshold
// to 0.2·0.83 ≈ 0.17, so only the top token survives and sampling is decisive.
func ExampleSampler_minP() {
	s := nlp.NewSampler(1, nlp.WithMinP(0.2))
	fmt.Println(s.Sample([]float64{3, 0, 0, 0}))
	// Output: 0
}
