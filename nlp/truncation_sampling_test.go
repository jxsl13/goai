package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// lnLogits returns logits whose temperature-1 softmax reproduces probs exactly
// (softmax(ln p) = p when p sums to 1), so a Sampler's Dist sees the given distribution.
func lnLogits(probs []float64) []float64 {
	z := make([]float64, len(probs))
	for i, p := range probs {
		z[i] = math.Log(p)
	}
	return z
}

// §V16 tier-1 (§R91): epsilon sampling keeps exactly the tokens with prob ≥ eps and
// renormalizes them; here 0.04 and 0.01 fall below 0.05 and are removed.
func TestEpsilonSampling(t *testing.T) {
	probs := []float64{0.5, 0.3, 0.15, 0.04, 0.01}
	s := nlp.NewSampler(1, nlp.WithEpsilon(0.05))
	dist := s.Dist(lnLogits(probs))

	ksum := 0.5 + 0.3 + 0.15
	want := []float64{0.5 / ksum, 0.3 / ksum, 0.15 / ksum, 0, 0}
	for i := range want {
		if math.Abs(dist[i]-want[i]) > 1e-9 {
			t.Errorf("epsilon dist[%d] = %.6f, want %.6f", i, dist[i], want[i])
		}
	}
}

// §V16 tier-1 (§R91): eta sampling uses the entropy-adaptive threshold
// η = min(ε, √ε·exp(−H)) with H the Shannon entropy in nats. This distribution is flat
// enough that η < ε (the high-entropy regime), and the survivor set exactly matches an
// independent recomputation of the threshold.
func TestEtaSampling(t *testing.T) {
	probs := []float64{0.25, 0.2, 0.15, 0.15, 0.1, 0.08, 0.04, 0.03}
	const eps = 0.05

	var h float64
	for _, p := range probs {
		h -= p * math.Log(p)
	}
	eta := math.Min(eps, math.Sqrt(eps)*math.Exp(-h))
	if eta >= eps {
		t.Fatalf("test distribution should exercise the η<ε regime, got η=%v ε=%v", eta, eps)
	}

	dist := nlp.NewSampler(1, nlp.WithEta(eps)).Dist(lnLogits(probs))
	top := 0
	for i := range probs {
		if probs[i] > probs[top] {
			top = i
		}
	}
	var ksum float64
	for i, p := range probs {
		if p >= eta || i == top {
			ksum += p
		}
	}
	for i, p := range probs {
		want := 0.0
		if p >= eta || i == top {
			want = p / ksum
		}
		if math.Abs(dist[i]-want) > 1e-9 {
			t.Errorf("eta dist[%d] = %.6f, want %.6f (η=%.6f)", i, dist[i], want, eta)
		}
	}
}

// §V16 tier-1: both filters always keep the arg-max (min_tokens_to_keep=1) — a
// threshold above every probability still leaves the top token, one-hot.
func TestTruncationKeepsTopToken(t *testing.T) {
	probs := []float64{0.5, 0.3, 0.2}
	for _, s := range []*nlp.Sampler{
		nlp.NewSampler(1, nlp.WithEpsilon(0.9)),
		nlp.NewSampler(1, nlp.WithEta(0.9)),
	} {
		dist := s.Dist(lnLogits(probs))
		if math.Abs(dist[0]-1) > 1e-9 || dist[1] != 0 || dist[2] != 0 {
			t.Errorf("threshold above all probs must keep only the arg-max, got %v", dist)
		}
	}
}

// Eta sampling adapts its cutoff to the model's confidence. On a peaked distribution
// (token 0 at 0.9) the entropy is low, so η rises to the ε floor and the low-probability
// tail is dropped, leaving only the confident token.
func ExampleWithEta() {
	s := nlp.NewSampler(1, nlp.WithEta(0.1))
	dist := s.Dist([]float64{math.Log(0.9), math.Log(0.08), math.Log(0.02)})
	fmt.Printf("%.2f %.2f %.2f\n", dist[0], dist[1], dist[2])
	// Output: 1.00 0.00 0.00
}
