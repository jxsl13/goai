package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// logitsFromProbs builds logits whose softmax at temperature 1 reproduces probs.
func logitsFromProbs(probs []float64) []float64 {
	l := make([]float64, len(probs))
	for i, p := range probs {
		l[i] = math.Log(p)
	}
	return l
}

// §V16 tier-1 (§T574): with firing probability 1 the top choices above the threshold
// are excluded except the least probable qualifier — over many seeded draws only the
// survivor and the below-threshold tail appear, in the renormalized 3:1 ratio.
func TestXTCFires(t *testing.T) {
	probs := []float64{0.5, 0.3, 0.15, 0.05} // qualifiers ≥0.1: {0,1,2}; survivor: 2
	s := nlp.NewSampler(7, nlp.WithXTC(1.0), nlp.WithXTCThreshold(0.1))
	counts := make([]int, 4)
	const draws = 4000
	for range draws {
		counts[s.Sample(logitsFromProbs(probs))]++
	}
	if counts[0] != 0 || counts[1] != 0 {
		t.Fatalf("excluded top choices were drawn: %v", counts)
	}
	// renormalized survivors: 0.15/0.20 vs 0.05/0.20 = 3:1.
	got := float64(counts[2]) / float64(counts[3])
	if got < 2.4 || got > 3.7 {
		t.Errorf("survivor ratio = %.2f, want ≈3 (counts %v)", got, counts)
	}
}

// §V16 tier-1 (§T574): does-not-fire cases — probability 0, threshold above 0.5 (at
// most one qualifier), a single qualifier, and greedy decoding are all untouched.
func TestXTCDoesNotFire(t *testing.T) {
	probs := []float64{0.5, 0.3, 0.15, 0.05}
	for name, s := range map[string]*nlp.Sampler{
		"probability 0":  nlp.NewSampler(3, nlp.WithXTC(0), nlp.WithXTCThreshold(0.1)),
		"threshold >0.5": nlp.NewSampler(3, nlp.WithXTC(1.0), nlp.WithXTCThreshold(0.6)),
		"one qualifier":  nlp.NewSampler(3, nlp.WithXTC(1.0), nlp.WithXTCThreshold(0.4)),
	} {
		counts := make([]int, 4)
		for range 4000 {
			counts[s.Sample(logitsFromProbs(probs))]++
		}
		if counts[0] == 0 {
			t.Errorf("%s: the top token must still be drawable, counts %v", name, counts)
		}
	}
	g := nlp.NewSampler(3, nlp.WithTemperature(0), nlp.WithXTC(1.0), nlp.WithXTCThreshold(0.1))
	for range 8 {
		if got := g.Sample(logitsFromProbs(probs)); got != 0 {
			t.Fatalf("greedy must ignore XTC, picked %d", got)
		}
	}
}

// §V16 tier-1 (§T574): partial firing — with probability 0.5 both behaviors occur
// across seeded draws (top token appears, but markedly less often than without XTC).
func TestXTCPartialFiring(t *testing.T) {
	probs := []float64{0.6, 0.4}
	s := nlp.NewSampler(11, nlp.WithXTC(0.5), nlp.WithXTCThreshold(0.1))
	counts := make([]int, 2)
	const draws = 6000
	for range draws {
		counts[s.Sample(logitsFromProbs(probs))]++
	}
	// fired half the time → expected P(top) ≈ 0.5·0.6 = 0.3 (vs 0.6 unfired).
	frac := float64(counts[0]) / draws
	if frac < 0.24 || frac > 0.36 {
		t.Errorf("top-token fraction = %.3f, want ≈0.30", frac)
	}
}

// §V16 tier-1: Dist stays pure — XTC must not perturb the distribution the
// speculative accept/reject math reads.
func TestXTCLeavesDistPure(t *testing.T) {
	probs := []float64{0.5, 0.3, 0.15, 0.05}
	s := nlp.NewSampler(5, nlp.WithXTC(1.0), nlp.WithXTCThreshold(0.1))
	d := s.Dist(logitsFromProbs(probs))
	for i, want := range probs {
		if math.Abs(d[i]-want) > 1e-9 {
			t.Fatalf("Dist perturbed at %d: got %v", i, d)
		}
	}
}
