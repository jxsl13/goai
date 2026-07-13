package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// logitsFor returns logits whose softmax is exactly p (logit = ln p).
func logitsFor(p []float64) []float64 {
	z := make([]float64, len(p))
	for i, v := range p {
		z[i] = math.Log(v)
	}
	return z
}

// §V16 tier-1: hand-computed locally typical filtering. For p=[0.4,0.3,0.2,0.1] the
// entropy is H≈1.2799 nats and the typicality order (|−log p − H| ascending) is tokens
// 1,2,0,3 (cumulative mass 0.3,0.5,0.9,1.0). τ=0.7 keeps {1,2,0} (drops the least
// typical token 3); τ=0.45 keeps only {1,2}. (τ values off the exact cumulative
// boundaries to avoid float ties.)
func TestTypicalHandComputed(t *testing.T) {
	p := []float64{0.4, 0.3, 0.2, 0.1}
	z := logitsFor(p)

	// τ=0.7 → keep {0,1,2}, drop 3; renormalize over mass 0.9
	d := nlp.NewSampler(1, nlp.WithTypical(0.7)).Dist(z)
	want07 := []float64{0.4 / 0.9, 0.3 / 0.9, 0.2 / 0.9, 0}
	for i := range p {
		if math.Abs(d[i]-want07[i]) > 1e-9 {
			t.Errorf("τ=0.7 d[%d] = %.10g, want %.10g", i, d[i], want07[i])
		}
	}

	// τ=0.45 → keep {1,2} (most typical), renormalize over mass 0.5
	d = nlp.NewSampler(1, nlp.WithTypical(0.45)).Dist(z)
	want045 := []float64{0, 0.3 / 0.5, 0.2 / 0.5, 0}
	for i := range p {
		if math.Abs(d[i]-want045[i]) > 1e-9 {
			t.Errorf("τ=0.45 d[%d] = %.10g, want %.10g", i, d[i], want045[i])
		}
	}
}

// §V16 tier-1: τ ≥ 1 (or 0) is a no-op — the distribution is unchanged.
func TestTypicalNoOp(t *testing.T) {
	p := []float64{0.5, 0.25, 0.15, 0.1}
	z := logitsFor(p)
	base := nlp.NewSampler(1).Dist(z)                      // no typical filter
	off := nlp.NewSampler(1, nlp.WithTypical(1.0)).Dist(z) // τ=1 disabled
	for i := range p {
		if math.Abs(base[i]-off[i]) > 1e-12 {
			t.Errorf("τ=1 changed d[%d]: %.12g vs %.12g", i, off[i], base[i])
		}
	}
}

// property: every kept token has typicality score ≤ every dropped token's score
// (the filter keeps a prefix of the ascending-by-|surprisal−H| order), and ≥1 kept.
func TestTypicalPrefixProperty(t *testing.T) {
	p := []float64{0.30, 0.28, 0.20, 0.12, 0.06, 0.04}
	z := logitsFor(p)
	d := nlp.NewSampler(1, nlp.WithTypical(0.75)).Dist(z)

	var h float64
	for _, pi := range p {
		h -= pi * math.Log(pi)
	}
	score := func(i int) float64 { return math.Abs(-math.Log(p[i]) - h) }

	kept, dropped := 0, 0
	var maxKept, minDropped = math.Inf(-1), math.Inf(1)
	for i := range p {
		if d[i] > 0 {
			kept++
			maxKept = math.Max(maxKept, score(i))
		} else {
			dropped++
			minDropped = math.Min(minDropped, score(i))
		}
	}
	if kept == 0 {
		t.Fatal("kept 0 tokens; must keep ≥1")
	}
	if dropped > 0 && maxKept > minDropped+1e-12 {
		t.Errorf("not a score-prefix: max kept score %.6g > min dropped %.6g", maxKept, minDropped)
	}
	// probabilities renormalize to 1 over the kept set
	var sum float64
	for _, v := range d {
		sum += v
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("kept probs sum = %.10g, want 1", sum)
	}
}

func ExampleWithTypical() {
	// p=[0.4,0.3,0.2,0.1]: locally typical (τ=0.7) drops the least-typical tail token.
	z := []float64{math.Log(0.4), math.Log(0.3), math.Log(0.2), math.Log(0.1)}
	d := nlp.NewSampler(1, nlp.WithTypical(0.7)).Dist(z)
	fmt.Printf("%.3f %.3f %.3f %.3f\n", d[0], d[1], d[2], d[3])
	// Output:
	// 0.444 0.333 0.222 0.000
}
