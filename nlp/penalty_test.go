package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §R57 CTRL repetition penalty is sign-aware: a positive logit of a seen token is
// divided by θ, a negative one is multiplied by θ (pushed further negative).
func TestRepetitionPenaltySignAware(t *testing.T) {
	const theta = 1.5
	logits := []float64{2.0, -2.0, 0.0, 5.0}
	nlp.ApplyPenalties(logits, []int{0, 1, 2}, theta, 0, 0)
	if math.Abs(logits[0]-2.0/theta) > 1e-12 {
		t.Errorf("positive seen logit: %.6f, want %.6f", logits[0], 2.0/theta)
	}
	if math.Abs(logits[1]-(-2.0*theta)) > 1e-12 {
		t.Errorf("negative seen logit: %.6f, want %.6f", logits[1], -2.0*theta)
	}
	if logits[2] != 0 {
		t.Errorf("zero logit must stay 0, got %.6f", logits[2])
	}
	if logits[3] != 5.0 {
		t.Errorf("unseen token 3 must be unchanged, got %.6f", logits[3])
	}
}

// θ=1 (or ≤0) is a no-op.
func TestRepetitionPenaltyNoOp(t *testing.T) {
	logits := []float64{2, -2, 3}
	orig := append([]float64(nil), logits...)
	nlp.ApplyPenalties(logits, []int{0, 1, 2}, 1, 0, 0)
	for i := range logits {
		if logits[i] != orig[i] {
			t.Errorf("θ=1 changed logit %d", i)
		}
	}
}

// §R57 OpenAI frequency penalty scales with occurrence count; presence is a flat
// one-time subtraction for any appeared token.
func TestFrequencyAndPresencePenalty(t *testing.T) {
	// token 0 appears 3×, token 1 once, token 2 never.
	logits := []float64{10, 10, 10}
	nlp.ApplyPenalties(logits, []int{0, 0, 0, 1}, 1 /*no rep*/, 0.5 /*freq*/, 2.0 /*presence*/)
	if math.Abs(logits[0]-(10-0.5*3-2.0)) > 1e-12 { // -1.5 freq -2 presence
		t.Errorf("freq+presence token0: %.6f, want %.6f", logits[0], 10-0.5*3-2.0)
	}
	if math.Abs(logits[1]-(10-0.5*1-2.0)) > 1e-12 {
		t.Errorf("freq+presence token1: %.6f, want %.6f", logits[1], 10-0.5-2.0)
	}
	if logits[2] != 10 {
		t.Errorf("unseen token2 must be unchanged, got %.6f", logits[2])
	}
}

// The whole point: penalizing a repeated token lowers its post-softmax probability.
func TestPenaltyReducesRepeatProbability(t *testing.T) {
	base := []float64{3.0, 1.0, 0.5}
	pen := append([]float64(nil), base...)
	nlp.ApplyPenalties(pen, []int{0}, 1.5, 0, 0) // token 0 was generated

	softmax := func(z []float64) []float64 {
		m := z[0]
		for _, v := range z {
			if v > m {
				m = v
			}
		}
		var s float64
		out := make([]float64, len(z))
		for i, v := range z {
			out[i] = math.Exp(v - m)
			s += out[i]
		}
		for i := range out {
			out[i] /= s
		}
		return out
	}
	p0 := softmax(base)[0]
	p1 := softmax(pen)[0]
	if p1 >= p0 {
		t.Errorf("repetition penalty must lower repeated-token prob: %.4f → %.4f", p0, p1)
	}
}

// Out-of-range history entries are ignored (no panic, no effect).
func TestPenaltyIgnoresOutOfRange(t *testing.T) {
	logits := []float64{1, 2}
	nlp.ApplyPenalties(logits, []int{-1, 5, 0}, 2, 0, 0)
	if math.Abs(logits[0]-0.5) > 1e-12 || logits[1] != 2 {
		t.Errorf("only in-range token 0 should change: %v", logits)
	}
}
