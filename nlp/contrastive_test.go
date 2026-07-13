package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// independent log-softmax for the differential test.
func logsm(x []float64) []float64 {
	m := math.Inf(-1)
	for _, v := range x {
		m = math.Max(m, v)
	}
	var s float64
	for _, v := range x {
		s += math.Exp(v - m)
	}
	lse := m + math.Log(s)
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = v - lse
	}
	return out
}

// §V16 tier-1: ContrastiveLogits masks tokens the expert finds implausible
// (le < log α + max le) to −∞ and scores the rest (1+β)·log p_e − β·log p_a —
// checked against an independent recomputation (Li 2023 / O'Brien & Lewis 2023).
func TestContrastiveLogitsFormula(t *testing.T) {
	expert := []float64{3, 1, 0, -2}
	amateur := []float64{0.5, -0.5, 1, 0}
	const alpha, beta = 0.1, 1.0

	got := nlp.ContrastiveLogits(expert, amateur, alpha, beta)
	le, la := logsm(expert), logsm(amateur)
	maxLe := math.Inf(-1)
	for _, v := range le {
		maxLe = math.Max(maxLe, v)
	}
	thresh := math.Log(alpha) + maxLe

	nPlaus, nMasked := 0, 0
	for v := range expert {
		if le[v] < thresh {
			nMasked++
			if !math.IsInf(got[v], -1) {
				t.Errorf("token %d implausible (le=%v < %v) → want −inf, got %v", v, le[v], thresh, got[v])
			}
		} else {
			nPlaus++
			want := (1+beta)*le[v] - beta*la[v]
			if math.Abs(got[v]-want) > 1e-12 {
				t.Errorf("token %d score = %v, want (1+β)le−β·la = %v", v, got[v], want)
			}
		}
	}
	if nPlaus == 0 || nMasked == 0 {
		t.Fatalf("test needs both plausible and masked tokens (got %d/%d)", nPlaus, nMasked)
	}
}

// §V16 tier-1 property: with beta=0 the amateur is ignored, so contrastive decoding
// reduces EXACTLY to the expert's own decoding — a deterministic equivalence
// (greedy). The plausibility constraint never changes the arg-max (the top expert
// token is always plausible).
func TestContrastiveDecodeBetaZeroEqualsExpert(t *testing.T) {
	model, tokens := loadGPTModel(t)
	prompt := tokens[:min(4, len(tokens))]
	const n = 8

	want, err := model.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	got, err := nlp.ContrastiveDecode(model, model, prompt, n, 0.1, 0, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("contrastive produced %d tokens, expert Generate %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d] = %d, want %d — β=0 contrastive must equal expert greedy", i, got[i], want[i])
		}
	}
}

// The plausibility constraint excludes tokens the expert finds unlikely even when
// the amateur loves them — here the expert is confident on token 0, so the amateur's
// favourite (token 2, low expert probability) is masked out of the contrast.
func ExampleContrastiveLogits() {
	expert := []float64{5, 1, -3} // token 2: very low expert probability
	amateur := []float64{0, 0, 5} // amateur strongly prefers token 2
	s := nlp.ContrastiveLogits(expert, amateur, 0.1, 1.0)
	fmt.Println(math.IsInf(s[2], -1)) // excluded by the plausibility constraint
	// Output: true
}
