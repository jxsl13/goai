package nlp_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// constNext returns the same logits at every step (independent of the prefix).
func constNext(logits []float64) nlp.NextLogits {
	return func([]int) []float64 { return append([]float64(nil), logits...) }
}

// lastTokens returns the final token of each beam.
func lastTokens(beams []nlp.Beam) []int {
	out := make([]int, len(beams))
	for i, b := range beams {
		out[i] = b.Tokens[len(b.Tokens)-1]
	}
	slices.Sort(out)
	return out
}

// §V16 tier-1 (Algorithm 1, Hamming diversity): with one beam per group and a strong penalty the
// groups pick DISTINCT tokens — the top-G by likelihood, one per group.
func TestDiverseBeamDiversity(t *testing.T) {
	next := constNext([]float64{0, -1, -2, -3, -4}) // token 0 > 1 > 2 > …
	beams, err := nlp.DiverseBeamSearch(next, []int{9}, 3, 3, 1, -1, 0, 100)
	if err != nil {
		t.Fatal(err)
	}
	if got := lastTokens(beams); !slices.Equal(got, []int{0, 1, 2}) {
		t.Errorf("diverse first tokens = %v, want distinct {0,1,2}", got)
	}
}

// §V16 tier-1 (λ=0 ⇒ no diversity): with no penalty every group collapses onto the same best
// token — DBS degenerates to independent beam searches.
func TestDiverseBeamNoPenaltyCollapses(t *testing.T) {
	next := constNext([]float64{0, -1, -2, -3, -4})
	beams, err := nlp.DiverseBeamSearch(next, []int{9}, 3, 3, 1, -1, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := lastTokens(beams); !slices.Equal(got, []int{0, 0, 0}) {
		t.Errorf("λ=0 first tokens = %v, want all 0", got)
	}
}

// §V16 tier-1 (exact penalty magnitude): the log-softmax gap equals the logit gap, so a second
// group switches off the first group's token exactly when λ exceeds that gap. With logits gap
// 0.4 between token 0 and token 1, λ=0.5 flips group 2 to token 1, but λ=0.3 does not.
func TestDiverseBeamHammingThreshold(t *testing.T) {
	next := constNext([]float64{0, -0.4, -20}) // logp0 − logp1 = 0.4
	flip, err := nlp.DiverseBeamSearch(next, []int{0}, 2, 2, 1, -1, 0, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if got := lastTokens(flip); !slices.Equal(got, []int{0, 1}) {
		t.Errorf("λ=0.5 (>gap 0.4): tokens %v, want {0,1}", got)
	}
	stay, err := nlp.DiverseBeamSearch(next, []int{0}, 2, 2, 1, -1, 0, 0.3)
	if err != nil {
		t.Fatal(err)
	}
	if got := lastTokens(stay); !slices.Equal(got, []int{0, 0}) {
		t.Errorf("λ=0.3 (<gap 0.4): tokens %v, want {0,0}", got)
	}
}

// §V16 tier-1 (reduces to beam search): one group with no penalty is ordinary beam search — the
// best hypothesis matches nlp.BeamSearch.
func TestDiverseBeamReducesToBeamSearch(t *testing.T) {
	// deterministic chain: the best next token is (last+1) mod 5.
	next := func(prefix []int) []float64 {
		last := prefix[len(prefix)-1]
		l := make([]float64, 5)
		for i := range l {
			l[i] = -10
		}
		l[(last+1)%5] = 0
		return l
	}
	dbs, err := nlp.DiverseBeamSearch(next, []int{0}, 2, 1, 3, -1, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	bs := nlp.BeamSearch(next, []int{0}, 2, 3, -1, 0)
	if !slices.Equal(dbs[0].Tokens, bs[0].Tokens) {
		t.Errorf("groups=1,λ=0 best %v != BeamSearch best %v", dbs[0].Tokens, bs[0].Tokens)
	}
	if want := []int{0, 1, 2, 3}; !slices.Equal(dbs[0].Tokens, want) {
		t.Errorf("best chain %v, want %v", dbs[0].Tokens, want)
	}
}

// §V2 (multi-step diversity): over several steps each group still yields a full-length,
// distinct-leading hypothesis.
func TestDiverseBeamMultiStep(t *testing.T) {
	next := constNext([]float64{0, -0.5, -1, -1.5, -2, -2.5})
	beams, err := nlp.DiverseBeamSearch(next, []int{0}, 3, 3, 4, -1, 0, 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(beams) != 3 {
		t.Fatalf("got %d beams, want 3", len(beams))
	}
	firsts := map[int]bool{}
	for _, b := range beams {
		if len(b.Tokens) != 5 { // start + 4 new
			t.Errorf("beam length %d, want 5", len(b.Tokens))
		}
		firsts[b.Tokens[1]] = true
	}
	if len(firsts) != 3 {
		t.Errorf("first new tokens not distinct across groups: %v", firsts)
	}
}

func TestDiverseBeamErrors(t *testing.T) {
	next := constNext([]float64{0, -1})
	if _, err := nlp.DiverseBeamSearch(next, []int{0}, 3, 2, 1, -1, 0, 1); err == nil {
		t.Error("width 3 not divisible by groups 2: want error")
	}
	if _, err := nlp.DiverseBeamSearch(next, []int{0}, 0, 1, 1, -1, 0, 1); err == nil {
		t.Error("width 0: want error")
	}
	if _, err := nlp.DiverseBeamSearch(next, []int{0}, 2, 0, 1, -1, 0, 1); err == nil {
		t.Error("groups 0: want error")
	}
}

func ExampleDiverseBeamSearch() {
	// Three groups, one beam each, strong diversity: the groups return the three most likely
	// yet DISTINCT first tokens instead of three near-copies.
	next := constNext([]float64{0, -1, -2, -3})
	beams, _ := nlp.DiverseBeamSearch(next, []int{7}, 3, 3, 1, -1, 0, 100)
	fmt.Println(lastTokens(beams))
	// Output:
	// [0 1 2]
}
