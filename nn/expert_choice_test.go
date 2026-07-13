package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
)

// §V16 tier-1 (the defining property): every expert selects EXACTLY `capacity` tokens,
// so the load is perfectly balanced regardless of the score distribution.
func TestExpertChoiceLoadBalance(t *testing.T) {
	// 6 tokens, 3 experts; a skewed score matrix (token 0 loved by all experts)
	scores := [][]float64{
		{0.9, 0.8, 0.7}, {0.1, 0.2, 0.3}, {0.4, 0.5, 0.1},
		{0.3, 0.1, 0.6}, {0.2, 0.6, 0.2}, {0.5, 0.3, 0.4},
	}
	cap := nn.ExpertChoiceCapacity(6, 3, 2) // k = round(6·2/3) = 4
	if cap != 4 {
		t.Fatalf("capacity = %d, want 4", cap)
	}
	tokens, gates := nn.ExpertChoiceRoute(scores, cap)
	total := 0
	for e := range tokens {
		if len(tokens[e]) != cap {
			t.Errorf("expert %d got %d tokens, want %d (perfect balance)", e, len(tokens[e]), cap)
		}
		if len(gates[e]) != cap {
			t.Errorf("expert %d has %d gates, want %d", e, len(gates[e]), cap)
		}
		total += len(tokens[e])
	}
	if total != 3*cap {
		t.Errorf("total assignments = %d, want %d", total, 3*cap)
	}
}

// §V16 tier-1: each expert's selected tokens are its top-`capacity` by affinity (every
// selected score ≥ every non-selected score for that expert), and gates equal those
// affinities.
func TestExpertChoiceTopK(t *testing.T) {
	scores := [][]float64{
		{0.9, 0.1}, {0.2, 0.8}, {0.7, 0.3}, {0.4, 0.6}, {0.5, 0.5},
	}
	const cap = 2
	tokens, gates := nn.ExpertChoiceRoute(scores, cap)
	for e := range tokens {
		selected := map[int]bool{}
		var minSel = math.Inf(1)
		for i, tok := range tokens[e] {
			selected[tok] = true
			minSel = math.Min(minSel, scores[tok][e])
			if gates[e][i] != scores[tok][e] {
				t.Errorf("expert %d gate[%d] = %g, want affinity %g", e, i, gates[e][i], scores[tok][e])
			}
		}
		for tok := range scores {
			if !selected[tok] && scores[tok][e] > minSel+1e-12 {
				t.Errorf("expert %d dropped token %d (score %g) over a lower selected %g", e, tok, scores[tok][e], minSel)
			}
		}
	}
	// expert 0's top-2 tokens by column 0 (0.9,0.2,0.7,0.4,0.5) are 0 and 2
	if !(tokens[0][0] == 0 && tokens[0][1] == 2) {
		t.Errorf("expert 0 selected %v, want [0 2]", tokens[0])
	}
}

// §V16 tier-1: a token may be selected by several experts or by none (heterogeneous
// per-token compute), unlike token-choice which routes every token to exactly k experts.
func TestExpertChoiceHeterogeneous(t *testing.T) {
	// token 0 is every expert's favorite; token 2 is nobody's top-1
	scores := [][]float64{{0.9, 0.9}, {0.5, 0.4}, {0.1, 0.2}, {0.6, 0.7}}
	tokens, _ := nn.ExpertChoiceRoute(scores, 2) // 2 experts, cap 2
	count := make([]int, 4)
	for e := range tokens {
		for _, tok := range tokens[e] {
			count[tok]++
		}
	}
	if count[0] != 2 {
		t.Errorf("token 0 selected by %d experts, want 2", count[0])
	}
	if count[2] != 0 {
		t.Errorf("token 2 selected by %d experts, want 0", count[2])
	}
}

// combine: y[token] = Σ over experts that selected it of gate·expertOut.
func TestExpertChoiceCombine(t *testing.T) {
	// 3 tokens, 2 experts, capacity 1, dim 2
	scores := [][]float64{{0.9, 0.2}, {0.3, 0.8}, {0.5, 0.5}}
	tokens, gates := nn.ExpertChoiceRoute(scores, 1) // expert0→token0, expert1→token1
	// expert outputs: expert e, slot i → a vector
	expertOut := [][][]float64{
		{{1, 2}}, // expert 0, slot 0 (token 0)
		{{3, 4}}, // expert 1, slot 0 (token 1)
	}
	y := nn.ExpertChoiceCombine(tokens, gates, expertOut, 3, 2)
	// token 0: gate 0.9 · [1,2] = [0.9,1.8]; token 1: 0.8·[3,4]=[2.4,3.2]; token 2: none → 0
	want := [][]float64{{0.9, 1.8}, {2.4, 3.2}, {0, 0}}
	for tk := range 3 {
		for d := range 2 {
			if math.Abs(y[tk][d]-want[tk][d]) > 1e-12 {
				t.Errorf("y[%d][%d] = %g, want %g", tk, d, y[tk][d], want[tk][d])
			}
		}
	}
}

func ExampleExpertChoiceRoute() {
	// 4 tokens, 2 experts, capacity 2: each expert picks its 2 highest-affinity tokens.
	scores := [][]float64{{0.9, 0.1}, {0.2, 0.8}, {0.7, 0.3}, {0.4, 0.6}}
	tokens, _ := nn.ExpertChoiceRoute(scores, 2)
	fmt.Printf("expert0=%v expert1=%v\n", tokens[0], tokens[1])
	// Output:
	// expert0=[0 2] expert1=[1 3]
}
