package nlp_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

func sumInts(s []int) int {
	t := 0
	for _, v := range s {
		t += v
	}
	return t
}

// §V16 tier-1 (arXiv:2406.02069, pyramid): budgets decrease from bottom (layer 0) to top,
// with the total conserved.
func TestPyramidKVBudgetsMonotoneConserved(t *testing.T) {
	const layers, total, minB = 8, 800, 40
	b := nlp.PyramidKVBudgets(layers, total, minB)
	if len(b) != layers {
		t.Fatalf("got %d budgets, want %d", len(b), layers)
	}
	if sumInts(b) != total {
		t.Errorf("Σ budgets = %d, want %d (conserved)", sumInts(b), total)
	}
	for l := 1; l < layers; l++ {
		if b[l] > b[l-1] {
			t.Errorf("not a decreasing pyramid at layer %d: %v", l, b)
		}
	}
	if b[0] <= total/layers { // bottom gets more than the uniform average
		t.Errorf("bottom budget %d not above the average %d", b[0], total/layers)
	}
}

// §V16 tier-1 (endpoints): top layer ≈ minBudget, bottom ≈ 2·avg − minBudget.
func TestPyramidKVBudgetsEndpoints(t *testing.T) {
	const layers, total, minB = 10, 1000, 20 // avg=100 ⇒ bottom≈180
	b := nlp.PyramidKVBudgets(layers, total, minB)
	if b[layers-1] != minB {
		t.Errorf("top budget = %d, want minBudget %d", b[layers-1], minB)
	}
	if bottom := 2*(total/layers) - minB; b[0] < bottom-1 || b[0] > bottom+1 {
		t.Errorf("bottom budget = %d, want ≈ 2·avg−min = %d", b[0], bottom)
	}
}

// §V2: minBudget = average gives a uniform (flat) allocation.
func TestPyramidKVBudgetsUniform(t *testing.T) {
	b := nlp.PyramidKVBudgets(5, 500, 100) // avg = 100
	for l, v := range b {
		if v != 100 {
			t.Errorf("min=avg should be uniform, layer %d = %d", l, v)
		}
	}
}

// §V2 (edges): single layer takes the whole budget; a too-large minBudget is clamped so the
// allocation stays a valid non-increasing pyramid summing to the total.
func TestPyramidKVBudgetsEdges(t *testing.T) {
	if b := nlp.PyramidKVBudgets(1, 300, 10); len(b) != 1 || b[0] != 300 {
		t.Errorf("single layer = %v, want [300]", b)
	}
	b := nlp.PyramidKVBudgets(4, 400, 999) // min > avg=100 ⇒ clamped
	if sumInts(b) != 400 {
		t.Errorf("clamped: Σ = %d, want 400", sumInts(b))
	}
	for l := 1; l < len(b); l++ {
		if b[l] > b[l-1] {
			t.Errorf("clamped allocation not non-increasing: %v", b)
		}
	}
}

// §V2 (end-to-end): each layer's pyramid budget drives a SnapKV-style token selection,
// keeping at most that layer's budget positions.
func TestPyramidKVWithSnapKV(t *testing.T) {
	obsAttn := [][]float64{ // 2 observation queries over a 12-token prompt
		{0.1, 0.4, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.03},
		{0.2, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02},
	}
	budgets := nlp.PyramidKVBudgets(4, 24, 4) // avg 6 ⇒ e.g. [8,7,5,4]
	for layer, budget := range budgets {
		keep := nlp.SnapKVKeep(obsAttn, budget, 1)
		if len(keep) > budget {
			t.Errorf("layer %d kept %d > budget %d", layer, len(keep), budget)
		}
	}
}

func ExamplePyramidKVBudgets() {
	// 6 layers, total budget 600 (avg 100), top-layer minimum 40: a decreasing pyramid whose
	// budgets still sum to 600.
	fmt.Println(nlp.PyramidKVBudgets(6, 600, 40))
	// Output:
	// [160 136 112 88 64 40]
}
