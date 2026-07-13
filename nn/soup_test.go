package nn_test

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// scalarModels wraps each value as a one-model, one-[1]-tensor parameter list.
func scalarModels(vals []float64) [][]*tensor.Tensor {
	ms := make([][]*tensor.Tensor, len(vals))
	for i, v := range vals {
		ms[i] = []*tensor.Tensor{vec([]float64{v})}
	}
	return ms
}

// §V16 tier-1: the uniform soup is the unweighted mean of every model's weights.
func TestUniformSoup(t *testing.T) {
	soup, err := nn.UniformSoup(scalarModels([]float64{1.0, 5.5, 3.0}))
	if err != nil {
		t.Fatal(err)
	}
	if got := soup[0].AtF64(0); math.Abs(got-9.5/3) > 1e-12 {
		t.Errorf("uniform soup = %.12g, want %.12g", got, 9.5/3)
	}
}

func TestUniformSoupMultiParam(t *testing.T) {
	mk := func(a, b, c float64) []*tensor.Tensor {
		return []*tensor.Tensor{vec([]float64{a, b}), vec([]float64{c})}
	}
	soup, err := nn.UniformSoup([][]*tensor.Tensor{mk(1, 2, 10), mk(3, 6, 20)})
	if err != nil {
		t.Fatal(err)
	}
	if soup[0].AtF64(0) != 2 || soup[0].AtF64(1) != 4 || soup[1].AtF64(0) != 15 {
		t.Errorf("multi-param soup = [%g %g],[%g], want [2 4],[15]", soup[0].AtF64(0), soup[0].AtF64(1), soup[1].AtF64(0))
	}
}

// peakAt2 is a synthetic held-out accuracy that peaks when the (single-param) soup averages to 2.
func peakAt2(soup []*tensor.Tensor) float64 { return -math.Pow(soup[0].AtF64(0)-2, 2) }

// §V16 tier-1 (Algorithm 1): greedy soup walks models in decreasing val-accuracy order and keeps a
// model only when it does not lower the soup's held-out accuracy. Hand-worked: models 1.0/5.5/3.0
// (already accuracy-sorted) with a peak at avg=2 ⇒ keep {1.0} (acc −1), reject 5.5 (avg 3.25 ⇒
// −1.5625 < −1), keep 3.0 (avg 2.0 ⇒ 0 ≥ −1). Result: ingredients {0,2}, soup 2.0.
func TestGreedySoupSelection(t *testing.T) {
	soup, chosen, err := nn.GreedySoup(scalarModels([]float64{1.0, 5.5, 3.0}), []float64{0.9, 0.85, 0.8}, peakAt2)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(chosen, []int{0, 2}) {
		t.Errorf("chosen ingredients = %v, want [0 2]", chosen)
	}
	if got := soup[0].AtF64(0); math.Abs(got-2.0) > 1e-12 {
		t.Errorf("greedy soup = %.12g, want 2.0", got)
	}
}

// §V16 tier-1 (sort by val accuracy): the greedy order follows valAcc, not input order — the
// highest-accuracy model is always the first ingredient even when passed last.
func TestGreedySoupSortsByValAcc(t *testing.T) {
	// same values reindexed: model 1 has the highest valAcc (0.9) and value 1.0.
	_, chosen, err := nn.GreedySoup(scalarModels([]float64{3.0, 1.0, 5.5}), []float64{0.8, 0.9, 0.85}, peakAt2)
	if err != nil {
		t.Fatal(err)
	}
	if chosen[0] != 1 {
		t.Errorf("first ingredient = %d, want 1 (highest valAcc)", chosen[0])
	}
	if !slices.Equal(chosen, []int{1, 0}) { // keep 1.0, reject 5.5, keep 3.0
		t.Errorf("chosen = %v, want [1 0]", chosen)
	}
}

// §V16 tier-1 (key guarantee): the greedy soup never scores below the best individual model on the
// held-out metric (it only ever accepts non-decreasing steps from that starting point).
func TestGreedySoupNeverWorseThanBest(t *testing.T) {
	models := scalarModels([]float64{1.0, 5.5, 3.0})
	soup, _, err := nn.GreedySoup(models, []float64{0.9, 0.85, 0.8}, peakAt2)
	if err != nil {
		t.Fatal(err)
	}
	best, _ := nn.UniformSoup(models[:1]) // the top-accuracy model alone
	if peakAt2(soup) < peakAt2(best) {
		t.Errorf("greedy soup acc %.6g < best single %.6g", peakAt2(soup), peakAt2(best))
	}
}

func TestGreedySoupSingle(t *testing.T) {
	soup, chosen, err := nn.GreedySoup(scalarModels([]float64{7}), []float64{0.5}, peakAt2)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(chosen, []int{0}) || soup[0].AtF64(0) != 7 {
		t.Errorf("single model: chosen %v soup %g, want [0] / 7", chosen, soup[0].AtF64(0))
	}
}

func TestSoupErrors(t *testing.T) {
	if _, err := nn.UniformSoup(nil); err == nil {
		t.Error("empty models: want error")
	}
	bad := [][]*tensor.Tensor{{vec([]float64{1, 2})}, {vec([]float64{1})}} // shape mismatch
	if _, err := nn.UniformSoup(bad); err == nil {
		t.Error("shape mismatch: want error")
	}
	ms := scalarModels([]float64{1, 2})
	if _, _, err := nn.GreedySoup(ms, []float64{0.5}, peakAt2); err == nil {
		t.Error("valAcc length mismatch: want error")
	}
	if _, _, err := nn.GreedySoup(ms, []float64{0.5, 0.6}, nil); err == nil {
		t.Error("nil eval: want error")
	}
}

func ExampleGreedySoup() {
	// Three fine-tunes (scalar weights 1.0/5.5/3.0), held-out accuracy peaking at an average of 2.
	// Greedy keeps the best model, skips 5.5, and adds 3.0 — landing exactly on the peak.
	models := scalarModels([]float64{1.0, 5.5, 3.0})
	soup, chosen, _ := nn.GreedySoup(models, []float64{0.9, 0.85, 0.8}, peakAt2)
	fmt.Printf("%v %.1f\n", chosen, soup[0].AtF64(0))
	// Output:
	// [0 2] 2.0
}
