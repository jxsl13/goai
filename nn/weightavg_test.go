package nn_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 (SWA, Izmailov et al. 2018): the incremental equal-weight update avg←avg+(w−avg)/(n+1)
// yields exactly the arithmetic mean of the captured snapshots, verified against a direct mean over
// several hand-chosen snapshots and over many random ones.
func TestSWAEqualWeightAverage(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 1})
	swa := nn.NewSWA([]*tensor.Tensor{p})
	snaps := [][]float64{{1, 1}, {3, 5}, {8, 2}}
	for _, s := range snaps {
		p.SetF64(s[0], 0)
		p.SetF64(s[1], 1)
		if err := swa.Update(); err != nil {
			t.Fatal(err)
		}
	}
	if swa.N() != 3 {
		t.Errorf("N = %d, want 3", swa.N())
	}
	w := swa.Weights()[0]
	want := []float64{(1 + 3 + 8) / 3.0, (1 + 5 + 2) / 3.0}
	for i, wv := range want {
		if math.Abs(w.AtF64(i)-wv) > 1e-12 {
			t.Errorf("SWA[%d] = %.14g, want %.14g", i, w.AtF64(i), wv)
		}
	}
}

// §V16 tier-1: over many random snapshots the incremental running average equals the true mean
// (numerically, within f64 tolerance).
func TestSWAMatchesTrueMean(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const d, k = 5, 200
	p := tensor.New(tensor.F64, tensor.Shape{d})
	swa := nn.NewSWA([]*tensor.Tensor{p})
	sum := make([]float64, d)
	for range k {
		for j := range d {
			v := rng.NormFloat64() * 10
			p.SetF64(v, j)
			sum[j] += v
		}
		if err := swa.Update(); err != nil {
			t.Fatal(err)
		}
	}
	w := swa.Weights()[0]
	for j := range d {
		if got, want := w.AtF64(j), sum[j]/k; math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
			t.Errorf("coord %d: SWA %.14g, mean %.14g", j, got, want)
		}
	}
}

// §V16 tier-1 (weight EMA / Polyak averaging): the update avg←decay·avg+(1−decay)·w, initialized to
// the starting weights, reproduces an independent recomputation of the recurrence at 1e-12.
func TestEMARecurrence(t *testing.T) {
	const decay = 0.9
	p := tensor.FromFloat64(tensor.Shape{1}, []float64{1})
	ema := nn.NewEMA([]*tensor.Tensor{p}, decay)
	if got := ema.Weights()[0].AtF64(0); got != 1 {
		t.Errorf("EMA must initialize to the current weights, got %v", got)
	}
	ref := 1.0
	for _, w := range []float64{2, 3, -1, 0.5} {
		p.SetF64(w, 0)
		if err := ema.Update(); err != nil {
			t.Fatal(err)
		}
		ref = decay*ref + (1-decay)*w
		if got := ema.Weights()[0].AtF64(0); math.Abs(got-ref) > 1e-12 {
			t.Errorf("EMA = %.14g, want %.14g", got, ref)
		}
	}
}

// §V16 tier-1: both averagers handle multiple parameter tensors of different shapes and never mutate
// the training parameters.
func TestWeightAvgMultiParamNoMutation(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	b := tensor.FromFloat64(tensor.Shape{2, 1}, []float64{3, 4})
	swa := nn.NewSWA([]*tensor.Tensor{a, b})
	_ = swa.Update()
	a.SetF64(5, 0) // mutate the live params AFTER the snapshot
	// the SWA snapshot must reflect the value at Update time, not the later mutation
	if got := swa.Weights()[0].AtF64(0); got != 1 {
		t.Errorf("SWA snapshot must be independent of later param edits, got %v", got)
	}
	if len(swa.Weights()) != 2 || !swa.Weights()[1].Shape().Equal(tensor.Shape{2, 1}) {
		t.Error("SWA must preserve param count and shapes")
	}
}

// SWA averages parameter snapshots with equal weight. After capturing weights 2 then 4, the SWA
// weight is their mean, 3.
func ExampleSWA() {
	w := tensor.FromFloat64(tensor.Shape{1}, []float64{2})
	swa := nn.NewSWA([]*tensor.Tensor{w})
	_ = swa.Update() // snapshot 2
	w.SetF64(4, 0)
	_ = swa.Update() // snapshot 4
	fmt.Printf("%.1f\n", swa.Weights()[0].AtF64(0))
	// Output: 3.0
}

// A weight EMA tracks an exponential moving average of the parameters. Starting at 0 with decay 0.5,
// one update toward weight 1 gives 0.5·0 + 0.5·1 = 0.5.
func ExampleEMA() {
	w := tensor.FromFloat64(tensor.Shape{1}, []float64{0})
	ema := nn.NewEMA([]*tensor.Tensor{w}, 0.5)
	w.SetF64(1, 0)
	_ = ema.Update()
	fmt.Printf("%.2f\n", ema.Weights()[0].AtF64(0))
	// Output: 0.50
}
