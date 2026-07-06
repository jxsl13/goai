package ops_test

import (
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/ref" // register the reference backend
)

func TestEagerOps(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	b := tensor.FromFloat64(tensor.Shape{3}, []float64{10, 20, 30})

	sum, err := ops.Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float64{11, 22, 33} {
		if sum.AtF64(i) != want {
			t.Errorf("Add[%d] = %v, want %v", i, sum.AtF64(i), want)
		}
	}

	r, err := ops.ReLU(tensor.FromFloat64(tensor.Shape{3}, []float64{-1, 0, 2}))
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float64{0, 0, 2} {
		if r.AtF64(i) != want {
			t.Errorf("ReLU[%d] = %v, want %v", i, r.AtF64(i), want)
		}
	}
}
