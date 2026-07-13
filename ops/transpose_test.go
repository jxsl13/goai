package ops_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1 (DEFINITIONAL, numpy.T): out[j,i] = x[i,j].
func TestTransposeForward(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	y, err := ops.Transpose(x)
	if err != nil {
		t.Fatal(err)
	}
	if !y.Shape().Equal(tensor.Shape{3, 2}) {
		t.Fatalf("shape %v, want [3,2]", y.Shape())
	}
	for i := range 2 {
		for j := range 3 {
			if y.AtF64(j, i) != x.AtF64(i, j) {
				t.Errorf("T[%d,%d] = %g, want %g", j, i, y.AtF64(j, i), x.AtF64(i, j))
			}
		}
	}
	if _, err := ops.Transpose(tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})); err == nil {
		t.Error("rank-1: want error")
	}
}

func ExampleTranspose() {
	x := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})
	y, _ := ops.Transpose(x)
	fmt.Printf("%g %g / %g %g\n", y.AtF64(0, 0), y.AtF64(0, 1), y.AtF64(1, 0), y.AtF64(1, 1))
	// Output:
	// 1 3 / 2 4
}
