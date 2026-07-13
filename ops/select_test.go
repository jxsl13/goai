package ops_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func TestMaximumMinimumWhereForward(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 5, 3})
	b := tensor.FromFloat64(tensor.Shape{3}, []float64{4, 2, 3})
	check := func(name string, got *tensor.Tensor, want []float64) {
		for i, w := range want {
			if v := got.AtF64(i); v != w {
				t.Errorf("%s[%d]=%g, want %g", name, i, v, w)
			}
		}
	}
	mx, _ := ops.Maximum(a, b)
	check("max", mx, []float64{4, 5, 3})
	mn, _ := ops.Minimum(a, b)
	check("min", mn, []float64{1, 2, 3})

	cond := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 0, 1})
	w, _ := ops.Where(cond, tensor.FromFloat64(tensor.Shape{3}, []float64{10, 20, 30}), tensor.FromFloat64(tensor.Shape{3}, []float64{40, 50, 60}))
	check("where", w, []float64{10, 50, 30})

	if _, err := ops.Maximum(a, tensor.New(tensor.F64, tensor.Shape{2})); err == nil {
		t.Error("shape mismatch must error")
	}
	if _, err := ops.Where(cond, a, tensor.New(tensor.F64, tensor.Shape{2})); err == nil {
		t.Error("where shape mismatch must error")
	}
}

// Where picks elementwise from two tensors by a condition mask, like numpy.where —
// keeping a where the mask is nonzero and b elsewhere.
func ExampleWhere() {
	cond := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 0, 0, 1})
	a := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 2, 3, 4})
	b := tensor.FromFloat64(tensor.Shape{4}, []float64{-1, -2, -3, -4})
	out, _ := ops.Where(cond, a, b)
	fmt.Printf("%.0f %.0f %.0f %.0f\n", out.AtF64(0), out.AtF64(1), out.AtF64(2), out.AtF64(3))
	// Output: 1 -2 -3 4
}
