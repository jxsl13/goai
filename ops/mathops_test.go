package ops_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func TestSqrtAbsClipForward(t *testing.T) {
	check := func(name string, got *tensor.Tensor, want []float64) {
		for i, w := range want {
			if v := got.AtF64(tensor.Unravel(i, got.Shape())...); math.Abs(v-w) > 1e-12 {
				t.Errorf("%s[%d]=%g, want %g", name, i, v, w)
			}
		}
	}
	sq, _ := ops.Sqrt(tensor.FromFloat64(tensor.Shape{3}, []float64{1, 4, 9}))
	check("sqrt", sq, []float64{1, 2, 3})

	ab, _ := ops.Abs(tensor.FromFloat64(tensor.Shape{3}, []float64{-2, 3, -0.5}))
	check("abs", ab, []float64{2, 3, 0.5})

	cl, _ := ops.Clip(tensor.FromFloat64(tensor.Shape{4}, []float64{-1, 0.5, 2, 0}), 0, 1)
	check("clip", cl, []float64{0, 0.5, 1, 0})

	if _, err := ops.Clip(tensor.New(tensor.F64, tensor.Shape{2}), 1, 0); err == nil {
		t.Error("clip lo>hi must error")
	}
}

// Clip clamps values to a range, like numpy.clip — here to [0,1].
func ExampleClip() {
	x := tensor.FromFloat64(tensor.Shape{4}, []float64{-0.5, 0.25, 1.5, 0.75})
	c, _ := ops.Clip(x, 0, 1)
	fmt.Printf("%.2f %.2f %.2f %.2f\n", c.AtF64(0), c.AtF64(1), c.AtF64(2), c.AtF64(3))
	// Output: 0.00 0.25 1.00 0.75
}
