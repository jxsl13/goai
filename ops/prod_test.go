package ops_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1 (DEFINITIONAL, numpy.prod): products over all axes / axis 0 / axis 1,
// with and without keepdims, against numpy 2.5.1 goldens.
func TestProdVsNumpy(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})

	all, _ := ops.Prod(a, nil, false)
	if all.AtF64() != 720 {
		t.Errorf("prod all = %g, want 720", all.AtF64())
	}
	ax0, _ := ops.Prod(a, []int{0}, false) // [4,10,18]
	for i, w := range []float64{4, 10, 18} {
		if ax0.AtF64(i) != w {
			t.Errorf("prod ax0 [%d] = %g, want %g", i, ax0.AtF64(i), w)
		}
	}
	ax1, _ := ops.Prod(a, []int{1}, false) // [6,120]
	for i, w := range []float64{6, 120} {
		if ax1.AtF64(i) != w {
			t.Errorf("prod ax1 [%d] = %g, want %g", i, ax1.AtF64(i), w)
		}
	}
	keep, _ := ops.Prod(a, []int{1}, true) // [[6],[120]] shape [2,1]
	if !keep.Shape().Equal(tensor.Shape{2, 1}) || keep.AtF64(0, 0) != 6 || keep.AtF64(1, 0) != 120 {
		t.Errorf("prod ax1 keepdims = %v %g,%g", keep.Shape(), keep.AtF64(0, 0), keep.AtF64(1, 0))
	}
}

func ExampleProd() {
	a := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 2, 3, 4})
	p, _ := ops.Prod(a, nil, false)
	fmt.Printf("%g\n", p.AtF64())
	// Output:
	// 24
}
