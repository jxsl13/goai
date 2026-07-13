package ops_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// numpy.concatenate: joining [2,3] tensors along axis 0 gives [4,3] (rows stacked)
// and along axis 1 gives [2,6] (columns joined), preserving element values.
func TestConcatForward(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	b := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{7, 8, 9, 10, 11, 12})

	ax0, err := ops.Concat(0, a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !ax0.Shape().Equal(tensor.Shape{4, 3}) {
		t.Fatalf("axis0 shape %v", ax0.Shape())
	}
	want0 := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	for i, w := range want0 {
		if ax0.AtF64(tensor.Unravel(i, ax0.Shape())...) != w {
			t.Errorf("axis0[%d]=%g, want %g", i, ax0.AtF64(tensor.Unravel(i, ax0.Shape())...), w)
		}
	}

	ax1, err := ops.Concat(1, a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !ax1.Shape().Equal(tensor.Shape{2, 6}) {
		t.Fatalf("axis1 shape %v", ax1.Shape())
	}
	want1 := []float64{1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}
	for i, w := range want1 {
		if ax1.AtF64(tensor.Unravel(i, ax1.Shape())...) != w {
			t.Errorf("axis1[%d]=%g, want %g", i, ax1.AtF64(tensor.Unravel(i, ax1.Shape())...), w)
		}
	}
}

// Three rank-3 inputs with different extents along a negative axis join correctly,
// and slicing the result back along the axis recovers each input (round-trip).
func TestConcatNDNegAxisRoundTrip(t *testing.T) {
	mk := func(d2 int, base float64) *tensor.Tensor {
		n := 2 * 2 * d2
		data := make([]float64, n)
		for i := range data {
			data[i] = base + float64(i)
		}
		return tensor.FromFloat64(tensor.Shape{2, 2, d2}, data)
	}
	a, b, c := mk(1, 0), mk(3, 100), mk(2, 200) // last-axis extents 1,3,2
	out, err := ops.Concat(-1, a, b, c)         // axis -1 == 2
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{2, 2, 6}) {
		t.Fatalf("shape %v, want [2 2 6]", out.Shape())
	}
	// round-trip: the axis-2 segments [0:1],[1:4],[4:6] must equal a,b,c
	inputs := []*tensor.Tensor{a, b, c}
	off := 0
	for _, in := range inputs {
		d2 := in.Shape()[2]
		for i0 := range 2 {
			for i1 := range 2 {
				for j := range d2 {
					if out.AtF64(i0, i1, off+j) != in.AtF64(i0, i1, j) {
						t.Errorf("segment mismatch at [%d,%d,%d]", i0, i1, j)
					}
				}
			}
		}
		off += d2
	}
}

func TestConcatErrors(t *testing.T) {
	a := tensor.New(tensor.F64, tensor.Shape{2, 3})
	if _, err := ops.Concat(0); err == nil {
		t.Error("zero inputs must error")
	}
	if _, err := ops.Concat(0, a, tensor.New(tensor.F64, tensor.Shape{2, 4})); err == nil {
		t.Error("non-axis dim mismatch must error")
	}
	if _, err := ops.Concat(0, a, tensor.New(tensor.F64, tensor.Shape{2, 3, 1})); err == nil {
		t.Error("rank mismatch must error")
	}
	if _, err := ops.Concat(5, a, a); err == nil {
		t.Error("axis out of range must error")
	}
}

// Concat joins arrays along an axis, like numpy.concatenate. Here two rows are
// stacked along axis 0 into a 2×2 result.
func ExampleConcat() {
	a := tensor.FromFloat64(tensor.Shape{1, 2}, []float64{1, 2})
	b := tensor.FromFloat64(tensor.Shape{1, 2}, []float64{3, 4})
	out, _ := ops.Concat(0, a, b)
	fmt.Println(out.Shape(), out.AtF64(1, 0), out.AtF64(1, 1))
	// Output: (2, 2) 3 4
}
