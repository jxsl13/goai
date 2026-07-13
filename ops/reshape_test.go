package ops_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// numpy.reshape preserves row-major element order: [2,3] → [3,2] and → [6] keep
// the flat sequence.
func TestReshapeForward(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	for _, sh := range [][]int{{3, 2}, {6}, {1, 6}, {6, 1}, {2, 3}} {
		r, err := ops.Reshape(x, sh...)
		if err != nil {
			t.Fatalf("%v: %v", sh, err)
		}
		if !r.Shape().Equal(tensor.Shape(sh)) {
			t.Fatalf("shape %v want %v", r.Shape(), sh)
		}
		for i := range 6 { // flat order preserved
			if r.AtF64(tensor.Unravel(i, r.Shape())...) != x.AtF64(tensor.Unravel(i, x.Shape())...) {
				t.Errorf("%v: flat[%d] changed", sh, i)
			}
		}
	}
	if _, err := ops.Reshape(x, 4, 2); err == nil {
		t.Error("incompatible element count must error")
	}
}

// ExpandDims inserts a size-1 axis; Squeeze removes it — they round-trip, and the
// flat values are preserved.
func TestExpandSqueezeRoundTrip(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	cases := []struct {
		axis int
		want tensor.Shape
	}{{0, tensor.Shape{1, 2, 3}}, {1, tensor.Shape{2, 1, 3}}, {2, tensor.Shape{2, 3, 1}}, {-1, tensor.Shape{2, 3, 1}}}
	for _, c := range cases {
		e, err := ops.ExpandDims(x, c.axis)
		if err != nil {
			t.Fatalf("axis %d: %v", c.axis, err)
		}
		if !e.Shape().Equal(c.want) {
			t.Fatalf("axis %d: shape %v want %v", c.axis, e.Shape(), c.want)
		}
		for i := range 6 {
			if e.AtF64(tensor.Unravel(i, e.Shape())...) != x.AtF64(tensor.Unravel(i, x.Shape())...) {
				t.Errorf("axis %d flat[%d] changed", c.axis, i)
			}
		}
		ax := c.axis
		if ax < 0 {
			ax += 3 // normalized axis in the expanded tensor
		}
		s, err := ops.Squeeze(e, ax)
		if err != nil {
			t.Fatalf("squeeze axis %d: %v", ax, err)
		}
		if !s.Shape().Equal(tensor.Shape{2, 3}) {
			t.Errorf("round-trip shape %v", s.Shape())
		}
	}
	if _, err := ops.Squeeze(x, 0); err == nil {
		t.Error("squeezing a size-2 axis must error")
	}
}

// numpy.stack joins along a NEW axis: stacking two [2,3] along axis 0 gives [2,2,3]
// with result[0]=a, result[1]=b; along axis 1 gives [2,2,3] with result[:,0]=a.
func TestStack(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	b := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{7, 8, 9, 10, 11, 12})

	s0, err := ops.Stack(0, a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !s0.Shape().Equal(tensor.Shape{2, 2, 3}) {
		t.Fatalf("axis0 shape %v", s0.Shape())
	}
	for i := range 2 {
		for j := range 3 {
			if s0.AtF64(0, i, j) != a.AtF64(i, j) || s0.AtF64(1, i, j) != b.AtF64(i, j) {
				t.Errorf("stack axis0 mismatch at [%d,%d]", i, j)
			}
		}
	}
	s1, _ := ops.Stack(1, a, b)
	if !s1.Shape().Equal(tensor.Shape{2, 2, 3}) {
		t.Fatalf("axis1 shape %v", s1.Shape())
	}
	for i := range 2 {
		for j := range 3 {
			if s1.AtF64(i, 0, j) != a.AtF64(i, j) || s1.AtF64(i, 1, j) != b.AtF64(i, j) {
				t.Errorf("stack axis1 mismatch at [%d,%d]", i, j)
			}
		}
	}
	if _, err := ops.Stack(0, a, tensor.New(tensor.F64, tensor.Shape{2, 4})); err == nil {
		t.Error("mismatched shapes must error")
	}
	if _, err := ops.Stack(0); err == nil {
		t.Error("zero inputs must error")
	}
}

// ExpandDims adds a size-1 dimension, like numpy.expand_dims. Here a length-3
// vector becomes a 1×3 row.
func ExampleExpandDims() {
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})
	e, _ := ops.ExpandDims(x, 0)
	fmt.Println(e.Shape())
	// Output: (1, 3)
}

// Stack joins tensors along a new axis, like numpy.stack — turning a list of N
// [2,3] tensors into one [N,2,3] batch.
func ExampleStack() {
	a := tensor.New(tensor.F64, tensor.Shape{2, 3})
	b := tensor.New(tensor.F64, tensor.Shape{2, 3})
	s, _ := ops.Stack(0, a, b)
	fmt.Println(s.Shape())
	// Output: (2, 2, 3)
}
