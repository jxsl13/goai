package ops_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func flatOf(t *tensor.Tensor) []float64 {
	out := make([]float64, t.Numel())
	for i := range out {
		out[i] = t.AtF64(tensor.Unravel(i, t.Shape())...)
	}
	return out
}

// numpy x[start:end] along an axis: rows 1:3 of a [4,3] give [2,3]; cols 0:2 give [4,2].
func TestSliceForward(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{4, 3}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	r, err := ops.Slice(x, 0, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("axis0 shape %v", r.Shape())
	}
	want := []float64{4, 5, 6, 7, 8, 9}
	for i, w := range want {
		if flatOf(r)[i] != w {
			t.Errorf("axis0[%d]=%g want %g", i, flatOf(r)[i], w)
		}
	}
	c, _ := ops.Slice(x, -1, 0, 2) // negative axis
	if !c.Shape().Equal(tensor.Shape{4, 2}) {
		t.Fatalf("axis-1 shape %v", c.Shape())
	}
	wantc := []float64{1, 2, 4, 5, 7, 8, 10, 11}
	for i, w := range wantc {
		if flatOf(c)[i] != w {
			t.Errorf("axis-1[%d]=%g want %g", i, flatOf(c)[i], w)
		}
	}
}

// Split cuts along an axis into sized pieces; re-concatenating recovers the input.
func TestSplitConcatInverse(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{2, 6}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	for _, sizes := range [][]int{{2, 2, 2}, {1, 2, 3}, {6}, {3, 3}} {
		parts, err := ops.Split(x, 1, sizes...)
		if err != nil {
			t.Fatalf("sizes %v: %v", sizes, err)
		}
		if len(parts) != len(sizes) {
			t.Fatalf("sizes %v: %d parts", sizes, len(parts))
		}
		for i, s := range sizes {
			if !parts[i].Shape().Equal(tensor.Shape{2, s}) {
				t.Errorf("sizes %v part %d shape %v", sizes, i, parts[i].Shape())
			}
		}
		joined, err := ops.Concat(1, parts...)
		if err != nil {
			t.Fatal(err)
		}
		for i := range x.Numel() {
			idx := tensor.Unravel(i, x.Shape())
			if joined.AtF64(idx...) != x.AtF64(idx...) {
				t.Errorf("sizes %v: concat(split) != x at %v", sizes, idx)
			}
		}
	}
}

func TestSliceSplitErrors(t *testing.T) {
	x := tensor.New(tensor.F64, tensor.Shape{4, 3})
	if _, err := ops.Slice(x, 0, 2, 2); err == nil {
		t.Error("empty range (start>=end) must error")
	}
	if _, err := ops.Slice(x, 0, 0, 5); err == nil {
		t.Error("end past extent must error")
	}
	if _, err := ops.Slice(x, 3, 0, 1); err == nil {
		t.Error("axis out of range must error")
	}
	if _, err := ops.Split(x, 0, 1, 2); err == nil {
		t.Error("sizes not summing to extent must error")
	}
	if _, err := ops.Split(x, 0, 2, -2, 4); err == nil {
		t.Error("non-positive size must error")
	}
}

// Slice extracts a sub-range along an axis, like numpy x[start:end]. Here rows 1:3
// of a 4×2 tensor give a 2×2 result.
func ExampleSlice() {
	x := tensor.FromFloat64(tensor.Shape{4, 2}, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	r, _ := ops.Slice(x, 0, 1, 3)
	fmt.Println(r.Shape(), r.AtF64(0, 0), r.AtF64(1, 1))
	// Output: (2, 2) 3 6
}

// Split cuts a tensor into sized pieces along an axis — e.g. a fused [seq, 3·d]
// QKV projection into three [seq, d] tensors.
func ExampleSplit() {
	qkv := tensor.New(tensor.F64, tensor.Shape{5, 12})
	parts, _ := ops.Split(qkv, 1, 4, 4, 4)
	fmt.Println(len(parts), parts[0].Shape())
	// Output: 3 (5, 4)
}
