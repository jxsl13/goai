package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// firstNonFinite must catch NaN/±Inf in every float dtype — including the raw-
// bit f16/bf16 paths, the small dtypes where overflow bites first — and in
// non-contiguous views via the logical fallback.
func TestFirstNonFiniteAllDtypes(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64, tensor.F16, tensor.BF16} {
		for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			x := tensor.Full(dt, tensor.Shape{2, 3}, 1.5)
			x.SetF64(bad, 1, 1) // flat index 4
			pos, v, hit := firstNonFinite(x)
			if !hit {
				t.Errorf("%v/%v: not detected", dt, classify(bad))
				continue
			}
			if pos != 4 {
				t.Errorf("%v/%v: pos %d, want 4", dt, classify(bad), pos)
			}
			if classify(v) != classify(bad) {
				t.Errorf("%v: classified %s, want %s", dt, classify(v), classify(bad))
			}
		}
		// all-finite: no hit
		if _, _, hit := firstNonFinite(tensor.Full(dt, tensor.Shape{4}, 2.25)); hit {
			t.Errorf("%v: false positive on finite data", dt)
		}
	}
}

// A transposed (non-contiguous) view routes through the logical accessor and
// must report the position in the VIEW's row-major order.
func TestFirstNonFiniteNonContiguous(t *testing.T) {
	x := tensor.Full(tensor.F64, tensor.Shape{2, 3}, 1.0)
	x.SetF64(math.Inf(1), 1, 0)  // base coords (1,0)
	xt, err := x.Transpose(0, 1) // shape {3,2}; Inf now at view coords (0,1) = flat 1
	if err != nil {
		t.Fatal(err)
	}
	if xt.IsContiguous() {
		t.Fatal("test premise: transpose view must be non-contiguous")
	}
	pos, v, hit := firstNonFinite(xt)
	if !hit || pos != 1 || !math.IsInf(v, 1) {
		t.Errorf("got pos=%d v=%v hit=%v, want pos=1 v=+Inf hit=true", pos, v, hit)
	}
}
