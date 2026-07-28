package tensor_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// TestGatherCastBitIdentical pins the run-hoisted strided gather against an
// independent oracle: the strided view read one element at a time through AtF64,
// which does not share the gather traversal. Comparison is on RAW BITS, not a
// tolerance. Covers same-type (S==D) and cross-type (S!=D via Cast) instantiations.
func TestGatherCastBitIdentical(t *testing.T) {
	perms := [][]int{{1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {2, 1, 0}}
	shapes := []tensor.Shape{{3, 5, 7}, {2, 1, 9}, {4, 4, 4}}
	for _, sh := range shapes {
		for _, p := range perms {
			for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
				name := dt.String()
				t.Run(name, func(t *testing.T) {
					var base *tensor.Tensor
					if dt == tensor.F32 {
						base = bench.RandF32(sh, 1)
					} else {
						base = bench.RandF64(sh, 1)
					}
					v, err := base.Permute(p...)
					if err != nil {
						t.Fatal(err)
					}
					checkBits(t, v)
					// Cross-type instantiation: S != D.
					if dt == tensor.F64 {
						checkBits(t, v.Cast(tensor.F32))
					} else {
						checkBits(t, v.Cast(tensor.F64))
					}
				})
			}
		}
	}
}

func checkBits(t *testing.T, v *tensor.Tensor) {
	t.Helper()
	got := v.Contiguous()
	if got.Numel() != v.Numel() {
		t.Fatalf("numel %d != %d", got.Numel(), v.Numel())
	}
	for i := range v.Numel() {
		coord := tensor.Unravel(i, v.Shape())
		want := v.AtF64(coord...) // independent oracle: no shared traversal
		g := got.AtF64(tensor.Unravel(i, got.Shape())...)
		if math.Float64bits(g) != math.Float64bits(want) {
			t.Fatalf("element %d %v: got bits %#x, want %#x (%v vs %v)",
				i, coord, math.Float64bits(g), math.Float64bits(want), g, want)
		}
	}
}
