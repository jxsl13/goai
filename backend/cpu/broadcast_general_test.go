package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// TestGeneralBroadcastBitIdenticalToRef pins the run-hoisted broadcast materialization
// against the ref backend BIT-for-bit (raw bits, not a tolerance). Add does no
// reassociation here — both backends sum the same pairs — so any bit difference is a
// materialization bug, which is exactly what this guards.
//
// Coverage, established by mutation rather than assumed: these cases reach the
// CONTIGUOUS-RUN arm (innermost effective stride 1) — corrupting it turns this test
// red. They do NOT reach the fill arm (innermost stride 0) despite the shape below
// looking like they should; a fill-arm defect is caught by the wider package suite,
// not here. Do not read the case names as proof of which arm ran.
func TestGeneralBroadcastBitIdenticalToRef(t *testing.T) {
	cases := []struct {
		name string
		a, b tensor.Shape
	}{
		{"mid-axis stride1", tensor.Shape{4, 1, 8}, tensor.Shape{4, 3, 8}},
		{"innermost size1 operand", tensor.Shape{4, 3, 1}, tensor.Shape{4, 3, 8}},
		{"leading absent", tensor.Shape{8}, tensor.Shape{4, 3, 8}},
		{"both broadcast", tensor.Shape{4, 1, 8}, tensor.Shape{1, 3, 8}},
		{"scalar", tensor.Shape{}, tensor.Shape{4, 3, 8}},
		{"rank1 broadcast", tensor.Shape{1}, tensor.Shape{5}},
	}
	for _, tc := range cases {
		for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
			t.Run(tc.name+"/"+dt.String(), func(t *testing.T) {
				var x, y *tensor.Tensor
				if dt == tensor.F32 {
					x, y = bench.RandF32(tc.a, 1), bench.RandF32(tc.b, 2)
				} else {
					x, y = bench.RandF64(tc.a, 1), bench.RandF64(tc.b, 2)
				}
				want := execOn(t, backend.Ref, x, y)
				got := execOn(t, backend.CPU, x, y)
				if len(want) != len(got) {
					t.Fatalf("length %d != %d", len(got), len(want))
				}
				for i := range want {
					if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
						t.Fatalf("element %d: cpu bits %#x != ref bits %#x (%v vs %v)",
							i, math.Float64bits(got[i]), math.Float64bits(want[i]), got[i], want[i])
					}
				}
			})
		}
	}
}

func execOn(t *testing.T, name backend.Name, x, y *tensor.Tensor) []float64 {
	t.Helper()
	be, ok := backend.Get(name)
	if !ok {
		t.Fatalf("backend %v unavailable", name)
	}
	ctx := backend.NewContext().WithBackend(be)
	out, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{x, y}, nil)
	if err != nil {
		t.Fatal(err)
	}
	o := out[0].Contiguous()
	res := make([]float64, o.Numel())
	for i := range res {
		res[i] = o.AtF64(tensor.Unravel(i, o.Shape())...)
	}
	return res
}
