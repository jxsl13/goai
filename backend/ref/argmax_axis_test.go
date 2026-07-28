package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// TestArgMaxAxisMatchesNaive pins the run-hoisted axis argmax against a naive
// independent oracle on RAW BITS, over every axis of a rank-3 and a rank-1 input,
// and — the case this transform can actually break — over an input full of TIES,
// where the contract is that the lowest index wins.
func TestArgMaxAxisMatchesNaive(t *testing.T) {
	shapes := []tensor.Shape{{3, 5, 7}, {4, 4, 4}, {6}}
	for _, sh := range shapes {
		for axis := range len(sh) {
			for _, tied := range []bool{false, true} {
				x := bench.RandF32(sh, 1)
				if tied {
					// Every value equal: any index is a valid maximum, so only the
					// lowest-index tie rule distinguishes correct from incorrect.
					for i := range x.Numel() {
						x.SetF64(1.5, tensor.Unravel(i, x.Shape())...)
					}
				}
				got := argmaxAxis(t, x, axis)
				want := naiveArgMaxAxis(x, axis)
				if len(got) != len(want) {
					t.Fatalf("shape %v axis %d: length %d != %d", sh, axis, len(got), len(want))
				}
				for i := range want {
					if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
						t.Fatalf("shape %v axis %d tied=%v elem %d: got %v want %v",
							sh, axis, tied, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func argmaxAxis(t *testing.T, x *tensor.Tensor, axis int) []float64 {
	t.Helper()
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	out, err := backend.Execute(ctx, backend.OpArgMax, []*tensor.Tensor{x}, backend.ArgMaxAttrs{Axis: axis})
	if err != nil {
		t.Fatal(err)
	}
	o := out[0]
	res := make([]float64, o.Numel())
	for i := range res {
		res[i] = o.AtF64(tensor.Unravel(i, o.Shape())...)
	}
	return res
}

// naiveArgMaxAxis scans each lane directly through AtF64 — no shared traversal with
// the kernel. Strict > keeps the lowest index on ties.
func naiveArgMaxAxis(x *tensor.Tensor, axis int) []float64 {
	sh := x.Shape()
	outShape := tensor.Shape{}
	for a := range len(sh) {
		if a != axis {
			outShape = append(outShape, sh[a])
		}
	}
	n := outShape.Numel()
	res := make([]float64, n)
	for i := range n {
		oc := tensor.Unravel(i, outShape)
		coord := make([]int, len(sh))
		k := 0
		for a := range len(sh) {
			if a == axis {
				continue
			}
			coord[a] = oc[k]
			k++
		}
		best, bi := math.Inf(-1), 0
		for j := range sh[axis] {
			coord[axis] = j
			if v := x.AtF64(coord...); v > best {
				best, bi = v, j
			}
		}
		res[i] = float64(bi)
	}
	return res
}
