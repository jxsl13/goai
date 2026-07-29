package backend

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestEinsumF32FastPathParity verifies the allF32 fast path is bit-identical to the
// generic variadic AtF64/SetF64 fallback it replaces, across several contraction specs.
func TestEinsumF32FastPathParity(t *testing.T) {
	cases := []struct {
		spec string
		dims [][]int
	}{
		{"ij,jk->ik", [][]int{{6, 5}, {5, 7}}},
		{"td,tmd->tm", [][]int{{4, 8}, {4, 3, 8}}},
		{"tm,tmd->td", [][]int{{4, 3}, {4, 3, 8}}},
		{"trh,trd->thd", [][]int{{5, 4, 3}, {5, 4, 6}}},
		{"tjk,jk->tk", [][]int{{4, 5, 6}, {5, 6}}},
	}
	for _, tc := range cases {
		ops := make([]*tensor.Tensor, len(tc.dims))
		for k, d := range tc.dims {
			sh := tensor.Shape{}
			for _, x := range d {
				sh = append(sh, x)
			}
			ops[k] = tensor.New(tensor.F32, sh)
			s := ops[k].Storage().F32()
			for i := range s {
				s[i] = float32(math.Sin(float64((k+1)*97+i) * 0.013))
			}
		}
		ctx := NewContext()
		got, err := Execute(ctx, OpEinsum, ops, EinsumAttrs{Spec: tc.spec})
		if err != nil {
			t.Fatalf("%s: %v", tc.spec, err)
		}
		// Reference: replicate the exact generic fallback (float32 storage read via
		// float64 widen, prod in f64, per-combo float32 narrowing of the output).
		inSubs, outSub, err := ParseEinsum(tc.spec, len(ops))
		if err != nil {
			t.Fatal(err)
		}
		size := map[byte]int{}
		for k, sub := range inSubs {
			for pos := 0; pos < len(sub); pos++ {
				size[sub[pos]] = ops[k].Shape()[pos]
			}
		}
		var order []byte
		seen := map[byte]bool{}
		for _, sub := range inSubs {
			for i := 0; i < len(sub); i++ {
				if !seen[sub[i]] {
					seen[sub[i]] = true
					order = append(order, sub[i])
				}
			}
		}
		outShape := tensor.Shape{}
		for i := 0; i < len(outSub); i++ {
			outShape = append(outShape, size[outSub[i]])
		}
		ref := tensor.New(tensor.F32, outShape)
		rd := ref.Storage().F32()
		os := tensor.RowMajorStrides(outShape)
		total := 1
		for _, ix := range order {
			total *= size[ix]
		}
		val := map[byte]int{}
		for combo := 0; combo < total; combo++ {
			rem := combo
			for _, ix := range order {
				val[ix] = rem % size[ix]
				rem /= size[ix]
			}
			prod := 1.0
			for k, sub := range inSubs {
				st := tensor.RowMajorStrides(ops[k].Shape())
				off := 0
				for pos := 0; pos < len(sub); pos++ {
					off += val[sub[pos]] * st[pos]
				}
				prod *= float64(ops[k].Storage().F32()[off])
			}
			of := 0
			for i := 0; i < len(outSub); i++ {
				of += val[outSub[i]] * os[i]
			}
			rd[of] = float32(float64(rd[of]) + prod)
		}
		gs := got[0].Storage().F32()
		for i := range gs {
			if gs[i] != rd[i] {
				t.Fatalf("%s: element %d fast=%v ref=%v (not bit-identical)", tc.spec, i, gs[i], rd[i])
			}
		}
	}
}
