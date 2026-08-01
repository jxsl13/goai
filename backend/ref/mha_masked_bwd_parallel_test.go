package ref

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestMHAMaskedBackwardParallelBitExact locks the head-parallel F64 path (GOMAXPROCS>1)
// BYTE-FOR-BYTE identical to the serial path (GOMAXPROCS=1). The per-head dMask
// contributions are reduced in head order, so the shared-mask accumulation matches the
// serial h=0,1,2… order exactly; dQ/dK/dV are disjoint per head. Covers shared and
// per-head masks and rectangular sq≠sk.
func TestMHAMaskedBackwardParallelBitExact(t *testing.T) {
	fill := func(shape tensor.Shape, seed float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, shape)
		s := x.Storage().F64()
		for i := range s {
			s[i] = math.Sin(float64(i)*0.017 + seed)
		}
		return x
	}
	run := func(in []*tensor.Tensor, attrs backend.AttnAttrs) []*tensor.Tensor {
		out, err := backend.Execute(backend.NewContext(), backend.OpMHAMaskedBackward, in, attrs)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	cases := []struct {
		sq, sk, dm, heads int
		perHead           bool
	}{
		{16, 16, 32, 4, false},
		{16, 16, 32, 4, true},
		{24, 16, 48, 4, false}, // rectangular (cross-attention)
		{32, 32, 64, 8, false},
		{13, 13, 30, 3, true},
	}
	for _, c := range cases {
		q := fill(tensor.Shape{c.sq, c.dm}, 0.1)
		k := fill(tensor.Shape{c.sk, c.dm}, 0.2)
		v := fill(tensor.Shape{c.sk, c.dm}, 0.3)
		g := fill(tensor.Shape{c.sq, c.dm}, 0.4)
		var mask *tensor.Tensor
		if c.perHead {
			mask = tensor.New(tensor.F64, tensor.Shape{c.heads, c.sq, c.sk})
		} else {
			mask = tensor.New(tensor.F64, tensor.Shape{c.sq, c.sk})
		}
		ms := mask.Storage().F64()
		// causal-ish: mask out j>i within each (optional head) plane
		planes := 1
		if c.perHead {
			planes = c.heads
		}
		for p := 0; p < planes; p++ {
			for i := 0; i < c.sq; i++ {
				for j := 0; j < c.sk; j++ {
					if j > i {
						ms[(p*c.sq+i)*c.sk+j] = math.Inf(-1)
					}
				}
			}
		}
		in := []*tensor.Tensor{q, k, v, mask, g}
		attrs := backend.AttnAttrs{Heads: c.heads}

		prev := runtime.GOMAXPROCS(1)
		serial := run(in, attrs)
		runtime.GOMAXPROCS(8)
		par := run(in, attrs)
		runtime.GOMAXPROCS(prev)

		names := []string{"dQ", "dK", "dV", "dMask"}
		for oi := range serial {
			ss := serial[oi].Storage().F64()
			ps := par[oi].Storage().F64()
			if len(ss) != len(ps) {
				t.Fatalf("case %+v %s len mismatch", c, names[oi])
			}
			for idx := range ss {
				if ss[idx] != ps[idx] {
					t.Fatalf("case %+v %s[%d]: serial %v != parallel %v (want byte-identical)",
						c, names[oi], idx, ss[idx], ps[idx])
				}
			}
		}
	}
}
