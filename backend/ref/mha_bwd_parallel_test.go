package ref

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestMHABackwardParallelBitExact locks the head-parallel core (GOMAXPROCS>1)
// byte-for-byte identical to serial (GOMAXPROCS=1). With no mask and rep==1 every head
// writes disjoint dQ/dK/dV bands, so order is irrelevant; covers F64+F32, causal and
// non-causal, and the GQA (rep>1) serial fallback.
func TestMHABackwardParallelBitExact(t *testing.T) {
	mk := func(dt tensor.Dtype, r, c int, s float64) *tensor.Tensor {
		x := tensor.New(dt, tensor.Shape{r, c})
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				x.SetF64(math.Sin(float64(i*c+j)*0.013+s), i, j)
			}
		}
		return x
	}
	cases := []struct {
		dt              tensor.Dtype
		seq, dm, heads  int
		kvHeads         int
		causal          bool
	}{
		{tensor.F64, 32, 64, 8, 8, true},
		{tensor.F64, 24, 48, 4, 4, false},
		{tensor.F32, 32, 64, 8, 8, true},
		{tensor.F64, 32, 64, 8, 2, true}, // GQA rep=4 → serial fallback (still must match)
		{tensor.F32, 17, 30, 3, 3, false},
	}
	for _, c := range cases {
		q := mk(c.dt, c.seq, c.dm, 0.1)
		k := mk(c.dt, c.seq, c.dm, 0.2)
		v := mk(c.dt, c.seq, c.dm, 0.3)
		g := mk(c.dt, c.seq, c.dm, 0.4)
		in := []*tensor.Tensor{q, k, v, g}
		attrs := backend.AttnAttrs{Heads: c.heads, KVHeads: c.kvHeads, Causal: c.causal}
		exec := func() []*tensor.Tensor {
			out, err := backend.Execute(backend.NewContext(), backend.OpMHABackward, in, attrs)
			if err != nil {
				t.Fatal(err)
			}
			return out
		}
		prev := runtime.GOMAXPROCS(1)
		serial := exec()
		runtime.GOMAXPROCS(8)
		par := exec()
		runtime.GOMAXPROCS(prev)
		names := []string{"dQ", "dK", "dV"}
		for oi := range serial {
			ss, ps := serial[oi].Storage(), par[oi].Storage()
			if c.dt == tensor.F64 {
				a, b := ss.F64(), ps.F64()
				for x := range a {
					if a[x] != b[x] {
						t.Fatalf("%+v %s[%d]: %v != %v", c, names[oi], x, a[x], b[x])
					}
				}
			} else {
				a, b := ss.F32(), ps.F32()
				for x := range a {
					if a[x] != b[x] {
						t.Fatalf("%+v %s[%d]: %v != %v", c, names[oi], x, a[x], b[x])
					}
				}
			}
		}
	}
}
