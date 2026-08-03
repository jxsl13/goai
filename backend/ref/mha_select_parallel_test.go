package ref

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestMHASelectParallelBitExact locks the head-parallel selective-attention forward
// (GOMAXPROCS>1) byte-for-byte identical to serial (GOMAXPROCS=1): each head writes disjoint
// output columns. Covers F32/F64, GQA, rectangular sq≠sk, and the sel branch (0 vs nonzero).
func TestMHASelectParallelBitExact(t *testing.T) {
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
		dt                    tensor.Dtype
		sq, sk, dm, heads, kv int
	}{
		{tensor.F32, 32, 32, 64, 8, 8},
		{tensor.F64, 32, 32, 64, 8, 8},
		{tensor.F32, 24, 16, 48, 4, 4},
		{tensor.F32, 32, 32, 64, 8, 2}, // GQA
	}
	for _, c := range cases {
		kvdm := c.kv * (c.dm / c.heads)
		q1 := mk(c.dt, c.sq, c.dm, 0.1)
		k1 := mk(c.dt, c.sk, kvdm, 0.2)
		q2 := mk(c.dt, c.sq, c.dm, 0.3)
		k2 := mk(c.dt, c.sk, kvdm, 0.4)
		v := mk(c.dt, c.sk, kvdm, 0.5)
		sel := tensor.New(c.dt, tensor.Shape{c.sq, c.sk})
		for i := 0; i < c.sq; i++ {
			for j := 0; j < c.sk; j++ {
				val := 0.0
				if (i+j)%3 == 0 {
					val = 1.0 // exercise the q2/k2 branch
				}
				if j > i+2 {
					val = math.Inf(-1) // exercise masking
				}
				sel.SetF64(val, i, j)
			}
		}
		in := []*tensor.Tensor{q1, k1, q2, k2, v, sel}
		attrs := backend.AttnAttrs{Heads: c.heads, KVHeads: c.kv}
		exec := func() *tensor.Tensor {
			out, err := backend.Execute(backend.NewContext(), backend.OpMHASelect, in, attrs)
			if err != nil {
				t.Fatalf("%+v: %v", c, err)
			}
			return out[0]
		}
		prev := runtime.GOMAXPROCS(1)
		serial := exec()
		runtime.GOMAXPROCS(8)
		par := exec()
		runtime.GOMAXPROCS(prev)
		if c.dt == tensor.F64 {
			a, b := serial.Storage().F64(), par.Storage().F64()
			for x := range a {
				if a[x] != b[x] {
					t.Fatalf("%+v out[%d]: %v != %v", c, x, a[x], b[x])
				}
			}
		} else {
			a, b := serial.Storage().F32(), par.Storage().F32()
			for x := range a {
				if a[x] != b[x] {
					t.Fatalf("%+v out[%d]: %v != %v", c, x, a[x], b[x])
				}
			}
		}
	}
}
