package ref

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestMHAMaskedForwardParallelBitExact locks the head-parallel forward (GOMAXPROCS>1)
// byte-for-byte identical to serial (GOMAXPROCS=1). Each head writes disjoint output
// columns, so order is irrelevant. Covers F32/F64, shared and per-head masks, GQA, and
// rectangular sq≠sk.
func TestMHAMaskedForwardParallelBitExact(t *testing.T) {
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
		perHead               bool
	}{
		{tensor.F32, 32, 32, 64, 8, 8, false},
		{tensor.F64, 32, 32, 64, 8, 8, false},
		{tensor.F32, 24, 16, 48, 4, 4, false}, // rectangular
		{tensor.F32, 32, 32, 64, 8, 2, false}, // GQA rep=4
		{tensor.F64, 16, 16, 32, 4, 4, true},  // per-head mask
	}
	for _, c := range cases {
		q := mk(c.dt, c.sq, c.dm, 0.1)
		k := mk(c.dt, c.sk, c.kv*(c.dm/c.heads), 0.2)
		v := mk(c.dt, c.sk, c.kv*(c.dm/c.heads), 0.3)
		var mask *tensor.Tensor
		if c.perHead {
			mask = tensor.New(c.dt, tensor.Shape{c.heads, c.sq, c.sk})
		} else {
			mask = tensor.New(c.dt, tensor.Shape{c.sq, c.sk})
		}
		planes := 1
		if c.perHead {
			planes = c.heads
		}
		for p := 0; p < planes; p++ {
			for i := 0; i < c.sq; i++ {
				for j := 0; j < c.sk; j++ {
					if j > i {
						mask.SetF64(math.Inf(-1), sliceIdx(c.perHead, p, i, j)...)
					}
				}
			}
		}
		in := []*tensor.Tensor{q, k, v, mask}
		attrs := backend.AttnAttrs{Heads: c.heads, KVHeads: c.kv}
		exec := func() *tensor.Tensor {
			out, err := backend.Execute(backend.NewContext(), backend.OpMHAMasked, in, attrs)
			if err != nil {
				t.Fatalf("case %+v: %v", c, err)
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
					t.Fatalf("%+v out[%d]: serial %v != parallel %v", c, x, a[x], b[x])
				}
			}
		} else {
			a, b := serial.Storage().F32(), par.Storage().F32()
			for x := range a {
				if a[x] != b[x] {
					t.Fatalf("%+v out[%d]: serial %v != parallel %v", c, x, a[x], b[x])
				}
			}
		}
	}
}

func sliceIdx(perHead bool, p, i, j int) []int {
	if perHead {
		return []int{p, i, j}
	}
	return []int{i, j}
}
