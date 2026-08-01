package backend

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestEinsumContractParallelBitExact locks the output-parallel contraction (GOMAXPROCS>1)
// byte-for-byte identical to serial (GOMAXPROCS=1): each output position sums over the
// contracted indices independently, so parallelizing over outputs preserves the exact
// per-output accumulation order.
func TestEinsumContractParallelBitExact(t *testing.T) {
	mk := func(dt tensor.Dtype, sh tensor.Shape, seed float64) *tensor.Tensor {
		x := tensor.New(dt, sh)
		for i := 0; i < x.Numel(); i++ {
			co := tensor.Unravel(i, sh)
			x.SetF64(math.Sin(float64(i)*0.017+seed), co...)
		}
		return x
	}
	cases := []struct {
		spec   string
		shapes []tensor.Shape
	}{
		{"bic,ijc->bij", []tensor.Shape{{12, 10, 4}, {10, 8, 4}}},
		{"trh,trd->thd", []tensor.Shape{{20, 5, 3}, {20, 5, 6}}},
		{"ij,jk->ik", []tensor.Shape{{16, 24}, {24, 9}}},
		{"ijm,im->ij", []tensor.Shape{{7, 5, 6}, {7, 6}}},
		{"bhqk,bhkd->bhqd", []tensor.Shape{{2, 3, 5, 4}, {2, 3, 4, 6}}},
	}
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, c := range cases {
			inSubs, outSub, err := ParseEinsum(c.spec, len(c.shapes))
			if err != nil {
				t.Fatalf("%s: %v", c.spec, err)
			}
			ops := make([]*tensor.Tensor, len(c.shapes))
			for k, sh := range c.shapes {
				ops[k] = mk(dt, sh, float64(k)*0.3)
			}
			run := func() *tensor.Tensor {
				out, err := EinsumContract(inSubs, outSub, ops)
				if err != nil {
					t.Fatalf("%s: %v", c.spec, err)
				}
				return out
			}
			prev := runtime.GOMAXPROCS(1)
			serial := run()
			runtime.GOMAXPROCS(8)
			par := run()
			runtime.GOMAXPROCS(prev)
			if dt == tensor.F64 {
				a, b := serial.Storage().F64(), par.Storage().F64()
				for i := range a {
					if a[i] != b[i] {
						t.Fatalf("dt=%v spec=%s out[%d]: serial %v != parallel %v", dt, c.spec, i, a[i], b[i])
					}
				}
			} else {
				a, b := serial.Storage().F32(), par.Storage().F32()
				for i := range a {
					if a[i] != b[i] {
						t.Fatalf("dt=%v spec=%s out[%d]: serial %v != parallel %v", dt, c.spec, i, a[i], b[i])
					}
				}
			}
		}
	}
}
