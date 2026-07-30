package ref

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestMoEBalanceParallelBitExact locks the token-parallel MoE balance loss (GOMAXPROCS>1)
// byte-for-byte identical to serial (GOMAXPROCS=1): per-token softmax runs in parallel into
// disjoint probs rows, then P and the dispatch counts f fold serially in token order.
func TestMoEBalanceParallelBitExact(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, dims := range [][2]int{{200, 8}, {51, 33}, {1000, 64}} {
			tks, n := dims[0], dims[1]
			x := tensor.New(dt, tensor.Shape{tks, n})
			for i := 0; i < tks*n; i++ {
				co := tensor.Unravel(i, x.Shape())
				x.SetF64(math.Sin(float64(i)*0.017)*2, co...)
			}
			as := tensor.New(tensor.F64, tensor.Shape{tks})
			for tt := 0; tt < tks; tt++ {
				as.SetF64(float64((tt*7)%n), tt)
			}
			attrs := backend.MoEBalanceAttrs{Alpha: 0.01}
			exec := func() float64 {
				out, err := backend.Execute(backend.NewContext(), backend.OpMoEBalance, []*tensor.Tensor{x, as}, attrs)
				if err != nil {
					t.Fatal(err)
				}
				return out[0].AtF64()
			}
			prev := runtime.GOMAXPROCS(1)
			serial := exec()
			runtime.GOMAXPROCS(8)
			par := exec()
			runtime.GOMAXPROCS(prev)
			if serial != par {
				t.Fatalf("dt=%v tks=%d n=%d: serial %v != parallel %v (want byte-identical)", dt, tks, n, serial, par)
			}
		}
	}
}
