package ref

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestZLossParallelBitExact locks the row-parallel z-loss (GOMAXPROCS>1) byte-for-byte
// identical to serial (GOMAXPROCS=1): each row's lse² is computed in parallel into contrib[i],
// then summed serially in row order (no reassociation).
func TestZLossParallelBitExact(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, dims := range [][2]int{{40, 128}, {7, 513}, {200, 64}} {
			b, c := dims[0], dims[1]
			x := tensor.New(dt, tensor.Shape{b, c})
			for i := 0; i < b*c; i++ {
				co := tensor.Unravel(i, x.Shape())
				x.SetF64(math.Sin(float64(i)*0.01)*3, co...)
			}
			attrs := backend.ZLossAttrs{Coeff: 1e-3}
			exec := func() float64 {
				out, err := backend.Execute(backend.NewContext(), backend.OpZLoss, []*tensor.Tensor{x}, attrs)
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
				t.Fatalf("dt=%v b=%d c=%d: serial %v != parallel %v (want byte-identical)", dt, b, c, serial, par)
			}
		}
	}
}
