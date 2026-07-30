package ref

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestReduceTrailingParallelBitExact locks the parallel trailing-axis reduction (GOMAXPROCS>1)
// byte-for-byte identical to serial (GOMAXPROCS=1): each output segment reduces its own
// contiguous run, so parallelizing over segments preserves the exact per-segment combine order.
func TestReduceTrailingParallelBitExact(t *testing.T) {
	ops := []backend.Op{backend.OpSum, backend.OpMax, backend.OpMin, backend.OpMean, backend.OpProd}
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, sh := range []tensor.Shape{{64, 200}, {40, 33}, {8, 5, 50}} {
			x := tensor.New(dt, sh)
			for i := 0; i < x.Numel(); i++ {
				co := tensor.Unravel(i, sh)
				x.SetF64(0.5+0.3*math.Sin(float64(i)*0.03), co...) // positive for Prod stability
			}
			axis := sh.Numel() // placeholder
			_ = axis
			lastAx := len(sh) - 1
			for _, op := range ops {
				run := func() *tensor.Tensor {
					out, err := backend.Execute(backend.NewContext(), op, []*tensor.Tensor{x},
						backend.ReduceAttrs{Axes: []int{lastAx}, KeepDims: true})
					if err != nil {
						t.Fatalf("%v: %v", op, err)
					}
					return out[0]
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
							t.Fatalf("op=%v dt=%v sh=%v out[%d]: %v != %v", op, dt, sh, i, a[i], b[i])
						}
					}
				} else {
					a, b := serial.Storage().F32(), par.Storage().F32()
					for i := range a {
						if a[i] != b[i] {
							t.Fatalf("op=%v dt=%v sh=%v out[%d]: %v != %v", op, dt, sh, i, a[i], b[i])
						}
					}
				}
			}
		}
	}
}
