package ref

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestArgMaxTrailingParallelBitExact locks the parallel trailing-axis argmax (GOMAXPROCS>1)
// byte-for-byte identical to serial (GOMAXPROCS=1), including the lowest-index tie rule — each
// segment scans its own contiguous run so parallelizing over segments changes nothing.
func TestArgMaxTrailingParallelBitExact(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, sh := range []tensor.Shape{{64, 200}, {40, 33}, {8, 5, 50}, {129, 128}} {
			x := tensor.New(dt, sh)
			for i := 0; i < x.Numel(); i++ {
				co := tensor.Unravel(i, sh)
				v := math.Sin(float64(i) * 0.3)
				// force frequent ties by quantizing → exercise the lowest-index rule
				x.SetF64(math.Round(v*4)/4, co...)
			}
			lastAx := len(sh) - 1
			run := func() *tensor.Tensor {
				out, err := backend.Execute(backend.NewContext(), backend.OpArgMax, []*tensor.Tensor{x}, backend.ArgMaxAttrs{Axis: lastAx})
				if err != nil {
					t.Fatal(err)
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
						t.Fatalf("dt=%v sh=%v out[%d]: serial %v != parallel %v", dt, sh, i, a[i], b[i])
					}
				}
			} else {
				a, b := serial.Storage().F32(), par.Storage().F32()
				for i := range a {
					if a[i] != b[i] {
						t.Fatalf("dt=%v sh=%v out[%d]: serial %v != parallel %v", dt, sh, i, a[i], b[i])
					}
				}
			}
		}
	}
}
