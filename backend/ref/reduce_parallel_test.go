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

// TestReduceLeadingParallelBitExact locks the leading-prefix (outermost reduced axes) parallel
// path byte-for-byte identical to serial, across ops/dtypes. Reduces axis 0 (and {0,1}).
func TestReduceLeadingParallelBitExact(t *testing.T) {
	ops := []backend.Op{backend.OpSum, backend.OpMax, backend.OpMin, backend.OpMean, backend.OpProd}
	type tc struct {
		sh   tensor.Shape
		axes []int
	}
	cases := []tc{
		{tensor.Shape{200, 64}, []int{0}},
		{tensor.Shape{33, 40}, []int{0}},
		{tensor.Shape{8, 5, 50}, []int{0, 1}}, // leading prefix {0,1}, kept {2}
	}
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, c := range cases {
			x := tensor.New(dt, c.sh)
			for i := 0; i < x.Numel(); i++ {
				co := tensor.Unravel(i, c.sh)
				x.SetF64(0.5+0.3*math.Sin(float64(i)*0.03), co...)
			}
			for _, op := range ops {
				run := func() *tensor.Tensor {
					out, err := backend.Execute(backend.NewContext(), op, []*tensor.Tensor{x},
						backend.ReduceAttrs{Axes: c.axes, KeepDims: true})
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
							t.Fatalf("op=%v dt=%v sh=%v axes=%v out[%d]: %v != %v", op, dt, c.sh, c.axes, i, a[i], b[i])
						}
					}
				} else {
					a, b := serial.Storage().F32(), par.Storage().F32()
					for i := range a {
						if a[i] != b[i] {
							t.Fatalf("op=%v dt=%v sh=%v axes=%v out[%d]: %v != %v", op, dt, c.sh, c.axes, i, a[i], b[i])
						}
					}
				}
			}
		}
	}
}
