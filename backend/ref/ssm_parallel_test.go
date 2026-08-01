package ref

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestSSMParallelBitExact locks the channel-parallel selective scan (GOMAXPROCS>1)
// byte-for-byte identical to serial (GOMAXPROCS=1). The loop interchange (channel-outer)
// and per-worker state reproduce each channel's time recurrence in the same order; channels
// write disjoint output columns. Covers F32/F64, with and without the D-skip input.
func TestSSMParallelBitExact(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, withSkip := range []bool{false, true} {
			for _, dims := range [][3]int{{40, 48, 16}, {17, 13, 8}, {64, 96, 16}} {
				L, D, N := dims[0], dims[1], dims[2]
				mk := func(shape tensor.Shape, s float64) *tensor.Tensor {
					x := tensor.New(dt, shape)
					for i := 0; i < x.Numel(); i++ {
						co := tensor.Unravel(i, shape)
						x.SetF64(math.Sin(float64(i)*0.017+s)*0.3, co...)
					}
					return x
				}
				in := []*tensor.Tensor{
					mk(tensor.Shape{L, D}, 0.1), mk(tensor.Shape{L, D}, 0.2),
					mk(tensor.Shape{D, N}, 0.3), mk(tensor.Shape{L, N}, 0.4), mk(tensor.Shape{L, N}, 0.5),
				}
				if withSkip {
					in = append(in, mk(tensor.Shape{D}, 0.6))
				}
				exec := func() *tensor.Tensor {
					out, err := backend.Execute(backend.NewContext(), backend.OpSSM, in, nil)
					if err != nil {
						t.Fatal(err)
					}
					return out[0]
				}
				prev := runtime.GOMAXPROCS(1)
				serial := exec()
				runtime.GOMAXPROCS(8)
				par := exec()
				runtime.GOMAXPROCS(prev)
				if dt == tensor.F64 {
					a, b := serial.Storage().F64(), par.Storage().F64()
					for x := range a {
						if a[x] != b[x] {
							t.Fatalf("dt=%v skip=%v L=%d D=%d N=%d out[%d]: %v != %v", dt, withSkip, L, D, N, x, a[x], b[x])
						}
					}
				} else {
					a, b := serial.Storage().F32(), par.Storage().F32()
					for x := range a {
						if a[x] != b[x] {
							t.Fatalf("dt=%v skip=%v L=%d D=%d N=%d out[%d]: %v != %v", dt, withSkip, L, D, N, x, a[x], b[x])
						}
					}
				}
			}
		}
	}
}
