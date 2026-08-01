package ref

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestWKVParallelBitExact locks the channel-parallel WKV recurrence (GOMAXPROCS>1)
// byte-for-byte identical to serial (GOMAXPROCS=1). Each channel has independent state and
// a disjoint output column, so order is irrelevant. Covers F32 and F64.
func TestWKVParallelBitExact(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, dims := range [][2]int{{40, 48}, {17, 13}, {64, 128}} {
			seq, d := dims[0], dims[1]
			mk := func(r, c int, s float64) *tensor.Tensor {
				x := tensor.New(dt, tensor.Shape{r, c})
				for i := 0; i < r; i++ {
					for j := 0; j < c; j++ {
						x.SetF64(math.Sin(float64(i*c+j)*0.017+s)*0.5, i, j)
					}
				}
				return x
			}
			vec := func(n int, s float64) *tensor.Tensor {
				x := tensor.New(dt, tensor.Shape{n})
				for i := 0; i < n; i++ {
					x.SetF64(0.3+0.1*math.Cos(float64(i)*0.03+s), i)
				}
				return x
			}
			in := []*tensor.Tensor{mk(seq, d, 0.1), mk(seq, d, 0.2), vec(d, 0.3), vec(d, 0.4)}
			exec := func() *tensor.Tensor {
				out, err := backend.Execute(backend.NewContext(), backend.OpWKV, in, nil)
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
						t.Fatalf("dt=%v seq=%d d=%d out[%d]: %v != %v", dt, seq, d, x, a[x], b[x])
					}
				}
			} else {
				a, b := serial.Storage().F32(), par.Storage().F32()
				for x := range a {
					if a[x] != b[x] {
						t.Fatalf("dt=%v seq=%d d=%d out[%d]: %v != %v", dt, seq, d, x, a[x], b[x])
					}
				}
			}
		}
	}
}
