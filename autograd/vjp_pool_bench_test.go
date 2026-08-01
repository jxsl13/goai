package autograd

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// benchPoolVJP times a pooling backward at a shape a convolutional training step actually produces:
// a batch of feature maps, 2x2 pooling at stride 2.
//
// The forward output is built here rather than by pooling, because the rule reads only y, g and x.
// For the MAX rule that matters more than it looks: y must actually BE the window maximum, or the
// equality search routes to the fallback on every window and the benchmark measures the NaN path
// instead of the real one. So y is computed by an explicit max over each window.
func benchPoolVJP(b *testing.B, op backend.Op, dt tensor.Dtype, n, c, hw, k, s int) {
	vjp := vjps[op]
	if vjp == nil {
		b.Fatalf("no VJP registered for %v", op)
	}
	ho := (hw-k)/s + 1
	x := tensor.New(dt, tensor.Shape{n, c, hw, hw})
	for i := range x.Numel() {
		v := math.Sin(float64(i)*0.037) * 3
		x.SetF64(v, i/(c*hw*hw), (i/(hw*hw))%c, (i/hw)%hw, i%hw)
	}
	y := tensor.New(dt, tensor.Shape{n, c, ho, ho})
	g := tensor.New(dt, tensor.Shape{n, c, ho, ho})
	for pl := range n * c {
		for oy := range ho {
			for ox := range ho {
				m := math.Inf(-1)
				var sum float64
				for ky := range k {
					for kx := range k {
						v := x.AtF64(pl/c, pl%c, oy*s+ky, ox*s+kx)
						sum += v
						if v > m {
							m = v
						}
					}
				}
				if op == backend.OpAvgPool2D {
					m = sum / float64(k*k)
				}
				y.SetF64(m, pl/c, pl%c, oy, ox)
				g.SetF64(math.Cos(float64(pl*7+oy*3+ox))*0.5, pl/c, pl%c, oy, ox)
			}
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{y}, backend.PoolAttrs{Kernel: k, Stride: s}, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMaxPool2DBackwardF32(b *testing.B) {
	benchPoolVJP(b, backend.OpMaxPool2D, tensor.F32, 8, 32, 64, 2, 2)
}
func BenchmarkMaxPool2DBackwardF64(b *testing.B) {
	benchPoolVJP(b, backend.OpMaxPool2D, tensor.F64, 8, 32, 64, 2, 2)
}
func BenchmarkAvgPool2DBackwardF32(b *testing.B) {
	benchPoolVJP(b, backend.OpAvgPool2D, tensor.F32, 8, 32, 64, 2, 2)
}
func BenchmarkAvgPool2DBackwardF64(b *testing.B) {
	benchPoolVJP(b, backend.OpAvgPool2D, tensor.F64, 8, 32, 64, 2, 2)
}
