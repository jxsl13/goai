package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSwiGLUForward drives the SwiGLU FFN forward on a plain (non-recording)
// context — the inference decode/prefill path where its five ops (3 matmul + 1 mul + 1
// SiLU) route through the recorder-guarded execPool helpers (T964). It measures the
// per-op input-slice allocation the pool removes.
func BenchmarkSwiGLUForward(b *testing.B) {
	const dim, hidden = 256, 1024
	s := nn.NewSwiGLU(tensor.F64, dim, hidden, 1)
	x := tensor.New(tensor.F64, tensor.Shape{1, dim})
	xd := x.Storage().F64()
	for i := range xd {
		xd[i] = 0.01 * float64(i%7)
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := s.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}
