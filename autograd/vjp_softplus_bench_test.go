package autograd

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu" // register the CPU backend as default so the dispatch hits vsoftplusGradF64
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSoftplusVJP_F64 times the OpSoftplus backward (Mamba/Jamba Δ-projection) on a
// Mamba-shaped [1024,2048] F64 tensor — the vectorized OpSoftplusBackward dispatch (4-wide
// AVX2 exp) versus the old per-element scalar math.Exp loop.
func BenchmarkSoftplusVJP_F64(b *testing.B) {
	vjp := vjps[backend.OpSoftplus]
	if vjp == nil {
		b.Skip("no softplus vjp")
	}
	mk := func() *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{1024, 2048})
		s := t.Storage().F64()
		for i := range s {
			s[i] = float64((i%37)-18) * 0.1
		}
		return t
	}
	x, y, g := mk(), mk(), mk()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{y}, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}
