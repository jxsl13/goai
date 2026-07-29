package autograd

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkTransposeVJP covers the transpose adjoint gᵀ over a realistic activation
// matrix (e.g. [seq, model] = [512, 768] gradient), F64.
func BenchmarkTransposeVJP(b *testing.B) {
	const m, n = 768, 512 // x is [m,n]; g (grad of xᵀ) is [n,m]
	x := tensor.New(tensor.F64, tensor.Shape{m, n})
	g := tensor.New(tensor.F64, tensor.Shape{n, m})
	gs := g.Storage().F64()
	for i := range gs {
		gs[i] = float64(i%97) * 0.5
	}
	ctx := backend.NewContext()
	fn := vjps[backend.OpTranspose]
	in := []*tensor.Tensor{x}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := fn(ctx, in, nil, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}
