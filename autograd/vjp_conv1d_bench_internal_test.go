package autograd

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func BenchmarkConv1DBackward(b *testing.B) {
	vjp := vjps[backend.OpConv1D]
	const L, D, K = 2048, 4096, 4 // Mamba d_inner scale
	rng := rand.New(rand.NewPCG(7, 0x9e3779b9))
	mk := func(shape tensor.Shape) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		for i, s := 0, t.Storage().F64(); i < len(s); i++ {
			s[i] = rng.NormFloat64()
		}
		return t
	}
	x := mk(tensor.Shape{L, D})
	w := mk(tensor.Shape{D, K})
	g := mk(tensor.Shape{L, D})
	in := []*tensor.Tensor{x, w}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := vjp(nil, in, nil, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}
