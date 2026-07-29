package autograd

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func BenchmarkMoECombineBackward(b *testing.B) {
	vjp := vjps[backend.OpMoECombine]
	const tks, e, d = 4096, 8, 256
	rng := rand.New(rand.NewPCG(3, 0x9e3779b9))
	w := tensor.New(tensor.F64, tensor.Shape{tks, e})
	for i, s := 0, w.Storage().F64(); i < len(s); i++ {
		s[i] = rng.Float64()
	}
	experts := make([]*tensor.Tensor, e)
	for i := range experts {
		experts[i] = tensor.New(tensor.F64, tensor.Shape{tks, d})
		for k, s := 0, experts[i].Storage().F64(); k < len(s); k++ {
			s[k] = rng.NormFloat64()
		}
	}
	g := tensor.New(tensor.F64, tensor.Shape{tks, d})
	for i, s := 0, g.Storage().F64(); i < len(s); i++ {
		s[i] = rng.NormFloat64()
	}
	in := append([]*tensor.Tensor{w}, experts...)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := vjp(nil, in, nil, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}
