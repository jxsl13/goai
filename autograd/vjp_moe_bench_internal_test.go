package autograd

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchMoECombineBackward(b *testing.B, dt tensor.Dtype) {
	vjp := vjps[backend.OpMoECombine]
	const tks, e, d = 4096, 8, 256
	rng := rand.New(rand.NewPCG(3, 0x9e3779b9))
	w := tensor.New(dt, tensor.Shape{tks, e})
	experts := make([]*tensor.Tensor, e)
	for i := range experts {
		experts[i] = tensor.New(dt, tensor.Shape{tks, d})
	}
	g := tensor.New(dt, tensor.Shape{tks, d})
	switch dt {
	case tensor.F64:
		for i, s := 0, w.Storage().F64(); i < len(s); i++ {
			s[i] = rng.Float64()
		}
		for i := range experts {
			for k, s := 0, experts[i].Storage().F64(); k < len(s); k++ {
				s[k] = rng.NormFloat64()
			}
		}
		for i, s := 0, g.Storage().F64(); i < len(s); i++ {
			s[i] = rng.NormFloat64()
		}
	case tensor.F32:
		for i, s := 0, w.Storage().F32(); i < len(s); i++ {
			s[i] = float32(rng.Float64())
		}
		for i := range experts {
			for k, s := 0, experts[i].Storage().F32(); k < len(s); k++ {
				s[k] = float32(rng.NormFloat64())
			}
		}
		for i, s := 0, g.Storage().F32(); i < len(s); i++ {
			s[i] = float32(rng.NormFloat64())
		}
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

func BenchmarkMoECombineBackward(b *testing.B)    { benchMoECombineBackward(b, tensor.F64) }
func BenchmarkMoECombineBackwardF32(b *testing.B) { benchMoECombineBackward(b, tensor.F32) }
