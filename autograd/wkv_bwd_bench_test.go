package autograd

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func benchWKVBackward(b *testing.B, seq, d int) {
	rng := rand.New(rand.NewPCG(1, 2))
	mk := func(shape ...int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape(shape))
		s := t.Storage().F64()
		for i := range s {
			s[i] = rng.NormFloat64() * 0.1
		}
		return t
	}
	k, v := mk(seq, d), mk(seq, d)
	w, u := mk(d), mk(d)
	g := mk(seq, d)
	vjp, ok := vjps[backend.OpWKV]
	if !ok {
		b.Fatal("no WKV VJP registered")
	}
	in := []*tensor.Tensor{k, v, w, u}
	b.ResetTimer()
	for range b.N {
		if _, err := vjp(nil, in, nil, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWKVBackward_256x1024(b *testing.B) { benchWKVBackward(b, 256, 1024) }
func BenchmarkWKVBackward_512x2048(b *testing.B) { benchWKVBackward(b, 512, 2048) }
