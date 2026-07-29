package cpu_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

func benchMHAMasked(b *testing.B, sq, sk, dk, heads int, ref bool) {
	rng := rand.New(rand.NewSource(1))
	dm := dk * heads
	f := func(shape tensor.Shape) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		for i, s := 0, t.Storage().F64(); i < len(s); i++ {
			s[i] = rng.NormFloat64()
		}
		return t
	}
	q, k, v := f(tensor.Shape{sq, dm}), f(tensor.Shape{sk, dm}), f(tensor.Shape{sk, dm})
	mask := tensor.New(tensor.F64, tensor.Shape{sq, sk}) // zeros
	in := []*tensor.Tensor{q, k, v, mask}
	ctx := backend.NewContext()
	if ref {
		ctx = ctx.WithBackend(backend.Reference())
	}
	attr := backend.AttnAttrs{Heads: heads}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMHAMasked, in, attr); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMHAMasked_cpu_512(b *testing.B) { benchMHAMasked(b, 512, 512, 64, 8, false) }
func BenchmarkMHAMasked_ref_512(b *testing.B) { benchMHAMasked(b, 512, 512, 64, 8, true) }
