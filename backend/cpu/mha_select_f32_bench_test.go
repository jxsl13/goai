package cpu_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

func benchMHASelectF32On(b *testing.B, name backend.Name, sq, sk, dk, heads int) {
	be, _ := backend.Get(name)
	rng := rand.New(rand.NewSource(1))
	dm := dk * heads
	f := func(shape tensor.Shape) *tensor.Tensor {
		tn := tensor.New(tensor.F32, shape)
		s := tn.Storage().F32()
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return tn
	}
	selT := tensor.New(tensor.F32, tensor.Shape{sq, sk})
	ss := selT.Storage().F32()
	for i := range ss {
		if rng.Float64() < 0.5 {
			ss[i] = 1
		}
	}
	in := []*tensor.Tensor{f(tensor.Shape{sq, dm}), f(tensor.Shape{sk, dm}), f(tensor.Shape{sq, dm}), f(tensor.Shape{sk, dm}), f(tensor.Shape{sk, dm}), selT}
	attr := backend.AttnAttrs{Heads: heads, KVHeads: heads}
	ctx := backend.NewContext().WithBackend(be)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMHASelect, in, attr); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMHASelectF32Ref_512x512x64x12(b *testing.B) {
	benchMHASelectF32On(b, backend.Ref, 512, 512, 64, 12)
}
func BenchmarkMHASelectF32CPU_512x512x64x12(b *testing.B) {
	benchMHASelectF32On(b, backend.CPU, 512, 512, 64, 12)
}
func BenchmarkMHASelectF32Ref_1024x1024x64x16(b *testing.B) {
	benchMHASelectF32On(b, backend.Ref, 1024, 1024, 64, 16)
}
func BenchmarkMHASelectF32CPU_1024x1024x64x16(b *testing.B) {
	benchMHASelectF32On(b, backend.CPU, 1024, 1024, 64, 16)
}
