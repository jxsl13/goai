package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

func TestMHASelectCPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	f := func(shape tensor.Shape) *tensor.Tensor {
		tn := tensor.New(tensor.F64, shape)
		for i, s := 0, tn.Storage().F64(); i < len(s); i++ {
			s[i] = rng.NormFloat64()
		}
		return tn
	}
	for _, cfg := range []struct{ sq, sk, dk, heads, kvh int }{{4, 4, 8, 2, 2}, {5, 7, 16, 4, 2}, {64, 96, 32, 8, 8}} {
		dm := cfg.dk * cfg.heads
		kd := cfg.kvh * cfg.dk
		selT := tensor.New(tensor.F64, tensor.Shape{cfg.sq, cfg.sk})
		for i, s := 0, selT.Storage().F64(); i < len(s); i++ {
			if rng.Float64() < 0.5 {
				s[i] = 1
			}
		}
		in := []*tensor.Tensor{f(tensor.Shape{cfg.sq, dm}), f(tensor.Shape{cfg.sk, kd}), f(tensor.Shape{cfg.sq, dm}), f(tensor.Shape{cfg.sk, kd}), f(tensor.Shape{cfg.sk, kd}), selT}
		attr := backend.AttnAttrs{Heads: cfg.heads, KVHeads: cfg.kvh}
		gc, err := backend.Execute(backend.NewContext(), backend.OpMHASelect, in, attr)
		if err != nil {
			t.Fatal(err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpMHASelect, in, attr)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < gc[0].Numel(); i++ {
			c := tensor.Unravel(i, gc[0].Shape())
			if math.Float64bits(gc[0].AtF64(c...)) != math.Float64bits(gr[0].AtF64(c...)) {
				t.Fatalf("cfg=%+v idx=%d cpu=%v ref=%v", cfg, i, gc[0].AtF64(c...), gr[0].AtF64(c...))
			}
		}
	}
}

func benchMHASelect(b *testing.B, sq, sk, dk, heads int, ref bool) {
	rng := rand.New(rand.NewSource(1))
	dm := dk * heads
	f := func(shape tensor.Shape) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		for i, s := 0, t.Storage().F64(); i < len(s); i++ {
			s[i] = rng.NormFloat64()
		}
		return t
	}
	selT := tensor.New(tensor.F64, tensor.Shape{sq, sk})
	for i, s := 0, selT.Storage().F64(); i < len(s); i++ {
		if rng.Float64() < 0.5 {
			s[i] = 1
		}
	}
	in := []*tensor.Tensor{f(tensor.Shape{sq, dm}), f(tensor.Shape{sk, dm}), f(tensor.Shape{sq, dm}), f(tensor.Shape{sk, dm}), f(tensor.Shape{sk, dm}), selT}
	attr := backend.AttnAttrs{Heads: heads}
	ctx := backend.NewContext()
	if ref {
		ctx = ctx.WithBackend(backend.Reference())
	}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMHASelect, in, attr); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMHASelect_cpu_512(b *testing.B) { benchMHASelect(b, 512, 512, 64, 8, false) }
func BenchmarkMHASelect_ref_512(b *testing.B) { benchMHASelect(b, 512, 512, 64, 8, true) }
