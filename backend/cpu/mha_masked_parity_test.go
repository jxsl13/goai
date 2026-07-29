package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

func TestMHAMaskedCPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	f := func(shape tensor.Shape) *tensor.Tensor {
		tn := tensor.New(tensor.F64, shape)
		for i, s := 0, tn.Storage().F64(); i < len(s); i++ {
			s[i] = rng.NormFloat64()
		}
		return tn
	}
	for _, cfg := range []struct{ sq, sk, dk, heads, kvh int; perHead bool }{
		{4, 4, 8, 2, 2, false}, {5, 7, 16, 4, 4, false}, {5, 7, 16, 4, 2, false},
		{5, 7, 16, 4, 4, true}, {64, 96, 32, 8, 8, false},
	} {
		dm := cfg.dk * cfg.heads
		q, k, v := f(tensor.Shape{cfg.sq, dm}), f(tensor.Shape{cfg.sk, cfg.kvh * cfg.dk}), f(tensor.Shape{cfg.sk, cfg.kvh * cfg.dk})
		var mask *tensor.Tensor
		if cfg.perHead {
			mask = f(tensor.Shape{cfg.heads, cfg.sq, cfg.sk})
		} else {
			mask = f(tensor.Shape{cfg.sq, cfg.sk})
		}
		in := []*tensor.Tensor{q, k, v, mask}
		attr := backend.AttnAttrs{Heads: cfg.heads, KVHeads: cfg.kvh}
		gc, err := backend.Execute(backend.NewContext(), backend.OpMHAMasked, in, attr)
		if err != nil {
			t.Fatal(err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpMHAMasked, in, attr)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < cfg.sq*dm; i++ {
			c := tensor.Unravel(i, gc[0].Shape())
			if math.Float64bits(gc[0].AtF64(c...)) != math.Float64bits(gr[0].AtF64(c...)) {
				t.Fatalf("cfg=%+v idx=%d cpu=%v ref=%v", cfg, i, gc[0].AtF64(c...), gr[0].AtF64(c...))
			}
		}
	}
}
