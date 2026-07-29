package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

func mlaInputs(rng *rand.Rand, seq, heads, dh, dR int) []*tensor.Tensor {
	f := func(shape tensor.Shape) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		for i, s := 0, t.Storage().F64(); i < len(s); i++ {
			s[i] = rng.NormFloat64()
		}
		return t
	}
	hdh := heads * dh
	return []*tensor.Tensor{f(tensor.Shape{seq, hdh}), f(tensor.Shape{seq, hdh}), f(tensor.Shape{seq, hdh}), f(tensor.Shape{seq, heads * dR}), f(tensor.Shape{seq, dR})}
}

func TestMLACPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, cfg := range []struct{ seq, heads, dh, dR int }{{4, 2, 8, 4}, {7, 4, 16, 8}, {64, 8, 32, 16}} {
		for _, causal := range []bool{false, true} {
			in := mlaInputs(rng, cfg.seq, cfg.heads, cfg.dh, cfg.dR)
			attr := backend.MLAAttrs{Heads: cfg.heads, Causal: causal}
			gc, err := backend.Execute(backend.NewContext(), backend.OpMLA, in, attr)
			if err != nil {
				t.Fatal(err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpMLA, in, attr)
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < gc[0].Numel(); i++ {
				c := tensor.Unravel(i, gc[0].Shape())
				if math.Float64bits(gc[0].AtF64(c...)) != math.Float64bits(gr[0].AtF64(c...)) {
					t.Fatalf("cfg=%+v causal=%v idx=%d cpu=%v ref=%v", cfg, causal, i, gc[0].AtF64(c...), gr[0].AtF64(c...))
				}
			}
		}
	}
}

func benchMLA(b *testing.B, seq, heads, dh, dR int, ref bool) {
	in := mlaInputs(rand.New(rand.NewSource(1)), seq, heads, dh, dR)
	attr := backend.MLAAttrs{Heads: heads, Causal: true}
	ctx := backend.NewContext()
	if ref {
		ctx = ctx.WithBackend(backend.Reference())
	}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMLA, in, attr); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMLA_cpu_512(b *testing.B) { benchMLA(b, 512, 8, 64, 32, false) }
func BenchmarkMLA_ref_512(b *testing.B) { benchMLA(b, 512, 8, 64, 32, true) }
