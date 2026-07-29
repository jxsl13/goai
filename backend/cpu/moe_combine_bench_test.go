package cpu_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

func moeCombineInputs(rng *rand.Rand, tks, d, e int) []*tensor.Tensor {
	f := func(shape tensor.Shape) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		for i, s := 0, t.Storage().F64(); i < len(s); i++ {
			s[i] = rng.NormFloat64()
		}
		return t
	}
	in := []*tensor.Tensor{f(tensor.Shape{tks, e})}
	for i := 0; i < e; i++ {
		in = append(in, f(tensor.Shape{tks, d}))
	}
	return in
}

func TestMoECombineCPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, cfg := range [][3]int{{1, 3, 2}, {5, 7, 4}, {256, 128, 8}} {
		in := moeCombineInputs(rng, cfg[0], cfg[1], cfg[2])
		gc, _ := backend.Execute(backend.NewContext(), backend.OpMoECombine, in, nil)
		gr, _ := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpMoECombine, in, nil)
		for i := 0; i < gc[0].Numel(); i++ {
			c := tensor.Unravel(i, gc[0].Shape())
			if math.Float64bits(gc[0].AtF64(c...)) != math.Float64bits(gr[0].AtF64(c...)) {
				t.Fatalf("cfg=%v idx=%d cpu=%v ref=%v", cfg, i, gc[0].AtF64(c...), gr[0].AtF64(c...))
			}
		}
	}
}

func benchMoECombine(b *testing.B, tks, d, e int, ref bool) {
	in := moeCombineInputs(rand.New(rand.NewSource(1)), tks, d, e)
	ctx := backend.NewContext()
	if ref {
		ctx = ctx.WithBackend(backend.Reference())
	}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMoECombine, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoECombine_cpu(b *testing.B) { benchMoECombine(b, 2048, 2048, 8, false) }
func BenchmarkMoECombine_ref(b *testing.B) { benchMoECombine(b, 2048, 2048, 8, true) }
