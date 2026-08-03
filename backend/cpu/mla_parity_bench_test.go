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
	return mlaInputsDtype(rng, seq, heads, dh, dR, tensor.F64)
}

// mlaInputsDtype builds the same inputs in a chosen dtype. The ROPE halves stay F64 in both
// arms of the kernel, so they stay F64 here: making them f32 would test a combination the
// kernel never sees.
func mlaInputsDtype(rng *rand.Rand, seq, heads, dh, dR int, dt tensor.Dtype) []*tensor.Tensor {
	f := func(shape tensor.Shape, d tensor.Dtype) *tensor.Tensor {
		t := tensor.New(d, shape)
		n := 1
		for _, x := range shape {
			n *= x
		}
		for i := range n {
			t.SetF64(rng.NormFloat64(), tensor.Unravel(i, shape)...)
		}
		return t
	}
	hdh := heads * dh
	return []*tensor.Tensor{f(tensor.Shape{seq, hdh}, dt), f(tensor.Shape{seq, hdh}, dt),
		f(tensor.Shape{seq, hdh}, dt), f(tensor.Shape{seq, heads * dR}, tensor.F64),
		f(tensor.Shape{seq, dR}, tensor.F64)}
}

func TestMLACPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	// BOTH DTYPE ARMS. They were near-duplicates and now share one scoring body and one
	// weighted sum, which is exactly when testing one of them proves nothing about the other.
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, cfg := range []struct{ seq, heads, dh, dR int }{{4, 2, 8, 4}, {7, 4, 16, 8}, {64, 8, 32, 16}} {
			for _, causal := range []bool{false, true} {
				in := mlaInputsDtype(rng, cfg.seq, cfg.heads, cfg.dh, cfg.dR, dt)
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
						t.Fatalf("dt=%v cfg=%+v causal=%v idx=%d cpu=%v ref=%v",
							dt, cfg, causal, i, gc[0].AtF64(c...), gr[0].AtF64(c...))
					}
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
