package nn_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func TestSigmoidAttnFusedMatchesDispatch(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, causal := range []bool{false, true} {
			opts := []nn.SigmoidAttentionOption{nn.WithSigmoidAttnBias(-1.5)}
			if causal {
				opts = append(opts, nn.WithSigmoidAttnCausal())
			}
			m, err := nn.NewSigmoidAttention(dt, 256, 4, 1, opts...)
			if err != nil {
				t.Fatal(err)
			}
			rng := rand.New(rand.NewPCG(3, 4))
			x := tensor.New(dt, tensor.Shape{192, 256})
			for i := 0; i < x.Numel(); i++ {
				x.SetF64(rng.NormFloat64()*0.2, tensor.Unravel(i, x.Shape())...)
			}
			fused, _ := m.Forward(backend.NewContext(), x)
			disp, _ := m.Forward(autograd.NewTape().Context(), x)
			tol := 1e-10
			if dt == tensor.F32 {
				tol = 1e-4
			}
			var maxRel float64
			for i := 0; i < fused.Numel(); i++ {
				idx := tensor.Unravel(i, fused.Shape())
				rel := math.Abs(fused.AtF64(idx...)-disp.AtF64(idx...)) / math.Max(1, math.Abs(disp.AtF64(idx...)))
				if rel > maxRel {
					maxRel = rel
				}
			}
			if maxRel > tol {
				t.Fatalf("%v causal=%v: maxRel %.3e > tol %.3e", dt, causal, maxRel, tol)
			}
			t.Logf("%v causal=%v: maxRel %.3e", dt, causal, maxRel)
		}
	}
}

func benchSigmoidFwd(b *testing.B, seq, dim, heads int) {
	m, _ := nn.NewSigmoidAttention(tensor.F64, dim, heads, 1, nn.WithSigmoidAttnCausal())
	rng := rand.New(rand.NewPCG(1, 2))
	x := tensor.New(tensor.F64, tensor.Shape{seq, dim})
	for i := 0; i < x.Numel(); i++ {
		x.SetF64(rng.NormFloat64()*0.1, tensor.Unravel(i, x.Shape())...)
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := m.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSigmoidFwd_512x512_h8(b *testing.B) { benchSigmoidFwd(b, 512, 512, 8) }
