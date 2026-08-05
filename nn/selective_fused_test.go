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

func TestSelectiveFusedMatchesDispatch(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		m, err := nn.NewSelectiveAttention(dt, 256, 4, 1)
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewPCG(3, 4))
		x := tensor.New(dt, tensor.Shape{160, 256})
		for i := 0; i < x.Numel(); i++ {
			x.SetF64(rng.NormFloat64()*0.3, tensor.Unravel(i, x.Shape())...)
		}
		fused, _ := m.Forward(backend.NewContext(), x)
		disp, _ := m.Forward(autograd.NewTape().Context(), x)
		tol := 1e-9
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
			t.Fatalf("%v: maxRel %.3e > tol %.3e", dt, maxRel, tol)
		}
		t.Logf("%v: maxRel %.3e", dt, maxRel)
	}
}

func BenchmarkSelectiveFwd_512x512_h8(b *testing.B) {
	m, _ := nn.NewSelectiveAttention(tensor.F64, 512, 8, 1)
	rng := rand.New(rand.NewPCG(1, 2))
	x := tensor.New(tensor.F64, tensor.Shape{512, 512})
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
