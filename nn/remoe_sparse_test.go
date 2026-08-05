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

func TestReMoESparseMatchesDense(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		m := nn.NewReMoE(dt, 128, 256, 8, 2.0, 1)
		rng := rand.New(rand.NewPCG(3, 4))
		x := tensor.New(dt, tensor.Shape{96, 128})
		for i := 0; i < x.Numel(); i++ {
			x.SetF64(rng.NormFloat64()*0.3, tensor.Unravel(i, x.Shape())...)
		}
		sp, _, _, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		dn, _, _, err := m.Forward(autograd.NewTape().Context(), x)
		if err != nil {
			t.Fatal(err)
		}
		tol := 1e-12
		if dt == tensor.F32 {
			tol = 1e-5
		}
		var maxRel float64
		for i := 0; i < sp.Numel(); i++ {
			idx := tensor.Unravel(i, sp.Shape())
			rel := math.Abs(sp.AtF64(idx...)-dn.AtF64(idx...)) / math.Max(1, math.Abs(dn.AtF64(idx...)))
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

func BenchmarkReMoEPrefill_512x512_e8(b *testing.B) {
	m := nn.NewReMoE(tensor.F64, 512, 1024, 8, 2.0, 1)
	rng := rand.New(rand.NewPCG(1, 2))
	x := tensor.New(tensor.F64, tensor.Shape{512, 512})
	for i := 0; i < x.Numel(); i++ {
		x.SetF64(rng.NormFloat64()*0.1, tensor.Unravel(i, x.Shape())...)
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, _, err := m.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}
