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

func TestMoESparsePrefillMatchesDense(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		m := nn.NewSparseMoE(dt, 128, 256, 8, 2, 1)
		rng := rand.New(rand.NewPCG(3, 4))
		x := tensor.New(dt, tensor.Shape{96, 128})
		for i := 0; i < x.Numel(); i++ {
			x.SetF64(rng.NormFloat64()*0.3, tensor.Unravel(i, x.Shape())...)
		}
		sparse, _, err := m.Forward(backend.NewContext(), x) // Recorder==nil → sparse
		if err != nil {
			t.Fatal(err)
		}
		dense, _, err := m.Forward(autograd.NewTape().Context(), x) // Recorder!=nil → dense
		if err != nil {
			t.Fatal(err)
		}
		tol := 1e-12
		if dt == tensor.F32 {
			tol = 1e-5
		}
		var maxRel float64
		for i := 0; i < sparse.Numel(); i++ {
			idx := tensor.Unravel(i, sparse.Shape())
			rel := math.Abs(sparse.AtF64(idx...)-dense.AtF64(idx...)) / math.Max(1, math.Abs(dense.AtF64(idx...)))
			if rel > maxRel {
				maxRel = rel
			}
		}
		if maxRel > tol {
			t.Fatalf("%v: sparse vs dense maxRel %.3e > tol %.3e", dt, maxRel, tol)
		}
		t.Logf("%v: maxRel %.3e", dt, maxRel)
	}
}

func benchMoEFwd(b *testing.B, tks, dim, hidden, e, k int) {
	m := nn.NewSparseMoE(tensor.F64, dim, hidden, e, k, 1)
	rng := rand.New(rand.NewPCG(1, 2))
	x := tensor.New(tensor.F64, tensor.Shape{tks, dim})
	for i := 0; i < x.Numel(); i++ {
		x.SetF64(rng.NormFloat64()*0.1, tensor.Unravel(i, x.Shape())...)
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, err := m.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoEPrefill_512x512_e8k2(b *testing.B) { benchMoEFwd(b, 512, 512, 1024, 8, 2) }
