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

// TestSoftpickFusedMatchesDispatch checks the Recorder==nil fused softmax-off-by-one path against the
// recorded OpConcat+OpSoftmax+OpSlice path within the model's softmax f64 tolerance.
func TestSoftpickFusedMatchesDispatch(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		m, err := nn.NewSoftpickAttention(dt, 256, 4, 1, nn.WithSoftpickCausal())
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewPCG(3, 4))
		x := tensor.New(dt, tensor.Shape{192, 256})
		for i := 0; i < x.Numel(); i++ {
			x.SetF64(rng.NormFloat64()*0.2, tensor.Unravel(i, x.Shape())...)
		}
		fused, err := m.Forward(backend.NewContext(), x) // Recorder==nil → fused
		if err != nil {
			t.Fatal(err)
		}
		disp, err := m.Forward(autograd.NewTape().Context(), x) // Recorder!=nil → op path
		if err != nil {
			t.Fatal(err)
		}
		tol := 1e-10
		if dt == tensor.F32 {
			tol = 1e-4
		}
		var maxRel float64
		for i := 0; i < fused.Numel(); i++ {
			idx := tensor.Unravel(i, fused.Shape())
			f, d := fused.AtF64(idx...), disp.AtF64(idx...)
			rel := math.Abs(f-d) / math.Max(1, math.Abs(d))
			if rel > maxRel {
				maxRel = rel
			}
		}
		if maxRel > tol {
			t.Fatalf("%v: fused vs dispatch maxRel %.3e > tol %.3e", dt, maxRel, tol)
		}
		t.Logf("%v: fused matches dispatch, maxRel %.3e", dt, maxRel)
	}
}

func benchSoftpickFwd(b *testing.B, seq, dim, heads int) {
	m, _ := nn.NewSoftpickAttention(tensor.F64, dim, heads, 1, nn.WithSoftpickCausal())
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

func BenchmarkSoftpickFwd_512x512_h8(b *testing.B) { benchSoftpickFwd(b, 512, 512, 8) }
