package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1: the power-iteration estimate σ ≈ uᵀWv converges to the TRUE largest
// singular value σ_max = s[0] of the SVD (cross-checked against ops.SVD).
func TestSpectralNormSigmaMatchesSVD(t *testing.T) {
	s := nn.NewSpectralNorm(tensor.F64, 5, 3, 42) // in≥out so ops.SVD applies to W
	_, sv, _, err := ops.SVD(s.W)
	if err != nil {
		t.Fatalf("SVD: %v", err)
	}
	want := sv.AtF64(0)
	if got := s.SigmaEst(); math.Abs(got-want) > 1e-6*want {
		t.Errorf("σ estimate = %.10g, want σ_max = %.10g", got, want)
	}
}

// §V16 tier-1: the normalized weight W_SN = W/σ has spectral norm 1 (its top singular
// value is 1) — the defining Lipschitz-1 property of the layer.
func TestSpectralNormUnitScale(t *testing.T) {
	s := nn.NewSpectralNorm(tensor.F64, 6, 4, 7)
	sigma := s.SigmaEst()
	wsn := tensor.New(tensor.F64, s.W.Shape())
	for i := range 6 {
		for j := range 4 {
			wsn.SetF64(s.W.AtF64(i, j)/sigma, i, j)
		}
	}
	_, sv, _, err := ops.SVD(wsn)
	if err != nil {
		t.Fatalf("SVD: %v", err)
	}
	if top := sv.AtF64(0); math.Abs(top-1) > 1e-6 {
		t.Errorf("σ_max(W_SN) = %.10g, want 1", top)
	}
}

// §V2: with the u,v buffers frozen (iters=0), the layer's gradient w.r.t. W — which
// autograd computes from the σ=uᵀWv, W/σ composition — matches central finite
// differences. This is the paper's Eq. 9 gradient, obtained without a hand-coded VJP.
func TestSpectralNormGradcheck(t *testing.T) {
	const in, out, batch = 3, 2, 2
	s := nn.NewSpectralNorm(tensor.F64, in, out, 3, nn.WithSpectralNormIters(0)) // freeze u,v after warm-up
	s.B = nil                                                                    // isolate the W gradient
	x := tensor.FromFloat64(tensor.Shape{batch, in}, []float64{1, -1, 0.5, 0.3, 1, -2})
	mask := tensor.FromFloat64(tensor.Shape{batch, out}, []float64{0.7, -1.2, 0.4, 0.9})

	lossOf := func() float64 {
		y, err := s.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var acc float64
		for i := range batch {
			for j := range out {
				acc += mask.AtF64(i, j) * y.AtF64(i, j)
			}
		}
		return acc
	}

	tape := autograd.NewTape()
	y, err := s.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	p, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{y, mask}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{p[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(s.W)
	if g == nil {
		t.Fatal("nil W gradient")
	}
	const h = 1e-6
	for i := range in {
		for j := range out {
			orig := s.W.AtF64(i, j)
			s.W.SetF64(orig+h, i, j)
			lp := lossOf()
			s.W.SetF64(orig-h, i, j)
			lm := lossOf()
			s.W.SetF64(orig, i, j)
			num := (lp - lm) / (2 * h)
			if ana := g.AtF64(i, j); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("W̄[%d,%d]: numeric %.8g vs analytic %.8g", i, j, num, ana)
			}
		}
	}
}

func ExampleSpectralNorm() {
	s := nn.NewSpectralNorm(tensor.F64, 4, 3, 1)
	// After spectral normalization the largest singular value of W/σ is 1.
	sigma := s.SigmaEst()
	wsn := tensor.New(tensor.F64, s.W.Shape())
	for i := range 4 {
		for j := range 3 {
			wsn.SetF64(s.W.AtF64(i, j)/sigma, i, j)
		}
	}
	_, sv, _, _ := ops.SVD(wsn)
	fmt.Printf("%.4f\n", sv.AtF64(0))
	// Output:
	// 1.0000
}
