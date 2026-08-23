package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func TestPreNormFFNReferenceMatchesComposite(t *testing.T) {
	be := backend.Reference()
	x := bench.RandF64(tensor.Shape{5, 7}, 1)
	gamma := bench.RandF64(tensor.Shape{7}, 2)
	beta := bench.RandF64(tensor.Shape{7}, 3)
	w1 := bench.RandF64(tensor.Shape{7, 11}, 4)
	b1 := bench.RandF64(tensor.Shape{11}, 5)
	w2 := bench.RandF64(tensor.Shape{11, 7}, 6)
	b2 := bench.RandF64(tensor.Shape{7}, 7)
	dOut := bench.RandF64(tensor.Shape{5, 7}, 8)

	controlTape := autograd.NewTapeOn(be)
	ctx := controlTape.Context()
	h, err := (&nn.LayerNorm{Gamma: gamma, Beta: beta, Eps: 1e-5}).Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	if h, err = (&nn.Linear{W: w1, B: b1}).Forward(ctx, h); err != nil {
		t.Fatal(err)
	}
	gelu, err := backend.Execute(ctx, backend.OpGELU, []*tensor.Tensor{h}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if h, err = (&nn.Linear{W: w2, B: b2}).Forward(ctx, gelu[0]); err != nil {
		t.Fatal(err)
	}
	control, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{x, h}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := controlTape.BackwardGrad(control[0], dOut); err != nil {
		t.Fatal(err)
	}

	fusedTape := autograd.NewTapeOn(be)
	fused, err := backend.Execute(fusedTape.Context(), backend.OpPreNormFFN,
		[]*tensor.Tensor{x, gamma, beta, w1, b1, w2, b2}, backend.NormAttrs{Eps: 1e-5})
	if err != nil {
		t.Fatal(err)
	}
	if err := fusedTape.BackwardGrad(fused[0], dOut); err != nil {
		t.Fatal(err)
	}
	check := func(name string, got, want *tensor.Tensor, tol float64) {
		t.Helper()
		for i := range got.Numel() {
			if d := math.Abs(got.Storage().F64()[i] - want.Storage().F64()[i]); d > tol {
				t.Fatalf("%s[%d] diff=%g > %g", name, i, d, tol)
			}
		}
	}
	check("Y", fused[0], control[0], 1e-12)
	for _, pair := range []struct {
		name string
		in   *tensor.Tensor
	}{
		{"dX", x}, {"dGamma", gamma}, {"dBeta", beta}, {"dW1", w1},
		{"dB1", b1}, {"dW2", w2}, {"dB2", b2},
	} {
		check(pair.name, fusedTape.Grad(pair.in), controlTape.Grad(pair.in), 1e-10)
	}
}
