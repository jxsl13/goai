package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func approxEqT(t *testing.T, got, want *tensor.Tensor, tol float64, msg string) {
	t.Helper()
	for i := range got.Numel() {
		c := tensor.Unravel(i, got.Shape())
		if math.Abs(got.AtF64(c...)-want.AtF64(c...)) > tol {
			t.Errorf("%s: at %v %.9g != %.9g", msg, c, got.AtF64(c...), want.AtF64(c...))
		}
	}
}

// §V16 tier-1 (arXiv:2404.02948): W_res + A·B = W, so a fresh PiSSA adapter reproduces x·W exactly
// at initialization — identity like LoRA, but seeded from the principal components.
func TestPiSSAIdentityAtInit(t *testing.T) {
	w := toT([][]float64{{2, -1, 0.5}, {1, 3, -2}, {0.5, 0.5, 4}, {-1, 2, 1}}) // [4,3], in≥out
	p, err := nn.NewPiSSA(w, 2)
	if err != nil {
		t.Fatal(err)
	}
	x := toT([][]float64{{1, -2, 0.5, 3}, {0, 1, -1, 2}})
	got, err := p.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	approxEqT(t, got, matmul(t, x, w), 1e-9, "PiSSA identity at init")
}

// §V16 tier-1 (in<out transpose branch): the tall/square SVD requirement is handled by
// decomposing Wᵀ, so a wide weight still reproduces x·W at init.
func TestPiSSAIdentityWideWeight(t *testing.T) {
	w := toT([][]float64{{1, 2, 3, 4}, {5, 6, 7, 8}}) // [2,4], in<out
	p, err := nn.NewPiSSA(w, 2)
	if err != nil {
		t.Fatal(err)
	}
	x := toT([][]float64{{1, -1}, {2, 0.5}})
	got, err := p.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	approxEqT(t, got, matmul(t, x, w), 1e-9, "PiSSA identity (wide)")
}

// §V16 tier-1 (principal components): for a diagonal weight the singular values are the diagonal,
// so a rank-1 PiSSA captures the LARGEST entry in A·B and leaves the smaller one in the frozen
// residual — proving the top-r principal reconstruction and the σ ordering.
func TestPiSSAPrincipalComponents(t *testing.T) {
	w := toT([][]float64{{3, 0}, {0, 1}}) // singular values 3 > 1
	p, err := nn.NewPiSSA(w, 1)
	if err != nil {
		t.Fatal(err)
	}
	// A·B should be the top-1 reconstruction [[3,0],[0,0]].
	ab := tensor.New(tensor.F64, tensor.Shape{2, 2})
	for i := range 2 {
		for j := range 2 {
			var s float64
			for k := 0; k < p.R; k++ {
				s += p.A.AtF64(i, k) * p.B.AtF64(k, j)
			}
			ab.SetF64(s, i, j)
		}
	}
	approxEqT(t, ab, toT([][]float64{{3, 0}, {0, 0}}), 1e-9, "principal A·B")
	// the residual base keeps the smaller component [[0,0],[0,1]].
	approxEqT(t, p.W, toT([][]float64{{0, 0}, {0, 1}}), 1e-9, "residual W_res")
}

// §V16 tier-1 (full rank ⇒ zero residual): with r = min(in,out) the principal reconstruction is
// the whole weight, so W_res ≈ 0.
func TestPiSSAFullRankZeroResidual(t *testing.T) {
	w := toT([][]float64{{2, 1, 0}, {1, 3, 1}, {0, 1, 2}})
	p, err := nn.NewPiSSA(w, 3)
	if err != nil {
		t.Fatal(err)
	}
	approxEqT(t, p.W, tensor.New(tensor.F64, tensor.Shape{3, 3}), 1e-9, "full-rank residual is 0")
}

// §V-PROP: PiSSA exposes A,B as trainable and a GD step on an MSE objective reduces the loss
// (it reuses LoRALinear's forward/params).
func TestPiSSATrains(t *testing.T) {
	w := toT([][]float64{{1, 0}, {0, 1}})
	p, err := nn.NewPiSSA(w, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Params()) != 2 {
		t.Fatalf("PiSSA should expose A,B (2 params), got %d", len(p.Params()))
	}
	x := toT([][]float64{{1, 1}})
	target := toT([][]float64{{2, -1}})
	mse := func(ctx *backend.Context) (*tensor.Tensor, error) {
		y, err := p.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		d, err := exec1(ctx, backend.OpSub, nil, y, target)
		if err != nil {
			return nil, err
		}
		sq, err := exec1(ctx, backend.OpMul, nil, d, d)
		if err != nil {
			return nil, err
		}
		return exec1(ctx, backend.OpMean, nil, sq)
	}
	tape := autograd.NewTape()
	loss0, err := mse(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	l0 := loss0.AtF64()
	if err := tape.Backward(loss0); err != nil {
		t.Fatal(err)
	}
	for _, par := range p.Params() {
		g := tape.Grad(par)
		for i := range par.Numel() {
			c := tensor.Unravel(i, par.Shape())
			par.SetF64(par.AtF64(c...)-0.1*g.AtF64(c...), c...)
		}
	}
	l1, err := mse(backend.NewContext())
	if err != nil {
		t.Fatal(err)
	}
	if l1.AtF64() >= l0 {
		t.Errorf("PiSSA GD step did not reduce loss: %g → %g", l0, l1.AtF64())
	}
}

func TestPiSSAErrors(t *testing.T) {
	if _, err := nn.NewPiSSA(toT([][]float64{{1, 2}}), 5); err == nil {
		t.Error("rank out of range: want error")
	}
	if _, err := nn.NewPiSSA(vec([]float64{1, 2, 3}), 1); err == nil {
		t.Error("non-2D base: want error")
	}
}

func ExampleNewPiSSA() {
	// PiSSA seeds the LoRA matrices from the top-1 singular component of W; at init the adapter
	// reproduces x·W exactly (W_res + A·B = W).
	w := toT([][]float64{{3, 0}, {0, 1}})
	p, _ := nn.NewPiSSA(w, 1)
	y, _ := p.Forward(backend.NewContext(), toT([][]float64{{1, 1}}))
	fmt.Printf("%.1f %.1f\n", y.AtF64(0, 0), y.AtF64(0, 1))
	// Output:
	// 3.0 1.0
}
