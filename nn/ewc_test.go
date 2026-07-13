package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func ones(shape ...int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape(shape))
	for e := range t.Numel() {
		t.SetF64(1, tensor.Unravel(e, t.Shape())...)
	}
	return t
}

// §V16 tier-1 (paper CONFIRMED, arXiv:1612.00796 Eq. 3): Ω = (λ/2)Σ F(θ−θ*)².
func TestEWCPenaltyFormula(t *testing.T) {
	params := []*tensor.Tensor{toT([][]float64{{1, 2}, {3, 4}}), rvec(0.5, -0.5)}
	ref := []*tensor.Tensor{toT([][]float64{{0, 1}, {2, 5}}), rvec(0, 0)}
	fisher := []*tensor.Tensor{toT([][]float64{{2, 0.5}, {1, 3}}), rvec(4, 0.25)}
	got, err := nn.EWCPenalty(backend.NewContext(), params, ref, fisher, 0.7)
	if err != nil {
		t.Fatal(err)
	}
	// Σ F(θ−θ*)²: 2·1+0.5·1+1·1+3·1 (2×2 block, all drifts ±1) + 4·0.25+0.25·0.25 = 6.5+1.0625;
	// ×(λ/2=0.35) = 2.646875.
	if want := 2.646875; math.Abs(got.AtF64()-want) > 1e-9 {
		t.Errorf("penalty=%.12g want %.12g", got.AtF64(), want)
	}
}

// §V16 tier-1 (uniform Fisher ⇒ scaled L2): with F=1 the penalty is (λ/2)‖θ−θ*‖².
func TestEWCUniformFisherIsL2(t *testing.T) {
	p := []*tensor.Tensor{rvec(3, -1, 2)}
	r := []*tensor.Tensor{rvec(1, 1, 1)}
	f := []*tensor.Tensor{ones(3)}
	got, _ := nn.EWCPenalty(backend.NewContext(), p, r, f, 2.0)
	l2 := (2.0 / 2) * ((3-1)*(3-1) + (-1-1)*(-1-1) + (2-1)*(2-1)) // (λ/2)Σ(θ−θ*)²
	if math.Abs(got.AtF64()-l2) > 1e-9 {
		t.Errorf("uniform-Fisher penalty=%g, want scaled L2 %g", got.AtF64(), l2)
	}
}

// §V16 tier-1 (at the optimum): θ = θ* ⇒ penalty 0.
func TestEWCAtOptimumIsZero(t *testing.T) {
	p := []*tensor.Tensor{rvec(1, 2, 3)}
	f := []*tensor.Tensor{rvec(5, 10, 0.1)}
	got, _ := nn.EWCPenalty(backend.NewContext(), p, p, f, 1.0) // ref == params
	if math.Abs(got.AtF64()) > 1e-12 {
		t.Errorf("penalty at optimum=%g, want 0", got.AtF64())
	}
}

// §V2 (Fisher weighting): the same drift on a high-Fisher parameter is penalized more than
// on a low-Fisher one — the defining behavior that protects important weights.
func TestEWCFisherWeighting(t *testing.T) {
	// two parameters, each drifted by 1 from the optimum; only the Fisher differs.
	p := []*tensor.Tensor{rvec(1), rvec(1)}
	r := []*tensor.Tensor{rvec(0), rvec(0)}
	important := []*tensor.Tensor{rvec(10), rvec(1)}
	pen, _ := nn.EWCPenalty(backend.NewContext(), p, r, important, 1.0)
	// contribution ratio should equal the Fisher ratio (10:1) ⇒ total (λ/2)(10+1)=5.5.
	if math.Abs(pen.AtF64()-5.5) > 1e-9 {
		t.Errorf("Fisher-weighted penalty=%g, want 5.5", pen.AtF64())
	}
}

// §V2: differentiable w.r.t. the parameters (∂Ω/∂θ = λ·F·(θ−θ*)).
func TestEWCGradcheck(t *testing.T) {
	p := []*tensor.Tensor{toT([][]float64{{0.7, -0.3}, {1.1, 0.2}})}
	r := []*tensor.Tensor{toT([][]float64{{0.1, 0.4}, {-0.5, 0.9}})}
	f := []*tensor.Tensor{toT([][]float64{{2.0, 0.5}, {1.5, 3.0}})}
	tape := autograd.NewTape()
	loss, err := nn.EWCPenalty(tape.Context(), p, r, f, 0.8)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	g := tape.Grad(p[0])
	for e := range p[0].Numel() {
		c := tensor.Unravel(e, p[0].Shape())
		o := p[0].AtF64(c...)
		p[0].SetF64(o+h, c...)
		lp, _ := nn.EWCPenalty(backend.NewContext(), p, r, f, 0.8)
		p[0].SetF64(o-h, c...)
		lm, _ := nn.EWCPenalty(backend.NewContext(), p, r, f, 0.8)
		p[0].SetF64(o, c...)
		if num, ana := (lp.AtF64()-lm.AtF64())/(2*h), g.AtF64(c...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad%v: numeric %.8g vs analytic %.8g", c, num, ana)
		}
	}
}

// §V16 tier-1 (diagonal Fisher = mean squared gradient): F_i = mean_s (grad_s[i])².
func TestEWCFisher(t *testing.T) {
	samples := [][]*tensor.Tensor{
		{rvec(2, -1)}, {rvec(0, 3)}, {rvec(-2, 1)},
	}
	f, err := nn.EWCFisher(samples)
	if err != nil {
		t.Fatal(err)
	}
	// F[0] = mean(2²,0²,(−2)²)=8/3; F[1] = mean((−1)²,3²,1²)=11/3.
	if math.Abs(f[0].AtF64(0)-8.0/3) > 1e-9 || math.Abs(f[0].AtF64(1)-11.0/3) > 1e-9 {
		t.Errorf("Fisher=[%g %g], want [%g %g]", f[0].AtF64(0), f[0].AtF64(1), 8.0/3, 11.0/3)
	}
}

func TestEWCErrors(t *testing.T) {
	p := []*tensor.Tensor{rvec(1, 2)}
	if _, err := nn.EWCPenalty(backend.NewContext(), p, p, []*tensor.Tensor{rvec(1)}, 1); err == nil {
		t.Error("length mismatch: want error")
	}
	if _, err := nn.EWCPenalty(backend.NewContext(), p, []*tensor.Tensor{rvec(1, 2, 3)}, []*tensor.Tensor{rvec(1, 2, 3)}, 1); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.EWCFisher(nil); err == nil {
		t.Error("no samples: want error")
	}
}

func ExampleEWCPenalty() {
	// One weight drifted by 2 from its task-A optimum, Fisher 3, λ=1 ⇒ (1/2)·3·2² = 6.
	p := []*tensor.Tensor{rvec(5)}
	ref := []*tensor.Tensor{rvec(3)}
	fisher := []*tensor.Tensor{rvec(3)}
	pen, _ := nn.EWCPenalty(backend.NewContext(), p, ref, fisher, 1.0)
	fmt.Printf("%.1f\n", pen.AtF64())
	// Output:
	// 6.0
}
