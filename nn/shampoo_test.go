package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1: with a DIAGONAL gradient the preconditioners L,R are diagonal, so the
// Shampoo update Ĝ = L^{−1/4}·G·R^{−1/4} is diagonal with Ĝ_ii = g_i/√(ε+g_i²). This
// makes the whole matrix step hand-checkable against Algorithm 1.
func TestShampooDiagonalStep(t *testing.T) {
	const eps, lr = 1e-4, 0.1
	w := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 5, -2, 3})
	opt := nn.NewShampoo([]*tensor.Tensor{w}, lr, nn.WithShampooEps(eps))
	g := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{2, 0, 0, 3}) // diagonal gradient
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
		t.Fatal(err)
	}
	// L=R=diag(ε+4, ε+9) ⇒ Ĝ=diag(2/√(ε+4), 3/√(ε+9)); off-diagonal Ĝ=0.
	want := [][]float64{
		{1 - lr*2/math.Sqrt(eps+4), 5},
		{-2, 3 - lr*3/math.Sqrt(eps+9)},
	}
	for i := range 2 {
		for j := range 2 {
			if math.Abs(w.AtF64(i, j)-want[i][j]) > 1e-9 {
				t.Errorf("W[%d,%d] = %.10g, want %.10g", i, j, w.AtF64(i, j), want[i][j])
			}
		}
	}
}

// §V16 tier-1: a non-matrix (vector) parameter reduces to AdaGrad (exponent −1/2):
// v += g², W_i −= lr·g_i/(√v_i + ε).
func TestShampooVectorAdaGrad(t *testing.T) {
	const eps, lr = 1e-8, 0.1
	w := tensor.FromFloat64(tensor.Shape{3}, []float64{1, 1, 1})
	opt := nn.NewShampoo([]*tensor.Tensor{w}, lr, nn.WithShampooEps(eps))
	gv := []float64{2, -4, 0.5}
	g := tensor.FromFloat64(tensor.Shape{3}, gv)
	_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g })
	for i, gi := range gv {
		want := 1 - lr*gi/(math.Sqrt(gi*gi)+eps) // v=g² after one step
		if math.Abs(w.AtF64(i)-want) > 1e-9 {
			t.Errorf("W[%d] = %.10g, want %.10g", i, w.AtF64(i), want)
		}
	}
}

// §V2/convergence: Shampoo minimizes a convex quadratic on a matrix parameter.
func TestShampooConverges(t *testing.T) {
	w := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{4, -2, 5, 1, -3, 2})
	target := []float64{1, 0, -1, 2, 1, 0}
	// Higher lr + more steps: Shampoo's AdaGrad-style accumulation decays the step
	// size (~1/√t), so a plain quadratic converges steadily but sublinearly.
	opt := nn.NewShampoo([]*tensor.Tensor{w}, 1.0)
	loss := func() float64 {
		var s float64
		for i := range 6 {
			idx := tensor.Unravel(i, w.Shape())
			d := w.AtF64(idx...) - target[i]
			s += d * d
		}
		return s
	}
	first := loss()
	for range 1500 {
		err := opt.Step(func(p *tensor.Tensor) *tensor.Tensor {
			g := tensor.New(tensor.F64, p.Shape())
			for i := range 6 {
				idx := tensor.Unravel(i, p.Shape())
				g.SetF64(2*(p.AtF64(idx...)-target[i]), idx...)
			}
			return g
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if last := loss(); last > 0.01*first {
		t.Errorf("did not converge: loss %.4g → %.4g", first, last)
	}
}

func ExampleShampoo() {
	// One step on a matrix parameter with a diagonal gradient (ε small, lr=0.5):
	// Ĝ_ii = g_i/√(ε+g_i²) ≈ sign(g_i), so W ≈ W − 0.5·sign(g).
	w := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{0, 0, 0, 0})
	opt := nn.NewShampoo([]*tensor.Tensor{w}, 0.5, nn.WithShampooEps(1e-12))
	g := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{3, 0, 0, -3})
	_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g })
	fmt.Printf("%.3f %.3f\n", w.AtF64(0, 0), w.AtF64(1, 1))
	// Output:
	// -0.500 0.500
}
