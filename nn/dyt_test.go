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

func mkDyT(t *testing.T, d int, alpha float64, gamma, beta []float64) *nn.DyT {
	t.Helper()
	l, err := nn.NewDyT(tensor.F64, d, alpha)
	if err != nil {
		t.Fatal(err)
	}
	for i := range gamma {
		l.Gamma.SetF64(gamma[i], i)
		l.Beta.SetF64(beta[i], i)
	}
	return l
}

// §V16 tier-1 (arXiv:2503.10622): DyT(x) = γ⊙tanh(α·x)+β. Verified against a hand computation.
func TestDyTForwardParity(t *testing.T) {
	l := mkDyT(t, 2, 0.5, []float64{2, 3}, []float64{0.1, -0.2})
	y, err := l.Forward(backend.NewContext(), toT([][]float64{{1, 2}}))
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{2*math.Tanh(0.5) + 0.1, 3*math.Tanh(1.0) - 0.2}
	for j := range want {
		if math.Abs(y.AtF64(0, j)-want[j]) > 1e-12 {
			t.Errorf("y[0,%d] = %.12g, want %.12g", j, y.AtF64(0, j), want[j])
		}
	}
}

// §V16 tier-1 (NO statistics): DyT is purely elementwise — changing one input channel does not
// affect any other output channel. This is the defining contrast with LayerNorm, which couples
// channels through the shared mean/variance.
func TestDyTNoCrossChannelCoupling(t *testing.T) {
	l := mkDyT(t, 3, 0.7, []float64{1, 1, 1}, []float64{0, 0, 0})
	base, _ := l.Forward(backend.NewContext(), toT([][]float64{{0.3, -0.5, 1.2}}))
	// perturb only channel 2; channels 0 and 1 must be unchanged (an affine LayerNorm would shift).
	pert, _ := l.Forward(backend.NewContext(), toT([][]float64{{0.3, -0.5, 9.9}}))
	for j := 0; j < 2; j++ {
		if math.Abs(base.AtF64(0, j)-pert.AtF64(0, j)) > 1e-12 {
			t.Errorf("channel %d changed when only channel 2 was perturbed: %g vs %g", j, base.AtF64(0, j), pert.AtF64(0, j))
		}
	}
}

// §V16 tier-1 (tanh saturation): large-magnitude inputs saturate to γ·sign(x)+β (bounded output).
func TestDyTSaturation(t *testing.T) {
	l := mkDyT(t, 2, 1.0, []float64{2, 2}, []float64{0.5, 0.5})
	y, _ := l.Forward(backend.NewContext(), toT([][]float64{{100, -100}}))
	// tanh(±100)≈±1 ⇒ y≈[2·1+0.5, 2·(−1)+0.5]=[2.5, −1.5].
	if math.Abs(y.AtF64(0, 0)-2.5) > 1e-9 || math.Abs(y.AtF64(0, 1)+1.5) > 1e-9 {
		t.Errorf("saturation y = [%g %g], want [2.5 −1.5]", y.AtF64(0, 0), y.AtF64(0, 1))
	}
}

// §V-GRAD: finite-difference check of the gradient into α, γ and β.
func TestDyTGradCheck(t *testing.T) {
	l := mkDyT(t, 3, 0.6, []float64{1.2, -0.7, 0.5}, []float64{0.1, 0.2, -0.3})
	x := toT([][]float64{{0.5, -1.0, 2.0}, {-0.3, 0.8, 1.5}})
	sumLoss := func(ctx *backend.Context) (*tensor.Tensor, error) {
		y, err := l.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		return exec1(ctx, backend.OpSum, backend.ReduceAttrs{Axes: []int{0, 1}}, y)
	}
	tape := autograd.NewTape()
	loss, err := sumLoss(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const eps = 1e-6
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil gradient", name)
		}
		for i := range p.Numel() {
			c := tensor.Unravel(i, p.Shape())
			o := p.AtF64(c...)
			p.SetF64(o+eps, c...)
			lp, _ := sumLoss(backend.NewContext())
			p.SetF64(o-eps, c...)
			lm, _ := sumLoss(backend.NewContext())
			p.SetF64(o, c...)
			if num, ana := (lp.AtF64()-lm.AtF64())/(2*eps), g.AtF64(c...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad %v: numeric %.8g vs analytic %.8g", name, c, num, ana)
			}
		}
	}
	check("alpha", l.Alpha)
	check("gamma", l.Gamma)
	check("beta", l.Beta)
}

// §V2: shape preserved; Params exposes α[1], γ[d], β[d].
func TestDyTShapeAndParams(t *testing.T) {
	l, err := nn.NewDyT(tensor.F64, 4, 0)
	if err != nil {
		t.Fatal(err)
	}
	if l.Alpha.AtF64(0) != nn.DyTDefaultAlphaInit {
		t.Errorf("default α = %g, want %g", l.Alpha.AtF64(0), nn.DyTDefaultAlphaInit)
	}
	x := toT([][]float64{{1, 2, 3, 4}, {5, 6, 7, 8}})
	y, _ := l.Forward(backend.NewContext(), x)
	if !y.Shape().Equal(x.Shape()) {
		t.Errorf("shape %v != input %v", y.Shape(), x.Shape())
	}
	ps := l.Params()
	if len(ps) != 3 || ps[0].Numel() != 1 || ps[1].Numel() != 4 || ps[2].Numel() != 4 {
		t.Errorf("Params sizes = [%d %d %d], want α=1 γ=4 β=4", ps[0].Numel(), ps[1].Numel(), ps[2].Numel())
	}
}

func TestDyTErrors(t *testing.T) {
	if _, err := nn.NewDyT(tensor.F64, 0, 0.5); err == nil {
		t.Error("d=0: want error")
	}
}

func ExampleDyT() {
	// DyT with γ=1, β=0 at init just squashes: DyT(x) = tanh(α·x). At α=0.5, DyT(2) = tanh(1).
	l, _ := nn.NewDyT(tensor.F64, 1, 0.5)
	y, _ := l.Forward(backend.NewContext(), toT([][]float64{{2}}))
	fmt.Printf("%.4f\n", y.AtF64(0, 0))
	// Output:
	// 0.7616
}
