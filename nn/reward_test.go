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

func rvec(vals ...float64) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{len(vals)})
	for i, v := range vals {
		t.SetF64(v, i)
	}
	return t
}

// btRef = mean −log σ(r_chosen − r_rejected − margin) = mean softplus(r_rejected − r_chosen + margin).
func btRef(chosen, rejected []float64, margin float64) float64 {
	var s float64
	for i := range chosen {
		s += math.Log1p(math.Exp(rejected[i] - chosen[i] + margin))
	}
	return s / float64(len(chosen))
}

// §V16 tier-1 (paper CONFIRMED, arXiv:1706.03741 etc.): L = mean −log σ(r_w − r_l − margin).
func TestBradleyTerryFormula(t *testing.T) {
	chosen := []float64{2.0, 0.5, -1.0, 3.0}
	rejected := []float64{1.0, 0.8, -1.5, 0.0}
	for _, m := range []float64{0, 0.5} {
		got, err := nn.BradleyTerryLoss(backend.NewContext(), rvec(chosen...), rvec(rejected...), m)
		if err != nil {
			t.Fatal(err)
		}
		if want := btRef(chosen, rejected, m); math.Abs(got.AtF64()-want) > 1e-9 {
			t.Errorf("margin=%g loss=%.12g want %.12g", m, got.AtF64(), want)
		}
	}
}

// §V16 tier-1 (closed form): equal rewards ⇒ σ(0)=0.5 ⇒ per-pair loss ln 2.
func TestBradleyTerryEqualRewardsIsLn2(t *testing.T) {
	r := rvec(1.0, -2.0, 3.0)
	loss, err := nn.BradleyTerryLoss(backend.NewContext(), r, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(loss.AtF64()-math.Ln2) > 1e-9 {
		t.Errorf("equal rewards loss=%g, want ln2=%g", loss.AtF64(), math.Ln2)
	}
}

// §V2: a reward model that already ranks chosen ≫ rejected has near-zero loss.
func TestBradleyTerryPerfectSeparation(t *testing.T) {
	loss, _ := nn.BradleyTerryLoss(backend.NewContext(), rvec(20, 20), rvec(-20, -20), 0)
	if loss.AtF64() > 1e-6 {
		t.Errorf("perfect separation loss=%g, want ~0", loss.AtF64())
	}
}

// §V2: differentiable w.r.t. both reward vectors.
func TestBradleyTerryGradcheck(t *testing.T) {
	chosen := rvec(1.2, -0.3, 0.7, 2.0)
	rejected := rvec(0.5, 0.1, -0.4, 1.0)
	tape := autograd.NewTape()
	loss, err := nn.BradleyTerryLoss(tape.Context(), chosen, rejected, 0.3)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	fwd := func() float64 {
		l, _ := nn.BradleyTerryLoss(backend.NewContext(), chosen, rejected, 0.3)
		return l.AtF64()
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for i := range p.Numel() {
			o := p.AtF64(i)
			p.SetF64(o+h, i)
			lp := fwd()
			p.SetF64(o-h, i)
			lm := fwd()
			p.SetF64(o, i)
			if num, ana := (lp-lm)/(2*h), g.AtF64(i); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", name, i, num, ana)
			}
		}
	}
	check("chosen", chosen)
	check("rejected", rejected)
}

// §V2 (the training signal): the gradient pushes the chosen reward UP (∂L/∂r_chosen < 0) and
// the rejected reward DOWN (∂L/∂r_rejected > 0).
func TestBradleyTerryGradientDirection(t *testing.T) {
	chosen := rvec(0.0, 0.0)
	rejected := rvec(0.0, 0.0)
	tape := autograd.NewTape()
	loss, _ := nn.BradleyTerryLoss(tape.Context(), chosen, rejected, 0)
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	if tape.Grad(chosen).AtF64(0) >= 0 {
		t.Errorf("∂L/∂r_chosen=%g, want < 0 (push chosen up)", tape.Grad(chosen).AtF64(0))
	}
	if tape.Grad(rejected).AtF64(0) <= 0 {
		t.Errorf("∂L/∂r_rejected=%g, want > 0 (push rejected down)", tape.Grad(rejected).AtF64(0))
	}
}

// §V16 tier-1 (Llama-2 margin): subtracting a positive margin demands a larger reward gap,
// raising the loss for a fixed reward pair.
func TestBradleyTerryMarginIncreasesLoss(t *testing.T) {
	c, r := rvec(1.0, 2.0), rvec(0.5, 1.0)
	lo, _ := nn.BradleyTerryLoss(backend.NewContext(), c, r, 0)
	hi, _ := nn.BradleyTerryLoss(backend.NewContext(), c, r, 1.0)
	if !(hi.AtF64() > lo.AtF64()) {
		t.Errorf("margin loss %g not > no-margin %g", hi.AtF64(), lo.AtF64())
	}
}

func TestBradleyTerryErrors(t *testing.T) {
	if _, err := nn.BradleyTerryLoss(backend.NewContext(), rvec(1, 2, 3), rvec(1, 2), 0); err == nil {
		t.Error("shape mismatch: want error")
	}
}

func ExampleBradleyTerryLoss() {
	// A reward model already scoring the chosen response above the rejected one has low loss;
	// equal scores would give ln2 ≈ 0.69.
	loss, _ := nn.BradleyTerryLoss(backend.NewContext(), rvec(3.0, 2.5), rvec(0.0, -1.0), 0)
	fmt.Printf("%.3f\n", loss.AtF64())
	// Output:
	// 0.039
}
