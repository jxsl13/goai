package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1 (paper CONFIRMED, arXiv:2006.11239 §4): linear β, α=1−β, ᾱ=cumprod.
func TestDDPMSchedule(t *testing.T) {
	s := nn.NewDDPMSchedule(5, 0.1, 0.5)
	wantBeta := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	ab := 1.0
	for i := range 5 {
		if math.Abs(s.Beta[i]-wantBeta[i]) > 1e-12 {
			t.Errorf("beta[%d]=%g want %g", i, s.Beta[i], wantBeta[i])
		}
		if math.Abs(s.Alpha[i]-(1-wantBeta[i])) > 1e-12 {
			t.Errorf("alpha[%d]=%g want %g", i, s.Alpha[i], 1-wantBeta[i])
		}
		ab *= 1 - wantBeta[i]
		if math.Abs(s.AlphaBar[i]-ab) > 1e-12 {
			t.Errorf("alphaBar[%d]=%g want cumprod %g", i, s.AlphaBar[i], ab)
		}
		if i > 0 && s.AlphaBar[i] >= s.AlphaBar[i-1] {
			t.Errorf("alphaBar not decreasing at %d", i)
		}
	}
	// the paper's default schedule: ᾱ starts ≈1 (little noise) and ends ≈0 (pure noise).
	def := nn.NewDDPMSchedule(1000, 1e-4, 0.02)
	if def.AlphaBar[0] < 0.999 || def.AlphaBar[999] > 0.01 {
		t.Errorf("default ᾱ range [%g,%g], want ≈1→≈0", def.AlphaBar[0], def.AlphaBar[999])
	}
}

// §V16 tier-1 (Eq.4): x_t = √ᾱ·x0 + √(1−ᾱ)·eps, with the endpoints x0 (ᾱ=1) and
// eps (ᾱ=0), and variance preservation (√ᾱ²+√(1−ᾱ)²=1).
func TestDDPMForward(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 100))
	x0, eps := randMat(rng, 4, 3), randMat(rng, 4, 3)
	for _, ab := range []float64{0, 0.3, 1} {
		xt, err := nn.DDPMForward(backend.NewContext(), x0, eps, ab)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 4; i++ {
			for j := 0; j < 3; j++ {
				want := math.Sqrt(ab)*x0.AtF64(i, j) + math.Sqrt(1-ab)*eps.AtF64(i, j)
				if math.Abs(xt.AtF64(i, j)-want) > 1e-9 {
					t.Errorf("ᾱ=%g x_t[%d,%d]=%g want %g", ab, i, j, xt.AtF64(i, j), want)
				}
			}
		}
	}
}

// §V2 (defining inverse): DDPMPredictX0 recovers x0 exactly from the forward-noised
// sample and the true noise — the algebraic inverse of DDPMForward.
func TestDDPMPredictX0RecoversX0(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 101))
	x0, eps := randMat(rng, 3, 4), randMat(rng, 3, 4)
	const ab = 0.4
	xt, _ := nn.DDPMForward(backend.NewContext(), x0, eps, ab)
	rec, err := nn.DDPMPredictX0(backend.NewContext(), xt, eps, ab)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			if math.Abs(rec.AtF64(i, j)-x0.AtF64(i, j)) > 1e-9 {
				t.Errorf("PredictX0[%d,%d]=%g != x0 %g", i, j, rec.AtF64(i, j), x0.AtF64(i, j))
			}
		}
	}
}

// §V16 tier-1 (Eq.14): L = mean‖eps − epsPred‖², zero when the prediction is exact.
func TestDDPMLoss(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 102))
	eps, epsPred := randMat(rng, 4, 3), randMat(rng, 4, 3)
	loss, err := nn.DDPMLoss(backend.NewContext(), epsPred, eps)
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	n := 0
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			d := eps.AtF64(i, j) - epsPred.AtF64(i, j)
			sum += d * d
			n++
		}
	}
	if want := sum / float64(n); math.Abs(loss.AtF64()-want) > 1e-9 {
		t.Errorf("loss=%.10g want %.10g", loss.AtF64(), want)
	}
	zero, _ := nn.DDPMLoss(backend.NewContext(), eps, eps)
	if math.Abs(zero.AtF64()) > 1e-12 {
		t.Errorf("perfect-prediction loss=%g, want 0", zero.AtF64())
	}
}

// §V2: the ε-prediction loss is differentiable w.r.t. the model output.
func TestDDPMGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 103))
	eps, epsPred := randMat(rng, 3, 3), randMat(rng, 3, 3)
	tape := autograd.NewTape()
	loss, err := nn.DDPMLoss(tape.Context(), epsPred, eps)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(epsPred)
	const h = 1e-6
	fwd := func() float64 {
		l, _ := nn.DDPMLoss(backend.NewContext(), epsPred, eps)
		return l.AtF64()
	}
	for idx := range epsPred.Numel() {
		coord := tensor.Unravel(idx, epsPred.Shape())
		o := epsPred.AtF64(coord...)
		epsPred.SetF64(o+h, coord...)
		lp := fwd()
		epsPred.SetF64(o-h, coord...)
		lm := fwd()
		epsPred.SetF64(o, coord...)
		if num, ana := (lp-lm)/(2*h), g.AtF64(coord...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad%v: numeric %.8g vs analytic %.8g", coord, num, ana)
		}
	}
}

func TestDDPMErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.DDPMForward(backend.NewContext(), r(2, 3), r(2, 4), 0.5); err == nil {
		t.Error("forward shape mismatch: want error")
	}
	if _, err := nn.DDPMForward(backend.NewContext(), r(2, 3), r(2, 3), 1.5); err == nil {
		t.Error("alphaBar>1: want error")
	}
	if _, err := nn.DDPMLoss(backend.NewContext(), r(2, 3), r(2, 4)); err == nil {
		t.Error("loss shape mismatch: want error")
	}
	if _, err := nn.DDPMPredictX0(backend.NewContext(), r(2, 3), r(2, 3), 0); err == nil {
		t.Error("predictX0 alphaBar=0: want error")
	}
}

func ExampleDDPMSchedule() {
	// a tiny linear schedule: β from 0.1 to 0.5 over 3 steps; ᾱ is the running product
	// of α=1−β, decreasing as more noise accumulates.
	s := nn.NewDDPMSchedule(3, 0.1, 0.5)
	fmt.Printf("beta=[%.1f %.1f %.1f] alphaBar[last]=%.3f\n", s.Beta[0], s.Beta[1], s.Beta[2], s.AlphaBar[2])
	// Output:
	// beta=[0.1 0.3 0.5] alphaBar[last]=0.315
}

func ExampleDDPMForward() {
	// clean data x0=(4,4), noise eps=(1,1); at ᾱ=0.25 the noised sample is
	// √0.25·x0 + √0.75·eps = 0.5·4 + 0.866·1 = 2.87.
	x0 := toT([][]float64{{4, 4}})
	eps := toT([][]float64{{1, 1}})
	xt, _ := nn.DDPMForward(backend.NewContext(), x0, eps, 0.25)
	fmt.Printf("%.2f\n", xt.AtF64(0, 0))
	// Output:
	// 2.87
}
