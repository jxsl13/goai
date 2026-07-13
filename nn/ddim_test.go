package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1 (paper CONFIRMED, arXiv:2010.02502 Eq.12): the deterministic (η=0) step
// is √ᾱ_prev·x̂0 + √(1−ᾱ_prev)·ε, with x̂0 = (x_t − √(1−ᾱ_t)ε)/√ᾱ_t.
func TestDDIMDeterministicFormula(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 110))
	xt, eps := randMat(rng, 4, 3), randMat(rng, 4, 3)
	const abT, abPrev = 0.3, 0.6
	got, err := nn.DDIMStep(backend.NewContext(), xt, eps, abT, abPrev, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		for j := 0; j < 3; j++ {
			x0 := (xt.AtF64(i, j) - math.Sqrt(1-abT)*eps.AtF64(i, j)) / math.Sqrt(abT)
			want := math.Sqrt(abPrev)*x0 + math.Sqrt(1-abPrev)*eps.AtF64(i, j)
			if math.Abs(got.AtF64(i, j)-want) > 1e-9 {
				t.Errorf("x_prev[%d,%d]=%.10g want %.10g", i, j, got.AtF64(i, j), want)
			}
		}
	}
}

// §V2 (defining property): with the TRUE noise, the deterministic DDIM step maps the
// forward-noised sample at step t to the forward-noised sample at the earlier step —
// x̂0 recovers x0 exactly, so DDIM(η=0) is consistent with the forward process.
func TestDDIMForwardConsistency(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 111))
	x0, eps := randMat(rng, 3, 4), randMat(rng, 3, 4)
	const abT, abPrev = 0.3, 0.6 // reverse: prev has more signal (less noise)
	xt, _ := nn.DDPMForward(backend.NewContext(), x0, eps, abT)
	got, err := nn.DDIMStep(backend.NewContext(), xt, eps, abT, abPrev, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := nn.DDPMForward(backend.NewContext(), x0, eps, abPrev) // forward at the earlier step
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			if math.Abs(got.AtF64(i, j)-want.AtF64(i, j)) > 1e-9 {
				t.Errorf("DDIM(η=0)[%d,%d]=%g != forward(x0,ᾱ_prev) %g", i, j, got.AtF64(i, j), want.AtF64(i, j))
			}
		}
	}
}

// §V16 tier-1 (Eq.12, η=1 stochastic): full three-term update with the DDPM-recovering
// σ = √((1−ᾱ_prev)/(1−ᾱ_t)·(1 − ᾱ_t/ᾱ_prev)).
func TestDDIMStochasticParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 112))
	xt, eps, z := randMat(rng, 3, 3), randMat(rng, 3, 3), randMat(rng, 3, 3)
	const abT, abPrev, eta = 0.3, 0.6, 1.0
	got, err := nn.DDIMStep(backend.NewContext(), xt, eps, abT, abPrev, eta, z)
	if err != nil {
		t.Fatal(err)
	}
	sigma := eta * math.Sqrt((1-abPrev)/(1-abT)) * math.Sqrt(1-abT/abPrev)
	dirCoef := math.Sqrt(math.Max(0, 1-abPrev-sigma*sigma))
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			x0 := (xt.AtF64(i, j) - math.Sqrt(1-abT)*eps.AtF64(i, j)) / math.Sqrt(abT)
			want := math.Sqrt(abPrev)*x0 + dirCoef*eps.AtF64(i, j) + sigma*z.AtF64(i, j)
			if math.Abs(got.AtF64(i, j)-want) > 1e-9 {
				t.Errorf("η=1 x_prev[%d,%d]=%.10g want %.10g", i, j, got.AtF64(i, j), want)
			}
		}
	}
}

// §V2: the final step (ᾱ_prev=1) yields the clean predicted x0, and η=0 ignores z.
func TestDDIMFinalStepAndDeterminism(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 113))
	xt, eps := randMat(rng, 2, 3), randMat(rng, 2, 3)
	// ᾱ_prev = 1 ⇒ σ=0, dirCoef=0 ⇒ x_prev = x̂0.
	got, err := nn.DDIMStep(backend.NewContext(), xt, eps, 0.4, 1, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	x0, _ := nn.DDPMPredictX0(backend.NewContext(), xt, eps, 0.4)
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(got.AtF64(i, j)-x0.AtF64(i, j)) > 1e-9 {
				t.Errorf("final step[%d,%d]=%g != predicted x0 %g", i, j, got.AtF64(i, j), x0.AtF64(i, j))
			}
		}
	}
	// η=0 must ignore the noise z entirely.
	z := randMat(rng, 2, 3)
	withZ, _ := nn.DDIMStep(backend.NewContext(), xt, eps, 0.4, 0.7, 0, z)
	noZ, _ := nn.DDIMStep(backend.NewContext(), xt, eps, 0.4, 0.7, 0, nil)
	for i := range withZ.Numel() {
		c := tensor.Unravel(i, withZ.Shape())
		if withZ.AtF64(c...) != noZ.AtF64(c...) {
			t.Error("η=0 step used the noise z (should be deterministic)")
		}
	}
}

func TestDDIMErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.DDIMStep(backend.NewContext(), r(2, 3), r(2, 4), 0.3, 0.6, 0, nil); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.DDIMStep(backend.NewContext(), r(2, 3), r(2, 3), 0, 0.6, 0, nil); err == nil {
		t.Error("alphaBarT=0: want error")
	}
	if _, err := nn.DDIMStep(backend.NewContext(), r(2, 3), r(2, 3), 0.3, 0.6, 1.5, r(2, 3)); err == nil {
		t.Error("eta>1: want error")
	}
	if _, err := nn.DDIMStep(backend.NewContext(), r(2, 3), r(2, 3), 0.3, 0.6, 1, r(2, 4)); err == nil {
		t.Error("z shape mismatch: want error")
	}
}

func ExampleDDIMStep() {
	// deterministic DDIM. With the true noise, one step from the forward sample at
	// ᾱ=0.25 lands exactly on the forward sample at ᾱ=1 — i.e. the clean data x0.
	x0 := toT([][]float64{{2, -2}})
	eps := toT([][]float64{{1, 1}})
	xt, _ := nn.DDPMForward(backend.NewContext(), x0, eps, 0.25)
	out, _ := nn.DDIMStep(backend.NewContext(), xt, eps, 0.25, 1, 0, nil)
	fmt.Printf("%.2f %.2f\n", out.AtF64(0, 0), out.AtF64(0, 1))
	// Output:
	// 2.00 -2.00
}
