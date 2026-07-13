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

// §V16 tier-1 (arXiv:2305.10425): the calibration loss is the hinge max(0, δ − logPθ(y⁺) +
// logPθ(y⁻)); with the regularizer it adds λ·mean(−logPθ(y_ref)). Verified against an
// INDEPENDENT hand computation.
func TestSLiCParity(t *testing.T) {
	pref := vec([]float64{2.0, -1.0})   // logPθ(y⁺)
	dispref := vec([]float64{1.0, 0.5}) // logPθ(y⁻)
	sft := vec([]float64{-3.0, -2.0})   // logPθ(y_ref)
	const delta, lambda = 0.5, 0.1

	// hinge: max(0, 0.5−2+1)=0, max(0, 0.5+1+0.5)=2 ⇒ L_cal = (0+2)/2 = 1.0
	bare, err := nn.SLiCLoss(backend.NewContext(), pref, dispref, nil, delta, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := bare.AtF64(); math.Abs(got-1.0) > 1e-12 {
		t.Errorf("bare calibration loss %.12g, want 1.0", got)
	}
	// reg: mean(3,2)=2.5 ⇒ total = 1.0 + 0.1·2.5 = 1.25
	full, err := nn.SLiCLoss(backend.NewContext(), pref, dispref, sft, delta, lambda)
	if err != nil {
		t.Fatal(err)
	}
	if got := full.AtF64(); math.Abs(got-1.25) > 1e-12 {
		t.Errorf("regularized loss %.12g, want 1.25", got)
	}
}

// §V16 tier-1 (max-margin geometry): once the preferred candidate already wins by ≥ δ the
// hinge is inactive — the calibration loss and its gradient are exactly zero (unlike the
// logistic DPO losses, which keep pushing). This is SLiC's defining property.
func TestSLiCMarginSatisfiedIsZero(t *testing.T) {
	pref := vec([]float64{5.0, 4.0}) // wins by 4.0 and 3.5, both ≥ δ=1
	dispref := vec([]float64{1.0, 0.5})
	tape := autograd.NewTape()
	loss, err := nn.SLiCLoss(tape.Context(), pref, dispref, nil, 1.0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := loss.AtF64(); got != 0 {
		t.Errorf("satisfied margin: loss %.12g, want exactly 0", got)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	if g := tape.Grad(pref); g != nil {
		for i := range g.Numel() {
			if g.AtF64(i) != 0 {
				t.Errorf("satisfied margin: preferred grad[%d]=%g, want 0", i, g.AtF64(i))
			}
		}
	}
}

// §V-GRAD: finite-difference gradient check for both the calibration and regularization terms.
func TestSLiCGradCheck(t *testing.T) {
	pref := []float64{2.0, -1.0, 0.5}
	dispref := []float64{1.0, 0.5, -2.0}
	sft := []float64{-3.0, -2.0, -1.5}
	const delta, lambda = 0.5, 0.2

	lossAt := func(p, d, s []float64) float64 {
		out, err := nn.SLiCLoss(backend.NewContext(), vec(p), vec(d), vec(s), delta, lambda)
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}
	pT, dT, sT := vec(pref), vec(dispref), vec(sft)
	tape := autograd.NewTape()
	loss, err := nn.SLiCLoss(tape.Context(), pT, dT, sT, delta, lambda)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const eps = 1e-6
	check := func(name string, data []float64, analytic *tensor.Tensor, which int) {
		if analytic == nil {
			t.Fatalf("%s: nil gradient", name)
		}
		for i := range data {
			o := data[i]
			data[i] = o + eps
			lp := pick(lossAt, which, data, pref, dispref, sft)
			data[i] = o - eps
			lm := pick(lossAt, which, data, pref, dispref, sft)
			data[i] = o
			num, ana := (lp-lm)/(2*eps), analytic.AtF64(i)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", name, i, num, ana)
			}
		}
	}
	check("preferred", append([]float64(nil), pref...), tape.Grad(pT), 0)
	check("dispreferred", append([]float64(nil), dispref...), tape.Grad(dT), 1)
	check("sftTarget", append([]float64(nil), sft...), tape.Grad(sT), 2)
}

// pick evaluates lossAt with `data` substituted for argument `which`.
func pick(lossAt func(p, d, s []float64) float64, which int, data, pref, dispref, sft []float64) float64 {
	switch which {
	case 0:
		return lossAt(data, dispref, sft)
	case 1:
		return lossAt(pref, data, sft)
	default:
		return lossAt(pref, dispref, data)
	}
}

// §V-PROP: one gradient-descent step on the calibration loss widens the preferred/dispreferred
// margin (raises logPθ(y⁺), lowers logPθ(y⁻)) on the examples where the margin is violated.
func TestSLiCTrainsPreference(t *testing.T) {
	pref := vec([]float64{0.0, 0.0})
	dispref := vec([]float64{0.5, 1.0}) // dispreferred currently ahead ⇒ hinge active
	tape := autograd.NewTape()
	loss, err := nn.SLiCLoss(tape.Context(), pref, dispref, nil, 1.0, 0)
	if err != nil {
		t.Fatal(err)
	}
	l0 := loss.AtF64()
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	gp, gd := tape.Grad(pref), tape.Grad(dispref)
	const lr = 0.1
	np := make([]float64, 2)
	nd := make([]float64, 2)
	for i := range np {
		np[i] = pref.AtF64(i) - lr*gp.AtF64(i)
		nd[i] = dispref.AtF64(i) - lr*gd.AtF64(i)
	}
	// preferred should have moved up, dispreferred down.
	for i := range np {
		if np[i] <= pref.AtF64(i) {
			t.Errorf("preferred[%d] did not increase: %g→%g", i, pref.AtF64(i), np[i])
		}
		if nd[i] >= dispref.AtF64(i) {
			t.Errorf("dispreferred[%d] did not decrease: %g→%g", i, dispref.AtF64(i), nd[i])
		}
	}
	l1, err := nn.SLiCLoss(backend.NewContext(), vec(np), vec(nd), nil, 1.0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if l1.AtF64() >= l0 {
		t.Errorf("loss did not decrease after a GD step: %g → %g", l0, l1.AtF64())
	}
}

func TestSLiCErrors(t *testing.T) {
	if _, err := nn.SLiCLoss(backend.NewContext(), vec([]float64{1, 2}), vec([]float64{1}), nil, 1, 0); err == nil {
		t.Error("shape mismatch (preferred/dispreferred): want error")
	}
	if _, err := nn.SLiCLoss(backend.NewContext(), vec([]float64{1, 2}), vec([]float64{1, 2}), vec([]float64{1}), 1, 0.1); err == nil {
		t.Error("shape mismatch (sftTarget): want error")
	}
}

func ExampleSLiCLoss() {
	// Preferred candidate lags the dispreferred one, so the margin is violated and the hinge
	// pays; a bare calibration loss (no regularizer) with margin δ=1.
	pref := vec([]float64{0.0})
	dispref := vec([]float64{0.5})
	loss, _ := nn.SLiCLoss(backend.NewContext(), pref, dispref, nil, 1.0, 0)
	fmt.Printf("%.3f\n", loss.AtF64())
	// Output:
	// 1.500
}
