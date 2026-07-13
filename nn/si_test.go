package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// step sets the parameter to newVals and folds one optimizer step (with task-loss gradient
// grad) into the SI accumulator — the (set θ, then Accumulate) order a real loop uses.
func siStep(t *testing.T, si *nn.SI, p *tensor.Tensor, newVals, grad []float64) {
	t.Helper()
	for i, v := range newVals {
		p.SetF64(v, i)
	}
	if err := si.Accumulate([]*tensor.Tensor{vec(grad)}); err != nil {
		t.Fatal(err)
	}
}

// §V16 tier-1 (arXiv:1703.04200 Eq.3/4): ω_k = Σ −g_k·Δθ_k, then Ω_k = ω_k/((Δθ_k^task)²+ξ).
// Verified against a hand-computed 2-parameter, 2-step trajectory (one Ω is negative — the
// path integral's sign must be preserved).
func TestSIParity(t *testing.T) {
	p := vec([]float64{0, 0})
	si := nn.NewSI([]*tensor.Tensor{p})
	// step 1: θ 0→[0.5,−0.5] (Δθ=[0.5,−0.5]), g=[−2,1] ⇒ ω += [1.0, 0.5]
	siStep(t, si, p, []float64{0.5, -0.5}, []float64{-2, 1})
	// step 2: θ→[1.0,0.0] (Δθ=[0.5,0.5]), g=[−1,2] ⇒ ω += [0.5, −1.0] ⇒ ω=[1.5,−0.5]
	siStep(t, si, p, []float64{1.0, 0.0}, []float64{-1, 2})
	si.Consolidate(0.1)
	// Δθ^task=[1.0,0.0] ⇒ Ω=[1.5/1.1, −0.5/0.1]=[1.363636…, −5.0]; anchor=[1.0,0.0]
	imp, ref := si.Importance()[0], si.RefParams()[0]
	wantImp := []float64{1.5 / 1.1, -5.0}
	wantRef := []float64{1.0, 0.0}
	for i := range wantImp {
		if math.Abs(imp.AtF64(i)-wantImp[i]) > 1e-12 {
			t.Errorf("Ω[%d]=%.12g, want %.12g", i, imp.AtF64(i), wantImp[i])
		}
		if math.Abs(ref.AtF64(i)-wantRef[i]) > 1e-12 {
			t.Errorf("ref[%d]=%.12g, want %.12g", i, ref.AtF64(i), wantRef[i])
		}
	}
}

// §V16 tier-1 (accumulation across tasks): Ω sums per-task importances and the anchor advances
// to the latest weights at each Consolidate.
func TestSIMultiTask(t *testing.T) {
	p := vec([]float64{0, 0})
	si := nn.NewSI([]*tensor.Tensor{p})
	// task A: one step θ→[1,1], g=[−1,−1] ⇒ ω=[1,1]; Δθ^task=[1,1]; Ω_A=[1/1.5,1/1.5]
	siStep(t, si, p, []float64{1, 1}, []float64{-1, -1})
	si.Consolidate(0.5)
	// task B: one step θ→[2,2], g=[−1,−1] ⇒ ω=[1,1]; Δθ^task=[1,1]; Ω_B=[1/1.5,1/1.5]
	siStep(t, si, p, []float64{2, 2}, []float64{-1, -1})
	si.Consolidate(0.5)
	imp, ref := si.Importance()[0], si.RefParams()[0]
	for i := range 2 {
		if want := 2.0 / 1.5; math.Abs(imp.AtF64(i)-want) > 1e-12 {
			t.Errorf("Ω[%d]=%.12g, want %.12g (Ω_A+Ω_B)", i, imp.AtF64(i), want)
		}
		if math.Abs(ref.AtF64(i)-2.0) > 1e-12 {
			t.Errorf("anchor[%d]=%.12g, want 2.0 (latest task end)", i, ref.AtF64(i))
		}
	}
}

// §V16 tier-1 (composition): SI's Ω and anchor drop straight into EWCPenalty — zero at the
// anchor, and the hand-computed quadratic once the weights move.
func TestSIComposesWithEWCPenalty(t *testing.T) {
	p := vec([]float64{0, 0})
	si := nn.NewSI([]*tensor.Tensor{p})
	siStep(t, si, p, []float64{1, 1}, []float64{-1, -1}) // ω=[1,1]
	si.Consolidate(0.5)                                  // Ω=[1/1.5,1/1.5], anchor=[1,1]

	// at θ = anchor the penalty is exactly 0.
	atAnchor := vec([]float64{1, 1})
	z, err := nn.EWCPenalty(backend.NewContext(), []*tensor.Tensor{atAnchor}, si.RefParams(), si.Importance(), 2.0)
	if err != nil {
		t.Fatal(err)
	}
	if z.AtF64() != 0 {
		t.Errorf("penalty at anchor = %.12g, want 0", z.AtF64())
	}
	// move to [1.3,1.4]: penalty = (λ/2)·Σ Ω·diff² = 1·(2/3)(0.09+0.16) = (2/3)·0.25
	moved := vec([]float64{1.3, 1.4})
	pen, err := nn.EWCPenalty(backend.NewContext(), []*tensor.Tensor{moved}, si.RefParams(), si.Importance(), 2.0)
	if err != nil {
		t.Fatal(err)
	}
	if want := (2.0 / 3.0) * 0.25; math.Abs(pen.AtF64()-want) > 1e-12 {
		t.Errorf("penalty = %.12g, want %.12g", pen.AtF64(), want)
	}
}

// §V2 (default damping): Consolidate(0) uses SIDefaultDamping (0.1).
func TestSIDefaultDamping(t *testing.T) {
	p := vec([]float64{0})
	si := nn.NewSI([]*tensor.Tensor{p})
	siStep(t, si, p, []float64{1}, []float64{-1}) // ω=1, Δθ^task=1
	si.Consolidate(0)                             // xi≤0 ⇒ SIDefaultDamping
	if want := 1.0 / (1.0 + nn.SIDefaultDamping); math.Abs(si.Importance()[0].AtF64(0)-want) > 1e-12 {
		t.Errorf("default-damping Ω=%.12g, want %.12g", si.Importance()[0].AtF64(0), want)
	}
}

func TestSIErrors(t *testing.T) {
	p := vec([]float64{0, 0})
	si := nn.NewSI([]*tensor.Tensor{p})
	if err := si.Accumulate([]*tensor.Tensor{vec([]float64{1}), vec([]float64{1})}); err == nil {
		t.Error("wrong grad count: want error")
	}
	if err := si.Accumulate([]*tensor.Tensor{vec([]float64{1})}); err == nil {
		t.Error("grad shape mismatch: want error")
	}
}

func ExampleSI() {
	// One task: a single step moves θ from 0 to 1 with task-loss gradient −1, so the parameter
	// helped reduce the loss and earns positive importance ω=1 ⇒ Ω=1/(1²+0.1)≈0.909.
	p := vec([]float64{0})
	si := nn.NewSI([]*tensor.Tensor{p})
	p.SetF64(1, 0)
	si.Accumulate([]*tensor.Tensor{vec([]float64{-1})})
	si.Consolidate(0.1)
	fmt.Printf("%.4f\n", si.Importance()[0].AtF64(0))
	// Output:
	// 0.9091
}
