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

// §V16 tier-1 (arXiv:1708.02002, γ=0): focal loss reduces EXACTLY to cross-entropy when the
// focusing parameter is 0 and α=1.
func TestFocalReducesToCE(t *testing.T) {
	logits := toT([][]float64{{2, 1, -0.5}, {0.3, 2.5, 1.0}})
	targets := vec([]float64{0, 1})
	focal, err := nn.FocalLoss(backend.NewContext(), logits, targets, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	ce, err := nn.CrossEntropy(backend.NewContext(), logits, targets)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(focal.AtF64()-ce.AtF64()) > 1e-10 {
		t.Errorf("FocalLoss(γ=0,α=1) = %.12g, want CrossEntropy %.12g", focal.AtF64(), ce.AtF64())
	}
}

// §V16 tier-1: FL = −α·(1−p_t)^γ·log(p_t). Verified against a hand computation.
func TestFocalParity(t *testing.T) {
	logits := toT([][]float64{{2, 1}})
	targets := vec([]float64{0})
	const gamma, alpha = 2.0, 0.25
	loss, err := nn.FocalLoss(backend.NewContext(), logits, targets, gamma, alpha)
	if err != nil {
		t.Fatal(err)
	}
	pt := math.Exp(2) / (math.Exp(2) + math.Exp(1))
	want := -alpha * math.Pow(1-pt, gamma) * math.Log(pt)
	if math.Abs(loss.AtF64()-want) > 1e-12 {
		t.Errorf("focal loss = %.12g, want %.12g", loss.AtF64(), want)
	}
}

// §V16 tier-1 (down-weighting hard vs easy): the modulating factor makes a confident-correct
// example contribute far less than an uncertain one for γ>0.
func TestFocalDownweightsEasy(t *testing.T) {
	easy, _ := nn.FocalLoss(backend.NewContext(), toT([][]float64{{6, 0}}), vec([]float64{0}), 2, 1)   // p_t≈0.9975
	hard, _ := nn.FocalLoss(backend.NewContext(), toT([][]float64{{0.2, 0}}), vec([]float64{0}), 2, 1) // p_t≈0.55
	if !(easy.AtF64() < 0.05*hard.AtF64()) {
		t.Errorf("easy example not down-weighted: easy %.6g vs hard %.6g", easy.AtF64(), hard.AtF64())
	}
	// with γ=0 (plain CE) the modulation is gone, so the ratio is far smaller.
	easyCE, _ := nn.FocalLoss(backend.NewContext(), toT([][]float64{{6, 0}}), vec([]float64{0}), 0, 1)
	hardCE, _ := nn.FocalLoss(backend.NewContext(), toT([][]float64{{0.2, 0}}), vec([]float64{0}), 0, 1)
	if easyCE.AtF64()/hardCE.AtF64() < easy.AtF64()/hard.AtF64() {
		t.Error("γ=2 should down-weight the easy example MORE than γ=0")
	}
}

// §V16 tier-1 (α scaling): the loss is linear in α.
func TestFocalAlphaScales(t *testing.T) {
	logits := toT([][]float64{{2, 1, 0}})
	targets := vec([]float64{2})
	a1, _ := nn.FocalLoss(backend.NewContext(), logits, targets, 2, 1)
	a2, _ := nn.FocalLoss(backend.NewContext(), logits, targets, 2, 0.5)
	if math.Abs(a1.AtF64()-2*a2.AtF64()) > 1e-12 {
		t.Errorf("α=1 loss %.10g != 2×(α=0.5 loss %.10g)", a1.AtF64(), a2.AtF64())
	}
}

// §V-GRAD: finite-difference check of the gradient into the logits.
func TestFocalGradCheck(t *testing.T) {
	logits := toT([][]float64{{1.0, -0.5, 0.3}, {0.2, 1.5, -1.0}})
	targets := vec([]float64{0, 2})
	const gamma, alpha = 2.0, 0.25
	lossAt := func() float64 {
		out, err := nn.FocalLoss(backend.NewContext(), logits, targets, gamma, alpha)
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}
	tape := autograd.NewTape()
	loss, err := nn.FocalLoss(tape.Context(), logits, targets, gamma, alpha)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(logits)
	const eps = 1e-6
	for i := range logits.Numel() {
		c := tensor.Unravel(i, logits.Shape())
		o := logits.AtF64(c...)
		logits.SetF64(o+eps, c...)
		lp := lossAt()
		logits.SetF64(o-eps, c...)
		lm := lossAt()
		logits.SetF64(o, c...)
		if num, ana := (lp-lm)/(2*eps), g.AtF64(c...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("logit grad %v: numeric %.8g vs analytic %.8g", c, num, ana)
		}
	}
}

func TestFocalErrors(t *testing.T) {
	if _, err := nn.FocalLoss(backend.NewContext(), vec([]float64{1, 2}), vec([]float64{0}), 2, 1); err == nil {
		t.Error("rank-1 logits: want error")
	}
	if _, err := nn.FocalLoss(backend.NewContext(), toT([][]float64{{1, 2}}), vec([]float64{5}), 2, 1); err == nil {
		t.Error("target out of range: want error")
	}
	if _, err := nn.FocalLoss(backend.NewContext(), toT([][]float64{{1, 2}}), vec([]float64{0, 1}), 2, 1); err == nil {
		t.Error("batch mismatch: want error")
	}
}

func ExampleFocalLoss() {
	// A confident-correct example (logit 6 vs 0, p_t≈0.9975) incurs almost no focal loss at γ=2.
	loss, _ := nn.FocalLoss(backend.NewContext(), toT([][]float64{{6, 0}}), vec([]float64{0}), 2, 0.25)
	fmt.Printf("%.5f\n", loss.AtF64())
	// Output:
	// 0.00000
}

// softplus is a stable log(1+e^x) for the test references.
func softplusRef(x float64) float64 { return math.Log1p(math.Exp(-math.Abs(x))) + math.Max(x, 0) }

// §V16 tier-1 (§R228, γ=0): the binary focal loss reduces to (α-disabled) binary cross-entropy
// mean(softplus((1−2y)·x)).
func TestSigmoidFocalReducesToBCE(t *testing.T) {
	logits := toT([][]float64{{0.5, -1.0}, {2.0, -0.3}})
	targets := toT([][]float64{{1, 0}, {0, 1}})
	loss, err := nn.SigmoidFocalLoss(backend.NewContext(), logits, targets, 0, -1) // γ=0, α disabled
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	for _, r := range [][2]float64{{0.5, 1}, {-1.0, 0}, {2.0, 0}, {-0.3, 1}} {
		sum += softplusRef((1 - 2*r[1]) * r[0]) // BCE per element
	}
	if want := sum / 4; math.Abs(loss.AtF64()-want) > 1e-12 {
		t.Errorf("γ=0 sigmoid focal = %.12g, want BCE %.12g", loss.AtF64(), want)
	}
}

// §V16 tier-1 (Eq.4/5): FL = α_t·softplus(z)·exp(−γ·softplus(−z)), z=(1−2y)x. Hand check for a
// single positive element x=2, y=1, γ=2, α=0.25.
func TestSigmoidFocalParity(t *testing.T) {
	loss, err := nn.SigmoidFocalLoss(backend.NewContext(), toT([][]float64{{2}}), toT([][]float64{{1}}), 2, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	z := -2.0 // (1−2)·2
	want := 0.25 * softplusRef(z) * math.Exp(-2*softplusRef(-z))
	if math.Abs(loss.AtF64()-want) > 1e-12 {
		t.Errorf("sigmoid focal = %.12g, want %.12g", loss.AtF64(), want)
	}
}

// §V16 tier-1 (α-balance): α_t weights positives by α and negatives by 1−α (relative to the
// α-disabled loss).
func TestSigmoidFocalAlphaBalance(t *testing.T) {
	pos, posN := toT([][]float64{{0.7}}), toT([][]float64{{1}})
	neg, negN := toT([][]float64{{0.7}}), toT([][]float64{{0}})
	base := func(x, y *tensor.Tensor) float64 {
		l, _ := nn.SigmoidFocalLoss(backend.NewContext(), x, y, 2, -1)
		return l.AtF64()
	}
	weighted := func(x, y *tensor.Tensor, a float64) float64 {
		l, _ := nn.SigmoidFocalLoss(backend.NewContext(), x, y, 2, a)
		return l.AtF64()
	}
	if got, want := weighted(pos, posN, 0.25), 0.25*base(pos, posN); math.Abs(got-want) > 1e-12 {
		t.Errorf("positive α weighting: %.10g, want 0.25×base %.10g", got, want)
	}
	if got, want := weighted(neg, negN, 0.25), 0.75*base(neg, negN); math.Abs(got-want) > 1e-12 {
		t.Errorf("negative α weighting: %.10g, want 0.75×base %.10g", got, want)
	}
}

// §V16 tier-1 (down-weighting): a confident-correct element loses far less than an uncertain one.
func TestSigmoidFocalDownweightsEasy(t *testing.T) {
	easy, _ := nn.SigmoidFocalLoss(backend.NewContext(), toT([][]float64{{6}}), toT([][]float64{{1}}), 2, 0.25) // p≈0.9975
	hard, _ := nn.SigmoidFocalLoss(backend.NewContext(), toT([][]float64{{0.2}}), toT([][]float64{{1}}), 2, 0.25)
	if !(easy.AtF64() < 0.02*hard.AtF64()) {
		t.Errorf("easy not down-weighted: easy %.6g vs hard %.6g", easy.AtF64(), hard.AtF64())
	}
}

// §V-GRAD: finite-difference check of the gradient into the logits.
func TestSigmoidFocalGradCheck(t *testing.T) {
	logits := toT([][]float64{{0.5, -1.2, 0.3}, {1.5, -0.4, 0.8}})
	targets := toT([][]float64{{1, 0, 1}, {0, 1, 0}})
	const gamma, alpha = 2.0, 0.25
	lossAt := func() float64 {
		out, err := nn.SigmoidFocalLoss(backend.NewContext(), logits, targets, gamma, alpha)
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}
	tape := autograd.NewTape()
	loss, err := nn.SigmoidFocalLoss(tape.Context(), logits, targets, gamma, alpha)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(logits)
	const eps = 1e-6
	for i := range logits.Numel() {
		c := tensor.Unravel(i, logits.Shape())
		o := logits.AtF64(c...)
		logits.SetF64(o+eps, c...)
		lp := lossAt()
		logits.SetF64(o-eps, c...)
		lm := lossAt()
		logits.SetF64(o, c...)
		if num, ana := (lp-lm)/(2*eps), g.AtF64(c...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("logit grad %v: numeric %.8g vs analytic %.8g", c, num, ana)
		}
	}
}

func TestSigmoidFocalErrors(t *testing.T) {
	if _, err := nn.SigmoidFocalLoss(backend.NewContext(), toT([][]float64{{1, 2}}), toT([][]float64{{1}}), 2, 0.25); err == nil {
		t.Error("shape mismatch: want error")
	}
}

func ExampleSigmoidFocalLoss() {
	// A confident-correct positive (logit 6, label 1) incurs almost no loss at γ=2.
	loss, _ := nn.SigmoidFocalLoss(backend.NewContext(), toT([][]float64{{6}}), toT([][]float64{{1}}), 2, 0.25)
	fmt.Printf("%.5f\n", loss.AtF64())
	// Output:
	// 0.00000
}
