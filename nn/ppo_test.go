package nn_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func loadPPO(t *testing.T) (eps float64, lpNew, lpOld, adv []float64, loss float64) {
	t.Helper()
	raw, err := os.ReadFile("testdata/ppo.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g struct {
		Eps   float64   `json:"epsilon"`
		LpNew []float64 `json:"lp_new"`
		LpOld []float64 `json:"lp_old"`
		Adv   []float64 `json:"adv"`
		Loss  float64   `json:"loss"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g.Eps, g.LpNew, g.LpOld, g.Adv, g.Loss
}

// §V1 + §V16: the fused PPO clipped loss matches an INDEPENDENT numpy computation
// of Schulman et al. 2017 Eq. 7 at f64 rtol 1e-12.
func TestPPOParity(t *testing.T) {
	eps, lpNew, lpOld, adv, want := loadPPO(t)
	got, err := nn.PPOClipLoss(backend.NewContext(), vec(lpNew), vec(lpOld), vec(adv), eps)
	if err != nil {
		t.Fatal(err)
	}
	if v := got.AtF64(); math.Abs(v-want) > 1e-12*math.Max(1, math.Abs(want)) {
		t.Errorf("PPO loss = %.17g, want %.17g", v, want)
	}
}

// §V2: central finite differences of the PPO loss w.r.t. the policy log-probs
// match the analytic VJP; the frozen behaviour log-probs and advantages get no
// gradient. Inputs are chosen away from the clip kinks (r not near 1±ε).
func TestPPOGradCheck(t *testing.T) {
	eps := 0.2
	// r = exp(lpNew-lpOld) = [1.051, 0.951, 1.492, 0.670, 1.0]: two in-range, one
	// clamped-high, one clamped-low, one at 1 — all far from the 0.8/1.2 kinks.
	lpNew := []float64{0.05, -0.05, 0.4, -0.4, 0.0}
	lpOld := []float64{0, 0, 0, 0, 0}
	adv := []float64{1.0, -1.0, 1.5, 2.0, -0.5}
	oldT, advT := vec(lpOld), vec(adv)

	lossAt := func(d []float64) float64 {
		out, err := nn.PPOClipLoss(backend.NewContext(), vec(d), oldT, advT, eps)
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}

	newT := vec(lpNew)
	tape := autograd.NewTape()
	loss, err := nn.PPOClipLoss(tape.Context(), newT, oldT, advT, eps)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	if tape.Grad(oldT) != nil || tape.Grad(advT) != nil {
		t.Error("behaviour log-probs and advantages must be frozen (nil gradient)")
	}
	g := tape.Grad(newT)
	if g == nil {
		t.Fatal("policy log-probs must receive gradients")
	}
	const h = 1e-6
	work := append([]float64(nil), lpNew...)
	for i := range lpNew {
		o := work[i]
		work[i] = o + h
		lp := lossAt(work)
		work[i] = o - h
		lm := lossAt(work)
		work[i] = o
		num := (lp - lm) / (2 * h)
		if ana := g.AtF64(i); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", i, num, ana)
		}
	}
}

// §R49: the clip forms a trust region — when the ratio leaves [1−ε,1+ε] in the
// pessimistic direction (A>0 & r>1+ε, or A<0 & r<1−ε) the gradient is ZERO, so
// one batch cannot push the policy further. In-range ratios keep a live gradient.
func TestPPOClipTrustRegion(t *testing.T) {
	eps := 0.2
	// r = exp(lpNew): [e^1≈2.72 (>1.2), e^-1≈0.37 (<0.8), 1.0 (in), 1.0 (in)]
	lpNew := vec([]float64{1.0, -1.0, 0.0, 0.0})
	lpOld := vec([]float64{0, 0, 0, 0})
	adv := vec([]float64{1.0, -1.0, 1.0, -1.0})

	tape := autograd.NewTape()
	loss, err := nn.PPOClipLoss(tape.Context(), lpNew, lpOld, adv, eps)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(lpNew)
	if g.AtF64(0) != 0 {
		t.Errorf("A>0, r>1+ε must have zero grad (clamped), got %g", g.AtF64(0))
	}
	if g.AtF64(1) != 0 {
		t.Errorf("A<0, r<1−ε must have zero grad (clamped), got %g", g.AtF64(1))
	}
	if g.AtF64(2) == 0 || g.AtF64(3) == 0 {
		t.Errorf("in-range ratios must keep a live gradient, got %g, %g", g.AtF64(2), g.AtF64(3))
	}
}

// End-to-end: minimizing the PPO loss improves the policy — for a positive
// advantage the policy log-prob rises (until the clip caps it), and the loss
// falls. Frozen rollout log-probs and advantages.
func TestPPOTrainsPolicy(t *testing.T) {
	eps := 0.2
	lpNew := tensor.FromFloat64(tensor.Shape{2}, []float64{0.0, 0.0})
	lpOld := tensor.FromFloat64(tensor.Shape{2}, []float64{0.0, 0.0})
	adv := tensor.FromFloat64(tensor.Shape{2}, []float64{1.0, 1.0}) // both favourable
	opt := nn.NewAdam([]*tensor.Tensor{lpNew}, 0.02)

	var first, last float64
	start0 := lpNew.AtF64(0)
	for step := range 50 {
		tape := autograd.NewTape()
		loss, err := nn.PPOClipLoss(tape.Context(), lpNew, lpOld, adv, eps)
		if err != nil {
			t.Fatal(err)
		}
		if step == 0 {
			first = loss.AtF64()
		}
		last = loss.AtF64()
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	if last >= first {
		t.Fatalf("PPO did not reduce loss: %.4f → %.4f", first, last)
	}
	if lpNew.AtF64(0) <= start0 {
		t.Fatalf("PPO should raise log-prob for a favourable action: %.4f → %.4f", start0, lpNew.AtF64(0))
	}
	t.Logf("PPO improved: loss %.4f→%.4f, logπ %.4f→%.4f (clip caps at ratio 1+ε)", first, last, start0, lpNew.AtF64(0))
}
