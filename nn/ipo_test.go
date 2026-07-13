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

func loadIPO(t *testing.T) (beta float64, pc, pl, rc, rl []float64, loss float64) {
	t.Helper()
	raw, err := os.ReadFile("testdata/ipo.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g struct {
		Beta float64   `json:"beta"`
		PC   []float64 `json:"pc"`
		PL   []float64 `json:"pl"`
		RC   []float64 `json:"rc"`
		RL   []float64 `json:"rl"`
		Loss float64   `json:"loss"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g.Beta, g.PC, g.PL, g.RC, g.RL, g.Loss
}

// §V1 + §V16: the fused IPO loss matches an INDEPENDENT numpy computation of
// (h − 1/(2β))² (Azar et al. 2023) at f64 rtol 1e-12.
func TestIPOParity(t *testing.T) {
	beta, pc, pl, rc, rl, want := loadIPO(t)
	got, err := nn.IPO(backend.NewContext(), vec(pc), vec(pl), vec(rc), vec(rl), nn.Beta(beta))
	if err != nil {
		t.Fatal(err)
	}
	if v := got.AtF64(); math.Abs(v-want) > 1e-12*math.Max(1, math.Abs(want)) {
		t.Errorf("IPO loss = %.17g, want %.17g", v, want)
	}
}

// §V2: central finite differences of the IPO loss w.r.t. the policy log-probs
// match the analytic VJP; the frozen reference gets no gradient.
func TestIPOGradCheck(t *testing.T) {
	beta, pc, pl, rc, rl, _ := loadIPO(t)
	rcT, rlT := vec(rc), vec(rl)

	lossAt := func(pcd, pld []float64) float64 {
		out, err := nn.IPO(backend.NewContext(), vec(pcd), vec(pld), rcT, rlT, nn.Beta(beta))
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}

	pcT, plT := vec(pc), vec(pl)
	tape := autograd.NewTape()
	loss, err := nn.IPO(tape.Context(), pcT, plT, rcT, rlT, nn.Beta(beta))
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	if tape.Grad(rcT) != nil || tape.Grad(rlT) != nil {
		t.Error("reference log-probs must be frozen (nil gradient)")
	}
	gpc, gpl := tape.Grad(pcT), tape.Grad(plT)
	const eps = 1e-6
	check := func(name string, data []float64, analytic *tensor.Tensor, which int) {
		for i := range data {
			o := data[i]
			data[i] = o + eps
			var lp float64
			if which == 0 {
				lp = lossAt(data, pl)
			} else {
				lp = lossAt(pc, data)
			}
			data[i] = o - eps
			var lm float64
			if which == 0 {
				lm = lossAt(data, pl)
			} else {
				lm = lossAt(pc, data)
			}
			data[i] = o
			if num, ana := (lp-lm)/(2*eps), analytic.AtF64(i); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", name, i, num, ana)
			}
		}
	}
	check("policyChosen", append([]float64(nil), pc...), gpc, 0)
	check("policyRejected", append([]float64(nil), pl...), gpl, 1)
}

// The loss is exactly zero when every example's log-ratio difference equals the
// target margin 1/(2β) — IPO's fixed point (a FINITE margin, unlike DPO).
func TestIPOMinimumAtFiniteMargin(t *testing.T) {
	beta := 0.25
	target := 1.0 / (2.0 * beta) // = 2
	// choose pc so that h = (pc−rc)−(pl−rl) = target for every example
	rc := []float64{-2.0, -3.0}
	rl := []float64{-2.5, -3.5}
	pl := []float64{-2.5, -3.5}
	pc := make([]float64, 2)
	for i := range pc {
		pc[i] = target + rc[i] + (pl[i] - rl[i]) // h = target
	}
	out, err := nn.IPO(backend.NewContext(), vec(pc), vec(pl), vec(rc), vec(rl), nn.Beta(beta))
	if err != nil {
		t.Fatal(err)
	}
	if v := out.AtF64(); math.Abs(v) > 1e-12 {
		t.Errorf("loss at h=1/(2β) = %g, want 0", v)
	}
}

// End-to-end: minimizing IPO drives the chosen/rejected margin toward the finite
// target 1/(2β) (not toward infinity), and the loss falls to ~0.
func TestIPOTrainsToTargetMargin(t *testing.T) {
	beta := 0.5
	target := 1.0 / (2.0 * beta) // = 1
	pc := tensor.FromFloat64(tensor.Shape{3}, []float64{-2.0, -1.0, -3.0})
	pl := tensor.FromFloat64(tensor.Shape{3}, []float64{-2.0, -1.0, -3.0}) // start h=0
	rc := tensor.FromFloat64(tensor.Shape{3}, []float64{-2.0, -1.0, -3.0})
	rl := tensor.FromFloat64(tensor.Shape{3}, []float64{-2.0, -1.0, -3.0})
	opt := nn.NewAdam([]*tensor.Tensor{pc, pl}, 0.05)

	var first, last float64
	for step := range 400 {
		tape := autograd.NewTape()
		loss, err := nn.IPO(tape.Context(), pc, pl, rc, rl, nn.Beta(beta))
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
	if last >= 0.01*first {
		t.Fatalf("IPO did not converge: %.4f → %.4f", first, last)
	}
	// every example's margin should sit at the finite target 1/(2β)
	for i := range 3 {
		h := (pc.AtF64(i) - rc.AtF64(i)) - (pl.AtF64(i) - rl.AtF64(i))
		if math.Abs(h-target) > 0.05 {
			t.Errorf("example %d margin %.4f, want target %.4f", i, h, target)
		}
	}
	t.Logf("IPO trained to finite margin: loss %.5f → %.5f, margin → %.4f", first, last, target)
}
