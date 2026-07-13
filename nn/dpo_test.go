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

func loadDPO(t *testing.T) (beta float64, pc, pl, rc, rl []float64, loss float64) {
	t.Helper()
	raw, err := os.ReadFile("testdata/dpo.json")
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

func vec(d []float64) *tensor.Tensor { return tensor.FromFloat64(tensor.Shape{len(d)}, d) }

// §V1 + §V16: the fused DPO loss matches an INDEPENDENT numpy computation of
// Rafailov et al. 2023 Eq. 7 at f64 rtol 1e-12.
func TestDPOParity(t *testing.T) {
	beta, pc, pl, rc, rl, want := loadDPO(t)
	got, err := nn.DPO(backend.NewContext(), vec(pc), vec(pl), vec(rc), vec(rl), nn.Beta(beta))
	if err != nil {
		t.Fatal(err)
	}
	if v := got.AtF64(); math.Abs(v-want) > 1e-12*math.Max(1, math.Abs(want)) {
		t.Errorf("DPO loss = %.17g, want %.17g", v, want)
	}
}

// §V2: central finite differences of the DPO loss w.r.t. the POLICY log-probs
// match the analytic VJP; the FROZEN reference inputs get no gradient.
func TestDPOGradCheck(t *testing.T) {
	beta, pc, pl, rc, rl, _ := loadDPO(t)
	rcT, rlT := vec(rc), vec(rl)

	lossAt := func(pcd, pld []float64) float64 {
		out, err := nn.DPO(backend.NewContext(), vec(pcd), vec(pld), rcT, rlT, nn.Beta(beta))
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}

	// analytic grads via the tape
	pcT, plT := vec(pc), vec(pl)
	tape := autograd.NewTape()
	loss, err := nn.DPO(tape.Context(), pcT, plT, rcT, rlT, nn.Beta(beta))
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
	if gpc == nil || gpl == nil {
		t.Fatal("policy log-probs must receive gradients")
	}

	const eps = 1e-6
	check := func(name string, data []float64, analytic *tensor.Tensor, which int) {
		for i := range data {
			orig := data[i]
			data[i] = orig + eps
			var lp float64
			if which == 0 {
				lp = lossAt(data, pl)
			} else {
				lp = lossAt(pc, data)
			}
			data[i] = orig - eps
			var lm float64
			if which == 0 {
				lm = lossAt(data, pl)
			} else {
				lm = lossAt(pc, data)
			}
			data[i] = orig
			num := (lp - lm) / (2 * eps)
			ana := analytic.AtF64(i)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", name, i, num, ana)
			}
		}
	}
	check("policyChosen", append([]float64(nil), pc...), gpc, 0)
	check("policyRejected", append([]float64(nil), pl...), gpl, 1)
}

// §V12: extreme margins never overflow — the stable softplus keeps the loss
// finite for both a huge win and a huge loss.
func TestDPOStability(t *testing.T) {
	big := []float64{1000, -1000}
	small := []float64{0, 0}
	for _, tc := range []struct {
		name           string
		pc, pl, rc, rl []float64
	}{
		{"huge-correct", big, small, small, small}, // Δ ≫ 0 → loss ≈ 0
		{"huge-wrong", small, big, small, small},   // Δ ≪ 0 → loss large but finite
	} {
		out, err := nn.DPO(backend.NewContext(), vec(tc.pc), vec(tc.pl), vec(tc.rc), vec(tc.rl), nn.Beta(1.0))
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if v := out.AtF64(); math.IsInf(v, 0) || math.IsNaN(v) || v < 0 {
			t.Errorf("%s: loss %v not finite/non-negative", tc.name, v)
		}
	}
}

// End-to-end: optimizing the policy log-probs on the DPO loss aligns the policy —
// the chosen/rejected margin grows and the loss falls (RLHF-free preference
// training, reference held frozen).
func TestDPOTrainsPreference(t *testing.T) {
	beta := 0.5
	pc := tensor.FromFloat64(tensor.Shape{4}, []float64{-2.0, -3.5, -1.0, -4.0})
	pl := tensor.FromFloat64(tensor.Shape{4}, []float64{-2.0, -3.5, -1.0, -4.0}) // start equal
	rc := tensor.FromFloat64(tensor.Shape{4}, []float64{-2.0, -3.5, -1.0, -4.0})
	rl := tensor.FromFloat64(tensor.Shape{4}, []float64{-2.0, -3.5, -1.0, -4.0})

	marginOf := func() float64 {
		var s float64
		for i := range 4 {
			s += pc.AtF64(i) - pl.AtF64(i)
		}
		return s / 4
	}
	opt := nn.NewAdam([]*tensor.Tensor{pc, pl}, 0.05)

	var first, last float64
	m0 := marginOf()
	for step := range 100 {
		tape := autograd.NewTape()
		loss, err := nn.DPO(tape.Context(), pc, pl, rc, rl, nn.Beta(beta))
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
		t.Fatalf("DPO did not reduce loss: %.4f → %.4f", first, last)
	}
	if marginOf() <= m0 {
		t.Fatalf("DPO did not grow the chosen/rejected margin: %.4f → %.4f", m0, marginOf())
	}
	t.Logf("DPO aligned: loss %.4f→%.4f, margin %.4f→%.4f", first, last, m0, marginOf())
}
