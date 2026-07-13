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

func loadKTO(t *testing.T) (beta, zRef, lamD, lamU float64, pl, rl, labels []float64, loss float64) {
	t.Helper()
	raw, err := os.ReadFile("testdata/kto.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g struct {
		Beta   float64   `json:"beta"`
		ZRef   float64   `json:"z_ref"`
		LamD   float64   `json:"lambda_d"`
		LamU   float64   `json:"lambda_u"`
		PL     []float64 `json:"pl"`
		RL     []float64 `json:"rl"`
		Labels []float64 `json:"labels"`
		Loss   float64   `json:"loss"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g.Beta, g.ZRef, g.LamD, g.LamU, g.PL, g.RL, g.Labels, g.Loss
}

// §V1 + §V16: the fused KTO loss matches an INDEPENDENT numpy computation of the
// desirable/undesirable σ(±(r−z_ref)) losses (Ethayarajh et al. 2024) @ f64 1e-12.
func TestKTOParity(t *testing.T) {
	beta, zRef, lamD, lamU, pl, rl, labels, want := loadKTO(t)
	got, err := nn.KTO(backend.NewContext(), vec(pl), vec(rl), vec(labels), nn.Beta(beta), nn.ReferencePoint(zRef), nn.DesirableWeight(lamD), nn.UndesirableWeight(lamU))
	if err != nil {
		t.Fatal(err)
	}
	if v := got.AtF64(); math.Abs(v-want) > 1e-12*math.Max(1, math.Abs(want)) {
		t.Errorf("KTO loss = %.17g, want %.17g", v, want)
	}
}

// §V2: central finite differences of the KTO loss w.r.t. the policy log-probs
// match the analytic VJP; the frozen reference and labels get no gradient.
func TestKTOGradCheck(t *testing.T) {
	beta, zRef, lamD, lamU, pl, rl, labels, _ := loadKTO(t)
	rlT, labT := vec(rl), vec(labels)

	lossAt := func(pld []float64) float64 {
		out, err := nn.KTO(backend.NewContext(), vec(pld), rlT, labT, nn.Beta(beta), nn.ReferencePoint(zRef), nn.DesirableWeight(lamD), nn.UndesirableWeight(lamU))
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}

	plT := vec(pl)
	tape := autograd.NewTape()
	loss, err := nn.KTO(tape.Context(), plT, rlT, labT, nn.Beta(beta), nn.ReferencePoint(zRef), nn.DesirableWeight(lamD), nn.UndesirableWeight(lamU))
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	if tape.Grad(rlT) != nil || tape.Grad(labT) != nil {
		t.Error("reference log-probs and labels must be frozen (nil gradient)")
	}
	gp := tape.Grad(plT)
	const eps = 1e-6
	work := append([]float64(nil), pl...)
	for i := range pl {
		o := work[i]
		work[i] = o + eps
		lp := lossAt(work)
		work[i] = o - eps
		lm := lossAt(work)
		work[i] = o
		if num, ana := (lp-lm)/(2*eps), gp.AtF64(i); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", i, num, ana)
		}
	}
}

// Directionality: raising a DESIRABLE example's log-prob (reward up) lowers its
// loss; raising an UNDESIRABLE example's log-prob raises its loss.
func TestKTODirectionality(t *testing.T) {
	rl := vec([]float64{-2.0, -2.0})
	base := []float64{-2.0, -2.0}
	up := []float64{-1.0, -1.0} // higher policy log-prob → higher reward
	labels := vec([]float64{1, 0})

	loss := func(pl []float64) float64 {
		out, _ := nn.KTO(backend.NewContext(), vec(pl), rl, labels, nn.Beta(0.5), nn.ReferencePoint(0.2))
		return out.AtF64()
	}
	// desirable-only view: compare loss with a raised desirable reward
	desLabels := vec([]float64{1, 1})
	desLoss := func(pl []float64) float64 {
		out, _ := nn.KTO(backend.NewContext(), vec(pl), rl, desLabels, nn.Beta(0.5), nn.ReferencePoint(0.2))
		return out.AtF64()
	}
	undLabels := vec([]float64{0, 0})
	undLoss := func(pl []float64) float64 {
		out, _ := nn.KTO(backend.NewContext(), vec(pl), rl, undLabels, nn.Beta(0.5), nn.ReferencePoint(0.2))
		return out.AtF64()
	}
	if desLoss(up) >= desLoss(base) {
		t.Error("raising a desirable reward must lower KTO loss")
	}
	if undLoss(up) <= undLoss(base) {
		t.Error("raising an undesirable reward must raise KTO loss")
	}
	_ = loss
}

// End-to-end: KTO training raises desirable rewards and lowers undesirable ones.
func TestKTOTrainsRewards(t *testing.T) {
	pl := tensor.FromFloat64(tensor.Shape{4}, []float64{-2.0, -2.0, -2.0, -2.0})
	rl := tensor.FromFloat64(tensor.Shape{4}, []float64{-2.0, -2.0, -2.0, -2.0})
	labels := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 1, 0, 0}) // 2 desirable, 2 undesirable
	opt := nn.NewAdam([]*tensor.Tensor{pl}, 0.05)

	reward := func(i int) float64 { return 0.5 * (pl.AtF64(i) - rl.AtF64(i)) }
	var first, last float64
	for step := range 200 {
		tape := autograd.NewTape()
		loss, err := nn.KTO(tape.Context(), pl, rl, labels, nn.Beta(0.5))
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
		t.Fatalf("KTO did not reduce loss: %.4f → %.4f", first, last)
	}
	// desirable rewards up (>0), undesirable down (<0)
	if reward(0) <= 0 || reward(1) <= 0 {
		t.Errorf("desirable rewards should be positive: %.3f, %.3f", reward(0), reward(1))
	}
	if reward(2) >= 0 || reward(3) >= 0 {
		t.Errorf("undesirable rewards should be negative: %.3f, %.3f", reward(2), reward(3))
	}
	t.Logf("KTO: loss %.4f→%.4f, desirable r=%.3f undesirable r=%.3f", first, last, reward(0), reward(2))
}
