package nn_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type grpoGolden struct {
	Eps      float64   `json:"epsilon"`
	Beta     float64   `json:"beta"`
	Rewards  []float64 `json:"rewards"`
	AdvGroup []float64 `json:"adv_group"`
	LpNew    []float64 `json:"lp_new"`
	LpOld    []float64 `json:"lp_old"`
	LpRef    []float64 `json:"lp_ref"`
	Adv      []float64 `json:"adv"`
	Loss     float64   `json:"loss"`
}

func loadGRPO(t *testing.T) grpoGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/grpo.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g grpoGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// §V16 tier-1: the group-relative advantage matches the independent numpy
// reference (population std + eps) at f64 1e-12, and the advantages are centered.
func TestGroupAdvantageParity(t *testing.T) {
	g := loadGRPO(t)
	adv := nn.GroupAdvantage(g.Rewards)
	var sum float64
	for i, want := range g.AdvGroup {
		if math.Abs(adv[i]-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Errorf("adv[%d] = %.17g, want %.17g", i, adv[i], want)
		}
		sum += adv[i]
	}
	if math.Abs(sum) > 1e-12 {
		t.Errorf("group advantages must be centered (mean 0), got sum %.3g", sum)
	}
	// all-equal rewards → no signal
	for _, a := range nn.GroupAdvantage([]float64{2, 2, 2}) {
		if a != 0 {
			t.Errorf("all-equal rewards must give zero advantage, got %v", a)
		}
	}
}

// §V16 tier-1: the fused GRPO loss matches the independent numpy computation of
// DeepSeekMath Eqs. 3-4 (clipped surrogate − β·k3-KL) at f64 1e-12.
func TestGRPOParity(t *testing.T) {
	g := loadGRPO(t)
	got, err := nn.GRPOLoss(backend.NewContext(), vec(g.LpNew), vec(g.LpOld), vec(g.LpRef), vec(g.Adv),
		nn.WithClipEpsilon(g.Eps), nn.WithKLBeta(g.Beta))
	if err != nil {
		t.Fatal(err)
	}
	if v := got.AtF64(); math.Abs(v-g.Loss) > 1e-12*math.Max(1, math.Abs(g.Loss)) {
		t.Errorf("GRPO loss = %.17g, want %.17g", v, g.Loss)
	}
}

// §V2: central finite differences of the GRPO loss w.r.t. the policy log-probs
// match the analytic VJP (surrogate gradient + k3-KL term); the rollout log-probs,
// reference log-probs and advantages are frozen. Inputs are away from clip kinks.
func TestGRPOGradCheck(t *testing.T) {
	eps, beta := 0.2, 0.04
	lpNew := []float64{0.05, -0.05, 0.4, -0.4, 0.0}
	lpOld := []float64{0, 0, 0, 0, 0}
	lpRef := []float64{0.1, -0.1, 0.2, -0.2, 0.05}
	adv := []float64{1.0, -1.0, 1.5, 2.0, -0.5}
	oldT, refT, advT := vec(lpOld), vec(lpRef), vec(adv)

	lossAt := func(d []float64) float64 {
		out, err := nn.GRPOLoss(backend.NewContext(), vec(d), oldT, refT, advT,
			nn.WithClipEpsilon(eps), nn.WithKLBeta(beta))
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}

	newT := vec(lpNew)
	tape := autograd.NewTape()
	loss, err := nn.GRPOLoss(tape.Context(), newT, oldT, refT, advT,
		nn.WithClipEpsilon(eps), nn.WithKLBeta(beta))
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	if tape.Grad(oldT) != nil || tape.Grad(refT) != nil || tape.Grad(advT) != nil {
		t.Error("rollout/reference log-probs and advantages must be frozen (nil gradient)")
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

// GRPO (DeepSeekMath, the RL method behind DeepSeek-R1) needs no value network:
// it scores several sampled answers to a prompt with a reward model and turns the
// rewards into a group-relative advantage by centering and normalizing them. The
// best answer gets the largest positive advantage, the worst the most negative,
// and the advantages average to zero — that is the entire learning signal.
func ExampleGroupAdvantage() {
	adv := nn.GroupAdvantage([]float64{1, 0, 0.5, -0.5}) // 4 sampled answers' rewards
	fmt.Printf("%.3f %.3f %.3f %.3f\n", adv[0], adv[1], adv[2], adv[3])
	// Output: 1.341 -0.447 0.447 -1.341
}
