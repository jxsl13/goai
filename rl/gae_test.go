package rl_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/rl"
)

func loadGAE(t *testing.T) (gamma, lam, nextValue float64, rewards, values, adv, returns []float64) {
	t.Helper()
	raw, err := os.ReadFile("testdata/gae.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g struct {
		Gamma      float64   `json:"gamma"`
		Lam        float64   `json:"lam"`
		NextValue  float64   `json:"next_value"`
		Rewards    []float64 `json:"rewards"`
		Values     []float64 `json:"values"`
		Advantages []float64 `json:"advantages"`
		Returns    []float64 `json:"returns"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g.Gamma, g.Lam, g.NextValue, g.Rewards, g.Values, g.Advantages, g.Returns
}

func noDones(n int) []bool { return make([]bool, n) }

// §V1 + §V16: the backward-recursion GAE matches an INDEPENDENT numpy forward-sum
// Â_t = Σ (γλ)^l δ_{t+l} (Schulman et al. 2016) at f64 rtol 1e-12, and returns =
// advantages + values.
func TestGAEParity(t *testing.T) {
	gamma, lam, nextValue, rewards, values, wantAdv, wantRet := loadGAE(t)
	adv, ret := rl.GAE(rewards, values, noDones(len(rewards)), nextValue, false, gamma, lam)
	for i := range wantAdv {
		if math.Abs(adv[i]-wantAdv[i]) > 1e-12*math.Max(1, math.Abs(wantAdv[i])) {
			t.Errorf("adv[%d] = %.17g, want %.17g", i, adv[i], wantAdv[i])
		}
		if math.Abs(ret[i]-wantRet[i]) > 1e-12*math.Max(1, math.Abs(wantRet[i])) {
			t.Errorf("return[%d] = %.17g, want %.17g", i, ret[i], wantRet[i])
		}
	}
}

// §V16 limiting case: λ=0 collapses GAE to the one-step TD advantage
// δ_t = r_t + γ·V(s_{t+1}) − V(s_t).
func TestGAELambda0IsTD(t *testing.T) {
	gamma := 0.9
	rewards := []float64{1.0, 0.0, 2.0}
	values := []float64{0.5, 0.7, 0.3}
	nextValue := 0.4
	adv, _ := rl.GAE(rewards, values, noDones(3), nextValue, false, gamma, 0)
	for t0 := range rewards {
		nextV := nextValue
		if t0+1 < len(rewards) {
			nextV = values[t0+1]
		}
		td := rewards[t0] + gamma*nextV - values[t0]
		if math.Abs(adv[t0]-td) > 1e-12 {
			t.Errorf("λ=0 adv[%d] = %.12g, want TD δ = %.12g", t0, adv[t0], td)
		}
	}
}

// §V16 limiting case: λ=1 collapses GAE to the Monte-Carlo advantage — the
// discounted return (with bootstrap) minus the baseline value.
func TestGAELambda1IsMonteCarlo(t *testing.T) {
	gamma := 0.9
	rewards := []float64{1.0, 0.0, 2.0, -1.0}
	values := []float64{0.5, 0.7, 0.3, 0.2}
	nextValue := 0.4
	adv, _ := rl.GAE(rewards, values, noDones(4), nextValue, false, gamma, 1)
	n := len(rewards)
	for t0 := range rewards {
		// G_t = Σ_{k=t}^{T-1} γ^{k-t} r_k + γ^{T-t}·bootstrap
		g := 0.0
		for k := t0; k < n; k++ {
			g += math.Pow(gamma, float64(k-t0)) * rewards[k]
		}
		g += math.Pow(gamma, float64(n-t0)) * nextValue
		want := g - values[t0]
		if math.Abs(adv[t0]-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Errorf("λ=1 adv[%d] = %.12g, want MC (G−V) = %.12g", t0, adv[t0], want)
		}
	}
}

// returns_t = Â_t + V(s_t) — the value-function regression targets.
func TestGAEReturnsAreAdvantagePlusValue(t *testing.T) {
	gamma, lam, nextValue, rewards, values, _, _ := loadGAE(t)
	adv, ret := rl.GAE(rewards, values, noDones(len(rewards)), nextValue, false, gamma, lam)
	for i := range ret {
		if math.Abs(ret[i]-(adv[i]+values[i])) > 1e-12 {
			t.Errorf("return[%d] != adv+value", i)
		}
	}
}

// A terminal step (done after it) resets the accumulator: its advantage is just
// its own TD residual with NO bootstrap, and no future residual leaks back past
// the boundary.
func TestGAEDonesTruncate(t *testing.T) {
	gamma, lam := 0.99, 0.95
	rewards := []float64{1.0, 5.0, 2.0}
	values := []float64{0.5, 0.6, 0.4}
	dones := []bool{false, true, false} // episode ends after step 1
	adv, _ := rl.GAE(rewards, values, dones, 0.3, false, gamma, lam)

	// step 1 is terminal → adv[1] = r_1 − V_1 (no bootstrap, no successor leak)
	wantTerminal := rewards[1] - values[1]
	if math.Abs(adv[1]-wantTerminal) > 1e-12 {
		t.Errorf("terminal adv[1] = %.12g, want r−V = %.12g (no bootstrap)", adv[1], wantTerminal)
	}
	// step 0's advantage must not depend on step 2 (across the boundary): it is
	// δ_0 + γλ·adv[1] with nonterminal_0 = 1 (s_1 is not terminal; step 1 is).
	delta0 := rewards[0] + gamma*values[1] - values[0]
	want0 := delta0 + gamma*lam*adv[1]
	if math.Abs(adv[0]-want0) > 1e-12 {
		t.Errorf("adv[0] = %.12g, want δ0+γλ·adv1 = %.12g", adv[0], want0)
	}
}
