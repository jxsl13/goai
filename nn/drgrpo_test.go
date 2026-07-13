package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/nn"
)

// §V16 tier-1 (§R95): the Dr. GRPO advantage is the mean-centered reward with NO std
// normalization; it sums to zero and a single reward gives zero.
func TestDrGRPOAdvantage(t *testing.T) {
	r := []float64{3, 1, 2, 0}
	got := nn.DrGRPOAdvantage(r)
	mean := (3.0 + 1 + 2 + 0) / 4
	var sum float64
	for i, v := range r {
		if math.Abs(got[i]-(v-mean)) > 1e-12 {
			t.Errorf("Â[%d] = %v, want %v", i, got[i], v-mean)
		}
		sum += got[i]
	}
	if math.Abs(sum) > 1e-12 {
		t.Errorf("advantages must sum to 0, got %v", sum)
	}
	if a := nn.DrGRPOAdvantage([]float64{5}); a[0] != 0 {
		t.Errorf("single reward → 0 advantage, got %v", a[0])
	}
}

// §V16 tier-1: Dr. GRPO fixes GRPO's question-level DIFFICULTY BIAS. Two groups with
// the same centering but 100× different reward spread: Dr. GRPO preserves the spread
// (the low-variance group keeps ~100× smaller advantages), while GRPO's /std rescales
// both to ~unit magnitude, amplifying the low-variance group's gradient ~100×.
func TestDrGRPONoDifficultyBias(t *testing.T) {
	high := []float64{2, 0}      // centered ±1, std 1
	low := []float64{1.01, 0.99} // centered ±0.01, std 0.01

	drHigh, drLow := nn.DrGRPOAdvantage(high), nn.DrGRPOAdvantage(low)
	grHigh, grLow := nn.GroupAdvantage(high), nn.GroupAdvantage(low)

	drRatio := math.Abs(drLow[0]) / math.Abs(drHigh[0]) // ≈ 0.01 (spread preserved)
	grRatio := math.Abs(grLow[0]) / math.Abs(grHigh[0]) // ≈ 0.99 (std bias amplifies)
	if drRatio > 0.02 {
		t.Errorf("Dr. GRPO should preserve the 100× reward-spread gap, ratio %.4f", drRatio)
	}
	if grRatio < 0.9 {
		t.Errorf("GRPO's /std should amplify the low-variance group to ~unit scale, ratio %.4f", grRatio)
	}
}

// §V16 tier-1: the Dr. GRPO advantage composes with GRPOLoss (with β=0, as the paper
// uses) — the loss is finite and differentiable through the policy log-probs, so
// Dr. GRPO trains end-to-end.
func TestDrGRPOWithGRPOLoss(t *testing.T) {
	// four tokens: two responses of a group, each token carries its response advantage
	adv := nn.DrGRPOAdvantage([]float64{2, 0}) // → [1, -1]
	advTok := []float64{adv[0], adv[0], adv[1], adv[1]}
	logpNew := []float64{-0.5, -0.8, -1.0, -0.6}
	logpOld := []float64{-0.6, -0.7, -1.1, -0.5}
	logpRef := []float64{-0.55, -0.75, -1.05, -0.55}

	newT := vec(logpNew)
	tape := autograd.NewTape()
	loss, err := nn.GRPOLoss(tape.Context(), newT, vec(logpOld), vec(logpRef), vec(advTok), nn.WithKLBeta(0))
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(loss.AtF64()) || math.IsInf(loss.AtF64(), 0) {
		t.Fatalf("Dr. GRPO loss not finite: %v", loss.AtF64())
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	if g := tape.Grad(newT); g == nil {
		t.Fatal("policy log-probs received no gradient")
	}
}

// Dr. GRPO centers the group's rewards without GRPO's divide-by-std, so a batch of
// answers keeps its raw reward spread as the learning signal (sum zero).
func ExampleDrGRPOAdvantage() {
	adv := nn.DrGRPOAdvantage([]float64{1, 0, 0.5, -0.5})
	fmt.Printf("%.3f %.3f %.3f %.3f\n", adv[0], adv[1], adv[2], adv[3])
	// Output: 0.750 -0.250 0.250 -0.750
}
