package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// dapoRef independently recomputes the DAPO objective (surrogate − β·k3-KL, then the
// chosen reduction) so the tests pin the composed op-graph to a hand-derived value.
func dapoRef(lpNew, lpOld, lpRef, adv []float64, lengths []int, epsLow, epsHigh, beta float64, seqLevel bool) float64 {
	n := len(lpNew)
	obj := make([]float64, n)
	for i := 0; i < n; i++ {
		r := math.Exp(lpNew[i] - lpOld[i])
		rc := math.Max(1-epsLow, math.Min(1+epsHigh, r))
		surr := math.Min(r*adv[i], rc*adv[i])
		if beta != 0 {
			d := lpRef[i] - lpNew[i]
			kl := math.Exp(d) - d - 1
			surr -= beta * kl
		}
		obj[i] = surr
	}
	if seqLevel {
		g := float64(len(lengths))
		var loss float64
		pos := 0
		for _, l := range lengths {
			var s float64
			for k := 0; k < l; k++ {
				s += obj[pos]
				pos++
			}
			loss += (s / float64(l)) / g
		}
		return -loss
	}
	var s float64
	for _, o := range obj {
		s += o
	}
	return -s / float64(n)
}

// §V16 COLLAPSE anchor: DAPOLoss with ε_low==ε_high (symmetric clip), sequence-level
// reduction over length-1 responses, and a matching β must equal the existing GRPO
// clipped objective at f64 1e-10 — pinning that DAPO's deltas are EXACTLY clip-higher
// and token-level (nothing else changed).
func TestDAPOCollapsesToGRPO(t *testing.T) {
	eps, beta := 0.2, 0.04
	lpNew := []float64{0.05, -0.05, 0.4, -0.4, 0.0}
	lpOld := []float64{0.0, 0.1, -0.1, 0.2, -0.2}
	lpRef := []float64{0.1, -0.1, 0.2, -0.2, 0.05}
	adv := []float64{1.0, -1.0, 1.5, 2.0, -0.5}
	lengths := []int{1, 1, 1, 1, 1} // each token is its own response ⇒ seq-level == flat mean

	grpo, err := nn.GRPOLoss(backend.NewContext(), vec(lpNew), vec(lpOld), vec(lpRef), vec(adv),
		nn.WithClipEpsilon(eps), nn.WithKLBeta(beta))
	if err != nil {
		t.Fatal(err)
	}
	dapo, err := nn.DAPOLoss(backend.NewContext(), vec(lpNew), vec(lpOld), vec(lpRef), vec(adv), lengths,
		nn.WithDAPOClipLow(eps), nn.WithDAPOClipHigh(eps), nn.WithDAPOKLBeta(beta), nn.WithDAPOSequenceLevel())
	if err != nil {
		t.Fatal(err)
	}
	if g, d := grpo.AtF64(), dapo.AtF64(); math.Abs(g-d) > 1e-10*math.Max(1, math.Abs(g)) {
		t.Errorf("DAPO(symmetric,seq-level,β)=%.17g must collapse to GRPO=%.17g", d, g)
	}
}

// §V16 CLIP-HIGHER anchor: with a ratio r above 1+ε_low but below 1+ε_high and a
// POSITIVE advantage, symmetric GRPO would clip the update to (1+ε_low)·Â while DAPO
// leaves it at r·Â (a larger update). Assert (a) the loss differs in the expected
// direction (DAPO's objective is larger ⇒ its loss more negative) and (b) the clip
// boundaries are exactly [1−ε_low, 1+ε_high].
func TestDAPOClipHigher(t *testing.T) {
	epsLow, epsHigh := 0.2, 0.28
	// Pick logπθ−logπ_old so r sits strictly between 1+ε_low (=1.2) and 1+ε_high (=1.28).
	r := 1.24
	lr := math.Log(r)
	lpNew := []float64{lr}
	lpOld := []float64{0}
	adv := []float64{1.0} // positive ⇒ min() selects the clipped branch when it is smaller
	lengths := []int{1}

	dapo, err := nn.DAPOLoss(backend.NewContext(), vec(lpNew), vec(lpOld), nil, vec(adv), lengths,
		nn.WithDAPOClipLow(epsLow), nn.WithDAPOClipHigh(epsHigh))
	if err != nil {
		t.Fatal(err)
	}
	// Symmetric GRPO-style clip at [1−ε_low, 1+ε_low]: DAPO with ε_high := ε_low.
	sym, err := nn.DAPOLoss(backend.NewContext(), vec(lpNew), vec(lpOld), nil, vec(adv), lengths,
		nn.WithDAPOClipLow(epsLow), nn.WithDAPOClipHigh(epsLow))
	if err != nil {
		t.Fatal(err)
	}
	// DAPO leaves r unclipped (r < 1+ε_high) ⇒ objective r·Â; symmetric clips to (1+ε_low)·Â.
	// Larger objective ⇒ more negative loss.
	if !(dapo.AtF64() < sym.AtF64()) {
		t.Errorf("clip-higher must give a larger update: DAPO loss %.17g should be < symmetric %.17g",
			dapo.AtF64(), sym.AtF64())
	}
	wantDAPO := -r * adv[0]           // unclipped
	wantSym := -(1 + epsLow) * adv[0] // clipped at 1+ε_low
	if math.Abs(dapo.AtF64()-wantDAPO) > 1e-10 {
		t.Errorf("DAPO loss %.17g, want unclipped %.17g", dapo.AtF64(), wantDAPO)
	}
	if math.Abs(sym.AtF64()-wantSym) > 1e-10 {
		t.Errorf("symmetric loss %.17g, want clipped %.17g", sym.AtF64(), wantSym)
	}

	// Boundaries are EXACTLY [1−ε_low, 1+ε_high]. The upper clip binds only when the
	// advantage is positive (min picks the smaller clipped·Â); the lower clip binds
	// only when the advantage is negative. want() clamps to [1−ε_low, 1+ε_high]
	// independently, so any wrong boundary shows up as a mismatch.
	check := func(ratio, a float64) {
		lp := []float64{math.Log(ratio)}
		got, err := nn.DAPOLoss(backend.NewContext(), vec(lp), vec([]float64{0}), nil, vec([]float64{a}), []int{1},
			nn.WithDAPOClipLow(epsLow), nn.WithDAPOClipHigh(epsHigh))
		if err != nil {
			t.Fatal(err)
		}
		clamped := math.Max(1-epsLow, math.Min(1+epsHigh, ratio))
		want := -math.Min(ratio*a, clamped*a)
		if math.Abs(got.AtF64()-want) > 1e-10 {
			t.Errorf("ratio %.4f adv %.1f: loss %.17g, want %.17g", ratio, a, got.AtF64(), want)
		}
	}
	check(1.40, 1.0)  // above upper boundary, adv>0 ⇒ clipped to 1+ε_high=1.28
	check(0.50, -1.0) // below lower boundary, adv<0 ⇒ clipped to 1−ε_low=0.80
}

// §V16 TOKEN-LEVEL vs SEQUENCE-LEVEL anchor: on a batch with one long and one short
// response the token-level loss weights tokens by count (a long response contributes
// proportionally) and differs from the sequence-level mean. Both match hand-computed
// references at f64 1e-10.
func TestDAPOTokenVsSequenceLevel(t *testing.T) {
	epsLow, epsHigh := 0.2, 0.28
	// response 1: 4 tokens (long); response 2: 1 token (short).
	lpNew := []float64{0.05, -0.05, 0.10, -0.10, 0.30}
	lpOld := []float64{0.0, 0.0, 0.0, 0.0, 0.0}
	adv := []float64{1.0, 1.0, 1.0, 1.0, -2.0}
	lengths := []int{4, 1}

	tok, err := nn.DAPOLoss(backend.NewContext(), vec(lpNew), vec(lpOld), nil, vec(adv), lengths,
		nn.WithDAPOClipLow(epsLow), nn.WithDAPOClipHigh(epsHigh))
	if err != nil {
		t.Fatal(err)
	}
	seq, err := nn.DAPOLoss(backend.NewContext(), vec(lpNew), vec(lpOld), nil, vec(adv), lengths,
		nn.WithDAPOClipLow(epsLow), nn.WithDAPOClipHigh(epsHigh), nn.WithDAPOSequenceLevel())
	if err != nil {
		t.Fatal(err)
	}
	wantTok := dapoRef(lpNew, lpOld, nil, adv, lengths, epsLow, epsHigh, 0, false)
	wantSeq := dapoRef(lpNew, lpOld, nil, adv, lengths, epsLow, epsHigh, 0, true)
	if math.Abs(tok.AtF64()-wantTok) > 1e-10*math.Max(1, math.Abs(wantTok)) {
		t.Errorf("token-level loss %.17g, want %.17g", tok.AtF64(), wantTok)
	}
	if math.Abs(seq.AtF64()-wantSeq) > 1e-10*math.Max(1, math.Abs(wantSeq)) {
		t.Errorf("sequence-level loss %.17g, want %.17g", seq.AtF64(), wantSeq)
	}
	if math.Abs(tok.AtF64()-seq.AtF64()) < 1e-6 {
		t.Errorf("token-level (%.17g) and sequence-level (%.17g) must differ on unequal lengths",
			tok.AtF64(), seq.AtF64())
	}
}

// §V2 GRADCHECK anchor: central finite differences of the token-level DAPO objective
// (with a KL term active, to exercise the reference-log-prob path) w.r.t. the policy
// log-probs match the analytic VJP; rollout/reference log-probs and advantages are
// frozen. Inputs are away from clip kinks.
func TestDAPOGradCheck(t *testing.T) {
	epsLow, epsHigh, beta := 0.2, 0.28, 0.05
	lpNew := []float64{0.05, -0.05, 0.4, -0.4, 0.0, 0.12}
	lpOld := []float64{0, 0, 0, 0, 0, 0}
	lpRef := []float64{0.1, -0.1, 0.2, -0.2, 0.05, -0.03}
	adv := []float64{1.0, -1.0, 1.5, 2.0, -0.5, 0.7}
	lengths := []int{3, 3}
	oldT, refT, advT := vec(lpOld), vec(lpRef), vec(adv)

	lossAt := func(d []float64) float64 {
		out, err := nn.DAPOLoss(backend.NewContext(), vec(d), oldT, refT, advT, lengths,
			nn.WithDAPOClipLow(epsLow), nn.WithDAPOClipHigh(epsHigh), nn.WithDAPOKLBeta(beta))
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}

	newT := vec(lpNew)
	tape := autograd.NewTape()
	loss, err := nn.DAPOLoss(tape.Context(), newT, oldT, refT, advT, lengths,
		nn.WithDAPOClipLow(epsLow), nn.WithDAPOClipHigh(epsHigh), nn.WithDAPOKLBeta(beta))
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	// DAPOLoss composes plain ops, so autograd also produces (ignored) gradients for
	// the conceptual constants logπ_old/logπ_ref/Â — the caller updates only the
	// policy params behind logπθ. The anchor is the logπθ gradient, verified below.
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

// §V16 helper anchor: DynamicSampleFilter drops groups whose responses all earn the
// SAME reward (zero advantage, zero gradient) and keeps groups with varied rewards.
func TestDynamicSampleFilter(t *testing.T) {
	rewards := [][]float64{
		{1, 1, 1},  // all equal → drop
		{1, 0, 1},  // varied → keep
		{0.5, 0.5}, // all equal → drop
		{-1, 2},    // varied → keep
		{3},        // single → drop
		{},         // empty → drop
	}
	want := []bool{false, true, false, true, false, false}
	got := nn.DynamicSampleFilter(rewards)
	if len(got) != len(want) {
		t.Fatalf("mask len %d != %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("group %d: keep=%v, want %v", i, got[i], want[i])
		}
	}
}

// §V16 helper anchor: OverlongPenalty is 0 below the soft threshold, ramps linearly
// to −1 across the cache buffer, saturates at −1 beyond maxLen, and is monotone
// non-increasing in length.
func TestOverlongPenalty(t *testing.T) {
	maxLen, cacheLen := 100, 20
	soft := maxLen - cacheLen // 80

	if p := nn.OverlongPenalty(50, maxLen, cacheLen); p != 0 {
		t.Errorf("below soft threshold must be 0, got %v", p)
	}
	if p := nn.OverlongPenalty(soft, maxLen, cacheLen); p != 0 {
		t.Errorf("at soft threshold must be 0, got %v", p)
	}
	if p := nn.OverlongPenalty(maxLen, maxLen, cacheLen); math.Abs(p-(-1)) > 1e-12 {
		t.Errorf("at maxLen must be −1, got %v", p)
	}
	if p := nn.OverlongPenalty(90, maxLen, cacheLen); math.Abs(p-(-0.5)) > 1e-12 {
		t.Errorf("midpoint of buffer must be −0.5, got %v", p)
	}
	if p := nn.OverlongPenalty(150, maxLen, cacheLen); p != -1 {
		t.Errorf("beyond maxLen must saturate at −1, got %v", p)
	}
	// Monotone non-increasing.
	prev := 1.0
	for l := 0; l <= 160; l++ {
		p := nn.OverlongPenalty(l, maxLen, cacheLen)
		if p > prev+1e-12 {
			t.Errorf("penalty must be non-increasing: length %d gave %v > previous %v", l, p, prev)
		}
		prev = p
	}
}

// DAPO (Yu et al. 2025) improves GRPO for reasoning RL with a decoupled "clip-higher"
// range that lets rare exploratory tokens grow, a token-level loss that weights long
// responses by their length, and two data-side helpers: dropping prompt groups whose
// samples all earn the same reward (no learning signal) and softly penalizing
// overlong responses instead of cutting them off.
func ExampleDAPOLoss() {
	// One prompt, two sampled responses (3 tokens + 1 token) with group-relative advantages.
	lpNew := []float64{-0.1, -0.2, -0.15, -0.3}
	lpOld := []float64{-0.1, -0.2, -0.15, -0.3} // fresh rollout ⇒ ratio 1
	adv := []float64{1.0, 1.0, 1.0, -1.0}
	lengths := []int{3, 1}

	loss, _ := nn.DAPOLoss(backend.NewContext(), vec(lpNew), vec(lpOld), nil, vec(adv), lengths)
	keep := nn.DynamicSampleFilter([][]float64{{1, 1, 1}, {1, 0, 1}})
	pen := nn.OverlongPenalty(95, 100, 20)
	fmt.Printf("loss=%.3f keep=%v penalty=%.2f\n", loss.AtF64(), keep, pen)
	// Output: loss=-0.500 keep=[false true] penalty=-0.75
}
