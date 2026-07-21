package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// logits whose softmax is exactly [0.5, 0.25, 0.125, 0.125] → surprises 1,2,3,3 bits.
func knownLogits() []float64 {
	p := []float64{0.5, 0.25, 0.125, 0.125}
	z := make([]float64, len(p))
	for i, v := range p {
		z[i] = math.Log(v)
	}
	return z
}

// §V16 tier-1: the feedback update is exactly μ ← μ − η·(S(X) − τ) with S(X) the
// base-2 surprise −log₂p(X) of the ORIGINAL (pre-truncation) probability of the
// sampled token, starting from μ=2τ.
func TestMirostatMuUpdate(t *testing.T) {
	const tau, eta = 5.0, 0.1
	m := nlp.NewMirostat(7, nlp.WithMirostatTau(tau), nlp.WithMirostatEta(eta))
	if m.Mu != 2*tau {
		t.Fatalf("μ must start at 2τ=%v, got %v", 2*tau, m.Mu)
	}
	logits := knownLogits()
	probs := []float64{0.5, 0.25, 0.125, 0.125}
	x := m.Sample(logits)
	wantMu := 2*tau - eta*(-math.Log2(probs[x])-tau)
	if math.Abs(m.Mu-wantMu) > 1e-12 {
		t.Errorf("after sampling token %d: μ=%.14g want %.14g", x, m.Mu, wantMu)
	}
}

// §V16 tier-1: with τ=0 the initial μ=0 is below every token's surprise, so the
// truncation keeps only the single most probable token — Mirostat degenerates to
// greedy and returns the arg-max deterministically.
func TestMirostatTruncationKeepsTopOnly(t *testing.T) {
	m := nlp.NewMirostat(1, nlp.WithMirostatTau(0))
	for _, logits := range [][]float64{
		{0.1, 3.0, 0.2, 1.5},
		{5.0, 4.9, 0.0, -1.0},
		{-2, -1, -0.5, -0.4},
	} {
		want := 0
		for i := 1; i < len(logits); i++ {
			if logits[i] > logits[want] {
				want = i
			}
		}
		if got := m.Sample(logits); got != want {
			t.Errorf("τ=0 must pick arg-max %d, got %d", want, got)
		}
	}
}

// §V16 tier-1: Mirostat's defining guarantee — it directly CONTROLS perplexity. On a
// fixed distribution the feedback loop drives the running-average surprise of the
// sampled tokens to the target τ (so perplexity → 2^τ), independent of the truncation
// dynamics. Averaged over many steps the mean observed surprise sits at τ.
func TestMirostatControlsSurprise(t *testing.T) {
	const tau = 3.0
	// a spread distribution over 64 tokens (surprises bracket τ)
	n := 64
	logits := make([]float64, n)
	for i := range logits {
		logits[i] = -0.15 * float64(i)
	}
	// original probabilities, to score the surprise of each sampled token
	mx := logits[0]
	probs := make([]float64, n)
	var sum float64
	for i, v := range logits {
		probs[i] = math.Exp(v - mx)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	m := nlp.NewMirostat(42, nlp.WithMirostatTau(tau), nlp.WithMirostatEta(0.1))
	const warm, steps = 2000, 8000
	var acc float64
	for s := 0; s < warm+steps; s++ {
		x := m.Sample(logits)
		if s >= warm {
			acc += -math.Log2(probs[x])
		}
	}
	mean := acc / float64(steps)
	if math.Abs(mean-tau) > 0.25 {
		t.Errorf("controlled mean surprise %.4f bits, want ≈ τ=%.1f (perplexity control failed)", mean, tau)
	}
}

// Mirostat holds the average per-token surprise at the target τ. With τ set to 0 the
// threshold μ starts below every token's surprise, so it keeps only the most probable
// token — a deterministic arg-max here selecting token 1.
func ExampleMirostat() {
	m := nlp.NewMirostat(1, nlp.WithMirostatTau(0))
	fmt.Println(m.Sample([]float64{0.2, 2.5, 1.0, 0.3}))
	// Output: 1
}

// BenchmarkMirostatSample exercises the full-vocab descending sort that
// dominates Mirostat on a flat distribution (the radix path, see
// sortIdxDescByProb). Pre-radix this ran ~4.4ms; the radix cut it to ~1.8ms
// (2.45×) at V=50000.
func BenchmarkMirostatSample(b *testing.B) {
	const V = 50000
	logits := make([]float64, V)
	for i := range logits {
		logits[i] = math.Sin(float64(i)*0.001) * 2 // flat-ish ⇒ large keep set
	}
	m := nlp.NewMirostat(1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = m.Sample(logits)
	}
}
