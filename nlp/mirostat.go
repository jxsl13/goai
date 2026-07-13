package nlp

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Mirostat is the Mirostat 2.0 decoding algorithm (Basu et al. 2020, "Mirostat: A
// Neural Text Decoding Algorithm that Directly Controls Perplexity", arXiv:2007.14966,
// ICLR 2021; the mirostat_v2 variant used in llama.cpp, §R85). Unlike top-k/top-p,
// which fix a truncation and let perplexity drift, Mirostat targets a fixed SURPRISE
// τ (the per-token cross-entropy, in bits) with a feedback loop: it truncates to the
// tokens whose surprise is within a running threshold μ, samples, measures the actual
// surprise of the chosen token, and nudges μ by the error — so the long-run average
// surprise (hence perplexity 2^τ) is held at the target regardless of context.
//
// Per step, with token surprise S(i) = −log₂ p(i):
//
//	truncate to candidates with S(i) ≤ μ   (always keep ≥ 1 — the top token)
//	sample X from the renormalized candidates
//	e = S(X) − τ ;  μ ← μ − η·e            (S(X) from the ORIGINAL prob of X)
//
// μ starts at 2·τ (Algorithm 2). It is mutable state carried across calls within one
// generated sequence; call Reset (or a fresh Mirostat) per new sequence.
type Mirostat struct {
	Tau float64 // target surprise τ in bits (per-token cross-entropy; default 5.0)
	Eta float64 // feedback learning rate η (default 0.1)
	Mu  float64 // running surprise threshold μ (state; initialized to 2·τ)
	rng *rand.Rand
}

// MirostatOption configures a Mirostat sampler (functional-options idiom, §C12).
type MirostatOption func(*Mirostat)

// WithMirostatTau sets the target surprise τ in bits (default 5.0); the controlled
// perplexity is 2^τ.
func WithMirostatTau(tau float64) MirostatOption { return func(m *Mirostat) { m.Tau = tau } }

// WithMirostatEta sets the feedback learning rate η (default 0.1).
func WithMirostatEta(eta float64) MirostatOption { return func(m *Mirostat) { m.Eta = eta } }

// NewMirostat builds a deterministic Mirostat 2.0 sampler seeded by seed, with the
// canonical defaults τ=5.0 bits and η=0.1, and μ initialized to 2·τ.
func NewMirostat(seed uint64, opts ...MirostatOption) *Mirostat {
	m := &Mirostat{Tau: 5.0, Eta: 0.1, rng: rand.New(rand.NewPCG(seed, 0x5a3d))}
	for _, o := range opts {
		o(m)
	}
	m.Mu = 2 * m.Tau
	return m
}

// Reset restores the surprise threshold μ to 2·τ, for reuse across sequences.
func (m *Mirostat) Reset() { m.Mu = 2 * m.Tau }

// SampleWithHistory satisfies TokenSampler; Mirostat ignores the history — it
// controls repetition through its surprise target rather than explicit penalties
// (a Mirostat run whose μ has adapted never collapses to a greedy loop the way
// untruncated small-model sampling can).
func (m *Mirostat) SampleWithHistory(logits []float64, _ []int) int { return m.Sample(logits) }

// Sample returns a token id for the given logits and advances the surprise
// threshold μ by the feedback rule. It softmaxes the logits, truncates to the
// tokens whose surprise −log₂p ≤ μ (keeping at least the most probable token),
// samples one from the renormalized survivors, then updates μ ← μ − η·(S(X) − τ).
func (m *Mirostat) Sample(logits []float64) int {
	n := len(logits)
	if n == 0 {
		return 0
	}
	// stable softmax
	mx := math.Inf(-1)
	for _, v := range logits {
		if v > mx {
			mx = v
		}
	}
	probs := make([]float64, n)
	var sum float64
	for i, v := range logits {
		probs[i] = math.Exp(v - mx)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	// sort ids by probability descending → surprise −log₂p ascending
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })

	// truncate to the prefix with surprise ≤ μ, always keeping the top token
	keep := 1
	for keep < n && surpriseBits(probs[idx[keep]]) <= m.Mu {
		keep++
	}

	// sample X from the renormalized survivors
	var ksum float64
	for _, i := range idx[:keep] {
		ksum += probs[i]
	}
	u := m.rng.Float64() * ksum
	x := idx[keep-1]
	var cum float64
	for _, i := range idx[:keep] {
		cum += probs[i]
		if u <= cum {
			x = i
			break
		}
	}

	// feedback: observed surprise from the ORIGINAL prob of X, then nudge μ
	observed := surpriseBits(probs[x])
	m.Mu -= m.Eta * (observed - m.Tau)
	return x
}

// surpriseBits returns the information content −log₂(p) in bits, guarding p≤0.
func surpriseBits(p float64) float64 {
	if p <= 0 {
		return math.Inf(1)
	}
	return -math.Log2(p)
}
