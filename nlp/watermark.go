package nlp

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// LLM watermarking (Kirchenbauer, Geiping, Wen, Katz, Miers & Goldstein 2023, "A Watermark for
// Large Language Models", ICML, arXiv:2301.10226). A watermark biases generation toward a
// pseudo-random subset of the vocabulary that only a key holder can reconstruct, so machine text
// can be detected without access to the model — without perceptibly changing quality. At each
// step a "green list" of γ·|V| tokens is chosen by a PRNG SEEDED FROM THE PREVIOUS TOKEN and the
// secret key; the soft scheme adds a constant δ to the green tokens' logits (Algorithm 2). A
// detector recomputes the same green lists and counts green tokens: under the no-watermark null
// each token is green with probability γ, so the one-proportion z-statistic
//
//	z = (|s|_G − γ·T) / √(T·γ·(1−γ))
//
// (Section 4) is large for watermarked text; z > 4 flags a watermark at a ~3e−5 false-positive
// rate. The green partition is a deterministic function of (previous token, key), so generation
// and detection reconstruct the identical list. The previous-token hash is an implementation
// choice (the paper leaves the PRF open); this uses a stable PCG stream keyed by (Key, prevToken).

// WatermarkZThreshold is the default z-score above which Detect's result is treated as
// watermarked (Kirchenbauer et al. 2023 §4: z>4 ⇒ FPR ≈ 3e−5).
const WatermarkZThreshold = 4.0

// Watermark holds the red-green watermark parameters.
type Watermark struct {
	VocabSize int     // size of the vocabulary V
	Gamma     float64 // green-list fraction γ ∈ (0,1)
	Delta     float64 // logit bias δ added to green tokens (soft watermark)
	Key       uint64  // secret key seeding the green-list PRF
}

// WatermarkOption configures a Watermark via the functional-options idiom (§C12).
type WatermarkOption func(*Watermark)

// WithWatermarkGamma sets the green-list fraction γ (default 0.25).
func WithWatermarkGamma(g float64) WatermarkOption { return func(w *Watermark) { w.Gamma = g } }

// WithWatermarkDelta sets the green-list logit bias δ (default 2.0).
func WithWatermarkDelta(d float64) WatermarkOption { return func(w *Watermark) { w.Delta = d } }

// WithWatermarkKey sets the secret key seeding the green lists (default 0).
func WithWatermarkKey(k uint64) WatermarkOption { return func(w *Watermark) { w.Key = k } }

// NewWatermark builds a watermark over a vocabulary of vocabSize with the defaults γ=0.25,
// δ=2.0 and key 0 (γ=0.25 is the official lm-watermarking repo's recommended default; the
// paper's headline analysis uses γ=0.5 — set it with WithWatermarkGamma).
func NewWatermark(vocabSize int, opts ...WatermarkOption) (*Watermark, error) {
	if vocabSize <= 0 {
		return nil, fmt.Errorf("nlp: Watermark vocabSize must be positive, got %d", vocabSize)
	}
	w := &Watermark{VocabSize: vocabSize, Gamma: 0.25, Delta: 2.0}
	for _, o := range opts {
		o(w)
	}
	if w.Gamma <= 0 || w.Gamma >= 1 {
		return nil, fmt.Errorf("nlp: Watermark gamma must be in (0,1), got %g", w.Gamma)
	}
	return w, nil
}

// greenSize is ⌊γ·|V|⌋, the number of green tokens per step.
func (w *Watermark) greenSize() int { return int(w.Gamma * float64(w.VocabSize)) }

// GreenMask returns a length-VocabSize boolean mask of the green list seeded by prevToken: a
// partial Fisher–Yates shuffle picks the first ⌊γ·|V|⌋ ids of a (Key, prevToken)-seeded
// permutation as green. Deterministic — generation and detection call it identically.
func (w *Watermark) GreenMask(prevToken int) []bool {
	perm := make([]int, w.VocabSize)
	for i := range perm {
		perm[i] = i
	}
	rng := rand.New(rand.NewPCG(w.Key, uint64(prevToken)))
	g := w.greenSize()
	// partial Fisher–Yates: only the first g positions need to be finalized.
	for i := 0; i < g; i++ {
		j := i + rng.IntN(w.VocabSize-i)
		perm[i], perm[j] = perm[j], perm[i]
	}
	mask := make([]bool, w.VocabSize)
	for i := 0; i < g; i++ {
		mask[perm[i]] = true
	}
	return mask
}

// BiasLogits returns a copy of logits with δ added to every green-list token for the step whose
// preceding token is prevToken (the soft watermark, Algorithm 2). The caller samples from the
// result as usual. len(logits) must equal VocabSize.
func (w *Watermark) BiasLogits(logits []float64, prevToken int) ([]float64, error) {
	if len(logits) != w.VocabSize {
		return nil, fmt.Errorf("nlp: Watermark.BiasLogits logits length %d != vocab %d", len(logits), w.VocabSize)
	}
	mask := w.GreenMask(prevToken)
	out := make([]float64, len(logits))
	for i, l := range logits {
		if mask[i] {
			out[i] = l + w.Delta
		} else {
			out[i] = l
		}
	}
	return out, nil
}

// Sampler wraps inner so every sampled token is watermarked (§T439): before inner
// draws, BiasLogits adds δ to the green list seeded by the LAST history token, so the
// wrapper plugs straight into any generation loop that takes a TokenSampler
// (GPT/Llama/QuantLlama Generate, StreamGenerate, the llamagpu decoders). The
// history — prompt included — is what the loops already pass to SampleWithHistory.
// A history-less Sample (or a logits length ≠ VocabSize) draws unbiased: with no
// predecessor there is no green list to seed. Detect/IsWatermarked verify the output.
func (w *Watermark) Sampler(inner TokenSampler) TokenSampler {
	return &watermarkSampler{w: w, inner: inner}
}

type watermarkSampler struct {
	w     *Watermark
	inner TokenSampler
}

func (s *watermarkSampler) Sample(logits []float64) int { return s.inner.Sample(logits) }

func (s *watermarkSampler) SampleWithHistory(logits []float64, history []int) int {
	if len(history) > 0 {
		if b, err := s.w.BiasLogits(logits, history[len(history)-1]); err == nil {
			logits = b
		}
	}
	return s.inner.SampleWithHistory(logits, history)
}

// Detect scores a token sequence: each token from index 1 on is checked against the green list
// seeded by its predecessor. It returns the number of green tokens, the number scored (T = len−1)
// and the z-statistic z = (green − γ·T)/√(T·γ·(1−γ)) (0 when T=0). Compare z to WatermarkZThreshold.
func (w *Watermark) Detect(tokens []int) (z float64, green, scored int) {
	scored = len(tokens) - 1
	if scored <= 0 {
		return 0, 0, 0
	}
	for i := 1; i < len(tokens); i++ {
		if w.GreenMask(tokens[i-1])[tokens[i]] {
			green++
		}
	}
	T := float64(scored)
	z = (float64(green) - w.Gamma*T) / math.Sqrt(T*w.Gamma*(1-w.Gamma))
	return z, green, scored
}

// IsWatermarked reports whether Detect's z-score exceeds WatermarkZThreshold.
func (w *Watermark) IsWatermarked(tokens []int) bool {
	z, _, _ := w.Detect(tokens)
	return z > WatermarkZThreshold
}
