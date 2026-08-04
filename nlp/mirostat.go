package nlp

import (
	"math"
	"math/rand/v2"
	"slices"

	"github.com/jxsl13/goai/internal/simd"
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
	// per-token vocab-sized scratch, reused across Sample calls (Mirostat is already
	// stateful/per-sequence via Mu, so a struct-owned buffer — not a fresh make() every
	// token — is the natural home). Both are fully overwritten before any read.
	probsBuf []float64 // softmax buffer, filled by simd.ExpSumF64
	idxBuf   []int     // index buffer for the sortedKeep flat-distribution fallback
}

// MirostatOption configures a Mirostat sampler (functional-options idiom, §C12).
type MirostatOption func(*Mirostat)

// WithMirostatTau sets the target surprise τ, in bits.
//
// In plain terms: the "surprise dial" — how unpredictable you want the text to be on
// average. Low τ = safe and repetitive; high τ = adventurous and varied. Mirostat then
// steers generation to hold that level regardless of context, instead of letting it drift
// the way top-k/top-p do.
//
// Professional: τ is the per-token cross-entropy target; the controlled perplexity is 2^τ.
// Boundary behavior — as τ → 0 the sampler drives toward the single most likely token every
// step (greedy-like, minimal surprise); large τ (past ≈8) targets high perplexity and text
// grows incoherent. The feedback keeps the long-run average surprise at τ.
//
// Default 5.0 bits (research-grounded): the de-facto llama.cpp default; Basu et al. 2020
// (§R85) sweep τ ∈ {2.5, 3, 4, 5}, and 5 is the standard general-purpose setting.
func WithMirostatTau(tau float64) MirostatOption { return func(m *Mirostat) { m.Tau = tau } }

// WithMirostatEta sets the feedback learning rate η.
//
// In plain terms: how fast Mirostat corrects course when the last word was more or less
// surprising than the target — big η reacts quickly but jitters, small η adjusts slowly
// and smoothly.
//
// Professional: the step size of the μ ← μ − η·(S − τ) update. Boundary behavior — η → 0
// freezes the threshold μ (no adaptation, behaves like a fixed surprise cutoff); large η
// overshoots and oscillates around the target. SPECIAL VALUE: 0 = no feedback (static μ).
//
// Default 0.1 (research-grounded): the llama.cpp default and the authors' reference learning
// rate (Basu et al. 2020, §R85).
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
	if cap(m.probsBuf) < n {
		m.probsBuf = make([]float64, n)
	}
	probs := m.probsBuf[:n]                  // fully written by ExpSumF64 below, so a dirty reuse is defined
	sum := simd.ExpSumF64(probs, logits, mx) // 4-wide AVX2 f64 softmax exp (see Sampler.Dist)
	//perfscan:ignore PS5001 memory-bound normalize; reciprocal blocked by numerics note
	for i := range probs {
		probs[i] /= sum
	}

	// The survivor set is {i : surprise(i) ≤ μ} (always keeping the top token) — a tiny
	// prefix of the prob-descending order for a peaked distribution or a decayed μ. Rather
	// than radix-sort the WHOLE vocab to extract that prefix, pre-filter by the equivalent
	// prob bound (surprise = −log₂p is monotone, so surprise ≤ μ ⟺ p ≥ 2⁻ᵘ) and apply the
	// EXACT surpriseBits predicate only to the handful above the bound — the identical set,
	// with no full-vocab log2 and no full sort. Bit-identical to the sort-then-threshold
	// path: the 1−2⁻³⁰ margin keeps the bound safely below the true boundary (≫ log2/exp2
	// ULP) so no qualifying token is dropped, the exact predicate then decides membership,
	// and the survivors are stable-sorted desc-by-prob (ties → ascending id, matching the
	// stable LSD radix's prefix). A genuinely flat distribution that admits ≥ radixSortCutoff
	// candidates falls back to the original full radix sort — never worse than the sort path.
	var cand []int
	if n < radixSortCutoff {
		// Small vocab: the full sort is already cheap AND uses the comparison path
		// (slices.SortFunc), which is not stable — its tie order among equal-probability
		// tokens is not reproducible from a subset, so reproduce the original path exactly.
		cand = m.sortedKeep(probs, n)
	} else {
		// Large vocab (the stable LSD-radix regime, n ≥ radixSortCutoff): the full sort's tie
		// order is ascending-id, which SortStableFunc on the ascending-id candidate list
		// reproduces exactly — so we can skip the full sort. Pre-filter by the prob bound
		// equivalent to surprise ≤ μ, exact-refine the few survivors, then stable-sort them.
		argmax, maxp := 0, math.Inf(-1)
		if m.Mu >= 0 { // μ<0 ⇒ surprise≥0>μ ⇒ nothing qualifies; just find the forced top token
			thr := math.Exp2(-m.Mu) * (1 - 1.0/(1<<30))
			//perfscan:ignore PS3068 already-optimized prob pre-filter loop
			for i, p := range probs {
				if p > maxp {
					maxp, argmax = p, i
				}
				if p >= thr {
					cand = append(cand, i)
				}
			}
		} else {
			//perfscan:ignore PS3068 already-optimized argmax pre-filter loop
			for i, p := range probs {
				if p > maxp {
					maxp, argmax = p, i
				}
			}
		}
		if len(cand) >= radixSortCutoff {
			cand = m.sortedKeep(probs, n) // flat distribution: the full radix sort is the right tool
		} else {
			out := cand[:0] // exact refine on the pre-filtered few (in-place; ascending-id order kept)
			//perfscan:ignore PS4003 exact refine over few survivors, low trip-count
			for _, i := range cand {
				if surpriseBits(probs[i]) <= m.Mu {
					out = append(out, i)
				}
			}
			cand = out
			if len(cand) == 0 {
				cand = append(cand, argmax) // always keep the top token
			}
			//perfscan:ignore PS3002 sort closure over few candidates, low trip-count
			slices.SortStableFunc(cand, func(a, b int) int {
				switch {
				case probs[a] > probs[b]:
					return -1
				case probs[a] < probs[b]:
					return 1
				default:
					return 0
				}
			})
		}
	}

	// sample X from the renormalized survivors (identical accumulation order to the sort path)
	var ksum float64
	//perfscan:ignore PS3010 sum over few candidates, low trip-count
	for _, i := range cand {
		ksum += probs[i]
	}
	u := m.rng.Float64() * ksum
	x := cand[len(cand)-1]
	var cum float64
	for _, i := range cand {
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

// sortedKeep is the original full-vocab path: radix-sort all ids by descending probability
// and return the prefix with surprise ≤ μ (always ≥ 1 token). Used as the fallback when the
// prob pre-filter admits an unusually large candidate set, so Sample is never slower than the
// pure sort implementation on a genuinely flat distribution.
func (m *Mirostat) sortedKeep(probs []float64, n int) []int {
	if cap(m.idxBuf) < n {
		m.idxBuf = make([]int, n)
	}
	idx := m.idxBuf[:n] // fully written (idx[i]=i) before the sort, then consumed in-call
	for i := range idx {
		idx[i] = i
	}
	sortIdxDescByProb(idx, probs)
	keep := 1
	//perfscan:ignore PS4003 sortedKeep fallback path, rarely taken
	for keep < n && surpriseBits(probs[idx[keep]]) <= m.Mu {
		keep++
	}
	return idx[:keep]
}

// surpriseBits returns the information content −log₂(p) in bits, guarding p≤0.
func surpriseBits(p float64) float64 {
	if p <= 0 {
		return math.Inf(1)
	}
	return -math.Log2(p)
}
