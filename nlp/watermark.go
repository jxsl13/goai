package nlp

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sync"
)

// wmScratch holds the reusable per-call scratch of BiasLogits: the identity
// permutation, the transposition log, and a PCG/Rand pair (re-seeded per call).
// Pooled across calls — seedGreen re-seeds the pcg every call so the draw stream is
// independent of buffer provenance, and restoreIdentity returns perm to the identity
// before it goes back to the pool, so a same-length perm pulled from the pool is
// already identity (skips the O(V) re-init). Sized lazily to VocabSize.
type wmScratch struct {
	perm  []int
	swaps []int
	pcg   *rand.PCG
	rng   *rand.Rand
}

var wmScratchPool = sync.Pool{New: func() any {
	pcg := rand.NewPCG(0, 0)
	return &wmScratch{pcg: pcg, rng: rand.New(pcg)}
}}

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

// WithWatermarkGamma sets the green-list fraction γ — the share of the vocabulary marked
// "green" (favored) at each step.
//
// In plain terms: how big a slice of possible next words the watermark quietly nudges the
// model toward. A hidden reader who knows the key can later count how often green words were
// chosen and tell the text was machine-generated. Boundary behavior — γ→0 makes the green set
// tiny (a strong, easily-detected signal but a bigger dent in text quality); γ→1 makes almost
// everything green (invisible watermark, undetectable). Must be strictly in (0, 1).
//
// Default 0.25 (research-grounded): the official lm-watermarking repo's recommended default
// (Kirchenbauer et al. 2023, §R219; the paper's headline analysis uses γ=0.5 — 0.25 detects
// with fewer tokens at a modest quality cost).
func WithWatermarkGamma(g float64) WatermarkOption { return func(w *Watermark) { w.Gamma = g } }

// WithWatermarkDelta sets the green-list logit bias δ — how hard the watermark pushes toward
// green tokens.
//
// In plain terms: the size of the thumb on the scale. Bigger δ makes the watermark easier to
// detect but distorts the text more (the model is forced toward green words even when a red
// one fit better). Boundary behavior — δ→0 removes the watermark (green and red equally
// likely, undetectable); large δ (>5) nearly forbids red tokens and visibly degrades fluency.
//
// Default 2.0 (research-grounded): the value from Kirchenbauer et al. 2023 (§R219) — a good
// detectability/quality balance (with γ=0.25 an all-green run of 16 tokens already hits z≈4).
func WithWatermarkDelta(d float64) WatermarkOption { return func(w *Watermark) { w.Delta = d } }

// WithWatermarkKey sets the secret key that seeds the per-step green lists.
//
// In plain terms: the password. Generation and detection must use the SAME key to line up the
// same green lists; anyone without it cannot tell the text is watermarked or forge one.
// Boundary behavior — any uint64 is valid; different keys give independent, non-interfering
// watermarks. SPECIAL VALUE: 0 is the default seed (fine for demos; use a real secret in
// production, since a known key lets others detect OR strip the mark).
//
// Default 0 (research-grounded): a fixed seed for reproducibility; the scheme's security rests
// on the key being secret (Kirchenbauer et al. 2023, §R219).
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

// seedGreen runs the (Key, prevToken)-seeded partial Fisher–Yates over perm so
// that perm[0:g] hold the green-list ids (g = greenSize), recording each swap's
// partner in swaps[0:g]. perm MUST enter as the identity 0..VocabSize-1; after
// restoreIdentity(perm, swaps, g) it is the identity again, so ONE scratch perm
// serves an entire sequence instead of GreenMask's fresh VocabSize perm+mask per
// token. The caller's pcg is RESEEDED to (Key, prevToken) each call and rng wraps
// it — reseeding one generator is bit-identical to rand.New(rand.NewPCG(Key,
// prevToken)) (rand.Rand buffers no state) but allocates nothing per token, so
// the swap sequence is byte-for-byte the one GreenMask draws and the selected
// green set is identical.
func (w *Watermark) seedGreen(rng *rand.Rand, pcg *rand.PCG, perm, swaps []int, prevToken int) int {
	pcg.Seed(w.Key, uint64(prevToken))
	g := w.greenSize()
	for i := 0; i < g; i++ {
		j := i + rng.IntN(w.VocabSize-i)
		perm[i], perm[j] = perm[j], perm[i]
		swaps[i] = j
	}
	return g
}

// restoreIdentity reverses the g transpositions recorded by seedGreen in LIFO
// order, returning perm to the identity it entered with. A product of
// transpositions is inverted by replaying them in reverse (each is its own
// inverse), so this restores perm exactly regardless of index overlap — in O(g),
// never touching the untouched tail.
func restoreIdentity(perm, swaps []int, g int) {
	for i := g - 1; i >= 0; i-- {
		j := swaps[i]
		perm[i], perm[j] = perm[j], perm[i]
	}
}

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
	out := make([]float64, len(logits))
	copy(out, logits)
	// Add δ to the green tokens directly (the first g ids of the seeded partial
	// Fisher–Yates) — O(g) writes to distinct indices, no VocabSize bool mask.
	// Same index set and single +δ per green token as the mask form, so the
	// result is bit-identical.
	sc := wmScratchPool.Get().(*wmScratch)
	if len(sc.perm) != w.VocabSize { // fresh or different-vocab scratch: (re)build the identity
		sc.perm = make([]int, w.VocabSize)
		for i := range sc.perm {
			sc.perm[i] = i
		}
	} // else perm is already identity: restoreIdentity ran before it was pooled
	if g := w.greenSize(); cap(sc.swaps) < g {
		sc.swaps = make([]int, g)
	} else {
		sc.swaps = sc.swaps[:g]
	}
	g := w.seedGreen(sc.rng, sc.pcg, sc.perm, sc.swaps, prevToken)
	for i := 0; i < g; i++ {
		out[sc.perm[i]] += w.Delta
	}
	restoreIdentity(sc.perm, sc.swaps, g) // return perm to identity so the pooled buffer is reusable
	wmScratchPool.Put(sc)
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
	// One reusable identity permutation restored in O(g) after each token,
	// instead of GreenMask minting a fresh VocabSize perm+mask per token just to
	// read one entry (that O(T·|V|) allocation is GC-bound, not compute-bound).
	// The green set per token is identical, so green — and thus z — is unchanged.
	perm := make([]int, w.VocabSize)
	for i := range perm {
		perm[i] = i
	}
	swaps := make([]int, w.greenSize())
	pcg := rand.NewPCG(0, 0)
	rng := rand.New(pcg)
	// Fuse seedGreen's partial Fisher-Yates with the membership test: perm[k] is
	// finalized at step k (later steps only touch indices > k), so as soon as the
	// finalized green member equals tok we know tok is green and can stop — no
	// separate O(g) scan, and the remaining Fisher-Yates draws are skipped. Each
	// token reseeds pcg, so consuming fewer draws never perturbs a later token;
	// restoreIdentity reverses exactly the swaps performed. Green count (hence z)
	// is bit-identical to seedGreen + full scan.
	gsz := w.greenSize()
	for i := 1; i < len(tokens); i++ {
		pcg.Seed(w.Key, uint64(tokens[i-1]))
		tok := tokens[i]
		done := gsz
		for k := 0; k < gsz; k++ {
			j := k + rng.IntN(w.VocabSize-k)
			perm[k], perm[j] = perm[j], perm[k]
			swaps[k] = j
			if perm[k] == tok {
				green++
				done = k + 1
				break
			}
		}
		restoreIdentity(perm, swaps, done)
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
