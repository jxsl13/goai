package nlp

import (
	"math"
	"math/rand/v2"
	"slices"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Sampler turns a logit vector into a token id. Pipeline (matching HuggingFace
// generation): repetition penalties (via SampleWithHistory) → temperature scaling
// → top-k mask → softmax → top-p (nucleus) mask → min-p mask → epsilon/eta mask →
// multinomial sample. Temperature ≤ 0 selects greedy (argmax), which ignores the
// truncation filters and RNG — but not the penalties, which act on the logits
// before the greedy/temperature split.
//
// top-p follows Holtzman et al. 2019 (§R34): the nucleus is the smallest set of
// highest-probability tokens whose cumulative probability ≥ p; the token that
// crosses p is included, so at least the top token always survives. min-p follows
// Nguyen et al. 2024 (§R63): keep tokens with probability ≥ MinP·max-prob, an
// adaptive threshold that tightens when the model is confident. epsilon and eta
// sampling follow Hewitt et al. 2022 (§R91): an absolute floor, and an
// entropy-adaptive floor min(ε, √ε·exp(−H)). Kept probabilities are renormalized
// before sampling, and the arg-max token always survives every filter.
type Sampler struct {
	Temperature float64 // softmax temperature; <1 sharper, >1 flatter (default 1)
	TopK        int     // keep the K highest-prob tokens; 0 = off
	TopP        float64 // nucleus: keep smallest set with cumulative prob ≥ P; 0 = off
	MinP        float64 // keep tokens with prob ≥ MinP·maxProb; 0 = off
	Epsilon     float64 // epsilon sampling: keep tokens with prob ≥ Epsilon; 0 = off
	Eta         float64 // eta sampling: keep tokens with prob ≥ min(Eta, √Eta·exp(−H)); 0 = off
	Typical     float64 // locally typical: keep tokens whose surprisal is nearest the entropy until cum-prob ≥ τ; 0/≥1 = off

	// Repetition penalties (applied by SampleWithHistory over the recent history).
	RepeatPenalty   float64 // CTRL (Keskar et al. 2019): divide positive logits of seen tokens by this, multiply negative; 0/1 = off, typical 1.1–1.3
	FreqPenalty     float64 // subtract FreqPenalty · count(token in window) from its logit; 0 = off
	PresencePenalty float64 // subtract PresencePenalty once if the token appears in the window at all; 0 = off
	PenaltyLastN    int     // history window the penalties look at; 0 = the entire history

	// DRY sequence-repetition penalty (p-e-w 2024, see dry.go; applied by
	// SampleWithHistory). Off unless DRYMultiplier > 0.
	DRYMultiplier float64 // penalty strength; 0 = off, typical 0.8
	DRYBase       float64 // exponential growth per extra matched token; ≤0 → 1.75
	DRYAllowedLen int     // longest repetition left unpenalized; ≤0 → 2
	DRYRange      int     // history window DRY scans; 0 = the entire history
	DRYBreakers   []int   // token ids matches may not extend across (e.g. newline)

	// XTC top-choice exclusion (p-e-w 2024, see xtc.go; applied in Sample's final
	// draw, never in Dist or greedy decoding). Off unless both are > 0.
	XTCProbability float64 // chance per draw that exclusion fires; typical 0.5
	XTCThreshold   float64 // qualifying probability; typical 0.1 (>0.5 disables)

	rng *rand.Rand
}

// TokenSampler is anything that turns a logit row into a token id — the sequential
// generation loops (GPT/Llama/QuantLlama Generate, StreamGenerate, ContrastiveDecode,
// and the llamagpu decoders) accept any implementation. Sampler (temperature/top-k/
// top-p/… truncation) and Mirostat (adaptive surprise targeting) both satisfy it.
// SampleWithHistory receives the running sequence (prompt + generated) so
// history-aware strategies (repetition penalties) can act; implementations without
// history use are free to ignore it. The speculative paths (SpeculativeDecode,
// SpeculativeGenerate, prompt-lookup) still require a concrete *Sampler — their
// lossless accept/reject math needs the full distribution (Dist), not just a draw.
type TokenSampler interface {
	Sample(logits []float64) int
	SampleWithHistory(logits []float64, history []int) int
}

var (
	_ TokenSampler = (*Sampler)(nil)
	_ TokenSampler = (*Mirostat)(nil)
)

// SamplerOption configures a Sampler via the functional-options idiom (§C12).
type SamplerOption func(*Sampler)

// WithTemperature sets the softmax temperature — the single knob that trades safety
// for creativity.
//
// In plain terms: temperature is how adventurous the model is when it picks the next
// word. Low = it almost always takes the word it's most sure of (predictable, repetitive);
// high = it gives unlikely words a real chance (surprising, but more mistakes).
//
// Professional: logits are divided by t before the softmax, so t scales the entropy of
// the sampled distribution. Boundary behavior — as t→0⁺ the distribution collapses onto
// the argmax (this implementation treats t ≤ 0 as an explicit greedy short-circuit, a
// SPECIAL VALUE: no RNG draw, TopK/TopP/MinP ignored); as t grows the distribution
// flattens toward uniform and coherence degrades, with runaway incoherence typically past
// t≈1.5. t = 1 leaves the model's own trained distribution unchanged.
//
// Default 1 (research-grounded): 1 is the identity — it reproduces the distribution the
// model was trained to emit under maximum-likelihood, so out of the box the sampler adds
// no distortion (Holtzman et al. 2019, "The Curious Case of Neural Text Degeneration",
// §R34 — temperature scales the logits before truncation; llama.cpp and Hugging Face
// both default to 1). Lower it (≈0.7) for factual/deterministic tasks, raise it (≈1.2)
// for brainstorming.
func WithTemperature(t float64) SamplerOption { return func(s *Sampler) { s.Temperature = t } }

// WithTopK keeps only the k highest-probability tokens, zeroing the rest before the draw.
//
// In plain terms: only let the model choose among its k best guesses, throwing away the
// long tail of unlikely words that cause derailments.
//
// Professional: a fixed-size truncation of the tail (Fan et al. 2018, "Hierarchical Neural
// Story Generation"; the truncation family is catalogued in §R34). Boundary behavior — k = 1
// is equivalent to greedy (only the argmax survives); as k → vocabulary size the filter does
// nothing. Because k is a FIXED
// count it over-truncates flat distributions (many plausible tokens, most discarded) and
// under-truncates peaked ones — which is why nucleus/min-p, whose cutoff adapts to the
// distribution's shape, are usually preferred.
//
// Default 0 = DISABLED (special value), research-grounded: leaving top-k off is the modern
// default (llama.cpp ships top_k=40 but the nucleus/min-p line of work — Holtzman 2019,
// Nguyen 2024 — shows shape-adaptive cutoffs dominate a fixed count, so GoAI keeps it off
// and lets WithMinP carry the truncation). Set k≈40 only to emulate the classic pipeline.
func WithTopK(k int) SamplerOption { return func(s *Sampler) { s.TopK = k } }

// WithTopP keeps the smallest set of most-probable tokens whose probabilities sum to at
// least p (nucleus sampling), zeroing the rest.
//
// In plain terms: keep just enough of the model's top guesses to cover p of its confidence
// (e.g. 95%) and drop the improbable remainder — the kept set shrinks when the model is
// sure and grows when it's hesitant.
//
// Professional: nucleus sampling (Holtzman et al. 2019 arXiv:1904.09751, §R34). The cutoff is
// distribution-adaptive, unlike top-k. Boundary behavior — as p → 0⁺ only the single most
// probable token survives (greedy-like); p ≥ 1 keeps everything (disabled). SPECIAL VALUES:
// 0 or ≥1 disable the filter.
//
// Default 0 = DISABLED, research-grounded: GoAI's zero-config default truncates via min-p
// (WithMinP), which Nguyen et al. 2024 (§R63) show is more robust than nucleus at higher
// temperatures; enable nucleus explicitly at p≈0.9–0.95 (Holtzman's reported sweet spot,
// also Hugging Face's default when nucleus is used) to match the conventional pipeline.
func WithTopP(p float64) SamplerOption { return func(s *Sampler) { s.TopP = p } }

// WithMinP keeps tokens whose probability is at least p times the single most probable
// token's probability, zeroing the rest.
//
// In plain terms: drop any word that's much less likely than the model's favorite — the
// bar scales with how confident the top pick is, so it stays strict when the model is sure
// and lenient when the field is open.
//
// Professional: min-p sampling (Nguyen et al. 2024, "Turning Up the Heat", §R63). The
// relative cutoff p·maxProb makes it more temperature-robust than nucleus: raising
// temperature flattens the distribution but the cutoff tracks the peak, so quality holds
// at higher t. Boundary behavior — as p → 0⁺ nothing is filtered; as p → 1 only tokens
// essentially tied with the top token survive (greedy-like). SPECIAL VALUE: 0 disables it.
//
// Default 0 = DISABLED (a bare NewSampler samples untruncated at t=1); the RECOMMENDED
// enabled value is 0.05–0.1 — Nguyen et al. 2024 report 0.05–0.1 as the quality sweet spot
// across tasks, and llama.cpp defaults min_p to 0.05. This is GoAI's suggested single
// truncation knob: NewSampler(seed, WithMinP(0.05)) is a strong general-purpose setup.
func WithMinP(p float64) SamplerOption { return func(s *Sampler) { s.MinP = p } }

// WithEpsilon enables epsilon sampling (Hewitt et al. 2022, §R91).
//
// In plain terms: throw away any word below a fixed rarity floor, no matter the context.
//
// Professional: an absolute probability floor — keep tokens with p ≥ eps. Boundary behavior
// — tiny eps (≈3e-4) trims only the extreme tail; large eps clips into the plausible set and
// forces near-greedy output. Unlike min-p the floor is absolute, not relative to the peak.
// SPECIAL VALUE: 0 disables it.
//
// Default 0 = DISABLED, research-grounded: epsilon/eta are specialist desmoothing filters,
// off by default; Hewitt et al. 2022 suggest eps≈3e-4–2e-3 when enabled.
func WithEpsilon(eps float64) SamplerOption { return func(s *Sampler) { s.Epsilon = eps } }

// WithEta enables eta sampling (Hewitt et al. 2022, §R91).
//
// In plain terms: an automatic rarity floor that gets stricter when the model is confident
// and more permissive when it's unsure.
//
// Professional: an entropy-adaptive floor η = min(eps, √eps·exp(−H)), H the distribution's
// Shannon entropy in nats. Boundary behavior — in low-entropy (confident) states the
// √eps·exp(−H) term is large so η saturates at the eps cap and truncates hard; in
// high-entropy states η shrinks and keeps more. SPECIAL VALUE: 0 disables it.
//
// Default 0 = DISABLED, research-grounded: off by default; Hewitt et al. 2022 report
// eps≈9e-4 as a good working value (their desmoothing framework, §R91).
func WithEta(eps float64) SamplerOption { return func(s *Sampler) { s.Eta = eps } }

// WithTypical enables locally typical sampling (Meister, Pimentel, Wiher & Cotterell 2023,
// §R154).
//
// In plain terms: prefer words that are neither boringly obvious nor bizarrely surprising —
// the ones carrying an "average" amount of information, which reads as natural.
//
// Professional: keep the smallest set of tokens whose information content −log p is closest
// to the distribution's entropy H(p) until their cumulative probability reaches τ, trimming
// both the too-predictable and the too-surprising. Boundary behavior — small τ keeps only a
// few most-typical tokens (conservative); τ near 1 keeps almost everything. SPECIAL VALUES:
// 0 or ≥1 disable it.
//
// Default 0 = DISABLED, research-grounded: off by default; Meister et al. 2023 use τ≈0.9–0.95
// (also Hugging Face's TypicalLogitsWarper default 0.9).
func WithTypical(tau float64) SamplerOption { return func(s *Sampler) { s.Typical = tau } }

// WithRepeatPenalty enables the CTRL repetition penalty (Keskar et al. 2019, §R57).
//
// In plain terms: discourage the model from reusing words it has already said, so it
// stops looping on the same phrase.
//
// Professional: for every token seen in the penalty window, a positive logit is divided
// by p and a negative one multiplied by it (the asymmetric CTRL rule, which keeps the
// direction of the penalty consistent regardless of logit sign). Boundary behavior — at
// p = 1 the operation is identity (division and multiplication by 1); as p grows the
// suppression of already-seen tokens strengthens and above ≈1.3 the text starts avoiding
// necessary words (articles, common tokens) and degrades. SPECIAL VALUES: 0 or 1 disable it.
//
// Default 0 = DISABLED, research-grounded: repetition penalties distort the model's
// calibrated distribution, so GoAI leaves them off out of the box; when enabled, 1.1–1.3
// is the community-standard band (llama.cpp defaults repeat_penalty to 1.1) — start at 1.1.
func WithRepeatPenalty(p float64) SamplerOption { return func(s *Sampler) { s.RepeatPenalty = p } }

// WithFrequencyPenalty subtracts f · count(token in the penalty window) from each seen
// token's logit (the OpenAI-style frequency penalty).
//
// In plain terms: the more often the model has already used a word, the harder it is to
// use it again — proportional to the repeat count.
//
// Professional: a linear-in-count logit penalty. Boundary behavior — small f (≈0.1–0.5)
// gently reduces verbatim repeats; large f (>1) can blacklist frequent tokens entirely and
// break fluency. Unlike the multiplicative CTRL penalty it scales with how many times a
// token recurred, so runaway loops are punished progressively. SPECIAL VALUE: 0 disables it.
//
// Default 0 = DISABLED, research-grounded: the OpenAI API defaults frequency_penalty to 0
// (no distortion); a light 0.1–0.5 is the typical enabled range when repetition is a problem.
func WithFrequencyPenalty(f float64) SamplerOption { return func(s *Sampler) { s.FreqPenalty = f } }

// WithPresencePenalty subtracts p from the logit of every token that appears in the
// penalty window at all (the OpenAI-style presence penalty).
//
// In plain terms: a flat one-time nudge away from any word already used — encourages new
// vocabulary regardless of how many times a word appeared.
//
// Professional: a constant logit offset applied once per token present in the window (vs
// the frequency penalty's count-scaled offset). Boundary behavior — small p broadens topic
// coverage; large p (>1) forces the model onto unused, often less appropriate tokens.
// SPECIAL VALUE: 0 disables it.
//
// Default 0 = DISABLED, research-grounded: the OpenAI API defaults presence_penalty to 0;
// enable at 0.1–0.6 to push topical diversity.
func WithPresencePenalty(p float64) SamplerOption { return func(s *Sampler) { s.PresencePenalty = p } }

// WithPenaltyWindow limits every repetition penalty to the last n history tokens
// (llama.cpp's repeat_last_n).
//
// In plain terms: how far back the "have I said this already?" check looks — a short window
// only fights local loops, a long one fights repetition across the whole text.
//
// Professional: bounds the history slice the repeat/frequency/presence penalties scan.
// Boundary behavior — very small n misses longer-range repetition; very large n makes early
// common tokens permanently penalized, which can hurt fluency in long generations. SPECIAL
// VALUE: 0 = scan the ENTIRE history (not "no window").
//
// Default 0 = whole history, research-grounded: with penalties off by default the window is
// moot; when penalties are enabled, llama.cpp's repeat_last_n=64 is the usual practical bound.
func WithPenaltyWindow(n int) SamplerOption { return func(s *Sampler) { s.PenaltyLastN = n } }

// WithDRY enables the DRY ("Don't Repeat Yourself") sequence-repetition penalty (p-e-w
// 2024; llama.cpp standard sampler, see dry.go and §T573).
//
// In plain terms: unlike the per-word penalties, DRY catches the model repeating whole
// phrases it already produced — it spots when the next word would continue a chunk seen
// earlier and pushes back, harder the longer that repeated chunk gets.
//
// Professional: a token that would extend an already-seen suffix of length k is penalized by
// multiplier·base^(k−allowed). Boundary behavior — as the would-be repetition k grows the
// penalty grows exponentially, so long loops are crushed while short natural collocations
// (k ≤ allowed) stay free; larger multiplier raises the whole curve. SPECIAL VALUE: 0
// disables it.
//
// Default 0 = DISABLED, research-grounded: off by default (it reshapes the distribution);
// the reference/community default when enabled is 0.8 (koboldcpp, llama.cpp dry_multiplier).
func WithDRY(multiplier float64) SamplerOption {
	return func(s *Sampler) { s.DRYMultiplier = multiplier }
}

// WithDRYBase sets DRY's exponential growth base per extra matched token.
//
// In plain terms: how much steeper the penalty gets for each additional repeated word.
//
// Professional: the base of multiplier·base^(k−allowed). Boundary behavior — base near 1
// flattens DRY toward a constant penalty regardless of repetition length; larger base makes
// long repeats explode faster. SPECIAL VALUE: ≤0 falls back to the reference default 1.75.
//
// Default 1.75 (research-grounded): the koboldcpp/llama.cpp reference dry_base.
func WithDRYBase(b float64) SamplerOption { return func(s *Sampler) { s.DRYBase = b } }

// WithDRYAllowedLength sets the longest repeated run DRY leaves unpenalized.
//
// In plain terms: repeats up to this length are free, so common short phrases ("of the",
// "New York") aren't punished — only longer parroting is.
//
// Professional: the `allowed` exponent offset; penalty applies for match length k > allowed.
// Boundary behavior — small allowed penalizes even short natural n-grams (can hurt fluency);
// large allowed only catches very long loops. SPECIAL VALUE: ≤0 falls back to 2.
//
// Default 2 (research-grounded): the reference dry_allowed_length — bigrams stay free.
func WithDRYAllowedLength(n int) SamplerOption { return func(s *Sampler) { s.DRYAllowedLen = n } }

// WithDRYRange limits how far back DRY scans the history for repeats.
//
// In plain terms: how much of the text so far DRY checks against — the whole thing, or just
// the recent part.
//
// Professional: bounds the DRY history window (an O(window²) scan). Boundary behavior — small
// range only catches local loops and is cheaper; large range catches long-range repetition at
// higher cost. SPECIAL VALUE: 0 = scan the ENTIRE history.
//
// Default 0 = whole history, research-grounded: matches the reference dry_penalty_last_n=0
// ("whole context") default.
func WithDRYRange(n int) SamplerOption { return func(s *Sampler) { s.DRYRange = n } }

// WithDRYBreakers sets the token ids that DRY matches may not extend across.
//
// In plain terms: natural boundaries (newline, colon, quotes) where a "repeat" shouldn't
// count — DRY resets its matching when it hits one, so repeating across a paragraph break
// isn't treated as a loop.
//
// Professional: the token-level analogue of the reference implementation's "\n"/":"/quote
// sequence breakers. Pass the ids those strings tokenize to in the vocabulary in use.
// SPECIAL VALUE: empty/nil = no breakers (matches extend freely).
//
// Default empty, research-grounded: callers supply the ids because breakers are
// vocabulary-specific; the reference set is newline/colon/quotation marks.
func WithDRYBreakers(ids ...int) SamplerOption {
	return func(s *Sampler) { s.DRYBreakers = append([]int(nil), ids...) }
}

// WithXTC enables XTC ("eXclude Top Choices") sampling (p-e-w 2024, see xtc.go and §T574).
//
// In plain terms: occasionally the model drops its most obvious top answers and keeps a
// less-obvious one instead — a deliberate anti-cliché nudge that only fires sometimes, so
// coherence mostly survives.
//
// Professional: with the given per-draw probability, all tokens above WithXTCThreshold except
// the least-probable qualifier are removed before the draw (fires only when ≥2 tokens clear
// the threshold, so ≥1 always survives). Boundary behavior — probability near 1 excludes top
// choices almost every step (creative but risks derailment); near 0 rarely fires. SPECIAL
// VALUE: 0 disables it. Requires WithXTCThreshold to define "top choice".
//
// Default 0 = DISABLED, research-grounded: off by default; the community-typical firing
// probability when enabled is 0.5 (koboldcpp/llama.cpp xtc_probability).
func WithXTC(probability float64) SamplerOption {
	return func(s *Sampler) { s.XTCProbability = probability }
}

// WithXTCThreshold sets the probability a token needs to count as a "top choice" for XTC.
//
// In plain terms: how likely a word must be before XTC considers it an obvious pick eligible
// for exclusion.
//
// Professional: the qualifying probability for XTC's exclusion set. Boundary behavior — a low
// threshold makes many tokens "top choices" (aggressive exclusion); above 0.5 at most one
// token can ever qualify, so XTC can never fire (self-disabling). SPECIAL VALUE: >0.5 disables
// XTC naturally.
//
// Default 0 (with XTC off); the recommended enabled value is 0.1 (koboldcpp/llama.cpp
// xtc_threshold) — research-grounded as the reference default.
func WithXTCThreshold(t float64) SamplerOption {
	return func(s *Sampler) { s.XTCThreshold = t }
}

// NewSampler builds a deterministic sampler seeded by seed. Defaults: temperature
// 1, no top-k / top-p / min-p — configure with the options, e.g.
// NewSampler(seed, WithTemperature(0.8), WithTopP(0.95)).
func NewSampler(seed uint64, opts ...SamplerOption) *Sampler {
	s := &Sampler{Temperature: 1, rng: rand.New(rand.NewPCG(seed, 0x5a3d))}
	for _, o := range opts {
		o(s)
	}
	return s
}

// Greedy returns a deterministic argmax sampler. It still carries an rng (unused by
// argmax Sample) so callers that draw from its distribution — e.g. speculative
// decoding's residual/bonus draws — never hit a nil rng; the greedy outcome stays
// deterministic regardless.
func Greedy() *Sampler { return &Sampler{Temperature: 0, rng: rand.New(rand.NewPCG(0, 0x5a3d))} }

func argmax(x []float64) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

// kthLargest returns the k-th largest value in vals (k in [1, len(vals)]),
// permuting vals in place — pass a scratch COPY. It is quickselect with a
// deterministic median-of-three pivot: O(n) average, reproducible (no rng), used
// to threshold top-k without a full O(n log n) sort (§T626).
func kthLargest(vals []float64, k int) float64 {
	lo, hi := 0, len(vals)-1
	target := k - 1 // 0-indexed position in DESCENDING order
	for lo < hi {
		// median-of-three of vals[lo],vals[mid],vals[hi] → land the median at hi as
		// the pivot (guards against the sorted/reverse-sorted worst case).
		mid := lo + (hi-lo)/2
		if vals[mid] > vals[lo] {
			vals[lo], vals[mid] = vals[mid], vals[lo]
		}
		if vals[hi] > vals[lo] {
			vals[lo], vals[hi] = vals[hi], vals[lo]
		}
		if vals[mid] > vals[hi] {
			vals[mid], vals[hi] = vals[hi], vals[mid]
		}
		pivot := vals[hi] // the median of the three
		// Lomuto partition, DESCENDING: values strictly greater than the pivot move
		// to the front; equal values stay right (benign for distinct logits).
		i := lo
		for j := lo; j < hi; j++ {
			if vals[j] > pivot {
				vals[i], vals[j] = vals[j], vals[i]
				i++
			}
		}
		vals[i], vals[hi] = vals[hi], vals[i]
		switch {
		case i == target:
			return vals[i]
		case i < target:
			lo = i + 1
		default:
			hi = i - 1
		}
	}
	return vals[lo]
}

// Sample returns a token id for the given logits.
func (s *Sampler) Sample(logits []float64) int {
	if s.Temperature <= 0 {
		return argmax(logits) // greedy: deterministic, no rng draw
	}
	probs := s.Dist(logits)
	s.applyXTC(probs) // §T574: after truncation, before the draw; Dist stays pure
	u := s.rng.Float64()
	var cum float64
	for i, p := range probs {
		cum += p
		if u <= cum {
			return i
		}
	}
	return len(probs) - 1
}

// penalize returns the logits with the repetition penalties applied over the recent
// history via ApplyPenalties (on a copy; the input is never mutated). No-op when all
// penalties are off.
func (s *Sampler) penalize(logits []float64, history []int) []float64 {
	rp := s.RepeatPenalty > 0 && s.RepeatPenalty != 1
	if !rp && s.FreqPenalty == 0 && s.PresencePenalty == 0 {
		return logits
	}
	w := history
	if s.PenaltyLastN > 0 && len(w) > s.PenaltyLastN {
		w = w[len(w)-s.PenaltyLastN:]
	}
	if len(w) == 0 {
		return logits
	}
	out := append([]float64(nil), logits...)
	ApplyPenalties(out, w, s.RepeatPenalty, s.FreqPenalty, s.PresencePenalty)
	return out
}

// SampleWithHistory applies the sampler's repetition penalties (RepeatPenalty,
// FreqPenalty, PresencePenalty over the PenaltyLastN window) to the logits of tokens
// present in history, then samples like Sample. The penalties act before the
// greedy/temperature split, so they steer greedy decoding too — the classic cure for
// small-model repetition loops. history is the running sequence (prompt + generated),
// as maintained by the Generate loops. With no penalties configured it is exactly
// Sample. The speculative-decoding paths (SpeculativeGenerate, prompt-lookup) do NOT
// apply penalties: their lossless accept/reject math compares raw model
// distributions.
func (s *Sampler) SampleWithHistory(logits []float64, history []int) int {
	out := s.penalize(logits, history)
	if s.DRYMultiplier > 0 && len(out) > 0 {
		if &out[0] == &logits[0] { // penalize passed the original through: copy
			out = append([]float64(nil), logits...)
		}
		s.applyDRY(out, history) // DRY after the token-level penalties (§T573)
	}
	return s.Sample(out)
}

// Dist returns the probability distribution this sampler draws from for the given
// logits: temperature scaling, then top-k, top-p (nucleus) and min-p filtering,
// renormalized to sum 1. For greedy sampling (Temperature ≤ 0) it is a one-hot
// vector at the arg-max. Speculative decoding (§R53) uses it to obtain the target
// and draft distributions p and q that the accept/reject rule compares.
func (s *Sampler) Dist(logits []float64) []float64 {
	n := len(logits)
	if s.Temperature <= 0 {
		d := make([]float64, n)
		if n > 0 {
			d[argmax(logits)] = 1
		}
		return d
	}
	z := make([]float64, n)
	for i, v := range logits {
		z[i] = v / s.Temperature
	}

	// top-k: keep the k highest logits, mask the rest to −inf. Instead of a full
	// O(n log n) sort of the whole vocabulary just to drop the tail, quickselect
	// the k-th largest value in O(n) average and mask everything below it — the
	// kept set is identical for distinct logits (ties at the boundary keep all
	// their members, a benign, arguably-more-principled difference from the old
	// arbitrary sort tie-break). §T626.
	if s.TopK > 0 && s.TopK < n {
		scratch := make([]float64, n)
		copy(scratch, z)
		kv := kthLargest(scratch, s.TopK) // the k-th largest logit is the threshold
		for i := range z {
			if z[i] < kv {
				z[i] = math.Inf(-1)
			}
		}
	}

	// stable softmax
	m := math.Inf(-1)
	for _, v := range z {
		if v > m {
			m = v
		}
	}
	probs := make([]float64, n)
	var sum float64
	for i, v := range z {
		probs[i] = math.Exp(v - m)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}

	// top-p nucleus: keep smallest desc-prob prefix with cumsum ≥ p (crossing
	// token included), renormalize
	if s.TopP > 0 && s.TopP < 1 {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		// typed descending sort (no reflection, unlike sort.Slice) — same order.
		slices.SortFunc(idx, func(a, b int) int {
			switch {
			case probs[a] > probs[b]:
				return -1
			case probs[a] < probs[b]:
				return 1
			default:
				return 0
			}
		})
		var cum float64
		keep := make([]bool, n)
		for _, i := range idx {
			keep[i] = true
			cum += probs[i]
			if cum >= s.TopP {
				break // this token crosses p and is kept
			}
		}
		var ksum float64
		for i := range probs {
			if !keep[i] {
				probs[i] = 0
			} else {
				ksum += probs[i]
			}
		}
		for i := range probs {
			probs[i] /= ksum
		}
	}

	// min-p: keep tokens with prob ≥ MinP·max-prob (Nguyen et al. 2024, §R63), an
	// adaptive threshold — the top token always survives — then renormalize.
	if s.MinP > 0 {
		var pmax float64
		for _, p := range probs {
			if p > pmax {
				pmax = p
			}
		}
		thresh := s.MinP * pmax
		var ksum float64
		for i := range probs {
			if probs[i] < thresh {
				probs[i] = 0
			} else {
				ksum += probs[i]
			}
		}
		for i := range probs {
			probs[i] /= ksum
		}
	}

	// locally typical sampling (Meister et al. 2023): keep the smallest set of tokens
	// whose surprisal −log p is closest to the entropy H, by cumulative probability τ.
	if s.Typical > 0 && s.Typical < 1 {
		var h float64 // Shannon entropy in nats over the current distribution
		for _, p := range probs {
			if p > 0 {
				h -= p * math.Log(p)
			}
		}
		score := make([]float64, n) // |−log p − H|; masked tokens sort last (+inf)
		idx := make([]int, n)
		for i, p := range probs {
			idx[i] = i
			if p > 0 {
				score[i] = math.Abs(-math.Log(p) - h)
			} else {
				score[i] = math.Inf(1)
			}
		}
		slices.SortFunc(idx, func(a, b int) int { // typed ascending sort, no reflection
			switch {
			case score[a] < score[b]:
				return -1
			case score[a] > score[b]:
				return 1
			default:
				return 0
			}
		})
		var cum float64
		keep := make([]bool, n)
		for _, i := range idx {
			if probs[i] == 0 {
				break // no probability mass left to add
			}
			keep[i] = true // most-typical first; always keeps ≥1 token
			cum += probs[i]
			if cum >= s.Typical {
				break // this token crosses τ and is included
			}
		}
		var ksum float64
		for i := range probs {
			if !keep[i] {
				probs[i] = 0
			} else {
				ksum += probs[i]
			}
		}
		if ksum > 0 {
			for i := range probs {
				probs[i] /= ksum
			}
		}
	}

	// epsilon sampling: absolute-probability floor (Hewitt et al. 2022, §R91).
	if s.Epsilon > 0 {
		truncateAbove(probs, s.Epsilon)
	}
	// eta sampling: entropy-adaptive threshold η = min(ε, √ε·exp(−H)) over the current
	// distribution, H the Shannon entropy in nats (Hewitt et al. 2022, §R91).
	if s.Eta > 0 {
		var h float64
		for _, p := range probs {
			if p > 0 {
				h -= p * math.Log(p)
			}
		}
		truncateAbove(probs, math.Min(s.Eta, math.Sqrt(s.Eta)*math.Exp(-h)))
	}

	return probs
}

// truncateAbove zeroes every probability below thresh, always keeping the arg-max so
// at least one token survives (min_tokens_to_keep=1), then renormalizes.
func truncateAbove(probs []float64, thresh float64) {
	top := argmax(probs)
	var ksum float64
	for i := range probs {
		if probs[i] < thresh && i != top {
			probs[i] = 0
		} else {
			ksum += probs[i]
		}
	}
	if ksum == 0 {
		return
	}
	for i := range probs {
		probs[i] /= ksum
	}
}

// rowLogits copies row 0 of a [1,vocab] logits tensor to a slice. On the hot
// per-token sampling path, so it uses typed []T access (§base-perf) instead of a
// per-element AtF64 dispatch over the whole vocab.
func rowLogits(t *tensor.Tensor) []float64 {
	v := t.Shape()[1]
	out := make([]float64, v)
	tc := t.Contiguous()
	switch tc.Dtype() {
	case tensor.F64:
		copy(out, tc.Storage().F64()[:v]) // row 0 of a contiguous [1,v]
	case tensor.F32:
		src := tc.Storage().F32()
		for j := 0; j < v; j++ {
			out[j] = float64(src[j])
		}
	default:
		for j := 0; j < v; j++ {
			out[j] = tc.AtF64(0, j)
		}
	}
	return out
}

// GenerateOption configures a Generate call (functional options, §C12).
type GenerateOption func(*genConfig)

type genConfig struct{ be backend.Backend }

// WithBackend runs the decode loop on the given backend instead of the default.
// Single-token decode is dispatch-latency-bound, so for SMALL models the CPU is
// faster than the GPU (measured ~2.7× at dim 512), while large models still favour
// the GPU — the caller, who knows the model size and hardware, chooses (§T361).
func WithBackend(be backend.Backend) GenerateOption { return func(c *genConfig) { c.be = be } }

// Generate autoregressively produces up to maxNew tokens after the prompt,
// using the KV-cache. Returns prompt + generated tokens. Stops at the context
// limit. By default the decode runs on backend.Default(); WithBackend overrides it
// (single-token decode is often faster on the CPU for small models, §T361).
func (g *GPT) Generate(prompt []int, maxNew int, s TokenSampler, opts ...GenerateOption) ([]int, error) {
	var gc genConfig
	for _, o := range opts {
		o(&gc)
	}
	ctx := backend.NewContext()
	if gc.be != nil {
		ctx = ctx.WithBackend(gc.be)
	}
	cache := g.NewCache()
	out := append([]int(nil), prompt...)

	var logits *tensor.Tensor
	pos := 0
	for _, tok := range prompt {
		l, err := g.DecodeStep(ctx, cache, tok, pos)
		if err != nil {
			return nil, err
		}
		logits = l
		pos++
	}
	for range maxNew {
		if pos >= g.Config.Ctx {
			break
		}
		next := s.SampleWithHistory(rowLogits(logits), out)
		out = append(out, next)
		l, err := g.DecodeStep(ctx, cache, next, pos)
		if err != nil {
			return nil, err
		}
		logits = l
		pos++
	}
	return out, nil
}
