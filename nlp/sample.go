package nlp

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

// Sampler turns a logit vector into a token id. Pipeline (matching HuggingFace
// generation): repetition penalties (via SampleWithHistory) → temperature scaling
// → top-nσ logit mask → top-k mask → softmax → top-p (nucleus) mask → min-p mask
// → epsilon/eta mask → multinomial sample. Temperature ≤ 0 selects greedy (argmax), which ignores the
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
// before sampling.
//
// Every filter here EXCEPT locally-typical keeps the arg-max, so the distribution is
// never left empty: top-nσ, top-k and top-p keep it structurally, and the probability
// floors (min-p, epsilon, eta) are exempted from their own threshold in
// truncateAboveKeeping — enforced there rather than assumed, because min-p once
// skipped that guard and normalized an all-zero vector into NaN (§B73). Typical is the
// deliberate exception: trimming the too-predictable is the point of the method, so at
// a small τ it CAN zero the most probable token. It still always keeps at least one.
//
// # Building one
//
// [NewSampler] is the intended constructor: it seeds the RNG reproducibly and
// validates every knob, rejecting out-of-range values loudly (see [SamplerOption]).
// Because all knobs are exported fields, a struct literal also compiles:
//
//	s := &nlp.Sampler{Temperature: 0.8, TopK: 40, TopP: 0.9}
//
// That is supported, with two caveats worth knowing. Its RNG is seeded lazily from
// the process-global source on the first draw, so the sampler is random but NOT
// reproducible — pass a seed to [NewSampler] if you need a repeatable run. And no
// option ran, so nothing validated the fields; a value outside a knob's domain is
// simply used as given. The truncation filters are individually total (no field
// value can produce a NaN distribution), but they cannot tell you that MinP: 5 was
// meant to be 0.05.
//
// Note the zero value samples GREEDILY: Temperature defaults to 0 in a struct
// literal, and Temperature ≤ 0 short-circuits to the arg-max, ignoring TopK/TopP/
// MinP entirely. [NewSampler] defaults Temperature to 1 for that reason.
type Sampler struct {
	Temperature float64 // softmax temperature; <1 sharper, >1 flatter (default 1)
	TopK        int     // keep the K highest-prob tokens; 0 = off
	TopP        float64 // nucleus: keep smallest set with cumulative prob ≥ P; 0 = off
	MinP        float64 // keep tokens with prob ≥ MinP·maxProb; 0 = off
	Epsilon     float64 // epsilon sampling: keep tokens with prob ≥ Epsilon; 0 = off
	Eta         float64 // eta sampling: keep tokens with prob ≥ min(Eta, √Eta·exp(−H)); 0 = off
	Typical     float64 // locally typical: keep tokens whose surprisal is nearest the entropy until cum-prob ≥ τ; 0/1 = off
	TopNSigma   float64 // top-nσ: keep tokens with logit ≥ max − n·σ (Tang et al. 2024, see sample_topnsigma.go); ≤0 = off

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
//
// Out-of-range values are REJECTED, not clamped: an option given a value outside its
// knob's domain panics from NewSampler with a message naming the knob and the value.
// The rationale is that every such value is a typo no correct program can want —
// WithMinP(5) is min-p confused with a percentage, WithTopK(-5) is not a token count
// — and the alternatives are both worse. Clamping silently substitutes a setting the
// caller never asked for; leaving it alone is how WithMinP(5) came to emit the last
// vocabulary id on every step in the first place (§B73). Panicking fires at
// construction, once, on the first run, before a single token is generated.
//
// Documented SPECIAL VALUES are not out of range and never panic: 0 disables every
// truncation filter, TopP == 1 and Typical == 1 are the "keep everything" idiom, and
// Temperature ≤ 0 selects greedy decoding. Only genuinely undefined values — a
// probability above 1 or below 0, NaN, a negative count — are rejected.
type SamplerOption func(*Sampler)

// unitRange panics unless v is a probability in [0,1]. Written as a negated
// conjunction so NaN, which compares false against everything, is rejected too.
func unitRange(knob string, v float64) float64 {
	if !(v >= 0 && v <= 1) {
		panic(fmt.Sprintf("nlp: %s(%v) out of range [0,1] — sampler probabilities are "+
			"fractions, not percentages (min-p 0.05, not 5; top-p 0.9, not 90)", knob, v))
	}
	return v
}

// nonNegative panics unless v is a usable token count (0 = disabled).
func nonNegative(knob string, v int) int {
	if v < 0 {
		panic(fmt.Sprintf("nlp: %s(%d) is negative — pass a token count, or 0 to disable", knob, v))
	}
	return v
}

// WithTemperature sets the softmax temperature — the single knob that trades safety
// for creativity.
//
// In plain terms: temperature is how adventurous the model is when it picks the next
// word. Low = it almost always takes the word it's most sure of (predictable, repetitive);
// high = it gives unlikely words a real chance (surprising, but more mistakes).
//
// Professional: logits are divided by t before the softmax, so t scales the entropy of
// the sampled distribution. Boundary behavior — as t→0⁺ the distribution collapses onto
// the argmax; as t grows the distribution flattens toward uniform and coherence degrades,
// with runaway incoherence typically past t≈1.5. t = 1 leaves the model's own trained
// distribution unchanged.
//
// SPECIAL VALUE: t ≤ 0 is an explicit greedy short-circuit — no RNG draw, and every
// truncation knob is IGNORED. This is the one knob whose out-of-range values are kept
// rather than rejected (t = −0.5 means greedy, not an error), because greedy is the
// mathematical limit of t → 0⁺ and callers rely on it. The cost is that a sampler
// configured `WithTemperature(0), WithTopK(40), WithMinP(0.05)` silently discards the
// truncation: it is greedy, and the two filters do nothing. If you want truncated
// sampling, t must be > 0. Repetition penalties still apply either way — they act on
// the logits before the greedy/temperature split.
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
//
// PANICS on a negative k (not a token count under any reading; see [SamplerOption]).
// k ≥ the vocabulary size is legal and simply keeps everything.
func WithTopK(k int) SamplerOption {
	return func(s *Sampler) { s.TopK = nonNegative("WithTopK", k) }
}

// WithTopP keeps the smallest set of most-probable tokens whose probabilities sum to at
// least p (nucleus sampling), zeroing the rest.
//
// In plain terms: keep just enough of the model's top guesses to cover p of its confidence
// (e.g. 95%) and drop the improbable remainder — the kept set shrinks when the model is
// sure and grows when it's hesitant.
//
// Professional: nucleus sampling (Holtzman et al. 2019 arXiv:1904.09751, §R34). The cutoff is
// distribution-adaptive, unlike top-k. Boundary behavior — as p → 0⁺ only the single most
// probable token survives (greedy-like); p = 1 keeps everything (disabled). SPECIAL VALUES:
// 0 or 1 disable the filter.
//
// PANICS on p outside [0,1] — WithTopP(90) is p expressed as a percentage, which would
// otherwise silently disable nucleus sampling altogether (see [SamplerOption]).
//
// Default 0 = DISABLED, research-grounded: GoAI's zero-config default truncates via min-p
// (WithMinP), which Nguyen et al. 2024 (§R63) show is more robust than nucleus at higher
// temperatures; enable nucleus explicitly at p≈0.9–0.95 (Holtzman's reported sweet spot,
// also Hugging Face's default when nucleus is used) to match the conventional pipeline.
func WithTopP(p float64) SamplerOption {
	return func(s *Sampler) { s.TopP = unitRange("WithTopP", p) }
}

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
// at higher t. Boundary behavior — as p → 0⁺ nothing is filtered; at p = 1 the cutoff
// equals the peak itself, so only tokens exactly tied with the top token survive
// (greedy, or a uniform draw among the tied leaders on a flat distribution). The
// arg-max is exempt from the cutoff and therefore survives at EVERY p, which is what
// keeps the distribution normalizable. SPECIAL VALUE: 0 disables it.
//
// PANICS on p outside [0,1]. min-p is a FRACTION of the top token's probability, so
// WithMinP(5) is the percentage mistake, not a stricter filter; before §B73 it drove the
// threshold above the peak, zeroed every token including the arg-max, and normalized an
// all-zero vector into NaN — after which the sampler emitted the last vocabulary id, in
// silence, on every step. See [SamplerOption].
//
// Default 0 = DISABLED (a bare NewSampler samples untruncated at t=1); the RECOMMENDED
// enabled value is 0.05–0.1 — Nguyen et al. 2024 report 0.05–0.1 as the quality sweet spot
// across tasks, and llama.cpp defaults min_p to 0.05. This is GoAI's suggested single
// truncation knob: NewSampler(seed, WithMinP(0.05)) is a strong general-purpose setup.
func WithMinP(p float64) SamplerOption {
	return func(s *Sampler) { s.MinP = unitRange("WithMinP", p) }
}

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
//
// PANICS on eps outside [0,1] (a probability floor above 1 is unsatisfiable; see
// [SamplerOption]). An eps above the peak leaves only the arg-max, which is exempt.
func WithEpsilon(eps float64) SamplerOption {
	return func(s *Sampler) { s.Epsilon = unitRange("WithEpsilon", eps) }
}

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
//
// PANICS on eps outside [0,1] (see [SamplerOption]).
func WithEta(eps float64) SamplerOption {
	return func(s *Sampler) { s.Eta = unitRange("WithEta", eps) }
}

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
// 0 or 1 disable it.
//
// Default 0 = DISABLED, research-grounded: off by default; Meister et al. 2023 use τ≈0.9–0.95
// (also Hugging Face's TypicalLogitsWarper default 0.9).
//
// PANICS on τ outside [0,1] (see [SamplerOption]); τ = 1 stays the documented
// "keep everything" idiom.
func WithTypical(tau float64) SamplerOption {
	return func(s *Sampler) { s.Typical = unitRange("WithTypical", tau) }
}

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
// PANICS on a probability outside [0,1] (see [SamplerOption]).
func WithXTC(probability float64) SamplerOption {
	return func(s *Sampler) { s.XTCProbability = unitRange("WithXTC", probability) }
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
//
// PANICS on t outside [0,1] (see [SamplerOption]); t > 0.5 remains the documented
// self-disabling range, not an error.
func WithXTCThreshold(t float64) SamplerOption {
	return func(s *Sampler) { s.XTCThreshold = unitRange("WithXTCThreshold", t) }
}

// NewSampler builds a deterministic sampler seeded by seed. Defaults: temperature
// 1, no top-k / top-p / min-p — configure with the options, e.g.
// NewSampler(seed, WithTemperature(0.8), WithTopP(0.95)).
//
// The same seed always yields the same token sequence for the same logits, which is
// what makes generations reproducible; build a Sampler as a struct literal instead
// and you get a lazily seeded, non-reproducible RNG (see [Sampler]).
//
// It PANICS if any option carries an out-of-range value — WithMinP(5), WithTopP(90),
// WithTopK(-5). That is deliberate: those are typos, and the failure they used to
// cause was silent (§B73). [SamplerOption] documents the policy and the special
// values that are explicitly NOT errors.
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

// quickselectIdxDesc partitions idx in place so that idx[:k] hold the k indices
// with the LARGEST key[idx[·]] values (arbitrary internal order), O(n) average. It
// uses a THREE-WAY (Dutch-flag) partition so runs of equal keys are grouped in one
// pass — essential here because an earlier truncation (top-k, min-p) zeroes most of
// the vocabulary, and a two-way quickselect degrades to O(n²) on those thousands of
// identical zeros (§T629). Median-of-three pivot keeps it deterministic.
func quickselectIdxDesc(idx []int, key []float64, k int) {
	lo, hi := 0, len(idx)-1
	target := k - 1
	for lo < hi {
		mid := lo + (hi-lo)/2
		if key[idx[mid]] > key[idx[lo]] {
			idx[lo], idx[mid] = idx[mid], idx[lo]
		}
		if key[idx[hi]] > key[idx[lo]] {
			idx[lo], idx[hi] = idx[hi], idx[lo]
		}
		if key[idx[mid]] > key[idx[hi]] {
			idx[mid], idx[hi] = idx[hi], idx[mid]
		}
		pivot := key[idx[mid]] // median of three
		// 3-way: [lo..lt-1] > pivot, [lt..gt] == pivot, [gt+1..hi] < pivot.
		lt, gt, i := lo, hi, lo
		for i <= gt {
			switch {
			case key[idx[i]] > pivot:
				idx[lt], idx[i] = idx[i], idx[lt]
				lt++
				i++
			case key[idx[i]] < pivot:
				idx[gt], idx[i] = idx[i], idx[gt]
				gt--
			default:
				i++
			}
		}
		switch {
		case target < lt:
			hi = lt - 1
		case target <= gt:
			return // the k-th largest lies in the equal-to-pivot band → done
		default:
			lo = gt + 1
		}
	}
}

// sortIdxDescByKey sorts idx so key[idx[·]] is descending (typed, no reflection).
func sortIdxDescByKey(idx []int, key []float64) {
	// Small-n fallback for sortIdxDescByProb, which radix-sorts above radixSortCutoff.
	// The comparison sort is deliberate here: below the cutoff its lower constant
	// beats a radix pass, and it is also the reference the radix path is checked
	// against.
	//perfscan:ignore PS3002 small-n fallback; sortIdxDescByProb is the radix path
	slices.SortFunc(idx, func(a, b int) int {
		switch {
		case key[a] > key[b]:
			return -1
		case key[a] < key[b]:
			return 1
		default:
			return 0
		}
	})
}

// radixSortCutoff is the index count above which a full sort switches from the
// comparison sort to the LSD radix. The measured crossover is ~1024: radix is
// ~6% slower at n=512, ~1.17× faster at n=1024, and 2.4× at n=2048 (its fixed
// 8-pass + allocation overhead amortises out), so 1024 captures every sort that
// benefits while leaving small sorts on the lower-constant comparison path.
const radixSortCutoff = 1024

// sortIdxDescByProb orders idx so that key[idx[0]] ≥ key[idx[1]] ≥ … for
// NON-NEGATIVE keys (post-softmax probabilities). For large n it uses an 8-pass
// LSD radix sort on the IEEE-754 bits: for x ≥ 0 math.Float64bits is monotonic
// in x, so sorting the bit patterns orders the values, with no per-comparison
// closure call. The resulting value sequence is identical to sortIdxDescByKey
// for distinct keys — the nucleus threshold depends only on the value order, so
// top-p truncation is unchanged (§T627). Small n defers to the comparison sort.
// radixScratch holds the 3×n scratch buffers of the LSD radix sort. sortIdxDescByProb
// runs once per generated token (top-p / typical sampling), so a fresh scratch each
// call is O(T) large allocations over a decode; pooled + grown-on-demand it is
// amortized allocation-free. Pure scratch — the sort result is unchanged.
type radixScratch struct {
	bits    []uint64
	tmpBits []uint64
	tmpIdx  []int
}

var radixScratchPool = sync.Pool{New: func() any { return new(radixScratch) }}

func sortIdxDescByProb(idx []int, key []float64) {
	n := len(idx)
	if n < radixSortCutoff {
		sortIdxDescByKey(idx, key)
		return
	}
	sc := radixScratchPool.Get().(*radixScratch)
	if cap(sc.bits) < n {
		sc.bits = make([]uint64, n)
		sc.tmpBits = make([]uint64, n)
		sc.tmpIdx = make([]int, n)
	}
	// bits[i] = ^Float64bits(key[idx[i]]): inverting makes an ASCENDING radix
	// pass emit values in DESCENDING order (largest probability first).
	bits, tmpBits, tmpIdx := sc.bits[:n], sc.tmpBits[:n], sc.tmpIdx[:n]
	for i, id := range idx {
		bits[i] = ^math.Float64bits(key[id])
	}
	var count [256]int
	for shift := uint(0); shift < 64; shift += 8 {
		for i := range count {
			count[i] = 0
		}
		for i := 0; i < n; i++ {
			count[(bits[i]>>shift)&0xff]++
		}
		sum := 0
		for i := 0; i < 256; i++ {
			c := count[i]
			count[i] = sum
			sum += c
		}
		for i := 0; i < n; i++ {
			b := (bits[i] >> shift) & 0xff
			pos := count[b]
			count[b]++
			tmpIdx[pos] = idx[i]
			tmpBits[pos] = bits[i]
		}
		idx, tmpIdx = tmpIdx, idx
		bits, tmpBits = tmpBits, bits
	}
	// 8 passes (even) ⇒ the swapped idx alias is the caller's slice again, now
	// holding the sorted order; no final copy needed. sc.{bits,tmpBits,tmpIdx} still
	// alias their backing arrays (the swaps only rebound the locals), so it pools cleanly.
	radixScratchPool.Put(sc)
}

// nucleusIdxPool reuses nucleusTopP's vocab-sized index scratch across the per-token
// top-p calls (T945, the sibling of T944's radix-scratch pool: the remaining alloc).
// typicalTruncate (T947) shares it for its own idx scratch.
var nucleusIdxPool = sync.Pool{New: func() any { return []int(nil) }}

// typicalScorePool / typicalKeepPool reuse typicalTruncate's per-call score ([]float64)
// and keep ([]bool) scratch across the per-token typical calls (T947).
var typicalScorePool = sync.Pool{New: func() any { return []float64(nil) }}
var typicalKeepPool = sync.Pool{New: func() any { return []bool(nil) }}

// float64ScratchPool reuses Dist's per-call working buffers (the temperature-scaled
// logits z, and top-k's kthLargest copy) across per-token Dist calls (T948). The
// returned distribution is a separate slice, so these are pure scratch.
var float64ScratchPool = sync.Pool{New: func() any { return []float64(nil) }}

// penalizeBufPool reuses the per-token penalized-logits copy in SampleWithHistory (T949):
// penalize/DRY need a mutable copy of logits (the input is never mutated), which was a
// fresh make per token when penalties are on. The copy lives only until Sample returns,
// so SampleWithHistory pools it (Put once, iff a copy was actually taken).
var penalizeBufPool = sync.Pool{New: func() any { return []float64(nil) }}

// sampleProbsPool reuses the per-token probability buffer that Sample draws from
// (T950). Sample builds the distribution via distInto into this pooled buffer and
// discards it after the multinomial draw — it never escapes, so reuse is safe, unlike
// Dist's public return which speculative decoding retains to compare distributions.
var sampleProbsPool = sync.Pool{New: func() any { return []float64(nil) }}

// nucleusTopP applies top-p (nucleus) truncation to probs in place: keep the
// minimal set of highest-probability tokens whose mass ≥ p, then renormalize. The
// nucleus is almost always tiny relative to the vocabulary, so instead of sorting
// all n it quickselects the top-K candidates (K ≫ any realistic nucleus), sorts
// only those, and accumulates in exact descending order. Only when the nucleus
// genuinely exceeds K does it fall back to a full sort — so the result is identical
// to the old full-sort prefix (same accumulation order ⟹ bit-exact for distinct
// probs) and the cost is never worse than that sort (§T627).
func nucleusTopP(probs []float64, p float64) {
	n := len(probs)
	idx := nucleusIdxPool.Get().([]int)
	if cap(idx) < n {
		idx = make([]int, n)
	} else {
		idx = idx[:n]
	}
	defer nucleusIdxPool.Put(idx) // covers the early-return paths; idx is not retained
	var sumSq float64             // Σp² — accumulated for free in the index-init pass
	for i := range idx {
		idx[i] = i
		sumSq += probs[i] * probs[i]
	}
	const k = 512
	// Cheap provable test for whether the K-candidate prefix can possibly reach the
	// nucleus. By Cauchy–Schwarz the k largest probs sum to at most √(k·Σ_topk p²) ≤
	// √(k·Σp²), so when k·Σp² < p² the top-k prefix provably falls short of p — the
	// quickselect+partial-sort would fail and fall through to the full sort anyway.
	// Skip straight to it and save the wasted O(n) selection. This is the high-entropy
	// case (temperature ≳ 1, large vocab) where the nucleus spans a big fraction of the
	// vocabulary — the default temp-1 top-p-0.9 config included; measured ~1.3–1.4× on
	// Dist there. sumSq = Σp² was accumulated for free in the index-init loop above.
	// The result is unchanged either way — both paths accumulate in exact
	// descending order and find the same crossing token; only the routing differs, so
	// the §T627 full-sort parity (TestDistQuickselectParity, 1e-12) still holds. The
	// squared form avoids a sqrt: √(k·Σp²) < p ⟺ k·Σp² < p² (both sides ≥ 0).
	if k < n && float64(k)*sumSq >= p*p {
		quickselectIdxDesc(idx, probs, k)
		top := idx[:k]
		sortIdxDescByProb(top, probs)
		var cum float64
		for _, i := range top {
			cum += probs[i]
			if cum >= p { // crossing token inside the K-candidate prefix
				truncateNucleus(probs, probs[i])
				return
			}
		}
		// nucleus larger than K candidates → exact full-sort fallback below
	}
	sortIdxDescByProb(idx, probs)
	var cum float64
	for _, i := range idx {
		cum += probs[i]
		if cum >= p {
			truncateNucleus(probs, probs[i])
			return
		}
	}
}

// truncateNucleus zeroes every probability strictly below thresh (the crossing
// token's probability) and renormalizes over the survivors — the top-p keep-set for
// distinct probabilities, matching the old prefix mask.
func truncateNucleus(probs []float64, thresh float64) {
	var ksum float64
	for i := range probs {
		if probs[i] < thresh {
			probs[i] = 0
		} else {
			ksum += probs[i]
		}
	}
	if ksum > 0 {
		inv := 1 / ksum // reciprocal-multiply the renormalize
		for i := range probs {
			probs[i] *= inv
		}
	}
}

// quickselectIdxAsc partitions idx in place so that idx[:k] hold the k indices with
// the SMALLEST key[idx[·]] values (mirror of quickselectIdxDesc, three-way for the
// same duplicate-key robustness — masked tokens share a +inf score), O(n) average.
func quickselectIdxAsc(idx []int, key []float64, k int) {
	lo, hi := 0, len(idx)-1
	target := k - 1
	for lo < hi {
		mid := lo + (hi-lo)/2
		if key[idx[mid]] < key[idx[lo]] {
			idx[lo], idx[mid] = idx[mid], idx[lo]
		}
		if key[idx[hi]] < key[idx[lo]] {
			idx[lo], idx[hi] = idx[hi], idx[lo]
		}
		if key[idx[mid]] < key[idx[hi]] {
			idx[mid], idx[hi] = idx[hi], idx[mid]
		}
		pivot := key[idx[mid]] // median of three
		// 3-way: [lo..lt-1] < pivot, [lt..gt] == pivot, [gt+1..hi] > pivot.
		lt, gt, i := lo, hi, lo
		for i <= gt {
			switch {
			case key[idx[i]] < pivot:
				idx[lt], idx[i] = idx[i], idx[lt]
				lt++
				i++
			case key[idx[i]] > pivot:
				idx[gt], idx[i] = idx[i], idx[gt]
				gt--
			default:
				i++
			}
		}
		switch {
		case target < lt:
			hi = lt - 1
		case target <= gt:
			return // the k-th smallest lies in the equal-to-pivot band → done
		default:
			lo = gt + 1
		}
	}
}

// sortIdxAscByKey sorts idx so key[idx[·]] is ascending (typed, no reflection).
func sortIdxAscByKey(idx []int, key []float64) {
	// Small-n fallback for sortIdxAscByScore, which radix-sorts above radixSortCutoff.
	// The comparison sort is deliberate here: below the cutoff its lower constant
	// beats a radix pass, and it is also the reference the radix path is checked
	// against.
	//perfscan:ignore PS3002 small-n fallback; sortIdxAscByScore is the radix path
	slices.SortFunc(idx, func(a, b int) int {
		switch {
		case key[a] < key[b]:
			return -1
		case key[a] > key[b]:
			return 1
		default:
			return 0
		}
	})
}

// sortIdxAscByScore orders idx so that key[idx[0]] ≤ key[idx[1]] ≤ … for
// NON-NEGATIVE keys (the typical score |−log p − H|, with masked tokens at
// +Inf). Like sortIdxDescByProb it radix-sorts the IEEE-754 bits — but without
// the inversion, so an ascending pass emits values smallest-first; +Inf has the
// largest positive bit pattern and lands last, exactly where the masked tail
// belongs. Identical value order to sortIdxAscByKey for distinct scores, so the
// typical set is unchanged (§T628). Small n defers to the comparison sort.
func sortIdxAscByScore(idx []int, key []float64) {
	n := len(idx)
	if n < radixSortCutoff {
		sortIdxAscByKey(idx, key)
		return
	}
	sc := radixScratchPool.Get().(*radixScratch) // shared with sortIdxDescByProb (T944)
	if cap(sc.bits) < n {
		sc.bits = make([]uint64, n)
		sc.tmpBits = make([]uint64, n)
		sc.tmpIdx = make([]int, n)
	}
	bits, tmpBits, tmpIdx := sc.bits[:n], sc.tmpBits[:n], sc.tmpIdx[:n]
	for i, id := range idx {
		bits[i] = math.Float64bits(key[id]) // ascending bits ⇒ ascending value (key ≥ 0)
	}
	var count [256]int
	for shift := uint(0); shift < 64; shift += 8 {
		for i := range count {
			count[i] = 0
		}
		for i := 0; i < n; i++ {
			count[(bits[i]>>shift)&0xff]++
		}
		sum := 0
		for i := 0; i < 256; i++ {
			c := count[i]
			count[i] = sum
			sum += c
		}
		for i := 0; i < n; i++ {
			b := (bits[i] >> shift) & 0xff
			pos := count[b]
			count[b]++
			tmpIdx[pos] = idx[i]
			tmpBits[pos] = bits[i]
		}
		idx, tmpIdx = tmpIdx, idx
		bits, tmpBits = tmpBits, bits
	}
	radixScratchPool.Put(sc)
}

// applyKeep zeroes every probability whose keep flag is false and renormalizes over
// the survivors — the exact prefix mask the old typical/top-p loops used.
func applyKeep(probs []float64, keep []bool) {
	var ksum float64
	for i := range probs {
		if !keep[i] {
			probs[i] = 0
		} else {
			ksum += probs[i]
		}
	}
	if ksum > 0 {
		inv := 1 / ksum // reciprocal-multiply the renormalize
		for i := range probs {
			probs[i] *= inv
		}
	}
}

// typicalTruncate applies locally-typical truncation (Meister et al. 2023) to probs
// in place: keep the smallest set of tokens whose surprisal −log p is closest to the
// entropy H, in ascending |−log p − H| order, until their cumulative probability ≥ τ.
// Like nucleusTopP it bounds the work: quickselect the K lowest-score candidates,
// sort only those, and accumulate — falling back to a full sort only if the typical
// set exceeds K. The kept prefix (by index) is identical to the old full sort, so the
// result is bit-exact for distinct scores (§T628). Zero-probability tokens carry a
// +inf score and are never kept, matching the old `if probs[i]==0 break`.
func typicalTruncate(probs []float64, tau float64) {
	n := len(probs)
	// pool the three per-call vocab-sized scratch buffers (T947); defer Put covers the
	// early-return paths. |−log p − H|; masked (zero-prob) tokens sort last.
	score := typicalScorePool.Get().([]float64)
	if cap(score) < n {
		score = make([]float64, n)
	} else {
		score = score[:n]
	}
	defer typicalScorePool.Put(score)
	idx := nucleusIdxPool.Get().([]int)
	if cap(idx) < n {
		idx = make([]int, n)
	} else {
		idx = idx[:n]
	}
	defer nucleusIdxPool.Put(idx)
	// One log per token: stash log(p) in score while accumulating the Shannon entropy
	// H = −Σ p·log p in the same pass, then fold each score into the typicality
	// |−log p − H|. math.Log is deterministic, so reusing the stored value is
	// bit-identical to recomputing it (the §T628/1e-12 parity holds) while halving the
	// transcendental work — the dominant cost of this, the slowest truncation sampler.
	var h float64 // Shannon entropy in nats over the current distribution
	for i, p := range probs {
		idx[i] = i
		if p > 0 {
			lp := math.Log(p)
			score[i] = lp
			h -= p * lp
		} else {
			score[i] = math.Inf(1)
		}
	}
	for i, p := range probs {
		if p > 0 {
			score[i] = math.Abs(-score[i] - h)
		}
	}
	keep := typicalKeepPool.Get().([]bool)
	if cap(keep) < n {
		keep = make([]bool, n)
	} else {
		keep = keep[:n]
		clear(keep) // a reused buffer may hold stale true flags
	}
	defer typicalKeepPool.Put(keep)
	const k = 512
	if k < n {
		quickselectIdxAsc(idx, score, k)
		top := idx[:k]
		sortIdxAscByScore(top, score)
		var cum float64
		for _, i := range top {
			if probs[i] == 0 {
				applyKeep(probs, keep) // reached the masked tail → typical set complete
				return
			}
			keep[i] = true
			cum += probs[i]
			if cum >= tau {
				applyKeep(probs, keep)
				return
			}
		}
		// typical set larger than K candidates → exact full-sort fallback below
		for i := range keep {
			keep[i] = false
		}
	}
	sortIdxAscByScore(idx, score)
	var cum float64
	for _, i := range idx {
		if probs[i] == 0 {
			break
		}
		keep[i] = true
		cum += probs[i]
		if cum >= tau {
			break
		}
	}
	applyKeep(probs, keep)
}

// Sample returns a token id for the given logits.
//
// Temperature ≤ 0 short-circuits to the arg-max without drawing from the RNG;
// otherwise the id is drawn from [Sampler.Dist] (after XTC, which acts on the draw
// only). A [Sampler] built as a struct literal rather than by [NewSampler] gets its
// RNG seeded lazily here — see the Sampler type doc for what that costs you.
func (s *Sampler) Sample(logits []float64) int {
	if s.Temperature <= 0 {
		return argmax(logits) // greedy: deterministic, no rng draw
	}
	rng := s.source() // materialize BEFORE applyXTC, which draws from it too
	// Pooled distribution buffer: probs is consumed by the draw below and never
	// escapes Sample, so unlike Dist's retained public return it can be reused (T950).
	dst := sampleProbsPool.Get().([]float64)
	if cap(dst) < len(logits) {
		dst = make([]float64, len(logits))
	} else {
		dst = dst[:len(logits)]
	}
	probs := s.distInto(dst, logits)
	defer sampleProbsPool.Put(probs) // Put on every draw-loop return path
	s.applyXTC(probs)                // §T574: after truncation, before the draw; Dist stays pure
	u := rng.Float64()
	var cum float64
	for i, p := range probs {
		cum += p
		if u <= cum {
			return i
		}
	}
	// Fall-through. For a well-formed distribution this is reachable only by
	// floating-point rounding — the accumulated cum can land a few ulps below a u
	// very close to 1 — so the answer is the last token whose cumulative interval
	// really does contain 1: the last one with NON-ZERO probability. Returning a bare
	// len(probs)−1 instead would emit the last vocabulary id, which after any
	// truncation filter is almost always a masked, zero-probability token (§B73).
	for i := len(probs) - 1; i >= 0; i-- {
		if probs[i] > 0 {
			return i
		}
	}
	return argmax(probs) // degenerate input (empty or no positive mass)
}

// source returns the sampler's RNG, seeding it on first use when the Sampler was
// built as a struct literal instead of by NewSampler/Greedy (rng is unexported, so a
// literal leaves it nil and every knob-setting field is exported, which invites
// exactly that). Lazy seeding is preferred over panicking: a nil RNG is not a caller
// error the caller can even see, and a Sampler that samples is strictly more useful
// than one that crashes. The seed is drawn from the process-global source, so such a
// sampler is random-but-unreproducible; NewSampler(seed, …) is the reproducible path.
func (s *Sampler) source() *rand.Rand {
	if s.rng == nil {
		s.rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	return s.rng
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
	if len(w) == 0 || len(logits) == 0 {
		return logits
	}
	out := penalizeBufPool.Get().([]float64) // Put by SampleWithHistory after Sample returns
	if cap(out) < len(logits) {
		out = make([]float64, len(logits))
	} else {
		out = out[:len(logits)]
	}
	copy(out, logits)
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
	pooled := len(out) > 0 && &out[0] != &logits[0] // penalize returned a pooled copy
	if s.DRYMultiplier > 0 && len(out) > 0 {
		if !pooled { // penalize passed the original through: take a pooled copy for DRY
			c := penalizeBufPool.Get().([]float64)
			if cap(c) < len(logits) {
				c = make([]float64, len(logits))
			} else {
				c = c[:len(logits)]
			}
			copy(c, logits)
			out, pooled = c, true
		}
		s.applyDRY(out, history) // DRY after the token-level penalties (§T573)
	}
	tok := s.Sample(out)
	if pooled {
		penalizeBufPool.Put(out) // safe: Sample consumed out and returns an int, not out
	}
	return tok
}

// Dist returns the probability distribution this sampler draws from for the given
// logits: temperature scaling, then top-nσ, top-k, top-p (nucleus) and min-p
// filtering, renormalized to sum 1. For greedy sampling (Temperature ≤ 0) it is a one-hot
// vector at the arg-max. Speculative decoding (§R53) uses it to obtain the target
// and draft distributions p and q that the accept/reject rule compares.
func (s *Sampler) Dist(logits []float64) []float64 {
	if s.Temperature <= 0 {
		d := make([]float64, len(logits))
		if len(logits) > 0 {
			d[argmax(logits)] = 1
		}
		return d
	}
	// Fresh dst: Dist's callers (speculative decoding, §R53) retain the returned
	// distribution to compare p against q, so it must be a slice they own.
	return s.distInto(make([]float64, len(logits)), logits)
}

// distInto is Dist's temperature>0 core: it writes the sampling distribution into
// the caller-provided dst (len(dst) must equal len(logits)) and returns it.
// Splitting it out lets Sample pass a pooled buffer for the per-token probs it
// draws from and immediately discards (T950), while Dist keeps allocating a fresh
// dst for callers that retain the result. Precondition: Temperature > 0 — the
// greedy one-hot stays in Dist, and the ExpSumF64 softmax below overwrites every
// dst lane, so a dirty pooled buffer is fully defined before any filter reads it.
func (s *Sampler) distInto(dst, logits []float64) []float64 {
	n := len(logits)
	z := float64ScratchPool.Get().([]float64)
	if cap(z) < n {
		z = make([]float64, n)
	} else {
		z = z[:n]
	}
	defer float64ScratchPool.Put(z) // z is scratch; dst holds the returned distribution
	invT := 1 / s.Temperature       // reciprocal-multiply: one divide, then a mul per lane
	// Fuse the stable-softmax max-scan into the scale pass: track the running max of
	// z here instead of a separate full pass over the vocabulary before exp. The
	// intervening filters that run before softmax only ever LOWER values to −Inf
	// (topNSigmaMask masks below max−n·σ; top-k's quickselect masks below the k-th
	// largest) and never touch the arg-max, so max(z) is unchanged by them — the
	// pre-filter max computed here equals the post-filter max both branches need,
	// and the top-k branch's kept-set max is this same global max (the max is always
	// ≥ any keep threshold). Bit-identical m, one fewer O(n) pass over z.
	m := math.Inf(-1)
	for i, v := range logits {
		zi := v * invT
		z[i] = zi
		if zi > m {
			m = zi
		}
	}

	// top-nσ (Tang et al. 2024 arXiv:2411.07641, see sample_topnsigma.go): mask
	// logits below max − n·σ to −inf, at the LOGIT stage before any other filter.
	// It must precede top-k's −inf masking so σ is computed over the full finite
	// vocabulary; and because max and σ scale by 1/T together, the kept set on z
	// equals the kept set on the raw logits — temperature moves no token across
	// the threshold (the paper's defining temperature-invariance property).
	if s.TopNSigma > 0 {
		topNSigmaMask(z, s.TopNSigma)
	}

	// top-k: keep the k highest logits, mask the rest to −inf. Instead of a full
	// O(n log n) sort of the whole vocabulary just to drop the tail, quickselect
	// the k-th largest value in O(n) average and mask everything below it — the
	// kept set is identical for distinct logits (ties at the boundary keep all
	// their members, a benign, arguably-more-principled difference from the old
	// arbitrary sort tie-break). §T626.
	if s.TopK > 0 && s.TopK < n {
		// Fused top-k + stable softmax: quickselect the threshold, then run exp over
		// ONLY the kept lanes instead of masking the n−k tail to −Inf and exp'ing all
		// n (exp(−Inf)=0 for the dropped lanes is pure waste; exp was ~26% of
		// large-vocab Dist, at k=40/n=50257 only 0.08% of the exps matter). Gather the
		// kept set (z[i] ≥ kv keeps every boundary tie exactly as the old −Inf mask
		// did), exp the compact buffer in place, then scatter the normalized probs
		// into a zeroed dst so downstream filters still see a full-length distribution
		// (k nonzero, rest 0). §T626/§T950.
		scratch := float64ScratchPool.Get().([]float64)
		if cap(scratch) < n {
			scratch = make([]float64, n)
		} else {
			scratch = scratch[:n]
		}
		copy(scratch, z)
		kv := kthLargest(scratch, s.TopK) // the k-th largest logit is the threshold
		// scratch's contents are spent; reuse its backing array as the compact keep
		// buffer (no extra pool Get) — cap ≥ n ⇒ the append below never reallocates.
		keep := scratch[:0]
		for _, v := range z {
			if v >= kv { // ≥ threshold ⇒ all boundary ties kept, matching the old mask
				keep = append(keep, v)
			}
		}
		// m (the global max, computed in the scale pass) is the kept set's max too: the
		// global max is always ≥ kv, so it is always in the set — same shift as before.
		// exp over the compact kept set (in place; elementwise, aliasing-safe), returns
		// Σ. Masked lanes never enter the sum, so it equals the full-vocab Σ up to the
		// exp-sum reassociation the Dist f64 tolerance (§sample_test 1e-12) already rides.
		sum := simd.ExpSumF64(keep, keep, m)
		inv := 1 / sum // reciprocal-multiply the normalize
		// scatter: re-scan z in the SAME ascending order the gather used, so keep[j]
		// lands at its original index; zero the rest. No index buffer needed.
		j := 0
		for i, v := range z {
			if v >= kv {
				dst[i] = keep[j] * inv
				j++
			} else {
				dst[i] = 0
			}
		}
		float64ScratchPool.Put(scratch)
	} else {
		// stable softmax over the full vocabulary (no top-k truncation); m is the
		// vocabulary max already computed in the scale pass above.
		// dst[i] = exp(z[i]-m), returns Σ — 4-wide AVX2 f64 exp on the SIMD build
		// (the exp was ~26% of large-vocab Dist), scalar otherwise; rides §sample_test 1e-12.
		sum := simd.ExpSumF64(dst, z, m)
		inv := 1 / sum // reciprocal-multiply the normalize: one divide, a mul per lane
		for i := range dst {
			dst[i] *= inv
		}
	}

	// top-p nucleus: keep smallest desc-prob prefix with cumsum ≥ p (crossing
	// token included), renormalize (§T627: bounded selection, not a full sort).
	if s.TopP > 0 && s.TopP < 1 {
		nucleusTopP(dst, s.TopP)
	}

	// min-p: keep tokens with prob ≥ MinP·max-prob (Nguyen et al. 2024, §R63), an
	// adaptive threshold. It is truncateAbove with a peak-relative floor, so it
	// inherits the same min_tokens_to_keep=1 guard: at MinP ≤ 1 the threshold is at
	// most the peak and the arg-max clears it on its own, while at MinP > 1 the
	// explicit guard is what keeps the distribution a valid one-hot instead of an
	// all-zero vector normalized into NaN (§B73).
	if s.MinP > 0 && len(dst) > 0 { // len guard: dst[top] on an empty vocab
		top := argmax(dst)
		truncateAboveKeeping(dst, s.MinP*dst[top], top)
	}

	// locally typical sampling (Meister et al. 2023): keep the smallest set of tokens
	// whose surprisal −log p is closest to the entropy H, by cumulative probability τ
	// (§T628: bounded selection over the typical score, not a full sort).
	if s.Typical > 0 && s.Typical < 1 {
		typicalTruncate(dst, s.Typical)
	}

	// epsilon sampling: absolute-probability floor (Hewitt et al. 2022, §R91).
	if s.Epsilon > 0 {
		truncateAbove(dst, s.Epsilon)
	}
	// eta sampling: entropy-adaptive threshold η = min(ε, √ε·exp(−H)) over the current
	// distribution, H the Shannon entropy in nats (Hewitt et al. 2022, §R91).
	if s.Eta > 0 {
		var h float64
		for _, p := range dst {
			if p > 0 {
				h -= p * math.Log(p)
			}
		}
		truncateAbove(dst, math.Min(s.Eta, math.Sqrt(s.Eta)*math.Exp(-h)))
	}

	return dst
}

// truncateAbove zeroes every probability below thresh, always keeping the arg-max so
// at least one token survives (min_tokens_to_keep=1), then renormalizes.
func truncateAbove(probs []float64, thresh float64) {
	truncateAboveKeeping(probs, thresh, argmax(probs))
}

// truncateAboveKeeping is truncateAbove with the survivor index supplied by the
// caller, for filters that already had to locate the peak (min-p). It is the single
// place the two guards every probability-floor filter needs actually live:
//
//   - i != top — the arg-max is exempt from the floor, so at least one token always
//     survives however aggressive the threshold is (transformers' min_tokens_to_keep=1).
//     A threshold above the peak — MinP > 1, Epsilon > maxProb — would otherwise zero
//     the entire vector.
//   - ksum == 0 — never divide by a zero normalizer. Unreachable while top survives
//     with probs[top] > 0, and cheap insurance for an already-degenerate input (an
//     all-zero or NaN-laden distribution handed in by an earlier filter).
//
// Skipping either one produces silent wrongness rather than a crash: probs becomes
// all-NaN, every `u <= cum` comparison in Sample is false against NaN, and the draw
// falls through to a fixed token id on every step (§B73).
func truncateAboveKeeping(probs []float64, thresh float64, top int) {
	if top < 0 || top >= len(probs) {
		return
	}
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
	inv := 1 / ksum // reciprocal-multiply the renormalize
	for i := range probs {
		probs[i] *= inv
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

type genConfig struct {
	be backend.Backend

	// eos holds the stop ids passed to WithEOS. eosOn records that WithEOS was
	// called AT ALL, and that — not a non-empty eos — is what arms stopping: the
	// zero-argument form WithEOS() arms sampler-declared stops (StopTokener) with
	// no explicit ids of its own. eosOn false is the default and means the loop
	// runs the full maxNew steps, whatever it draws (§B76).
	eos   []int
	eosOn bool
}

// WithBackend runs the decode loop on the given backend instead of the default.
// Single-token decode is dispatch-latency-bound, so for SMALL models the CPU is
// faster than the GPU (measured ~2.7× at dim 512), while large models still favour
// the GPU — the caller, who knows the model size and hardware, chooses (§T361).
//
// SCOPE: honoured by the float decode loops. The Quant* loops accept GenerateOption
// so [WithEOS] reaches them, but they IGNORE this one and always decode on
// backend.Default() — their kernels are not exercised on non-default backends, so
// routing them there is not claimed rather than silently half-supported (§B76).
func WithBackend(be backend.Backend) GenerateOption { return func(c *genConfig) { c.be = be } }

// WithEOS stops a Generate loop early when it draws one of the given stop-token
// ids. Pass every id the model may end on — many checkpoints ship two or more
// (a base </s> plus a chat <|im_end|>, say) and any of them ends the sequence.
//
// The stop token IS INCLUDED in the returned slice: a model that draws its first
// EOS as the k-th generated token returns prompt+k tokens, the last of which is
// that EOS. This matches HuggingFace `generate()` (whose returned ids carry the
// eos_token_id — the reason callers pass skip_special_tokens=True when decoding)
// and llama.cpp (which appends the token, then breaks), as well as this package's
// own [BeamSearch], whose completed hypotheses end in eos. It also leaves the stop
// observable: `out[len(out)-1]` tells the caller whether the loop ended on EOS or
// ran out of budget, which an excluding contract would erase. Note that
// [T5Decoder.Generate] predates this and EXCLUDES its eosToken; it is unchanged.
//
// EOS stopping is strictly opt-in and DOES NOT CHANGE DEFAULT BEHAVIOUR. Without
// this option every loop keeps drawing until maxNew or the context limit, exactly
// as before — existing callers get byte-identical output.
//
// Calling WithEOS with no arguments arms stopping for sampler-declared stop ids
// only: if the sampler implements [StopTokener] — as the guided samplers from
// [RegexGuide.Sampler] and [GrammarGuide.Sampler] do, reporting the eosID they
// were built with — its ids stop the loop too, unioned with any passed here. That
// is the fix for a guided FSM that reaches an accepting state, draws EOS and then
// emits filler from the frozen state for the rest of maxNew (§B76).
//
// NOT COVERED: stop STRINGS that straddle token boundaries. A stop string can end
// mid-token, and the decode loops carry []int with no detokenizer in scope, so
// neither the match nor the trim can be expressed here; there is deliberately no
// half-implementation. Callers needing that must detokenize the returned tokens
// and trim the text themselves.
func WithEOS(ids ...int) GenerateOption {
	return func(c *genConfig) {
		c.eosOn = true
		c.eos = append(c.eos, ids...)
	}
}

// StopTokener is implemented by a TokenSampler that carries stop-token ids of its
// own, so a Generate loop armed by [WithEOS] can honour them without the caller
// repeating ids the sampler already knows. The guided samplers implement it.
type StopTokener interface {
	// StopTokens returns the ids that end generation, or nil for none.
	StopTokens() []int
}

// EOSReporter is implemented by a TokenSampler that records whether it has drawn a
// stop token. It makes an unstopped loop's overrun DETECTABLE after the fact even
// when the caller did not pass [WithEOS] — check it on the sampler once Generate
// returns to find out whether the tail of the output is post-EOS filler.
type EOSReporter interface {
	// EOSEmitted reports whether this sampler has drawn a stop token.
	EOSEmitted() bool
}

// stopEOS reports whether tok ends the decode loop. It is the single stop decision
// every Generate loop in the package calls, so the contract cannot drift between
// ~40 architectures (§B75 is what copy-pasted per-loop logic costs). Disarmed —
// the default — it is a predictable-branch no-op on the per-token path.
//
// The scan is linear because the id sets are tiny (models ship 1-3 eos ids); a map
// would cost more to build per call than it saves per step.
func (c *genConfig) stopEOS(tok int, s TokenSampler) bool {
	if !c.eosOn {
		return false
	}
	for _, id := range c.eos {
		if tok == id {
			return true
		}
	}
	if st, ok := s.(StopTokener); ok {
		for _, id := range st.StopTokens() {
			if tok == id {
				return true
			}
		}
	}
	return false
}

// Generate autoregressively produces up to maxNew tokens after the prompt,
// using the KV-cache. Returns prompt + generated tokens. Stops at the context
// limit. By default the decode runs on backend.Default(); WithBackend overrides it
// (single-token decode is often faster on the CPU for small models, §T361).
//
// Pass [WithEOS] to stop early on a stop token, which is INCLUDED in the returned
// slice. Without it the loop runs the full maxNew steps whatever it draws — the
// long-standing behaviour, deliberately preserved.
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
		if gc.stopEOS(next, s) {
			break
		}
		l, err := g.DecodeStep(ctx, cache, next, pos)
		if err != nil {
			return nil, err
		}
		logits = l
		pos++
	}
	return out, nil
}
