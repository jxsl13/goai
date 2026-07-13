package nlp

import (
	"math"
	"math/rand/v2"
	"sort"

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

// WithTemperature sets the softmax temperature (default 1; ≤0 = greedy).
func WithTemperature(t float64) SamplerOption { return func(s *Sampler) { s.Temperature = t } }

// WithTopK keeps only the k highest-logit tokens (0 = disabled).
func WithTopK(k int) SamplerOption { return func(s *Sampler) { s.TopK = k } }

// WithTopP keeps the smallest nucleus with cumulative probability ≥ p (Holtzman
// et al. 2019); 0 or ≥1 disables it.
func WithTopP(p float64) SamplerOption { return func(s *Sampler) { s.TopP = p } }

// WithMinP keeps tokens with probability ≥ p·max-prob (Nguyen et al. 2024); 0
// disables it. Typical 0.05–0.1.
func WithMinP(p float64) SamplerOption { return func(s *Sampler) { s.MinP = p } }

// WithEpsilon enables epsilon sampling (Hewitt et al. 2022, §R91): keep only tokens
// with probability ≥ eps, an absolute floor. 0 disables it; typical 3e-4–2e-3.
func WithEpsilon(eps float64) SamplerOption { return func(s *Sampler) { s.Epsilon = eps } }

// WithEta enables eta sampling (Hewitt et al. 2022, §R91): an entropy-adaptive
// threshold η = min(eps, √eps·exp(−H)) where H is the distribution's Shannon entropy
// in nats — it truncates hard when the model is confident (low entropy) and keeps more
// when uncertain. 0 disables it; typical eps≈9e-4.
func WithEta(eps float64) SamplerOption { return func(s *Sampler) { s.Eta = eps } }

// WithTypical enables locally typical sampling (Meister, Pimentel, Wiher & Cotterell
// 2023, "Locally Typical Sampling"): keep the smallest set of tokens whose information
// content −log p is closest to the distribution's entropy H(p) until their cumulative
// probability reaches τ, filtering out both the too-predictable and the too-surprising.
// 0 or ≥1 disables it; typical τ 0.9–0.95.
func WithTypical(tau float64) SamplerOption { return func(s *Sampler) { s.Typical = tau } }

// WithRepeatPenalty enables the CTRL repetition penalty (Keskar et al. 2019): for
// every token in the penalty window, a positive logit is divided by p and a negative
// one multiplied by it. 0 or 1 disables it; typical 1.1–1.3.
func WithRepeatPenalty(p float64) SamplerOption { return func(s *Sampler) { s.RepeatPenalty = p } }

// WithFrequencyPenalty subtracts f · count(token in the penalty window) from each
// seen token's logit (the OpenAI-style frequency penalty). 0 disables it.
func WithFrequencyPenalty(f float64) SamplerOption { return func(s *Sampler) { s.FreqPenalty = f } }

// WithPresencePenalty subtracts p from the logit of every token that appears in the
// penalty window at all (the OpenAI-style presence penalty). 0 disables it.
func WithPresencePenalty(p float64) SamplerOption { return func(s *Sampler) { s.PresencePenalty = p } }

// WithPenaltyWindow limits the repetition penalties to the last n history tokens
// (llama.cpp's repeat_last_n). 0 (the default) penalizes over the entire history.
func WithPenaltyWindow(n int) SamplerOption { return func(s *Sampler) { s.PenaltyLastN = n } }

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

// Sample returns a token id for the given logits.
func (s *Sampler) Sample(logits []float64) int {
	if s.Temperature <= 0 {
		return argmax(logits) // greedy: deterministic, no rng draw
	}
	probs := s.Dist(logits)
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
	return s.Sample(s.penalize(logits, history))
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

	// top-k: keep the k highest logits, mask the rest to −inf
	if s.TopK > 0 && s.TopK < n {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return z[idx[a]] > z[idx[b]] })
		for _, i := range idx[s.TopK:] {
			z[i] = math.Inf(-1)
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
		sort.Slice(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
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
		sort.Slice(idx, func(a, b int) bool { return score[idx[a]] < score[idx[b]] })
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

// rowLogits copies row 0 of a [1,vocab] logits tensor to a slice.
func rowLogits(t *tensor.Tensor) []float64 {
	v := t.Shape()[1]
	out := make([]float64, v)
	for j := range v {
		out[j] = t.AtF64(0, j)
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
