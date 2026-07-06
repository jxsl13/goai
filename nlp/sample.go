package nlp

import (
	"math"
	"math/rand/v2"
	"sort"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Sampler turns a logit vector into a token id. Pipeline (matching HuggingFace
// generation): temperature scaling → top-k mask → top-p (nucleus) mask →
// softmax → multinomial sample. Temperature ≤ 0 selects greedy (argmax), which
// ignores TopK/TopP and RNG.
//
// top-p follows Holtzman et al. 2019 (§R34): the nucleus is the smallest set of
// highest-probability tokens whose cumulative probability ≥ p; the token that
// crosses p is included, so at least the top token always survives. Kept
// probabilities are renormalized before sampling.
type Sampler struct {
	Temperature float64
	TopK        int
	TopP        float64
	rng         *rand.Rand
}

// NewSampler builds a sampler with an explicit seed (deterministic).
func NewSampler(temperature float64, topK int, topP float64, seed uint64) *Sampler {
	return &Sampler{Temperature: temperature, TopK: topK, TopP: topP,
		rng: rand.New(rand.NewPCG(seed, 0x5a3d))}
}

// Greedy returns a deterministic argmax sampler.
func Greedy() *Sampler { return &Sampler{Temperature: 0} }

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
	n := len(logits)
	if s.Temperature <= 0 {
		return argmax(logits)
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

	// multinomial sample
	u := s.rng.Float64()
	var cum float64
	for i, p := range probs {
		cum += p
		if u <= cum {
			return i
		}
	}
	return n - 1
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

// Generate autoregressively produces up to maxNew tokens after the prompt,
// using the KV-cache. Returns prompt + generated tokens. Stops at the context
// limit.
func (g *GPT) Generate(prompt []int, maxNew int, s *Sampler) ([]int, error) {
	ctx := backend.NewContext()
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
		next := s.Sample(rowLogits(logits))
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
