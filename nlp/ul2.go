package nlp

import "math/rand/v2"

// UL2 Mixture-of-Denoisers pretraining objective (Tay, Dehghani, Tran, Garcia, Wei, Wang,
// Chung, Bahri, Schuster, Zheng, Zhou, Houlsby & Metzler 2022, "UL2: Unifying Language
// Learning Paradigms", arXiv:2205.05131, §3.1). UL2 unifies the T5 span-corruption objective
// (§R213, nlp.T5Corrupt) and prefix language modeling under one seq2seq denoising framework by
// MIXING three denoisers, each a different (span, rate) regime, and prefixing every example
// with a paradigm token so the model learns which mode it is in:
//
//   - R-Denoiser (Regular): standard T5 span corruption, short spans and a low corruption
//     rate — the ordinary masked-span objective.
//   - S-Denoiser (Sequential): prefix language modeling — a SINGLE contiguous span at the end
//     of the sequence, so the model conditions on a prefix and predicts the continuation.
//   - X-Denoiser (eXtreme): aggressive denoising — long spans and/or a high corruption rate,
//     bridging span corruption and full generation.
//
// R and X reuse the identical span-corruption mechanism (contiguous spans → unique sentinels)
// and differ ONLY in (mean span, rate); S is a single split point. The objective is a lossless
// data transform, so it round-trips (UL2Reconstruct) and is fuzzed (§V15).

// UL2Mode identifies which denoiser produced an example; its value is also the offset of the
// example's paradigm token from the sentinel base (R→base, S→base+1, X→base+2).
type UL2Mode int

const (
	UL2R UL2Mode = iota // Regular: short-span T5 corruption
	UL2S                // Sequential: prefix-LM (single trailing span)
	UL2X                // eXtreme: long spans and/or high corruption rate
)

// ul2NumModes reserves the ids [sentinelBase, sentinelBase+ul2NumModes) for the three paradigm
// tokens; span sentinels start above them.
const ul2NumModes = 3

// UL2Denoiser is one denoiser in the mixture: a (Mode, Rate, MeanSpan) regime with a sampling
// Weight. Rate is the fraction of tokens corrupted (for S, the fraction placed in the trailing
// target span); MeanSpan is the mean corrupted-span length (ignored for S).
type UL2Denoiser struct {
	Mode     UL2Mode // R, S or X
	Rate     float64 // corruption rate r
	MeanSpan float64 // mean span μ (unused for S)
	Weight   float64 // mixing proportion (need not be normalized)
}

// UL2MixtureOfDenoisers returns the paper's standard 7-denoiser mixture (UL2 §3.1.2): two
// R-denoisers (μ∈{3,8}, r=0.15), four X-denoisers (μ∈{3,8} at r=0.5, μ=64 at r∈{0.15,0.5}) and
// one S-denoiser (r=0.25). The paper mixes the seven with equal probability (§3.1.3), so every
// Weight is 1.
func UL2MixtureOfDenoisers() []UL2Denoiser {
	return []UL2Denoiser{
		{Mode: UL2R, Rate: 0.15, MeanSpan: 3, Weight: 1},
		{Mode: UL2R, Rate: 0.15, MeanSpan: 8, Weight: 1},
		{Mode: UL2X, Rate: 0.5, MeanSpan: 3, Weight: 1},
		{Mode: UL2X, Rate: 0.5, MeanSpan: 8, Weight: 1},
		{Mode: UL2X, Rate: 0.15, MeanSpan: 64, Weight: 1},
		{Mode: UL2X, Rate: 0.5, MeanSpan: 64, Weight: 1},
		{Mode: UL2S, Rate: 0.25, MeanSpan: 0, Weight: 1},
	}
}

// SampleUL2Denoiser draws one denoiser from the mixture with probability proportional to Weight.
func SampleUL2Denoiser(mix []UL2Denoiser, rng *rand.Rand) UL2Denoiser {
	var total float64
	for _, d := range mix {
		total += d.Weight
	}
	r := rng.Float64() * total
	for _, d := range mix {
		if r < d.Weight {
			return d
		}
		r -= d.Weight
	}
	return mix[len(mix)-1]
}

// UL2Denoise applies denoiser d to a document, returning the encoder input and decoder target.
// The input is prefixed with d's paradigm token (sentinelBase+int(d.Mode)); span sentinels start
// at sentinelBase+ul2NumModes. R/X delegate to T5 span corruption; S emits a single trailing
// prefix-LM span. The result is exactly reversible with UL2Reconstruct. The original tokens are
// not modified.
func UL2Denoise(tokens []int, d UL2Denoiser, sentinelBase int, rng *rand.Rand) (input, target []int) {
	spanBase := sentinelBase + ul2NumModes
	if d.Mode == UL2S {
		input, target = ul2Sequential(tokens, d.Rate, spanBase)
	} else {
		input, target = T5Corrupt(tokens, d.Rate, d.MeanSpan, spanBase, rng)
	}
	modeTok := sentinelBase + int(d.Mode)
	input = append([]int{modeTok}, input...) // prepend the paradigm token
	return input, target
}

// ul2Sequential is the S-denoiser (prefix-LM): the last ⌈rate·n⌉ tokens become one span behind a
// single sentinel — the model sees the prefix and predicts the suffix. Emitted in the same
// sentinel format as T5Corrupt (one span → sentinels spanBase, spanBase+1) so T5Reconstruct
// reverses it.
func ul2Sequential(tokens []int, rate float64, spanBase int) (input, target []int) {
	n := len(tokens)
	sufLen := int(float64(n)*rate + 0.5)
	if sufLen < 1 {
		sufLen = 1
	}
	if sufLen > n {
		sufLen = n
	}
	k := n - sufLen // prefix = tokens[:k], suffix = tokens[k:]
	sent0, sent1 := spanBase, spanBase+1
	input = make([]int, 0, k+1)
	input = append(input, tokens[:k]...)
	input = append(input, sent0)
	target = make([]int, 0, sufLen+2)
	target = append(target, sent0)
	target = append(target, tokens[k:]...)
	target = append(target, sent1) // trailing final sentinel (T5 convention)
	return input, target
}

// UL2Reconstruct recovers the original document and the denoiser mode from a UL2Denoise pair. It
// strips the leading paradigm token and reverses the span corruption (T5Reconstruct). ok is false
// for a malformed pair (empty input, a leading token outside the paradigm range, or an
// irreversible span layout).
func UL2Reconstruct(input, target []int, sentinelBase int) (doc []int, mode UL2Mode, ok bool) {
	if len(input) == 0 {
		return nil, 0, false
	}
	m := input[0] - sentinelBase
	if m < 0 || m >= ul2NumModes {
		return nil, 0, false // leading token is not a paradigm token
	}
	doc, ok = T5Reconstruct(input[1:], target, sentinelBase+ul2NumModes)
	return doc, UL2Mode(m), ok
}
