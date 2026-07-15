package nlp

import "math"

// Top-nσ sampling (Tang et al. 2024, "Top-nσ: Not All Logits Are You Need",
// arXiv:2411.07641) truncates on the RAW LOGITS rather than on the softmax
// probabilities: keep exactly the tokens whose logit lies within n standard
// deviations of the maximum,
//
//	keep i  ⟺  logit_i ≥ M − n·σ,   M = max(logits), σ = stddev(logits).
//
// The paper's observation is that a trained LM's logit vector splits into a
// large Gaussian-distributed noise region (the irrelevant tail) and a small
// informative outlier region near the max. σ measures the noise scale, so the
// threshold M − n·σ separates the informative head from the noise statistically,
// without ever sorting or exponentiating the vocabulary.
//
// The defining property is TEMPERATURE INVARIANCE. Probability-space filters
// (top-p, typical, epsilon) see softmax(logits/T): raising T flattens the
// distribution and their kept set balloons, which is exactly how high-temperature
// sampling derails. The top-nσ threshold instead lives in σ units on the logits —
// dividing by T scales M, σ and every logit together, so the kept set never
// changes: {i : logit_i/T ≥ M/T − n·σ/T} = {i : logit_i ≥ M − n·σ}. Temperature
// then only redistributes probability among the survivors. Accordingly the mask
// is applied in Dist at the LOGIT stage — after the temperature division (a
// no-op for the set, per the identity above) and before top-k's −inf masking,
// so σ is always computed over the full finite logit vector.
//
// In plain terms: instead of asking "how probable is this word?", top-nσ asks
// "does this word's raw score stand out from the background noise?" — and the
// answer doesn't change when you turn the temperature up for creativity.

// WithTopNSigma enables top-nσ sampling: keep only the tokens whose logit is
// within n standard deviations of the maximum logit, masking the rest before the
// softmax (Tang et al. 2024, "Top-nσ: Not All Logits Are You Need",
// arXiv:2411.07641; see sample_topnsigma.go).
//
// In plain terms: the model's raw scores are mostly background noise with a few
// genuine candidates poking out the top; n sets how far above the noise a word
// must stand to stay in the running. Unlike the probability-based filters, this
// cut doesn't loosen when you raise the temperature — you can sample "hot" for
// variety without the long tail of nonsense words leaking back in.
//
// Professional: a pre-softmax truncation keep i ⟺ logit_i ≥ M − n·σ (M the max
// logit, σ the population standard deviation over the vocabulary). Because the
// threshold is expressed in σ units of the logits, it is invariant to temperature
// scaling (both sides divide by T), so the candidate-set size is stable across T
// where nucleus sampling's set grows — the paper's motivation. The arg-max always
// survives (M ≥ M − n·σ for n ≥ 0). Boundary behavior — as n → 0⁺ only tokens
// essentially tied with the arg-max survive (greedy-like); large n (≳ the logit
// range in σ units, typically n ≳ 10) keeps the entire vocabulary (no-op); σ = 0
// (all logits equal — nothing stands out) keeps everything. SPECIAL VALUE: n ≤ 0
// disables the filter (matching llama.cpp's top_n_sigma ≤ 0 = off; use Greedy()
// or WithTemperature(0) for exact arg-max decoding).
//
// Default 0 = DISABLED, research-grounded: like every other truncation knob it is
// off out of the box (a bare NewSampler samples the model's own distribution);
// the RECOMMENDED enabled value is n = 1.0 — Tang et al. 2024 report n ≈ 1.0 as
// the working default across their benchmarks (they explore 0.5–2.0: smaller n
// approaches greedy, larger admits more of the noise region), and llama.cpp
// adopts 1.0 as the enabled top_n_sigma reference value.
func WithTopNSigma(n float64) SamplerOption { return func(s *Sampler) { s.TopNSigma = n } }

// topNSigmaMask masks (to −inf) every logit strictly below max(z) − n·σ(z), in
// place, where σ is the population standard deviation of z. Callers gate on
// n > 0. The arg-max always survives; σ = 0 (all logits equal) keeps everything
// (there is no tail to cut); the boundary is inclusive (logit == threshold is
// kept), matching the paper's ≥.
func topNSigmaMask(z []float64, n float64) {
	if len(z) == 0 {
		return
	}
	m := math.Inf(-1)
	var sum float64
	for _, v := range z {
		if v > m {
			m = v
		}
		sum += v
	}
	mean := sum / float64(len(z))
	var ss float64
	for _, v := range z {
		d := v - mean
		ss += d * d
	}
	sigma := math.Sqrt(ss / float64(len(z)))
	thresh := m - n*sigma
	for i := range z {
		if z[i] < thresh {
			z[i] = math.Inf(-1)
		}
	}
}
