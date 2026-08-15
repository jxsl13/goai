package nlp_test

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/nlp"
)

// ExampleSampler_SampleTopPFromCandidates draws a nucleus (top-p) token from a device TopK
// shortlist instead of the whole vocabulary, and gets the token full-vocab Sample would have
// drawn from the same seed. The full-vocab softmax statistics are what make it exact: nucleus
// membership depends on the true normalizer, so a top-p draw over the shortlist alone with its
// own smaller normalizer would inflate every probability and pick too small a nucleus.
//
// The second return is an overflow flag: false means the nucleus needs more tokens than the
// shortlist holds, and the caller must fall back to full-vocab Sample. It is the correctness
// valve for flat, high-entropy distributions.
func ExampleSampler_SampleTopPFromCandidates() {
	const temp, topP = 1.0, 0.9
	logits := make([]float64, 64) // a peaked distribution: token 7 dominates
	for i := range logits {
		logits[i] = -6 + 0.05*float64(i%11)
	}
	logits[7], logits[19], logits[42] = 5.0, 3.5, 2.0

	// Full-vocab statistics the caller already has from the GPU softmax pass.
	maxL := math.Inf(-1)
	for _, v := range logits {
		if v > maxL {
			maxL = v
		}
	}
	var Zexp float64
	for _, v := range logits {
		Zexp += math.Exp((v - maxL) / temp)
	}

	// The shortlist: the three tokens that carry essentially all the mass, in ascending
	// vocab-index order, as a device TopK would hand them over.
	candIdx := []int32{7, 19, 42}
	candLogits := []float64{logits[7], logits[19], logits[42]}

	fast := nlp.NewSampler(1, nlp.WithTemperature(temp), nlp.WithTopP(topP))
	tok, ok := fast.SampleTopPFromCandidates(candLogits, candIdx, maxL, Zexp)

	ref := nlp.NewSampler(1, nlp.WithTemperature(temp), nlp.WithTopP(topP))
	fmt.Printf("token %d, resolved %v, matches full-vocab Sample %v\n", tok, ok, tok == ref.Sample(logits))
	// Output: token 7, resolved true, matches full-vocab Sample true
}
