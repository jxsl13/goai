package nlp

import "math/rand/v2"

// T5 span-corruption denoising objective (Raffel, Shazeer, Roberts, Lee, Narang, Matena,
// Zhou, Li & Liu 2020, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text
// Transformer" / T5, arXiv:1910.10683, §3.1.4). Unlike the causal-LM and FIM objectives here,
// this is the seq2seq DENOISING objective behind T5/UL2: a fraction of the tokens are dropped
// as consecutive SPANS, each span in the encoder INPUT is replaced by a single unique sentinel
// token, and the decoder TARGET is the dropped spans behind their sentinels — so the model
// learns to reconstruct the masked content from the surrounding context.

// T5Corrupt applies span corruption to a document. About `density` of the tokens are dropped
// (T5 default 0.15), grouped into spans of mean length `meanSpan` (T5 default 3). Each dropped
// span is replaced in `input` by a unique sentinel (sentinelBase, sentinelBase+1, …) and
// appears in `target` behind that sentinel; `target` ends with one final sentinel. The
// sentinel ids must be reserved (≥ sentinelBase, above every document token). rng draws the
// span/gap segmentation. The original tokens are not modified.
func T5Corrupt(tokens []int, density, meanSpan float64, sentinelBase int, rng *rand.Rand) (input, target []int) {
	n := len(tokens)
	if n == 0 {
		return nil, []int{sentinelBase} // nothing to mask: target is just the final sentinel
	}
	numNoise := int(float64(n)*density + 0.5)
	if numNoise < 1 {
		numNoise = 1
	}
	if numNoise > n {
		numNoise = n
	}
	numSpans := int(float64(numNoise)/meanSpan + 0.5)
	if numSpans < 1 {
		numSpans = 1
	}
	if numSpans > numNoise {
		numSpans = numNoise
	}
	spanLens := randComposition(numNoise, numSpans, 1, rng)    // each masked span ≥ 1
	gapLens := randComposition(n-numNoise, numSpans+1, 0, rng) // kept runs around the spans

	input = make([]int, 0, n)
	target = make([]int, 0, numNoise+numSpans+1)
	pos := 0
	for s := 0; s < numSpans; s++ {
		for g := 0; g < gapLens[s]; g++ { // kept run before span s
			input = append(input, tokens[pos])
			pos++
		}
		sent := sentinelBase + s
		input = append(input, sent) // span s → one sentinel in the input
		target = append(target, sent)
		for l := 0; l < spanLens[s]; l++ { // …and its tokens in the target
			target = append(target, tokens[pos])
			pos++
		}
	}
	for g := 0; g < gapLens[numSpans]; g++ { // trailing kept run
		input = append(input, tokens[pos])
		pos++
	}
	target = append(target, sentinelBase+numSpans) // final sentinel
	return input, target
}

// T5Reconstruct recovers the original document from a (input, target) span-corruption pair by
// splicing each sentinel in the input with its span from the target. It is the exact inverse
// of T5Corrupt (returning ok=false on a malformed pair). A token is treated as a sentinel iff
// it is ≥ sentinelBase.
func T5Reconstruct(input, target []int, sentinelBase int) (doc []int, ok bool) {
	spans := make(map[int][]int)
	cur := -1
	for _, t := range target {
		if t >= sentinelBase {
			cur = t
			if _, seen := spans[cur]; !seen {
				spans[cur] = nil
			}
		} else {
			if cur < 0 {
				return nil, false // a token before any sentinel
			}
			spans[cur] = append(spans[cur], t)
		}
	}
	doc = make([]int, 0, len(input))
	for _, t := range input {
		if t >= sentinelBase {
			span, seen := spans[t]
			if !seen {
				return nil, false // an input sentinel with no span in the target
			}
			doc = append(doc, span...)
		} else {
			doc = append(doc, t)
		}
	}
	return doc, true
}

// randComposition splits total into parts parts each ≥ minEach (a multinomial spread of the
// remaining mass over the parts).
func randComposition(total, parts, minEach int, rng *rand.Rand) []int {
	out := make([]int, parts)
	for i := range out {
		out[i] = minEach
	}
	for r := 0; r < total-minEach*parts; r++ {
		out[rng.IntN(parts)]++
	}
	return out
}
