package nlp

import "math"

// NoRepeatNGramBlock enforces the no_repeat_ngram_size decoding constraint
// (HuggingFace NoRepeatNGramLogitsProcessor / fairseq; tri-gram blocking from
// Paulus, Xiong & Socher 2017, "A Deep Reinforced Model for Abstractive
// Summarization", arXiv:1705.04304). It HARD-blocks the model from ever repeating
// any n-gram: a next-token candidate t is forbidden if appending it would recreate an
// n-gram (context + t) that already occurred in the generated sequence.
//
// The current context is the last n−1 generated tokens. For every earlier position
// whose (n−1)-gram equals that context, the token that FOLLOWED it is banned by
// setting its logit to −∞, so it can never be sampled this step. Unlike the soft
// repetition/frequency penalties (ApplyPenalties), this is a hard constraint. It is a
// no-op when n ≤ 0 or the sequence is shorter than n; n = 3 is the classic tri-gram
// blocking. logits is modified in place; out-of-range history tokens are ignored.
func NoRepeatNGramBlock(logits []float64, generated []int, n int) {
	l := len(generated)
	if n <= 0 || l < n {
		return
	}
	prefixLen := n - 1
	ctxStart := l - prefixLen // start of the current last-(n−1)-token context
	for i := 0; i+n <= l; i++ {
		match := true
		for j := 0; j < prefixLen; j++ {
			if generated[i+j] != generated[ctxStart+j] {
				match = false
				break
			}
		}
		if match {
			if t := generated[i+prefixLen]; t >= 0 && t < len(logits) {
				logits[t] = math.Inf(-1) // completing this n-gram would repeat it → block
			}
		}
	}
}
