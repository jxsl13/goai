package nlp

// Logit penalties that reduce repetition during generation, applied to the raw
// logits BEFORE temperature/top-k/top-p sampling (§R57).

// ApplyPenalties adjusts logits in place using the history of already-generated
// tokens:
//
//   - repetition penalty (CTRL, Keskar et al. 2019): for each seen token, divide
//     its logit by θ if the logit is positive, else multiply by θ. θ>1 discourages
//     repeats; θ=1 (or ≤0) is a no-op. The sign-aware form (HuggingFace) is used so
//     a negative logit is pushed further negative rather than toward zero.
//   - frequency penalty (OpenAI): subtract frequency·count_i (scales with how many
//     times the token already appeared).
//   - presence penalty (OpenAI): subtract presence once for any token that appeared.
//
// Tokens that have not been generated are left unchanged. Out-of-range history
// entries are ignored.
func ApplyPenalties(logits []float64, generated []int, repetition, frequency, presence float64) {
	if len(generated) == 0 {
		return
	}
	counts := make(map[int]int, len(generated))
	for _, t := range generated {
		if t >= 0 && t < len(logits) {
			counts[t]++
		}
	}
	for tok, c := range counts {
		l := logits[tok]
		if repetition > 0 && repetition != 1 { // CTRL, sign-aware
			if l > 0 {
				l /= repetition
			} else {
				l *= repetition
			}
		}
		l -= frequency*float64(c) + presence // OpenAI additive penalties
		logits[tok] = l
	}
}
