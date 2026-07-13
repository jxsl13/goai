package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// dryPenalties returns the logit deltas a greedy DRY sampler applies for the given
// history, by comparing penalized vs raw logits through SampleWithHistory's argmax:
// instead we recover deltas directly by probing one-hot logits. Simpler: run
// SampleWithHistory on crafted logits where the expected winner flips iff the penalty
// matches the hand-computed value.
func dryDelta(t *testing.T, history []int, tok, vocab int, opts ...nlp.SamplerOption) float64 {
	t.Helper()
	// Binary search the smallest logit advantage that keeps tok winning over token 0
	// under greedy decoding — that advantage equals the penalty applied to tok.
	base := append([]nlp.SamplerOption{nlp.WithTemperature(0)}, opts...)
	s := nlp.NewSampler(1, base...)
	lo, hi := 0.0, 64.0
	for range 60 {
		mid := (lo + hi) / 2
		logits := make([]float64, vocab)
		logits[tok] = mid // token 0 stays at 0: tok wins iff mid − penalty > 0
		if s.SampleWithHistory(logits, history) == tok {
			hi = mid
		} else {
			lo = mid
		}
	}
	return (lo + hi) / 2
}

// §V16 tier-1 (§T573): hand-computed DRY penalties — multiplier·base^(k−allowed) with
// k the longest suffix match, maximum over multiple matches, nothing below allowed.
func TestDRYHandComputed(t *testing.T) {
	// history: 5 6 7 9 5 6 7 — suffix [5 6 7] occurred at 0..2 followed by 9 (k=3);
	// suffix [5 6] precedes 7 at i=6? (k=2 for token 7 via i=6? i=6: w[:6] vs tail —
	// w[5]=6 vs w[6]=7 mismatch → k=0). i=2: token 7, w[1]=6 vs w[6]=7 → k=0.
	// So: token 9 gets k=3 → 0.8·1.75^(3−2)=1.4; no other token penalized.
	hist := []int{5, 6, 7, 9, 5, 6, 7}
	got := dryDelta(t, hist, 9, 16, nlp.WithDRY(0.8))
	if want := 0.8 * 1.75; math.Abs(got-want) > 1e-6 {
		t.Errorf("token 9 penalty = %.6f, want %.6f", got, want)
	}
	for _, tok := range []int{5, 6, 8, 3} {
		if got := dryDelta(t, hist, tok, 16, nlp.WithDRY(0.8)); got > 1e-6 {
			t.Errorf("token %d must be unpenalized, got %.6f", tok, got)
		}
	}
	// k exactly at allowed length: history 1 2 4 1 2 → token 4 has k=2 → 0.8·1.75^0.
	got = dryDelta(t, []int{1, 2, 4, 1, 2}, 4, 16, nlp.WithDRY(0.8))
	if want := 0.8; math.Abs(got-want) > 1e-6 {
		t.Errorf("k=allowed penalty = %.6f, want %.6f", got, want)
	}
	// below allowed: history 1 4 1 → k=1 → free.
	if got := dryDelta(t, []int{1, 4, 1}, 4, 16, nlp.WithDRY(0.8)); got > 1e-6 {
		t.Errorf("k<allowed must be free, got %.6f", got)
	}
	// short-period loop: a a a → continuing a has k=2 → penalized (the classic stutter).
	if got := dryDelta(t, []int{3, 3, 3}, 3, 16, nlp.WithDRY(0.8)); math.Abs(got-0.8) > 1e-6 {
		t.Errorf("period-1 loop penalty = %.6f, want 0.8", got)
	}
}

// §V16 tier-1 (§T573): parameters — base and allowed length change the exponent;
// range limits the scan window; breakers cut matches.
func TestDRYParameters(t *testing.T) {
	hist := []int{5, 6, 7, 9, 5, 6, 7} // k=3 for token 9
	got := dryDelta(t, hist, 9, 16, nlp.WithDRY(1.0), nlp.WithDRYBase(2.0), nlp.WithDRYAllowedLength(1))
	if want := 4.0; math.Abs(got-want) > 1e-6 { // 1.0·2^(3−1)
		t.Errorf("base/allowed: got %.6f, want %.6f", got, want)
	}
	// breaker inside the match caps k below allowed → free: history 1 2 3 4 1 2 3
	// gives token 4 k=3, but with 2 as a breaker the suffix stops at [3] → k=1.
	histB := []int{1, 2, 3, 4, 1, 2, 3}
	if got := dryDelta(t, histB, 4, 16, nlp.WithDRY(0.8)); math.Abs(got-0.8*1.75) > 1e-6 {
		t.Fatalf("baseline k=3: got %.6f", got)
	}
	if got := dryDelta(t, histB, 4, 16, nlp.WithDRY(0.8), nlp.WithDRYBreakers(2)); got > 1e-6 {
		t.Errorf("breaker must cut the match, got %.6f", got)
	}
	// range excludes the earlier occurrence → free.
	if got := dryDelta(t, hist, 9, 16, nlp.WithDRY(0.8), nlp.WithDRYRange(3)); got > 1e-6 {
		t.Errorf("out-of-range match must be free, got %.6f", got)
	}
}

// §V16 tier-1 (§T573): the loop-breaking property — a synthetic distribution that
// always prefers repeating a 4-token cycle loops forever under plain greedy decoding;
// with DRY the exponential penalty must break the cycle. (Grounded default: synthetic
// logits instead of a trained model — the property under test is the sampler's.)
func TestDRYBreaksLoops(t *testing.T) {
	const vocab = 8
	cycle := []int{1, 2, 3, 4}
	logitsFor := func(hist []int) []float64 {
		l := make([]float64, vocab)
		next := cycle[len(hist)%len(cycle)]
		l[next] = 4 // the model "wants" to repeat the cycle with margin 4
		return l
	}
	run := func(s nlp.TokenSampler) (loops int) {
		hist := append([]int(nil), cycle...)
		for range 64 {
			tok := s.SampleWithHistory(logitsFor(hist), hist)
			if tok != cycle[len(hist)%len(cycle)] {
				return loops
			}
			hist = append(hist, tok)
			loops++
		}
		return loops
	}
	if n := run(nlp.NewSampler(1, nlp.WithTemperature(0))); n != 64 {
		t.Fatalf("plain greedy must loop the full horizon, broke after %d", n)
	}
	if n := run(nlp.NewSampler(1, nlp.WithTemperature(0), nlp.WithDRY(0.8))); n >= 32 {
		t.Errorf("DRY must break the cycle well before the horizon, still looping after %d", n)
	}
}

// §V16 tier-1: SampleWithHistory must not mutate the caller's logits when only DRY is
// active (the penalize fast path returns the original slice).
func TestDRYDoesNotMutateLogits(t *testing.T) {
	s := nlp.NewSampler(1, nlp.WithTemperature(0), nlp.WithDRY(0.8))
	logits := []float64{0, 1, 2, 3}
	orig := append([]float64(nil), logits...)
	s.SampleWithHistory(logits, []int{1, 2, 3, 1, 2})
	for i := range logits {
		if logits[i] != orig[i] {
			t.Fatalf("caller logits mutated at %d: %v vs %v", i, logits, orig)
		}
	}
}
