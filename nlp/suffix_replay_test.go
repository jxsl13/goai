package nlp

import (
	"math/rand"
	"testing"
)

// replayAcceptTotal replays a drafter over a fixed reference sequence: at each position i it drafts from
// ref[:i] and counts the longest prefix that matches ref[i:] (the accept length a verified speculative
// step would get, since the reference IS the greedy output). Total accepted tokens ↑ ⇒ forward passes ↓.
// Returns (total accepted, forward passes) starting from a warm prefix so history exists.
func replayAcceptTotal(ref []int, drafter func([]int, int, int) []int, maxNgram, draftLen, warm int) (accepted, passes int) {
	i := warm
	for i < len(ref) {
		d := drafter(ref[:i], maxNgram, draftLen)
		a := 0
		for a < len(d) && i+a < len(ref) && d[a] == ref[i+a] {
			a++
		}
		accepted += a
		passes++   // one model forward verifies the draft + emits the bonus token
		i += a + 1 // a accepted draft tokens + 1 bonus (the model's own next token)
	}
	return accepted, passes
}

// buildRepetitiveRef builds a realistic repetitive token stream: templated "records" whose structure
// recurs but whose field values vary from a small pool — so the longest suffix (the template tail) often
// recurs with a DOMINANT-but-not-unique continuation, the exact regime where frequency scoring should
// accept more than copying an arbitrary earlier occurrence. Deterministic (seeded).
func buildRepetitiveRef(records int, seed int64, dominantProb float64) []int {
	rng := rand.New(rand.NewSource(seed))
	tmpl := []int{2, 3, 4, 5} // recurring template head
	var out []int
	for r := 0; r < records; r++ {
		out = append(out, tmpl...)
		if rng.Float64() < dominantProb {
			out = append(out, 100, 100)
		} else {
			v := 101 + rng.Intn(4)
			out = append(out, v, v)
		}
		out = append(out, 6) // record separator
	}
	return out
}

// buildRandomRef is a NON-repetitive control: SuffixLookup must not regress vs NgramLookup here.
func buildRandomRef(n int, seed int64) []int {
	rng := rand.New(rand.NewSource(seed))
	out := make([]int, n)
	for i := range out {
		out[i] = rng.Intn(200)
	}
	return out
}

// TestSuffixVsNgramAcceptRate MEASURES whether the frequency drafter accepts more tokens than the
// earliest-match drafter on a realistic repetitive stream — the gate for whether SuffixDecode is a real
// (validated) speedup over PromptLookupDecode or a marginal non-win.
func TestSuffixVsNgramAcceptRate(t *testing.T) {
	tpp := func(ref []int, drafter func([]int, int, int) []int) float64 {
		a, p := replayAcceptTotal(ref, drafter, 3, 8, 16)
		return float64(a+p) / float64(p)
	}
	// Repetitive-with-ambiguity sweep: dominant-continuation probability × seed. SuffixLookup should
	// win (more tok/pass) whenever the recurring suffix has a dominant-but-noisy continuation, and never
	// meaningfully regress.
	var wins, total int
	var sumSpeedup float64
	for _, dom := range []float64{0.5, 0.6, 0.7, 0.85} {
		for _, seed := range []int64{1, 7, 42} {
			ref := buildRepetitiveRef(400, seed, dom)
			n, s := tpp(ref, NgramLookup), tpp(ref, SuffixLookup)
			total++
			sumSpeedup += s / n
			if s >= n-1e-9 {
				wins++
			}
			t.Logf("dom=%.2f seed=%d: ngram=%.3f suffix=%.3f tok/pass  speedup=%.3f×", dom, seed, n, s, s/n)
		}
	}
	t.Logf("SUMMARY repetitive: SuffixLookup ≥ NgramLookup in %d/%d cases, avg speedup %.3f×", wins, total, sumSpeedup/float64(total))
	if wins < total {
		t.Errorf("SuffixLookup regressed in %d/%d repetitive cases (should be a strict-or-better superset in the target regime)", total-wins, total)
	}
	// Non-repetitive control: both fall back to ~1 tok/pass, SuffixLookup must not regress.
	rr := buildRandomRef(3000, 3)
	n, s := tpp(rr, NgramLookup), tpp(rr, SuffixLookup)
	t.Logf("random control: ngram=%.4f suffix=%.4f tok/pass (both ~1.0, no regression)", n, s)
	if s < n-0.02 {
		t.Errorf("SuffixLookup regressed on non-repetitive input: %.4f < %.4f", s, n)
	}
}
