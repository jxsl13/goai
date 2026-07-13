//go:build darwin && cgo

package llamagpu_test

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// §T452: prompt-lookup speculative decoding measured with a REAL trained model — the second
// free-drafting scheme (§T446's Medusa was the first). The char-grammar corpus is highly
// repetitive, exactly prompt-lookup's regime: a base trained on it greedily regenerates n-grams
// that already occur in the running sequence, so history matching drafts well. Losslessness is
// exact by construction (speculative accept/reject over one-hot drafts), so the sharp assertion
// is token-for-token equality with plain batched greedy Generate — plus the measured acceptance
// and A/B throughput.
func TestPromptLookupTrainedThroughput(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a model; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("metal not registered")
	}
	corpus := specCorpus(600)
	toks, vocab := charTokens(corpus)
	cfg := nlp.GPTConfig{Vocab: vocab, Ctx: 256, Dim: 96, Heads: 4, Layers: 3, Eps: 1e-5}
	m := trainableGPT(t, cfg, 1)
	tf, tl := trainCharGPT(t, m, be, toks, 240, 128, 3e-3)
	t.Logf("base trained: CE %.3f → %.3f", tf, tl)

	prompt := toks[:48] // long repetitive prompt: n-gram lookup has material to match
	const maxNew = 96

	plDec, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer plDec.Release()
	got, stats, err := llamagpu.PromptLookupGenerate(plDec, prompt, maxNew, 3, 8, nlp.Greedy(), 7)
	if err != nil {
		t.Fatal(err)
	}
	plainDec, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	defer plainDec.Release()
	want, err := plainDec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("length %d vs %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]: prompt-lookup %d != plain %d — losslessness broken", i, got[i], want[i])
		}
	}
	if stats.Proposed == 0 {
		t.Fatal("no n-gram drafts proposed on a repetitive corpus")
	}

	// interleaved A/B (§V22: a sequential block hands the first scheme any cold-start /
	// thermal outlier — the first run of this test measured plain at 220 ms vs its steady 86 ms).
	time1 := func(f func() error) float64 {
		s := time.Now()
		if err := f(); err != nil {
			t.Fatal(err)
		}
		return time.Since(s).Seconds() * 1e3
	}
	runPL := func() error {
		_, _, err := llamagpu.PromptLookupGenerate(plDec, prompt, maxNew, 3, 8, nlp.Greedy(), 7)
		return err
	}
	runPlain := func() error {
		_, err := plainDec.Generate(prompt, maxNew, nlp.Greedy())
		return err
	}
	var mPL, mPlain []float64
	for range 3 {
		mPL = append(mPL, time1(runPL))
		mPlain = append(mPlain, time1(runPlain))
	}
	median := func(x []float64) float64 {
		a, b, c := x[0], x[1], x[2]
		if a > b {
			a, b = b, a
		}
		if b > c {
			b = c
		}
		if a > b {
			b = a
		}
		return b
	}
	tPL, tPlain := median(mPL), median(mPlain)
	t.Logf("LOSSLESS; acceptance %.0f%% (%d/%d); plain %.0f ms (%.0f tok/s) vs prompt-lookup %.0f ms (%.0f tok/s) → speedup %.2fx",
		100*stats.AcceptanceRate(), stats.Accepted, stats.Proposed,
		tPlain, float64(maxNew)/tPlain*1e3, tPL, float64(maxNew)/tPL*1e3, tPlain/tPL)
}
