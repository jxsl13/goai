//go:build darwin && cgo

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T475: the StreamingLLM/attention-sink claim measured on a REAL trained model — decoding with a
// BOUNDED KV cache (a few sink tokens + a recent window, KVCache.EvictStreaming) must generate
// text of nearly full-cache quality, where quality = mean surprise (bits/token) of the generated
// tail under the model itself with FULL teacher-forced context. The sink-free ablation
// (recent-only eviction with the same total budget) is measured and reported — the paper's claim
// is that the sinks are what keep bounded-cache decoding stable.
func TestStreamingEvictionQualityOnTrainedModel(t *testing.T) {
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
	cfg := nlp.GPTConfig{Vocab: vocab, Ctx: 512, Dim: 96, Heads: 4, Layers: 3, Eps: 1e-5}
	m := trainableGPT(t, cfg, 1)
	tf, tl := trainCharGPT(t, m, be, toks, 240, 128, 3e-3)
	t.Logf("model: CE %.3f → %.3f", tf, tl)

	prompt := toks[:16]
	const maxNew = 300
	ctx := backend.NewContext().WithBackend(be)

	row := func(l *tensor.Tensor) []float64 {
		v := l.Shape()[1]
		out := make([]float64, v)
		for j := range v {
			out[j] = l.AtF64(0, j)
		}
		return out
	}
	// greedy decode; evict < 0 disables eviction, otherwise evict to (sink, recent) every step.
	decode := func(sink, recent int) []int {
		cache := m.NewCache()
		out := append([]int(nil), prompt...)
		var logits []float64
		pos := 0
		for _, tok := range prompt {
			l, err := m.DecodeStep(ctx, cache, tok, pos)
			if err != nil {
				t.Fatal(err)
			}
			logits = row(l)
			pos++
		}
		for range maxNew {
			next := 0
			for v := 1; v < len(logits); v++ {
				if logits[v] > logits[next] {
					next = v
				}
			}
			out = append(out, next)
			l, err := m.DecodeStep(ctx, cache, next, pos)
			if err != nil {
				t.Fatal(err)
			}
			logits = row(l)
			pos++
			if sink >= 0 {
				cache.EvictStreaming(sink, recent)
			}
		}
		return out
	}

	// quality: mean surprise (bits/token) of the generated tail under FULL teacher-forced context.
	quality := func(seq []int) float64 {
		logits, err := m.Forward(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		sub, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{logits},
			backend.SliceAttrs{Axis: 0, Start: len(prompt) - 1, End: len(seq) - 1})
		if err != nil {
			t.Fatal(err)
		}
		lp, err := nn.TokenLogProbs(ctx, sub[0], seq[len(prompt):])
		if err != nil {
			t.Fatal(err)
		}
		var bits float64
		for i := range lp.Shape()[0] {
			bits += -lp.AtF64(i) / math.Ln2
		}
		return bits / float64(lp.Shape()[0])
	}

	full := quality(decode(-1, 0))    // unbounded cache (the reference)
	sinks := quality(decode(4, 64))   // StreamingLLM: 4 sinks + 64 recent (68-row cache)
	noSinks := quality(decode(0, 68)) // ablation: same budget, recent-only
	t.Logf("mean surprise of 300 greedy tokens (bits): full cache %.3f | sink4+recent64 %.3f | recent68-no-sinks %.3f", full, sinks, noSinks)

	if sinks > full+0.3 {
		t.Fatalf("bounded cache WITH sinks degraded quality: %.3f vs full %.3f", sinks, full)
	}
	if noSinks > sinks {
		t.Logf("FINDING: the sink-free ablation is worse (%.3f vs %.3f) — the attention-sink claim reproduced", noSinks, sinks)
	} else {
		t.Logf("FINDING: the sink-free ablation matched sinks here (%.3f vs %.3f) — this small model shows no sink dependence at this budget", noSinks, sinks)
	}
}
