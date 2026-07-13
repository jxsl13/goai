//go:build darwin && cgo

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// §T501: K-QUANT quality — the practically dominant GGUF formats (§T477 covered Q8_0/Q4_0,
// whose 32-element blocks fit tiny models; the k-quants need 256-element super-blocks, hence a
// dim-256 Llama). The ladder Q6_K / Q4_K / Q2_K is measured against f32 by teacher-forced CE and
// argmax agreement: Q6_K must be near-lossless, Q4_K mildly degraded, and Q2_K is REPORTED — at
// 2 bits visible degradation is the expected, documented behavior.
func TestKQuantQualityLadderOnTrainedLlama(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a dim-256 llama on cpu; skipped in -short")
	}
	corpus := specCorpus(600)
	toks, vocab := charTokens(corpus)
	cfg := nlp.LlamaConfig{
		Vocab: vocab, Ctx: 128, Dim: 256, Heads: 4, KVHeads: 2, Layers: 2,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000, // all matmul k-dims %256: k-quantizable
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	tf, tl := trainCharLlama(t, m, toks, 120, 48, 3e-3)
	t.Logf("llama(dim256) trained: CE %.3f → %.3f", tf, tl)

	quant := func(qt gguf.QuantType) *nlp.QuantLlama {
		q, err := nlp.QuantizeLlama(m, qt)
		if err != nil {
			t.Fatal(err)
		}
		return q
	}
	q6, q4, q2 := quant(gguf.Q6_K), quant(gguf.Q4_K), quant(gguf.Q2_K)

	ctx := backend.NewContext()
	row := func(l *tensor.Tensor) []float64 {
		v := l.Shape()[1]
		out := make([]float64, v)
		for j := range v {
			out[j] = l.AtF64(0, j)
		}
		return out
	}
	held := toks[5000:5100]
	type stepFn func(cache *nlp.LlamaCache, token, pos int) []float64
	f32Step := func(c *nlp.LlamaCache, tok, pos int) []float64 {
		l, err := m.DecodeStep(ctx, c, tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		return row(l)
	}
	qStep := func(q *nlp.QuantLlama) stepFn {
		return func(c *nlp.LlamaCache, tok, pos int) []float64 {
			l, err := q.DecodeStep(ctx, c, tok, pos)
			if err != nil {
				t.Fatal(err)
			}
			return row(l)
		}
	}
	evalCE := func(step stepFn) float64 {
		cache := m.NewCache()
		var bits float64
		for i := 0; i < len(held)-1; i++ {
			logits := step(cache, held[i], i)
			var mx float64 = math.Inf(-1)
			for _, v := range logits {
				if v > mx {
					mx = v
				}
			}
			var sum float64
			for _, v := range logits {
				sum += math.Exp(v - mx)
			}
			p := math.Exp(logits[held[i+1]]-mx) / sum
			bits += -math.Log2(p)
		}
		return bits / float64(len(held)-1)
	}
	argmax := func(x []float64) int {
		b := 0
		for j := 1; j < len(x); j++ {
			if x[j] > x[b] {
				b = j
			}
		}
		return b
	}
	agree := func(step stepFn) float64 {
		rc, qc := m.NewCache(), m.NewCache()
		match := 0
		for i := 0; i < len(held)-1; i++ {
			if argmax(f32Step(rc, held[i], i)) == argmax(step(qc, held[i], i)) {
				match++
			}
		}
		return float64(match) / float64(len(held)-1)
	}

	ceF := evalCE(f32Step)
	ce6, ce4, ce2 := evalCE(qStep(q6)), evalCE(qStep(q4)), evalCE(qStep(q2))
	ag6, ag4, ag2 := agree(qStep(q6)), agree(qStep(q4)), agree(qStep(q2))
	t.Logf("teacher-forced CE (bits): f32 %.4f | Q6_K %.4f (Δ%+.4f) | Q4_K %.4f (Δ%+.4f) | Q2_K %.4f (Δ%+.4f)",
		ceF, ce6, ce6-ceF, ce4, ce4-ceF, ce2, ce2-ceF)
	t.Logf("argmax agreement vs f32: Q6_K %.0f%% | Q4_K %.0f%% | Q2_K %.0f%%", 100*ag6, 100*ag4, 100*ag2)

	if ce6 > ceF+0.05 {
		t.Fatalf("Q6_K costs %.4f bits — 6-bit k-quant should be near-lossless", ce6-ceF)
	}
	if ce4 > ceF+0.2 {
		t.Fatalf("Q4_K costs %.4f bits — beyond the 4-bit band", ce4-ceF)
	}
	if ag6 < 0.95 {
		t.Fatalf("Q6_K agreement %.0f%%", 100*ag6)
	}
	// judge the ladder on AGREEMENT: on a 99-token window the CE is noisy enough that Q2_K's
	// Δ can even come out negative while it flips 14% of the decisions — agreement is the
	// sensitive metric at aggressive quantization (measured 97/97/86 with CE Δ −0.03 for Q2_K).
	if ag2 < ag4 && ag2 < ag6 {
		t.Logf("FINDING: the ladder shows in agreement (%.0f/%.0f/%.0f%%) — Q2_K flips decisions even where its small-window CE looks harmless (Δ%+.3f)",
			100*ag6, 100*ag4, 100*ag2, ce2-ceF)
	} else {
		t.Logf("FINDING: Q2_K held even on agreement (%.0f%%)", 100*ag2)
	}
}
