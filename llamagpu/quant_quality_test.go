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

// §T477: quantization QUALITY on a real trained model — §T413–§T416 measured quant decode for
// speed and memory on random weights; the user-facing question is how much MODEL QUALITY each
// ggml level costs. On the trained char-Llama, Q8_0 and Q4_0 are compared against f32 by (a)
// teacher-forced cross-entropy on held-out corpus text and (b) greedy-generation agreement.
func TestQuantizationQualityOnTrainedLlama(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a model on cpu; skipped in -short")
	}
	corpus := specCorpus(600)
	toks, vocab := charTokens(corpus)
	cfg := nlp.LlamaConfig{
		Vocab: vocab, Ctx: 128, Dim: 64, Heads: 4, KVHeads: 2, Layers: 2,
		Hidden: 192, Eps: 1e-5, RopeBase: 10000, // hidden %32: quantizable blocks
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	tf, tl := trainCharLlama(t, m, toks, 150, 48, 3e-3)
	t.Logf("llama trained: CE %.3f → %.3f", tf, tl)

	q8m, err := nlp.QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	q4m, err := nlp.QuantizeLlama(m, gguf.Q4_0)
	if err != nil {
		t.Fatal(err)
	}

	type stepper func(cache any, token, pos int) []float64
	ctx := backend.NewContext()
	row := func(l *tensor.Tensor) []float64 {
		v := l.Shape()[1]
		out := make([]float64, v)
		for j := range v {
			out[j] = l.AtF64(0, j)
		}
		return out
	}
	f32Step := func(cache any, token, pos int) []float64 {
		l, err := m.DecodeStep(ctx, cache.(*nlp.LlamaCache), token, pos)
		if err != nil {
			t.Fatal(err)
		}
		return row(l)
	}
	q8Step := func(cache any, token, pos int) []float64 {
		l, err := q8m.DecodeStep(ctx, cache.(*nlp.LlamaCache), token, pos)
		if err != nil {
			t.Fatal(err)
		}
		return row(l)
	}
	q4Step := func(cache any, token, pos int) []float64 {
		l, err := q4m.DecodeStep(ctx, cache.(*nlp.LlamaCache), token, pos)
		if err != nil {
			t.Fatal(err)
		}
		return row(l)
	}

	// (a) teacher-forced CE (bits/token) over a held-out corpus window.
	held := toks[5000:5100]
	teacherCE := func(step stepper) float64 {
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
	ceF32 := teacherCE(f32Step)
	ceQ8 := teacherCE(q8Step)
	ceQ4 := teacherCE(q4Step)

	// (b) TEACHER-FORCED next-token agreement: both models see the identical held-out context at
	// every step (a free-running comparison would diverge at the first mismatch and then compare
	// different contexts — a metric artifact, not a quality signal).
	argmax := func(x []float64) int {
		b := 0
		for j := 1; j < len(x); j++ {
			if x[j] > x[b] {
				b = j
			}
		}
		return b
	}
	agree := func(step stepper) float64 {
		refCache, qCache := m.NewCache(), m.NewCache()
		match := 0
		for i := 0; i < len(held)-1; i++ {
			rl := f32Step(refCache, held[i], i)
			ql := step(qCache, held[i], i)
			if argmax(rl) == argmax(ql) {
				match++
			}
		}
		return float64(match) / float64(len(held)-1)
	}
	agQ8 := agree(q8Step)
	agQ4 := agree(q4Step)

	t.Logf("teacher-forced CE (bits/tok): f32 %.4f | Q8_0 %.4f (Δ%.4f) | Q4_0 %.4f (Δ%.4f)",
		ceF32, ceQ8, ceQ8-ceF32, ceQ4, ceQ4-ceF32)
	t.Logf("teacher-forced argmax agreement vs f32: Q8_0 %.0f%% | Q4_0 %.0f%%", 100*agQ8, 100*agQ4)

	if ceQ8 > ceF32+0.05 {
		t.Fatalf("Q8_0 costs %.4f bits — 8-bit should be near-lossless on this model", ceQ8-ceF32)
	}
	if ceQ4 > ceF32+0.5 {
		t.Fatalf("Q4_0 costs %.4f bits — beyond the expected 4-bit degradation band", ceQ4-ceF32)
	}
	if agQ8 < 0.95 {
		t.Fatalf("Q8_0 greedy agreement %.0f%% — should be near-perfect", 100*agQ8)
	}
}
