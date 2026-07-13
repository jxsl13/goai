//go:build darwin && cgo

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// trainCharLlama mirrors trainCharGPT for the RoPE architecture (§T476). Llama models are F64
// (§B44), so training runs on the cpu backend.
func trainCharLlama(t *testing.T, m *nlp.Llama, corpus []int, steps, window int, lr float64) (first, last float64) {
	t.Helper()
	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Skip("cpu backend not registered")
	}
	opt := nn.NewAdamW(m.Params(), lr, 0.01)
	targets := tensor.New(tensor.F64, tensor.Shape{window})
	for step := range steps {
		start := (step * 97) % (len(corpus) - window - 1)
		tokens := corpus[start : start+window]
		for i := range window {
			targets.SetF64(float64(corpus[start+1+i]), i)
		}
		tape := autograd.NewTapeOn(be)
		ctx := tape.Context()
		logits, err := m.Forward(ctx, tokens)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(m.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	if last >= first*0.6 {
		t.Fatalf("llama did not train: CE %.4f → %.4f", first, last)
	}
	return first, last
}

// §T476: StreamGenerate's defining claim measured on a REAL trained model — an attention-sink +
// sliding-window cache lets a RoPE model generate FAR BEYOND its training context at stable
// quality (the use case §T475's GPT could not exercise: learned positional embeddings cannot
// extrapolate, RoPE can). A char-Llama trained at context 64 streams 256 tokens (4× its context);
// quality = windowed teacher-forced surprise, compared between the earliest and latest generated
// stretch — no degeneration cliff allowed.
func TestStreamGenerateBeyondContextOnTrainedLlama(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a model on cpu; skipped in -short")
	}
	corpus := specCorpus(600)
	toks, vocab := charTokens(corpus)
	cfg := nlp.LlamaConfig{
		Vocab: vocab, Ctx: 64, Dim: 64, Heads: 4, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	tf, tl := trainCharLlama(t, m, toks, 150, 48, 3e-3)
	t.Logf("llama trained (cpu, F64): CE %.3f → %.3f", tf, tl)

	prompt := toks[:16]
	const maxNew = 256 // 4× the training context
	out, err := m.StreamGenerate(prompt, maxNew, 4 /*sinks*/, 48 /*window*/, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+maxNew {
		t.Fatalf("streamed length %d, want %d", len(out), len(prompt)+maxNew)
	}
	for _, tok := range out {
		if tok < 0 || tok >= vocab {
			t.Fatalf("invalid token %d", tok)
		}
	}

	// windowed surprise: score each token in span against the model with the previous
	// (ctx−1)-token window as teacher-forced context.
	ctx := backend.NewContext()
	spanSurprise := func(lo, hi int) float64 {
		var bits float64
		n := 0
		for i := lo; i < hi; i++ {
			start := i - (cfg.Ctx - 1)
			if start < 0 {
				start = 0
			}
			win := out[start : i+1]
			logits, err := m.Forward(ctx, win[:len(win)-1])
			if err != nil {
				t.Fatal(err)
			}
			last := logits.Shape()[0] - 1
			v := logits.Shape()[1]
			var mx float64 = math.Inf(-1)
			for j := range v {
				if x := logits.AtF64(last, j); x > mx {
					mx = x
				}
			}
			var sum float64
			for j := range v {
				sum += math.Exp(logits.AtF64(last, j) - mx)
			}
			p := math.Exp(logits.AtF64(last, out[i])-mx) / sum
			bits += -math.Log2(p)
			n++
		}
		return bits / float64(n)
	}
	near := spanSurprise(len(prompt), len(prompt)+48) // first generated stretch (in-context regime)
	far := spanSurprise(len(out)-48, len(out))        // last stretch, ~4× past the training context

	// ablation: same budget without the sinks (the paper predicts this is what collapses).
	outNoSink, err := m.StreamGenerate(prompt, maxNew, 0, 52, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	save := out
	out = outNoSink
	farNoSink := spanSurprise(len(outNoSink)-48, len(outNoSink))
	out = save

	t.Logf("windowed surprise: first 48 gen %.3f bits | last 48 at 4× context: sinks4 %.3f | NO sinks %.3f bits (train CE %.3f)",
		near, far, farNoSink, tl)
	// far-tail must stay in the coherent regime: within 1 bit of the in-context stretch and
	// clearly below random (log2 vocab).
	if far > near+1.0 {
		t.Fatalf("quality cliff far beyond the training context: %.3f → %.3f bits", near, far)
	}
	if far > math.Log2(float64(vocab))/2 {
		t.Fatalf("far-tail surprise %.3f bits approaches noise (log2 vocab = %.2f)", far, math.Log2(float64(vocab)))
	}
	if farNoSink > far {
		t.Logf("FINDING: sink-free streaming is worse (%.3f vs %.3f) — the attention-sink claim shows on the RoPE model", farNoSink, far)
	} else {
		t.Logf("FINDING: no sink dependence at this scale (%.3f vs %.3f) — consistent with §T475 on the GPT", farNoSink, far)
	}
}
