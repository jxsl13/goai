//go:build darwin && cgo

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// nf4FreezeInPlace replaces w's values with dequant(quant_NF4(w)) using the paper's
// DOUBLE quantization (block 64 weights, block-256 quantized scales) — the frozen
// 4-bit base a QLoRA adapter trains against.
func nf4FreezeInPlace(w *tensor.Tensor) {
	n := w.Numel()
	vals := make([]float64, n)
	for i := range vals {
		vals[i] = w.AtF64(tensor.Unravel(i, w.Shape())...)
	}
	packed, absmax := nn.QuantizeNF4(vals, 64)
	codes, absmax2, off := nn.QuantizeScalesNF4(absmax, 256)
	deqScales := nn.DequantizeScalesNF4(codes, absmax2, off, 256, len(absmax))
	deq := nn.DequantizeNF4(packed, deqScales, 64, n)
	for i := range deq {
		w.SetF64(deq[i], tensor.Unravel(i, w.Shape())...)
	}
}

// §T552: the QLoRA RECIPE (Dettmers et al. 2023, arXiv:2305.14314) end to end — the
// NF4 codec and LoRA existed separately; this proves their composition: (1) the base
// model's attention projections are frozen as NF4 (double-quantized, w ← deq(q(w)));
// (2) rank-4 LoRA adapters train on top; (3) the adapters recover the dialect BELOW
// the quantized base's CE while every NF4-frozen weight stays bit-identical. What
// this fire does NOT claim: packed-4-bit STORAGE during training (our in-memory
// base materializes the dequantized values; the math is the paper's).
func TestQLoRAFinetunesNF4FrozenBase(t *testing.T) {
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
	mixed := dialectCorpus(400, []string{" sees ", " finds ", " chases ", " fears "})
	corpusA := dialectCorpus(400, []string{" sees ", " finds "})
	toksMixed, vocab := charTokens(mixed)
	toksA := tokensWith(corpusA, mixed)
	cfg := nlp.GPTConfig{Vocab: vocab, Ctx: 256, Dim: 96, Heads: 4, Layers: 3, Eps: 1e-5}

	m := trainableGPT(t, cfg, 1)
	trainCharGPT(t, m, be, toksMixed, 200, 128, 3e-3)

	ce := func(toks []int) float64 {
		ctx := backend.NewContext().WithBackend(be)
		const window = 128
		logits, err := m.Forward(ctx, toks[:window])
		if err != nil {
			t.Fatal(err)
		}
		lp, err := nn.TokenLogProbs(ctx, logits, toks[1:window+1])
		if err != nil {
			t.Fatal(err)
		}
		var nats float64
		for i := range lp.Shape()[0] {
			nats += -lp.AtF64(i)
		}
		return nats / float64(lp.Shape()[0])
	}
	ceFull := ce(toksA)

	// (1) freeze the attention projections as double-quantized NF4.
	for _, b := range m.Blocks {
		for _, w := range []*tensor.Tensor{b.Attn.Wq, b.Attn.Wk, b.Attn.Wv, b.Attn.Wo} {
			nf4FreezeInPlace(w)
		}
	}
	ceNF4 := ce(toksA)

	snapshot := map[*tensor.Tensor][]float64{}
	for _, p := range m.Params() {
		vals := make([]float64, p.Numel())
		for i := range vals {
			vals[i] = p.AtF64(tensor.Unravel(i, p.Shape())...)
		}
		snapshot[p] = vals
	}

	// (2) rank-4 LoRA on the NF4 base, adapters only.
	adapters, err := nlp.ApplyLoRAGPT(m, 4, 8, 21)
	if err != nil {
		t.Fatal(err)
	}
	opt := nn.NewAdamW(adapters, 5e-3, 0)
	const window = 128
	targets := tensor.New(tensor.F32, tensor.Shape{window})
	for step := range 120 {
		start := (step * 97) % (len(toksA) - window - 1)
		tokens := toksA[start : start+window]
		for i := range window {
			targets.SetF64(float64(toksA[start+1+i]), i)
		}
		tape := autograd.NewTapeOn(be)
		c := tape.Context()
		logits, err := m.Forward(c, tokens)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(c, logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(adapters, tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	ceQLoRA := ce(toksA)
	t.Logf("dialect-A CE: full base %.3f → NF4-frozen %.3f → QLoRA-tuned %.3f", ceFull, ceNF4, ceQLoRA)
	if ceQLoRA >= ceNF4-0.05 {
		t.Fatalf("QLoRA did not improve over the NF4 base (%.3f → %.3f)", ceNF4, ceQLoRA)
	}
	// (3) the NF4-frozen base is bit-identical after adapter training.
	for _, p := range m.Params() {
		vals := snapshot[p]
		for i := range vals {
			if p.AtF64(tensor.Unravel(i, p.Shape())...) != vals[i] {
				t.Fatal("an NF4-frozen base weight changed during QLoRA training")
			}
		}
	}
}
