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

// §T504: LoRA fine-tuning of the built-in GPT — the seam §T449 recorded as blocked turned out
// open (the attention projections are ordinary matmuls around the fused SDPA core, not inside
// it). Verified: (1) attaching adapters changes NOTHING until training (B=0 ⇒ logits
// bit-identical); (2) training ONLY the adapters on a dialect improves that dialect's CE
// substantially while (3) every base weight stays BIT-identical — the PEFT contract.
func TestLoRAFinetunesGPTFrozenBase(t *testing.T) {
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
	probe := toksA[:32]
	ctx := backend.NewContext().WithBackend(be)
	before, err := m.Forward(ctx, probe)
	if err != nil {
		t.Fatal(err)
	}
	ceBase := ce(toksA)

	// snapshot every base weight for the bit-identity check.
	baseSnapshot := map[*tensor.Tensor][]float64{}
	for _, p := range m.Params() {
		vals := make([]float64, p.Numel())
		for i := range vals {
			vals[i] = p.AtF64(tensor.Unravel(i, p.Shape())...)
		}
		baseSnapshot[p] = vals
	}

	adapters, err := nlp.ApplyLoRAGPT(m, 4, 8, 21)
	if err != nil {
		t.Fatal(err)
	}
	if len(adapters) != cfg.Layers*4*2 {
		t.Fatalf("adapter params %d, want %d", len(adapters), cfg.Layers*4*2)
	}
	// (1) B=0: attaching adapters is a bit-exact no-op.
	after, err := m.Forward(ctx, probe)
	if err != nil {
		t.Fatal(err)
	}
	for i := range before.Numel() {
		idx := tensor.Unravel(i, before.Shape())
		if before.AtF64(idx...) != after.AtF64(idx...) {
			t.Fatalf("attaching zero-init LoRA changed logits at %v", idx)
		}
	}

	// (2) train ONLY the adapters on dialect A — the base is frozen by omission.
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
	ceTuned := ce(toksA)
	t.Logf("dialect-A CE: base %.3f → LoRA-tuned %.3f (adapters: %d tensors, rank 4)", ceBase, ceTuned, len(adapters))
	if ceTuned >= ceBase-0.05 {
		t.Fatalf("LoRA fine-tune did not improve the dialect (%.3f → %.3f)", ceBase, ceTuned)
	}
	// (3) the base weights are BIT-identical — the PEFT contract.
	for _, p := range m.Params() {
		vals := baseSnapshot[p]
		for i := range vals {
			if p.AtF64(tensor.Unravel(i, p.Shape())...) != vals[i] {
				t.Fatal("a base weight changed during LoRA training — the freeze is broken")
			}
		}
	}
}
