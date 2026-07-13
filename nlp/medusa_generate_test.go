package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T444: with an impossible acceptance threshold (ε=1: no probability exceeds 1) every head
// proposal is rejected, so each round emits exactly the base greedy token — MedusaGenerate must
// equal plain greedy Generate token for token, whatever the (untrained) heads propose.
func TestMedusaGenerateAllRejectIsGreedy(t *testing.T) {
	model, prompt := loadGPTModel(t)
	prompt = prompt[:2]
	maxNew := model.Config.Ctx - len(prompt)
	heads, err := nlp.NewMedusaHeads(2, model.Config.Dim, model.Config.Vocab, 3, nlp.WithMedusaHeadsDtype(tensor.F64))
	if err != nil {
		t.Fatal(err)
	}
	got, stats, err := nlp.MedusaGenerate(model, heads, prompt, maxNew, 1.0, 1e9)
	if err != nil {
		t.Fatal(err)
	}
	want, err := model.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("length %d vs %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]: medusa %d != greedy %d", i, got[i], want[i])
		}
	}
	if stats.Accepted != 0 {
		t.Fatalf("ε=1 must reject everything, accepted %d", stats.Accepted)
	}
	if stats.Proposed == 0 {
		t.Fatal("no proposals made")
	}
}

// §T444: with a vanishing threshold every proposal is accepted — the window bookkeeping must
// stay exact (length budget, counts) and greedy chaining deterministic.
func TestMedusaGenerateAcceptAll(t *testing.T) {
	model, prompt := loadGPTModel(t)
	prompt = prompt[:2]
	maxNew := model.Config.Ctx - len(prompt)
	heads, err := nlp.NewMedusaHeads(2, model.Config.Dim, model.Config.Vocab, 3, nlp.WithMedusaHeadsDtype(tensor.F64))
	if err != nil {
		t.Fatal(err)
	}
	out1, stats, err := nlp.MedusaGenerate(model, heads, prompt, maxNew, 1e-12, 1e-12)
	if err != nil {
		t.Fatal(err)
	}
	if len(out1) != len(prompt)+maxNew {
		t.Fatalf("length %d, want %d", len(out1), len(prompt)+maxNew)
	}
	if stats.Accepted != stats.Proposed {
		t.Fatalf("accept-all: %d/%d", stats.Accepted, stats.Proposed)
	}
	for _, tok := range out1 {
		if tok < 0 || tok >= model.Config.Vocab {
			t.Fatalf("invalid token %d", tok)
		}
	}
	out2, _, err := nlp.MedusaGenerate(model, heads, prompt, maxNew, 1e-12, 1e-12)
	if err != nil {
		t.Fatal(err)
	}
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatal("MedusaGenerate must be deterministic")
		}
	}
}

// §T444 end-to-end: heads trained on the BASE MODEL'S OWN greedy rollouts (the Medusa training
// recipe — heads learn to predict what the base will say) reach real acceptance under the
// reference thresholds, and every emitted token is valid.
func TestMedusaGenerateWithAlignedHeads(t *testing.T) {
	model, _ := loadGPTModel(t)
	cfg := model.Config
	const K = 2
	heads, err := nlp.NewMedusaHeads(K, cfg.Dim, cfg.Vocab, 7, nlp.WithMedusaHeadsDtype(tensor.F64))
	if err != nil {
		t.Fatal(err)
	}

	// training data: the base's own greedy rollouts from every start token.
	ctx := backend.NewContext()
	var seqs [][]int
	var hiddens []*tensor.Tensor
	for s0 := range cfg.Vocab {
		seq, err := model.Generate([]int{s0}, cfg.Ctx-1, nlp.Greedy())
		if err != nil {
			t.Fatal(err)
		}
		h, err := model.ForwardHidden(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		seqs = append(seqs, seq)
		hiddens = append(hiddens, h)
	}
	opt := nn.NewAdamW(heads.Params(), 0.05, 0.0)
	for step := range 600 {
		n := step % len(seqs)
		s := seqs[n]
		tape := autograd.NewTape()
		tctx := tape.Context()
		logits, err := heads.Logits(tctx, hiddens[n])
		if err != nil {
			t.Fatal(err)
		}
		var total *tensor.Tensor
		for k := range K {
			rows := len(s) - 2 - k
			if rows <= 0 {
				continue
			}
			targets := tensor.New(tensor.F32, tensor.Shape{rows})
			for i := range rows {
				targets.SetF64(float64(s[i+2+k]), i)
			}
			sub, err := backend.Execute(tctx, backend.OpSlice, []*tensor.Tensor{logits[k]},
				backend.SliceAttrs{Axis: 0, Start: 0, End: rows})
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.CrossEntropy(tctx, sub[0], targets)
			if err != nil {
				t.Fatal(err)
			}
			if total == nil {
				total = loss
			} else {
				o, err := backend.Execute(tctx, backend.OpAdd, []*tensor.Tensor{total, loss}, nil)
				if err != nil {
					t.Fatal(err)
				}
				total = o[0]
			}
		}
		if err := tape.Backward(total); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(heads.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}

	prompt := []int{1, 2}
	out, stats, err := nlp.MedusaGenerate(model, heads, prompt, cfg.Ctx-len(prompt), -1, -1) // reference ε/δ
	if err != nil {
		t.Fatal(err)
	}
	for _, tok := range out {
		if tok < 0 || tok >= cfg.Vocab {
			t.Fatalf("invalid token %d", tok)
		}
	}
	if stats.Proposed == 0 {
		t.Fatal("no proposals")
	}
	if stats.Accepted == 0 {
		t.Fatalf("aligned heads accepted nothing (%d proposed) — the accept path never fired", stats.Proposed)
	}
	t.Logf("aligned Medusa heads: %d/%d proposals accepted (%.0f%%), %d tokens",
		stats.Accepted, stats.Proposed, 100*float64(stats.Accepted)/float64(stats.Proposed), len(out)-len(prompt))
}
