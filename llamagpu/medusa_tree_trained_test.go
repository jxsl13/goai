//go:build darwin && cgo

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T510 measurement: tree-Medusa vs chain-Medusa with TRAINED heads on the same base
// (§T434 infra, head recipe from §T446). Per round the chain's all-top-1 path is a
// member of the topK≥2 candidate tree, so on the SAME context the tree's deepest
// accepted path is at least the chain's — tokens/round (derived as emitted/rounds
// with rounds = emitted − Accepted, since every round emits exactly 1 + accepted)
// must not fall below the chain's. Both variants run nlp's analysis-scale loop —
// identical forward machinery, fair A/B; the tree's masked attention reaches the
// reference kernel through the §T461 fallback.
func TestMedusaTreeVsChainTrained(t *testing.T) {
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

	// heads trained on the base's own batched-greedy rollouts (§T446 recipe).
	roll, err := llamagpu.NewGPT(m)
	if err != nil {
		t.Fatal(err)
	}
	const K = 3
	heads, err := nlp.NewMedusaHeads(K, cfg.Dim, cfg.Vocab, 11)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext().WithBackend(be)
	var seqs [][]int
	var hiddens []*tensor.Tensor
	for r := range 6 {
		prompt := toks[r*31 : r*31+16]
		seq, err := roll.Generate(prompt, 120, nlp.Greedy())
		if err != nil {
			t.Fatal(err)
		}
		h, err := m.ForwardHidden(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		seqs = append(seqs, seq)
		hiddens = append(hiddens, h)
	}
	roll.Release()

	opt := nn.NewAdamW(heads.Params(), 0.02, 0.0)
	for step := range 300 {
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

	const maxNew = 64
	var chainAcc, treeAcc, chainRounds, treeRounds, emitted int
	for r := range 3 { // three prompts, aggregated
		prompt := toks[r*57 : r*57+16]
		chainOut, chainStats, err := nlp.MedusaGenerate(m, heads, prompt, maxNew, -1, -1)
		if err != nil {
			t.Fatal(err)
		}
		treeOut, treeStats, err := nlp.MedusaGenerateTree(m, heads, prompt, maxNew, -1, -1, 2)
		if err != nil {
			t.Fatal(err)
		}
		ce, te := len(chainOut)-len(prompt), len(treeOut)-len(prompt)
		if ce != maxNew || te != maxNew {
			t.Fatalf("prompt %d: emitted chain %d / tree %d, want %d", r, ce, te, maxNew)
		}
		emitted += maxNew
		chainAcc += chainStats.Accepted
		treeAcc += treeStats.Accepted
		chainRounds += maxNew - chainStats.Accepted // every round emits 1 + accepted
		treeRounds += maxNew - treeStats.Accepted
	}
	if treeAcc == 0 {
		t.Fatal("trained heads: tree acceptance never fired")
	}
	chainTPR := float64(emitted) / float64(chainRounds)
	treeTPR := float64(emitted) / float64(treeRounds)
	t.Logf("chain: %d accepted, %d rounds → %.2f tok/round; tree(topK=2): %d accepted, %d rounds → %.2f tok/round",
		chainAcc, chainRounds, chainTPR, treeAcc, treeRounds, treeTPR)
	// the chain path is a member of every round's tree — the tree must not do worse.
	if treeTPR < chainTPR {
		t.Fatalf("tree %.3f tok/round below chain %.3f — superset property violated", treeTPR, chainTPR)
	}
}
