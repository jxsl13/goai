package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §T509: with topK=1 the candidate tree is a single chain, so MedusaGenerateTree must
// reproduce MedusaGenerate EXACTLY — same emitted tokens, same proposal/acceptance
// stats (the collapse discipline: the general mechanism degenerates onto the verified
// special case).
func TestMedusaTreeCollapsesToChain(t *testing.T) {
	model, tokens := loadGPTModel(t)
	heads, err := nlp.NewMedusaHeads(3, model.Config.Dim, model.Config.Vocab, 42,
		nlp.WithMedusaHeadsDtype(model.TokEmb.Dtype()))
	if err != nil {
		t.Fatal(err)
	}
	prompt := tokens[:3]
	const maxNew = 4

	chain, chainStats, err := nlp.MedusaGenerate(model, heads, prompt, maxNew, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	tree, treeStats, err := nlp.MedusaGenerateTree(model, heads, prompt, maxNew, 0, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(tree) != len(chain) {
		t.Fatalf("lengths diverge: tree %d vs chain %d", len(tree), len(chain))
	}
	for i := range chain {
		if tree[i] != chain[i] {
			t.Fatalf("token %d diverges: tree %d vs chain %d", i, tree[i], chain[i])
		}
	}
	if treeStats != chainStats {
		t.Fatalf("stats diverge: tree %+v vs chain %+v", treeStats, chainStats)
	}
}

// §T509: topK=2 structural smoke — the branched tree respects the context window and
// the generation contract: output starts with the prompt, emits ≥1 token per round
// (the greedy anchor) up to maxNew, all tokens in-vocab, Accepted ≤ Proposed.
func TestMedusaTreeBranched(t *testing.T) {
	model, tokens := loadGPTModel(t)
	heads, err := nlp.NewMedusaHeads(2, model.Config.Dim, model.Config.Vocab, 7,
		nlp.WithMedusaHeadsDtype(model.TokEmb.Dtype()))
	if err != nil {
		t.Fatal(err)
	}
	prompt := tokens[:2]
	const maxNew = 4

	out, stats, err := nlp.MedusaGenerateTree(model, heads, prompt, maxNew, 0, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) <= len(prompt) || len(out) > len(prompt)+maxNew || len(out) > model.Config.Ctx {
		t.Fatalf("emitted %d tokens for prompt %d, maxNew %d, ctx %d",
			len(out)-len(prompt), len(prompt), maxNew, model.Config.Ctx)
	}
	for i, tok := range out {
		if i < len(prompt) && tok != prompt[i] {
			t.Fatalf("output does not start with the prompt at %d", i)
		}
		if tok < 0 || tok >= model.Config.Vocab {
			t.Fatalf("token %d out of vocab: %d", i, tok)
		}
	}
	if stats.Accepted > stats.Proposed {
		t.Fatalf("stats incoherent: %+v", stats)
	}
}
