package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// treeVerifyEmbed builds the embedded input for tree verification: row i is
// TokEmb[tok[i]] + PosEmb[pos[i]] — the same sum GPT.Embed produces, but with
// tree positions (prefix 0..P−1, node t at P+depth[t]) instead of 0..seq−1.
func treeVerifyEmbed(g *nlp.GPT, toks, pos []int) *tensor.Tensor {
	dim := g.TokEmb.Shape()[1]
	x := tensor.New(g.TokEmb.Dtype(), tensor.Shape{len(toks), dim})
	for i, tok := range toks {
		for d := range dim {
			x.SetF64(g.TokEmb.AtF64(tok, d)+g.PosEmb.AtF64(pos[i], d), i, d)
		}
	}
	return x
}

// treeVerifyMask assembles the full [P+N, P+N] additive mask: prefix rows are
// causal; tree rows see the whole prefix plus their MedusaTreeMask block.
func treeVerifyMask(prefixLen int, treeMask *tensor.Tensor) *tensor.Tensor {
	n := treeMask.Shape()[0]
	seq := prefixLen + n
	neg := math.Inf(-1)
	m := tensor.New(tensor.F64, tensor.Shape{seq, seq})
	for i := range seq {
		for j := range seq {
			switch {
			case i < prefixLen: // prefix row: ordinary causal
				if j > i {
					m.SetF64(neg, i, j)
				}
			case j < prefixLen: // tree row sees the entire prefix
			default: // tree row × tree column: the tree structure
				m.SetF64(treeMask.AtF64(i-prefixLen, j-prefixLen), i, j)
			}
		}
	}
	return m
}

// §T508: single-pass tree verification must equal sequential per-path verification.
// One ForwardFromEmbed over [prefix + candidate tree] — MHA.Mask carrying the causal-
// prefix + MedusaTreeMask structure, positions = P+depth — must reproduce, for EVERY
// tree node, the logits a plain causal Forward over [prefix + root-to-node path]
// produces at its last position. Bit-identical on the reference backend (the masked
// row's unmasked key set is exactly the sequential row's key set, §T507 collapse
// argument extended through the full model); ≤1e−9 on the default backend, where the
// sequential path runs cpu's fused OpMHA while OpMHAMasked reaches ref via the §T461
// fallback chain.
func TestTreeVerifyMatchesSequential(t *testing.T) {
	model, tokens := loadGPTModel(t)
	prefix := tokens[:4]
	// tree: two roots' worth of branching under one root — 0 → {1, 2}, 1 → 3.
	parents := []int{-1, 0, 0, 1}
	vocab := model.TokEmb.Shape()[0]
	treeToks := []int{tokens[4] % vocab, 2 % vocab, 5 % vocab, 1 % vocab}
	treeMask, depth := nlp.MedusaTreeMask(parents)

	toks := append(append([]int{}, prefix...), treeToks...)
	pos := make([]int, len(toks))
	for i := range prefix {
		pos[i] = i
	}
	for i, d := range depth {
		pos[len(prefix)+i] = len(prefix) + d
	}
	fullMask := treeVerifyMask(len(prefix), treeMask)

	pathToks := func(node int) []int { // prefix + root-to-node chain
		var chain []int
		for a := node; a != -1; a = parents[a] {
			chain = append([]int{treeToks[a]}, chain...)
		}
		return append(append([]int{}, prefix...), chain...)
	}

	run := func(ctx *backend.Context, name string, tol float64) {
		for _, b := range model.Blocks {
			b.Attn.Mask = fullMask
		}
		x := treeVerifyEmbed(model, toks, pos)
		tree, err := model.ForwardFromEmbed(ctx, x)
		for _, b := range model.Blocks {
			b.Attn.Mask = nil
		}
		if err != nil {
			t.Fatalf("%s: tree forward: %v", name, err)
		}
		for node := range parents {
			seqLogits, err := model.Forward(ctx, pathToks(node))
			if err != nil {
				t.Fatalf("%s: sequential forward node %d: %v", name, node, err)
			}
			last := seqLogits.Shape()[0] - 1
			row := len(prefix) + node
			for v := 0; v < tree.Shape()[1]; v++ {
				got, want := tree.AtF64(row, v), seqLogits.AtF64(last, v)
				if diff := math.Abs(got - want); diff > tol {
					t.Fatalf("%s: node %d vocab %d: tree %.17g vs sequential %.17g (Δ %.3g)",
						name, node, v, got, want, diff)
				}
			}
		}
	}

	run(backend.NewContext().WithBackend(backend.Reference()), "ref bit-identity", 0)
	run(backend.NewContext(), "default backend via §T461 fallback", 1e-9)
}
