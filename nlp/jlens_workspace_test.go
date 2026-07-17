package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// jlensPerm is the token map of the think-about-X task: a fixed derangement
// of the 8 content tokens (no fixed points, so X = π(s) never equals s).
var jlensPerm = [8]int{3, 6, 5, 1, 7, 0, 2, 4}

// jlensWorkspaceCorpus builds the think-about-X training set: lag-2
// permutation dynamics
//
//	s_t = π(s_{t-2})  for t ≥ 2,   s_0, s_1 seeded pseudo-random (s_1 ≠ s_0)
//
// so at EVERY position t ≥ 1 the model's next output is Y = π(s_{t-1}) —
// read off the neighbouring stream — while the token s_t it is currently
// holding will force X = π(s_t) two steps later. Every position therefore
// carries a "think X, say Y" state, which makes the carry transport the
// dominant shared structure of the averaged Jacobian — the miniature analogue
// of verbal content in the paper's web-text fits (§R250). The generator is a
// tiny deterministic LCG: identical corpus on every run.
func jlensWorkspaceCorpus(nSeq, seqLen int) [][]int {
	state := uint32(12345)
	next := func(n int) int {
		state = state*1664525 + 1013904223
		return int(state>>16) % n
	}
	seqs := make([][]int, nSeq)
	for i := range seqs {
		s := make([]int, seqLen)
		s[0] = next(8)
		for s[1] = next(8); s[1] == s[0]; {
			s[1] = next(8)
		}
		for t := 2; t < seqLen; t++ {
			s[t] = jlensPerm[s[t-2]]
		}
		seqs[i] = s
	}
	return seqs
}

// §T812 tier-2, the paper's think-about-X gate in miniature (§R250's core
// finding: "think about X, write Y" — the J-space shows X even though the
// output is Y). A tiny GPT char-LM is trained on the lag-2 permutation
// corpus above until the deterministic transitions are learned exactly. At a
// probe position t the model's actual next output is Y = π(s_{t-1}), but the
// residual must already carry X = π(s_t) for the emission two steps later.
// The gate: the J-lens transported readout at an intermediate lens layer
// ranks X far higher than the plain logit lens (no transport, W_U·norm(h_l))
// does on the SAME residual, and the final-layer readout agrees with the
// model's real output.
//
// The probed layer is 0, the post-embedding residual: in a 2-block model the
// think-X state is causally carried by h_0(t) — block-1 attention reads the
// PRE-block residuals of earlier positions, so by h_1(t) the payload
// information has already been copied downstream and the position-t stream no
// longer influences the future (probing layer 1 shows exactly that: near-
// chance ranks for both lenses). Layer 0 is a genuine lens layer — J_0
// transports through the whole stack and is far from identity — and the
// plain-lens baseline on the very same h_0 stays blind, which is the paper's
// contrast. Thresholds are aggregates over the probe set (miss count +
// mean-rank separation), not exact ranks — deterministic (seeded init, fixed
// corpus, f64 ref backend) but robust.
func TestJLensThinkAboutX(t *testing.T) {
	const vocab = 12 // 8 content tokens + 4 never-trained distractors
	corpus := jlensWorkspaceCorpus(16, 10)
	model := jlensGPT(t, vocab, 16, 16, 2, 2)

	// Overfit the char-LM (full-batch Adam, seeded init → deterministic).
	opt := nn.NewAdam(model.Params(), 1e-2)
	var last float64
	for step := 0; step < 500; step++ {
		tape := autograd.NewTape()
		ctx := tape.Context()
		var total *tensor.Tensor
		for _, toks := range corpus {
			logits, err := model.Forward(ctx, toks[:len(toks)-1])
			if err != nil {
				t.Fatal(err)
			}
			tgt := tensor.New(tensor.F64, tensor.Shape{len(toks) - 1})
			for i, tk := range toks[1:] {
				tgt.SetF64(float64(tk), i)
			}
			loss, err := nn.CrossEntropy(ctx, logits, tgt)
			if err != nil {
				t.Fatal(err)
			}
			if total == nil {
				total = loss
			} else if total, err = exec1t(ctx, backend.OpAdd, total, loss); err != nil {
				t.Fatal(err)
			}
		}
		if err := tape.Backward(total); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
		last = total.Contiguous().Storage().F64()[0]
	}
	// The raw loss keeps an irreducible floor (s_0, s_1 are unpredictable);
	// the overfit gate is the part that matters: every DETERMINISTIC
	// transition (t ≥ 1) must be predicted exactly.
	t.Logf("char-LM training loss after 500 steps (16 seqs, incl. irreducible s0/s1 entropy): %.4f", last)
	for li, toks := range corpus {
		logits, err := model.Forward(backend.NewContext(), toks)
		if err != nil {
			t.Fatal(err)
		}
		for p := 1; p < len(toks)-1; p++ {
			if got, want := argmaxAt(logits, p), toks[p+1]; got != want {
				t.Fatalf("seq %d pos %d: transition not learned (predicts %d, want %d) — the gate below would be meaningless", li, p, got, want)
			}
		}
	}

	// Fit the lens on the training sequences themselves.
	jl, err := nlp.FitJLens(model, corpus)
	if err != nil {
		t.Fatal(err)
	}

	const midLayer = 0 // the causal carrier layer (see the doc comment)
	var jWorse int
	var jSum, plainSum float64
	probes := 0
	for li, toks := range corpus {
		var resids []*tensor.Tensor
		if _, err := model.ForwardResiduals(backend.NewContext(), toks, func(l int, h *tensor.Tensor) {
			resids = append(resids, h)
		}); err != nil {
			t.Fatal(err)
		}
		logits, err := model.Forward(backend.NewContext(), toks)
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range []int{4, 6} {
			x := jlensPerm[toks[p]]   // X: forced two steps later by s_p
			y := jlensPerm[toks[p-1]] // Y: the actual next output
			if x == y {
				continue // π is a derangement but streams can collide; skip
			}
			probes++
			if got := argmaxAt(logits, p); got != y {
				t.Fatalf("seq %d pos %d: model says %d, want Y=%d", li, p, got, y)
			}

			jr, err := jl.ApplyAt(model, backend.NewContext(), toks, midLayer, p)
			if err != nil {
				t.Fatal(err)
			}
			jRank := jr.RankOf(x)

			row := tensor.New(tensor.F64, tensor.Shape{1, jl.Dim})
			for b := range jl.Dim {
				row.SetF64(resids[midLayer].AtF64(p, b), 0, b)
			}
			plain, err := model.Unembed(backend.NewContext(), row)
			if err != nil {
				t.Fatal(err)
			}
			plainRank := rankAt(plain, 0, x)
			t.Logf("seq %2d pos %d: s=%d X=%d Y=%d — J-lens rank %2d, plain logit-lens rank %2d (of %d)",
				li, p, toks[p], x, y, jRank, plainRank, vocab)
			if jRank > 2 {
				jWorse++
			}
			jSum += float64(jRank)
			plainSum += float64(plainRank)

			// Final layer must agree with the model's real output.
			fr, err := jl.ApplyAt(model, backend.NewContext(), toks, jl.Layers, p)
			if err != nil {
				t.Fatal(err)
			}
			if fr.Ranked[0].Token != y {
				t.Fatalf("seq %d pos %d: final-layer readout top %d disagrees with the model output %d", li, p, fr.Ranked[0].Token, y)
			}
		}
	}
	jMean, plainMean := jSum/float64(probes), plainSum/float64(probes)
	t.Logf("probes: %d; J-lens rank > 2 at %d; mean rank J-lens %.2f vs plain logit lens %.2f (vocab %d)",
		probes, jWorse, jMean, plainMean, vocab)
	if probes < 20 {
		t.Fatalf("only %d probes — corpus degenerate", probes)
	}
	// The aggregated gate: the transported readout puts X near the top at
	// (almost) every probe, and the plain logit lens on the SAME residual is
	// clearly blinder (a few probes may leak π through embedding geometry —
	// aggregates, not per-probe bands, keep the gate robust).
	if jWorse > probes/10 {
		t.Errorf("J-lens missed the think-about-X signal at %d/%d probes (rank > 2)", jWorse, probes)
	}
	if plainMean-jMean < 3 {
		t.Errorf("plain logit lens mean rank %.2f vs J-lens %.2f — transport adds < 3 ranks of information", plainMean, jMean)
	}
}

// exec1t is a test-local single-output op helper (nlp_test cannot see the
// package-internal exec1).
func exec1t(ctx *backend.Context, op backend.Op, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, nil)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// argmaxAt returns the argmax of row p of a [seq, vocab] tensor.
func argmaxAt(logits *tensor.Tensor, p int) int {
	best, bestV := 0, logits.AtF64(p, 0)
	for j := 1; j < logits.Shape()[1]; j++ {
		if v := logits.AtF64(p, j); v > bestV {
			best, bestV = j, v
		}
	}
	return best
}

// rankAt returns the 0-based descending rank of token in row p.
func rankAt(logits *tensor.Tensor, p, token int) int {
	v := logits.AtF64(p, token)
	rank := 0
	for j := 0; j < logits.Shape()[1]; j++ {
		if w := logits.AtF64(p, j); w > v || (w == v && j < token) {
			rank++
		}
	}
	return rank
}
