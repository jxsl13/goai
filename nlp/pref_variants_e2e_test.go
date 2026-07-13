package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T468: the remaining preference losses run end-to-end on the real GPT through the §T464
// bridge — each variant has a slightly different contract (IPO mirrors DPO; SimPO takes
// LENGTH-NORMALIZED policy log-probs and no reference; CPO takes no reference and adds a chosen
// NLL term; KTO takes UNPAIRED examples with desirability labels), so "a loss-call swap away"
// (§T465) is verified rather than assumed.

type prefHarness struct {
	t       *testing.T
	policy  *nlp.GPT
	prompts [][]int
	chosen  []int
	reject  []int
}

const prefRespLen = 4

func newPrefHarness(t *testing.T) *prefHarness {
	t.Helper()
	policy, _ := loadGPTModel(t)
	return &prefHarness{
		t:       t,
		policy:  policy,
		prompts: [][]int{{1, 2}, {4, 0}, {2, 5}},
		chosen:  []int{3, 3, 3, 3},
		reject:  []int{7, 7, 7, 7},
	}
}

// seqLogp is log P(resp|prompt) under m; differentiable when ctx carries a tape.
func (h *prefHarness) seqLogp(ctx *backend.Context, m *nlp.GPT, prompt, resp []int) *tensor.Tensor {
	h.t.Helper()
	full := append(append([]int(nil), prompt...), resp...)
	logits, err := m.Forward(ctx, full)
	if err != nil {
		h.t.Fatal(err)
	}
	sub, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{logits},
		backend.SliceAttrs{Axis: 0, Start: len(prompt) - 1, End: len(full) - 1})
	if err != nil {
		h.t.Fatal(err)
	}
	s, err := nn.SequenceLogProbs(ctx, sub[0], resp)
	if err != nil {
		h.t.Fatal(err)
	}
	return s
}

func (h *prefHarness) r1(ctx *backend.Context, x *tensor.Tensor) *tensor.Tensor {
	h.t.Helper()
	out, err := backend.Execute(ctx, backend.OpReshape, []*tensor.Tensor{x},
		backend.ReshapeAttrs{Shape: tensor.Shape{1}})
	if err != nil {
		h.t.Fatal(err)
	}
	return out[0]
}

// train runs steps optimizer steps of lossFor summed over the prompts.
func (h *prefHarness) train(steps int, lossFor func(ctx *backend.Context, prompt []int) *tensor.Tensor) {
	h.t.Helper()
	opt := nn.NewAdamW(h.policy.Params(), 5e-3, 0)
	for range steps {
		tape := autograd.NewTape()
		ctx := tape.Context()
		var total *tensor.Tensor
		for _, p := range h.prompts {
			loss := lossFor(ctx, p)
			if total == nil {
				total = loss
			} else {
				o, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{total, loss}, nil)
				if err != nil {
					h.t.Fatal(err)
				}
				total = o[0]
			}
		}
		if err := tape.Backward(total); err != nil {
			h.t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(h.policy.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			h.t.Fatal(err)
		}
	}
}

// margins returns (chosen−rejected) sequence-logp per prompt under the current policy.
func (h *prefHarness) margins() []float64 {
	h.t.Helper()
	out := make([]float64, len(h.prompts))
	for i, p := range h.prompts {
		w := h.seqLogp(backend.NewContext(), h.policy, p, h.chosen)
		l := h.seqLogp(backend.NewContext(), h.policy, p, h.reject)
		out[i] = w.AtF64() - l.AtF64()
	}
	return out
}

func assertMarginsImproved(t *testing.T, name string, before, after []float64) {
	t.Helper()
	for i := range before {
		if after[i] <= before[i] {
			t.Fatalf("%s pair %d: margin did not improve (%.4f → %.4f)", name, i, before[i], after[i])
		}
	}
	t.Logf("%s: all %d chosen−rejected margins improved (e.g. %.3f → %.3f)", name, len(before), before[0], after[0])
}

// IPO mirrors DPO's contract (with a frozen reference).
func TestIPOTrainsGPT(t *testing.T) {
	h := newPrefHarness(t)
	ref, _ := loadGPTModel(t)
	before := h.margins()
	h.train(25, func(ctx *backend.Context, p []int) *tensor.Tensor {
		pw := h.r1(ctx, h.seqLogp(ctx, h.policy, p, h.chosen))
		pl := h.r1(ctx, h.seqLogp(ctx, h.policy, p, h.reject))
		rw := h.r1(backend.NewContext(), h.seqLogp(backend.NewContext(), ref, p, h.chosen))
		rl := h.r1(backend.NewContext(), h.seqLogp(backend.NewContext(), ref, p, h.reject))
		loss, err := nn.IPO(ctx, pw, pl, rw, rl)
		if err != nil {
			t.Fatal(err)
		}
		return loss
	})
	assertMarginsImproved(t, "IPO", before, h.margins())
}

// SimPO takes LENGTH-NORMALIZED policy log-probs and no reference.
func TestSimPOTrainsGPT(t *testing.T) {
	h := newPrefHarness(t)
	inv := 1.0 / float64(prefRespLen)
	avg := func(ctx *backend.Context, x *tensor.Tensor) *tensor.Tensor {
		scale := tensor.FromFloat64(tensor.Shape{1}, []float64{inv})
		out, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{h.r1(ctx, x), scale}, nil)
		if err != nil {
			t.Fatal(err)
		}
		return out[0]
	}
	before := h.margins()
	h.train(25, func(ctx *backend.Context, p []int) *tensor.Tensor {
		pw := avg(ctx, h.seqLogp(ctx, h.policy, p, h.chosen))
		pl := avg(ctx, h.seqLogp(ctx, h.policy, p, h.reject))
		loss, err := nn.SimPO(ctx, pw, pl)
		if err != nil {
			t.Fatal(err)
		}
		return loss
	})
	assertMarginsImproved(t, "SimPO", before, h.margins())
}

// CPO takes no reference and adds an NLL term on the chosen response.
func TestCPOTrainsGPT(t *testing.T) {
	h := newPrefHarness(t)
	beforeM := h.margins()
	beforeW := h.seqLogp(backend.NewContext(), h.policy, h.prompts[0], h.chosen).AtF64()
	h.train(25, func(ctx *backend.Context, p []int) *tensor.Tensor {
		pw := h.r1(ctx, h.seqLogp(ctx, h.policy, p, h.chosen))
		pl := h.r1(ctx, h.seqLogp(ctx, h.policy, p, h.reject))
		loss, err := nn.CPO(ctx, pw, pl)
		if err != nil {
			t.Fatal(err)
		}
		return loss
	})
	assertMarginsImproved(t, "CPO", beforeM, h.margins())
	// the α·NLL term must also push the chosen response itself UP.
	if afterW := h.seqLogp(backend.NewContext(), h.policy, h.prompts[0], h.chosen).AtF64(); afterW <= beforeW {
		t.Fatalf("CPO: chosen log-prob did not increase (%.4f → %.4f)", beforeW, afterW)
	}
}

// KTO takes UNPAIRED examples with desirability labels and a frozen reference.
func TestKTOTrainsGPT(t *testing.T) {
	h := newPrefHarness(t)
	ref, _ := loadGPTModel(t)
	before := h.margins()
	h.train(25, func(ctx *backend.Context, p []int) *tensor.Tensor {
		// one desirable and one undesirable example per prompt, as an unpaired batch of 2.
		pw := h.seqLogp(ctx, h.policy, p, h.chosen)
		pl := h.seqLogp(ctx, h.policy, p, h.reject)
		cat, err := backend.Execute(ctx, backend.OpConcat,
			[]*tensor.Tensor{h.r1(ctx, pw), h.r1(ctx, pl)}, backend.ConcatAttrs{Axis: 0})
		if err != nil {
			t.Fatal(err)
		}
		rw := h.seqLogp(backend.NewContext(), ref, p, h.chosen)
		rl := h.seqLogp(backend.NewContext(), ref, p, h.reject)
		refT := tensor.FromFloat64(tensor.Shape{2}, []float64{rw.AtF64(), rl.AtF64()})
		labels := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 0})
		loss, err := nn.KTO(ctx, cat[0], refT, labels)
		if err != nil {
			t.Fatal(err)
		}
		return loss
	})
	assertMarginsImproved(t, "KTO", before, h.margins())
}
