package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T465: DPO trains a REAL GPT end-to-end through the §T464 bridge — the RL-free half of the
// alignment story (§T464 proved GRPO). Fixed preference pairs over three prompts (chosen =
// responses made of one token id, rejected = of another); after training, (1) the DPO implicit-
// reward margin β·((πθ_w−πref_w)−(πθ_l−πref_l)) is positive for EVERY pair — 100% preference
// accuracy — and (2) the policy raised the chosen responses' log-probs relative to the frozen
// reference while lowering the rejected ones.
func TestDPOTrainsGPT(t *testing.T) {
	policy, _ := loadGPTModel(t)
	ref, _ := loadGPTModel(t)
	prompts := [][]int{{1, 2}, {4, 0}, {2, 5}}
	const respLen = 4
	const chosenTok, rejectedTok = 3, 7
	mkResp := func(tok int) []int {
		r := make([]int, respLen)
		for i := range r {
			r[i] = tok
		}
		return r
	}

	// seqLogp: log P(resp | prompt) under m, differentiable when ctx carries a tape.
	seqLogp := func(ctx *backend.Context, m *nlp.GPT, prompt, resp []int) (*tensor.Tensor, error) {
		full := append(append([]int(nil), prompt...), resp...)
		logits, err := m.Forward(ctx, full)
		if err != nil {
			return nil, err
		}
		sub, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{logits},
			backend.SliceAttrs{Axis: 0, Start: len(prompt) - 1, End: len(full) - 1})
		if err != nil {
			return nil, err
		}
		return nn.SequenceLogProbs(ctx, sub[0], resp)
	}
	// frozen reference log-probs, computed once.
	refW := make([]float64, len(prompts))
	refL := make([]float64, len(prompts))
	for i, p := range prompts {
		w, err := seqLogp(backend.NewContext(), ref, p, mkResp(chosenTok))
		if err != nil {
			t.Fatal(err)
		}
		l, err := seqLogp(backend.NewContext(), ref, p, mkResp(rejectedTok))
		if err != nil {
			t.Fatal(err)
		}
		refW[i], refL[i] = w.AtF64(), l.AtF64()
	}
	vec := func(vals []float64) *tensor.Tensor {
		return tensor.FromFloat64(tensor.Shape{len(vals)}, vals)
	}

	opt := nn.NewAdamW(policy.Params(), 5e-3, 0)
	for range 30 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		// batch the pairs: concat per-pair scalars into [batch] via host copy of tape values
		// is NOT differentiable — instead sum the per-pair DPO losses (mean over batch=1 each).
		var total *tensor.Tensor
		for i, p := range prompts {
			pw, err := seqLogp(ctx, policy, p, mkResp(chosenTok))
			if err != nil {
				t.Fatal(err)
			}
			pl, err := seqLogp(ctx, policy, p, mkResp(rejectedTok))
			if err != nil {
				t.Fatal(err)
			}
			// DPO wants [batch] tensors; reshape the scalars to [1].
			r1 := func(x *tensor.Tensor) (*tensor.Tensor, error) {
				out, err := backend.Execute(ctx, backend.OpReshape, []*tensor.Tensor{x},
					backend.ReshapeAttrs{Shape: tensor.Shape{1}})
				if err != nil {
					return nil, err
				}
				return out[0], nil
			}
			pw1, err := r1(pw)
			if err != nil {
				t.Fatal(err)
			}
			pl1, err := r1(pl)
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.DPO(ctx, pw1, pl1, vec(refW[i:i+1]), vec(refL[i:i+1]))
			if err != nil {
				t.Fatal(err)
			}
			if total == nil {
				total = loss
			} else {
				o, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{total, loss}, nil)
				if err != nil {
					t.Fatal(err)
				}
				total = o[0]
			}
		}
		if err := tape.Backward(total); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(policy.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}

	// evaluation: implicit-reward margin per pair must be positive (100% preference accuracy),
	// with chosen pushed UP and rejected pushed DOWN relative to the reference.
	for i, p := range prompts {
		w, err := seqLogp(backend.NewContext(), policy, p, mkResp(chosenTok))
		if err != nil {
			t.Fatal(err)
		}
		l, err := seqLogp(backend.NewContext(), policy, p, mkResp(rejectedTok))
		if err != nil {
			t.Fatal(err)
		}
		margin := (w.AtF64() - refW[i]) - (l.AtF64() - refL[i])
		if margin <= 0 {
			t.Fatalf("pair %d: implicit-reward margin %.4f not positive (chosen Δ %.4f, rejected Δ %.4f)",
				i, margin, w.AtF64()-refW[i], l.AtF64()-refL[i])
		}
		if w.AtF64() <= refW[i] {
			t.Fatalf("pair %d: chosen log-prob did not increase (%.4f → %.4f)", i, refW[i], w.AtF64())
		}
		if l.AtF64() >= refL[i] {
			t.Fatalf("pair %d: rejected log-prob did not decrease (%.4f → %.4f)", i, refL[i], l.AtF64())
		}
	}
	t.Logf("DPO: 3/3 pairs preferred (positive margins), chosen ↑ / rejected ↓ vs the frozen reference")
}
