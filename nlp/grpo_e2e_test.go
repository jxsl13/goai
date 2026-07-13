package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T464 flagship: GRPO (the DeepSeek-R1 objective) trains a REAL GPT policy end-to-end with the
// library's own pieces — Generate samples the rollouts, nn.TokenLogProbs bridges logits to
// per-token log-probs (the previously-missing link the preference/RL losses document), GroupAdvantage
// normalizes a programmatic reward, GRPOLoss backpropagates into the policy. The reward (fraction
// of generated tokens equal to a target id) must rise substantially.
func TestGRPOTrainsGPTPolicy(t *testing.T) {
	policy, _ := loadGPTModel(t)
	ref, _ := loadGPTModel(t) // frozen reference for the KL term
	cfg := policy.Config
	prompt := []int{1, 2}
	const (
		G      = 8 // rollouts per group
		maxNew = 6
		iters  = 40
		tau    = 3 // the rewarded token id
	)
	reward := func(gen []int) float64 {
		hits := 0
		for _, tok := range gen {
			if tok == tau {
				hits++
			}
		}
		return float64(hits) / float64(len(gen))
	}
	rows := func(seq []int) (starts, targets []int) {
		// logits row i scores token i+1: generated tokens live at rows len(prompt)-1 … len(seq)-2.
		return []int{len(prompt) - 1, len(seq) - 1}, seq[len(prompt):]
	}
	logpsUnder := func(ctx *backend.Context, m *nlp.GPT, seq []int) (*tensor.Tensor, error) {
		logits, err := m.Forward(ctx, seq)
		if err != nil {
			return nil, err
		}
		span, targets := rows(seq)
		sub, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{logits},
			backend.SliceAttrs{Axis: 0, Start: span[0], End: span[1]})
		if err != nil {
			return nil, err
		}
		return nn.TokenLogProbs(ctx, sub[0], targets)
	}

	opt := nn.NewAdamW(policy.Params(), 0.01, 0)
	var firstMean, lastMean float64
	for it := range iters {
		// 1. sample a group of rollouts from the current policy.
		gens := make([][]int, G)
		rewards := make([]float64, G)
		for g := range G {
			s := nlp.NewSampler(uint64(it*100+g+1), nlp.WithTemperature(1))
			out, err := policy.Generate(prompt, maxNew, s)
			if err != nil {
				t.Fatal(err)
			}
			gens[g] = out
			rewards[g] = reward(out[len(prompt):])
		}
		adv := nn.GroupAdvantage(rewards)
		mean := 0.0
		for _, r := range rewards {
			mean += r
		}
		mean /= G
		if it == 0 {
			firstMean = mean
		}
		lastMean = mean

		// 2. one tape over the whole group: per-rollout GRPO losses, averaged.
		tape := autograd.NewTape()
		ctx := tape.Context()
		var total *tensor.Tensor
		for g := range G {
			lpNew, err := logpsUnder(ctx, policy, gens[g])
			if err != nil {
				t.Fatal(err)
			}
			T := lpNew.Shape()[0]
			cp := func(src *tensor.Tensor) *tensor.Tensor { // rollout-time constants
				out := tensor.New(tensor.F64, tensor.Shape{T})
				for i := range T {
					out.SetF64(src.AtF64(i), i)
				}
				return out
			}
			lpOld := cp(lpNew) // single-step GRPO: old policy = sampling policy
			lpRef, err := logpsUnder(backend.NewContext(), ref, gens[g])
			if err != nil {
				t.Fatal(err)
			}
			advT := tensor.New(tensor.F64, tensor.Shape{T})
			for i := range T {
				advT.SetF64(adv[g], i)
			}
			loss, err := nn.GRPOLoss(ctx, lpNew, lpOld, cp(lpRef), advT, nn.WithKLBeta(0.001))
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
	if math.IsNaN(lastMean) || lastMean < firstMean+0.2 {
		t.Fatalf("GRPO did not improve the policy: mean reward %.3f → %.3f (vocab %d)", firstMean, lastMean, cfg.Vocab)
	}
	t.Logf("GRPO trained the GPT policy: mean reward %.3f → %.3f over %d iterations", firstMean, lastMean, iters)
}
