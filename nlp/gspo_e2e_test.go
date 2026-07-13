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

// §T549 e2e: GSPO (Qwen3's sequence-level RL objective) trains a REAL GPT policy on the
// §T464 harness — same rollouts/reward as the GRPO flagship, but ONE loss call over the
// flat concatenation of the group's per-token log-probs with per-sequence lengths and
// per-sequence advantages (no token-level broadcast, no KL). Single-step on-policy means
// the sequence ratios start at exactly 1, safely inside even GSPO's tight ε. The mean
// reward must rise like GRPO's.
func TestGSPOTrainsGPTPolicy(t *testing.T) {
	policy, _ := loadGPTModel(t)
	prompt := []int{1, 2}
	const (
		G      = 8
		maxNew = 6
		iters  = 40
		tau    = 3
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
	logpsUnder := func(ctx *backend.Context, seq []int) (*tensor.Tensor, error) {
		logits, err := policy.Forward(ctx, seq)
		if err != nil {
			return nil, err
		}
		sub, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{logits},
			backend.SliceAttrs{Axis: 0, Start: len(prompt) - 1, End: len(seq) - 1})
		if err != nil {
			return nil, err
		}
		return nn.TokenLogProbs(ctx, sub[0], seq[len(prompt):])
	}

	opt := nn.NewAdamW(policy.Params(), 0.01, 0)
	var firstMean, lastMean float64
	for it := range iters {
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

		tape := autograd.NewTape()
		ctx := tape.Context()
		lengths := make([]int, G)
		lps := make([]*tensor.Tensor, G)
		total := 0
		for g := range G {
			lp, err := logpsUnder(ctx, gens[g])
			if err != nil {
				t.Fatal(err)
			}
			lps[g] = lp
			lengths[g] = lp.Shape()[0]
			total += lengths[g]
		}
		flat, err := backend.Execute(ctx, backend.OpConcat, lps, backend.ConcatAttrs{Axis: 0})
		if err != nil {
			t.Fatal(err)
		}
		lpOld := tensor.New(tensor.F64, tensor.Shape{total}) // rollout-time constants
		for i := range total {
			lpOld.SetF64(flat[0].AtF64(i), i)
		}
		advT := tensor.FromFloat64(tensor.Shape{G}, adv)
		loss, err := nn.GSPOLoss(ctx, flat[0], lpOld, advT, lengths)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(policy.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	if math.IsNaN(lastMean) || lastMean < firstMean+0.2 {
		t.Fatalf("GSPO did not improve the policy: mean reward %.3f → %.3f", firstMean, lastMean)
	}
	t.Logf("GSPO trained the GPT policy: mean reward %.3f → %.3f over %d iterations", firstMean, lastMean, iters)
}
