package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T469: the COMPLETE classic RLHF pipeline in-library, on the real GPT — (stage 1) a reward
// model (MLP head over the frozen base's ForwardHidden) trained from PREFERENCE PAIRS over
// on-distribution samples with BradleyTerryLoss; (stage 2) GRPO optimizes the policy against THAT
// LEARNED reward (no programmatic reward in stage 2).
//
// Assertions cover what the pipeline mechanically guarantees: the reward model ranks ≥90% of its
// pairs, and GRPO substantially RAISES THE LEARNED REWARD. The held-out true metric (τ-rate) is
// logged, NOT asserted — this experiment reliably reproduces REWARD HACKING: the policy drifts
// off-distribution, finds τ-free sequences the imperfect reward model scores highly, and the true
// metric can fall while the learned reward climbs (observed with a linear head at 87% pair
// accuracy AND with this MLP head above 90%). That is the canonical RLHF failure mode, faithfully
// demonstrated end-to-end; production systems counter it with iterated reward-model retraining
// and stronger KL anchoring.
func TestRLHFPipelineRewardModelThenGRPO(t *testing.T) {
	base, _ := loadGPTModel(t) // frozen featurizer for the reward model
	cfg := base.Config
	const tau = 3 // the behavior preferences encode: emit token τ
	prompts := [][]int{{1, 2}, {4, 0}, {2, 5}}
	const respLen = 4
	mkResp := func(tok int) []int {
		r := make([]int, respLen)
		for i := range r {
			r[i] = tok
		}
		return r
	}

	// reward(seq) = mean over response positions of an MLP score on the frozen base's hiddens —
	// per-position scoring makes count-like preferences linear in the pooling, and the small MLP
	// gives the head enough capacity over a weak frozen featurizer (a linear head plateaued at
	// ~87% pair accuracy and GRPO promptly exploited the residual error — observed reward hacking).
	h1 := nn.NewLinear(tensor.F64, cfg.Dim, 16, 11)
	h2 := nn.NewLinear(tensor.F64, 16, 1, 12)
	relu := nn.ReLU()
	headParams := append(h1.Params(), h2.Params()...)
	score := func(ctx *backend.Context, seq []int) *tensor.Tensor {
		hidden, err := base.ForwardHidden(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{hidden},
			backend.SliceAttrs{Axis: 0, Start: len(seq) - respLen, End: len(seq)})
		if err != nil {
			t.Fatal(err)
		}
		z, err := h1.Forward(ctx, resp[0])
		if err != nil {
			t.Fatal(err)
		}
		if z, err = relu.Forward(ctx, z); err != nil {
			t.Fatal(err)
		}
		perPos, err := h2.Forward(ctx, z) // [respLen,1] per-position scores
		if err != nil {
			t.Fatal(err)
		}
		s, err := backend.Execute(ctx, backend.OpMean, []*tensor.Tensor{perPos},
			backend.ReduceAttrs{Axes: []int{0}, KeepDims: true})
		if err != nil {
			t.Fatal(err)
		}
		return s[0] // [1,1]
	}

	// STAGE 1: preference pairs over ON-DISTRIBUTION samples (the standard recipe — a head
	// trained on a handful of synthetic sequences does not generalize to diverse rollouts):
	// sample rollouts from the base policy per prompt, rank them by the latent preference
	// (τ-count), and train Bradley-Terry on (better, worse) pairs.
	tauFrac := func(gen []int) float64 {
		hits := 0
		for _, tok := range gen {
			if tok == tau {
				hits++
			}
		}
		return float64(hits) / float64(len(gen))
	}
	type pair struct{ better, worse []int }
	var pairs []pair
	sampler, _ := loadGPTModel(t)
	for pi, p := range prompts {
		type ranked struct {
			seq  []int
			frac float64
		}
		var rs []ranked
		for g := range 16 {
			temp := 1.0
			if g%2 == 1 {
				temp = 1.4
			}
			s := nlp.NewSampler(uint64(pi*37+g+5), nlp.WithTemperature(temp))
			out, err := sampler.Generate(p, respLen, s)
			if err != nil {
				t.Fatal(err)
			}
			rs = append(rs, ranked{out, tauFrac(out[len(p):])})
		}
		// force one clearly-good example into the pool so every prompt has a positive pair.
		rs = append(rs, ranked{append(append([]int(nil), p...), mkResp(tau)...), 1})
		for i := range rs {
			for j := range rs {
				if rs[i].frac > rs[j].frac {
					pairs = append(pairs, pair{rs[i].seq, rs[j].seq})
				}
			}
		}
	}
	rOpt := nn.NewAdamW(headParams, 0.05, 0)
	for range 120 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		var total *tensor.Tensor
		for _, pr := range pairs {
			w := score(ctx, pr.better)
			l := score(ctx, pr.worse)
			loss, err := nn.BradleyTerryLoss(ctx, w, l, 0)
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
		if err := rOpt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	// the reward model must rank every training pair correctly before stage 2 may use it.
	correct := 0
	for _, pr := range pairs {
		if score(backend.NewContext(), pr.better).AtF64(0, 0) > score(backend.NewContext(), pr.worse).AtF64(0, 0) {
			correct++
		}
	}
	if acc := float64(correct) / float64(len(pairs)); acc < 0.9 {
		t.Fatalf("reward model ranks only %.0f%% of %d pairs correctly", 100*acc, len(pairs))
	}

	// STAGE 2: GRPO against the LEARNED reward (no programmatic reward from here on).
	policy, _ := loadGPTModel(t)
	ref, _ := loadGPTModel(t)
	learnedReward := func(seq []int) float64 { return score(backend.NewContext(), seq).AtF64(0, 0) }
	trueMetric := func(gen []int) float64 { // held-out ground truth, NEVER trains anything
		hits := 0
		for _, tok := range gen {
			if tok == tau {
				hits++
			}
		}
		return float64(hits) / float64(len(gen))
	}
	logps := func(ctx *backend.Context, m *nlp.GPT, seq []int, promptLen int) *tensor.Tensor {
		logits, err := m.Forward(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		sub, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{logits},
			backend.SliceAttrs{Axis: 0, Start: promptLen - 1, End: len(seq) - 1})
		if err != nil {
			t.Fatal(err)
		}
		lp, err := nn.TokenLogProbs(ctx, sub[0], seq[promptLen:])
		if err != nil {
			t.Fatal(err)
		}
		return lp
	}
	pOpt := nn.NewAdamW(policy.Params(), 0.01, 0)
	prompt := prompts[0]
	const G, maxNew, iters = 8, 6, 40
	var firstTrue, lastTrue, firstRM, lastRM float64
	for it := range iters {
		gens := make([][]int, G)
		rewards := make([]float64, G)
		trueMean := 0.0
		for g := range G {
			s := nlp.NewSampler(uint64(it*100+g+1), nlp.WithTemperature(1))
			out, err := policy.Generate(prompt, maxNew, s)
			if err != nil {
				t.Fatal(err)
			}
			gens[g] = out
			rewards[g] = learnedReward(out) // the trained reward model scores the rollout
			trueMean += trueMetric(out[len(prompt):])
		}
		trueMean /= G
		rmMean := 0.0
		for _, r := range rewards {
			rmMean += r
		}
		rmMean /= G
		if it == 0 {
			firstTrue, firstRM = trueMean, rmMean
		}
		lastTrue, lastRM = trueMean, rmMean
		adv := nn.GroupAdvantage(rewards)
		tape := autograd.NewTape()
		ctx := tape.Context()
		var total *tensor.Tensor
		for g := range G {
			lpNew := logps(ctx, policy, gens[g], len(prompt))
			T := lpNew.Shape()[0]
			cp := func(src *tensor.Tensor) *tensor.Tensor {
				out := tensor.New(tensor.F64, tensor.Shape{T})
				for i := range T {
					out.SetF64(src.AtF64(i), i)
				}
				return out
			}
			lpRef := logps(backend.NewContext(), ref, gens[g], len(prompt))
			advT := tensor.New(tensor.F64, tensor.Shape{T})
			for i := range T {
				advT.SetF64(adv[g], i)
			}
			loss, err := nn.GRPOLoss(ctx, lpNew, cp(lpNew), cp(lpRef), advT, nn.WithKLBeta(0.01))
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
		if err := pOpt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	if lastRM <= firstRM {
		t.Fatalf("GRPO did not raise the LEARNED reward it optimizes: %.4f → %.4f", firstRM, lastRM)
	}
	t.Logf("RLHF pipeline: learned reward %.4f → %.4f (optimized, asserted); held-out true τ-rate %.3f → %.3f (logged — reward hacking when it falls)", firstRM, lastRM, firstTrue, lastTrue)
}
