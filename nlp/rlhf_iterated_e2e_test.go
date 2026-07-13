package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T470: ITERATED RLHF rescues the true objective that §T469's static pipeline lost to reward
// hacking. Identical setup — MLP reward head over the frozen base, Bradley-Terry preferences,
// GRPO against the LEARNED reward — plus the standard countermeasure: every few policy updates
// the reward model is RETRAINED on freshly labeled samples from the CURRENT policy (the labeler
// oracle plays the human), so hacked sequences enter the preference data and lose their inflated
// scores. Assertion: the held-out true τ-rate — the exact metric that FELL 0.042→0.000 in §T469 —
// must now rise substantially. Reward hacking and its canonical mitigation, both demonstrated.
func TestIteratedRLHFDefeatsRewardHacking(t *testing.T) {
	if testing.Short() {
		t.Skip("~45s of training; skipped in -short")
	}
	base, _ := loadGPTModel(t)
	cfg := base.Config
	const tau = 3
	prompt := []int{1, 2}
	const respLen = 6

	// reward model: per-position MLP scores over the frozen base's hiddens, mean-pooled.
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
		perPos, err := h2.Forward(ctx, z)
		if err != nil {
			t.Fatal(err)
		}
		s, err := backend.Execute(ctx, backend.OpMean, []*tensor.Tensor{perPos},
			backend.ReduceAttrs{Axes: []int{0}, KeepDims: true})
		if err != nil {
			t.Fatal(err)
		}
		return s[0]
	}
	tauFrac := func(gen []int) float64 {
		hits := 0
		for _, tok := range gen {
			if tok == tau {
				hits++
			}
		}
		return float64(hits) / float64(len(gen))
	}

	// preference store: the oracle "labeler" ranks any two sequences by τ-rate.
	type pair struct{ better, worse []int }
	var pairs []pair
	addLabeled := func(seqs [][]int) {
		for i := range seqs {
			for j := range seqs {
				if tauFrac(seqs[i][len(prompt):]) > tauFrac(seqs[j][len(prompt):]) {
					pairs = append(pairs, pair{seqs[i], seqs[j]})
				}
			}
		}
	}
	trainRM := func(steps int) {
		opt := nn.NewAdamW(headParams, 0.05, 0)
		for range steps {
			tape := autograd.NewTape()
			ctx := tape.Context()
			var total *tensor.Tensor
			for _, pr := range pairs {
				loss, err := nn.BradleyTerryLoss(ctx, score(ctx, pr.better), score(ctx, pr.worse), 0)
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
			if err := opt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
		}
	}

	policy, _ := loadGPTModel(t)
	ref, _ := loadGPTModel(t)
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

	// bootstrap: label initial policy samples (+ one anchor) and train the first reward model.
	var boot [][]int
	for g := range 12 {
		s := nlp.NewSampler(uint64(g+5), nlp.WithTemperature(1.2))
		out, err := policy.Generate(prompt, respLen, s)
		if err != nil {
			t.Fatal(err)
		}
		boot = append(boot, out)
	}
	anchor := append(append([]int(nil), prompt...), []int{tau, tau, tau, tau, tau, tau}...)
	boot = append(boot, anchor)
	addLabeled(boot)
	trainRM(80)

	pOpt := nn.NewAdamW(policy.Params(), 0.01, 0)
	const G, iters, refreshEvery = 8, 70, 5
	var firstTrue, lastTrue float64
	for it := range iters {
		gens := make([][]int, G)
		rewards := make([]float64, G)
		trueMean := 0.0
		for g := range G {
			s := nlp.NewSampler(uint64(it*100+g+1), nlp.WithTemperature(1))
			out, err := policy.Generate(prompt, respLen, s)
			if err != nil {
				t.Fatal(err)
			}
			gens[g] = out
			rewards[g] = score(backend.NewContext(), out).AtF64(0, 0)
			trueMean += tauFrac(out[len(prompt):])
		}
		trueMean /= G
		if it == 0 {
			firstTrue = trueMean
		}
		lastTrue = trueMean

		// the mitigation: freshly label the CURRENT policy's samples and retrain the reward
		// model, so newly hacked sequences lose their inflated scores.
		if it > 0 && it%refreshEvery == 0 {
			addLabeled(append(append([][]int(nil), gens...), anchor))
			trainRM(50)
		}

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
	if lastTrue < firstTrue+0.2 {
		t.Fatalf("iterated RLHF did not rescue the true objective: τ-rate %.3f → %.3f", firstTrue, lastTrue)
	}
	t.Logf("iterated RLHF beats reward hacking: true τ-rate %.3f → %.3f (static pipeline: → 0.000, §T469)", firstTrue, lastTrue)
}
