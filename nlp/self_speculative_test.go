package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
)

// §T869/§V16: the final early exit runs the same blocks, final norm, and head as
// Forward. Exact equality makes p==q at the last exit a structural guarantee rather
// than a tolerance-based coincidence.
func TestForwardExitFinalIsForward(t *testing.T) {
	model, tokens := loadGPTModel(t)
	ctx := backend.NewContext()
	want, err := model.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	got, err := model.ForwardExit(ctx, tokens, len(model.Blocks)-1)
	if err != nil {
		t.Fatal(err)
	}
	tensorsEqual(t, got, want)

	early, err := model.ForwardExit(ctx, tokens, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !early.Shape().Equal(want.Shape()) {
		t.Fatalf("early-exit shape %v, want %v", early.Shape(), want.Shape())
	}
}

// The early draft may disagree completely with the mature model; modified
// rejection sampling must still reproduce mature greedy decoding token-for-token.
func TestSelfSpeculativeGreedyIsTarget(t *testing.T) {
	model, tokens := loadGPTModel(t)
	prompt := tokens[:2]
	maxNew := model.Config.Ctx - len(prompt)
	want, err := model.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	got, stats, err := nlp.SelfSpeculativeDecode(model, prompt, maxNew, 0, 3, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("length %d, target Generate %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]=%d, target=%d", i, got[i], want[i])
		}
	}
	if stats.Proposed == 0 {
		t.Fatal("early exit proposed no tokens")
	}
}

// At the last block ForwardExit == Forward, hence q_i == p_i and every proposal
// must be accepted under both greedy and stochastic sampling.
func TestSelfSpeculativeFinalExitAcceptsAll(t *testing.T) {
	model, tokens := loadGPTModel(t)
	prompt := tokens[:2]
	last := len(model.Blocks) - 1
	for _, s := range []*nlp.Sampler{
		nlp.Greedy(),
		nlp.NewSampler(42, nlp.WithTemperature(0.8), nlp.WithTopK(12)),
	} {
		out, stats, err := nlp.SelfSpeculativeDecode(model, prompt, 5, last, 2, s)
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != len(prompt)+5 {
			t.Fatalf("generated length %d, want %d", len(out), len(prompt)+5)
		}
		if stats.Proposed == 0 || stats.Accepted != stats.Proposed {
			t.Fatalf("final exit accepted %d/%d, want all", stats.Accepted, stats.Proposed)
		}
	}
}

// Integration-level Monte Carlo: drafting from a different intermediate
// distribution and correcting it must reproduce the mature next-token distribution.
func TestSelfSpeculativeSamplingMatchesTarget(t *testing.T) {
	model, tokens := loadGPTModel(t)
	prompt := tokens[:2]
	logits, err := model.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	row := make([]float64, model.Config.Vocab)
	for v := range row {
		row[v] = logits.AtF64(len(prompt)-1, v)
	}
	want := nlp.NewSampler(1, nlp.WithTemperature(0.9)).Dist(row)

	const trials = 3000
	counts := make([]int, model.Config.Vocab)
	for seed := range trials {
		s := nlp.NewSampler(uint64(seed+1), nlp.WithTemperature(0.9))
		out, _, err := nlp.SelfSpeculativeDecode(model, prompt, 1, 0, 1, s)
		if err != nil {
			t.Fatal(err)
		}
		counts[out[len(prompt)]]++
	}
	for tok, p := range want {
		emp := float64(counts[tok]) / trials
		// Six standard deviations plus a small finite-sample floor.
		tol := 6*math.Sqrt(p*(1-p)/trials) + 0.01
		if math.Abs(emp-p) > tol {
			t.Errorf("token %d empirical %.4f, target %.4f, tol %.4f", tok, emp, p, tol)
		}
	}
}

func TestSelfSpeculativeBoundaries(t *testing.T) {
	model, tokens := loadGPTModel(t)
	validPrompt := tokens[:2]
	if _, _, err := nlp.SelfSpeculativeDecode(nil, validPrompt, 1, 0, 1, nlp.Greedy()); err == nil {
		t.Fatal("nil model accepted")
	}
	if _, _, err := nlp.SelfSpeculativeDecode(model, validPrompt, 1, 0, 1, nil); err == nil {
		t.Fatal("nil sampler accepted")
	}
	if _, _, err := nlp.SelfSpeculativeDecode(model, nil, 1, 0, 1, nlp.Greedy()); err == nil {
		t.Fatal("empty prompt accepted")
	}
	if _, _, err := nlp.SelfSpeculativeDecode(model, validPrompt, 1, -1, 1, nlp.Greedy()); err == nil {
		t.Fatal("negative exit accepted")
	}
	if _, _, err := nlp.SelfSpeculativeDecode(model, validPrompt, 1, len(model.Blocks), 1, nlp.Greedy()); err == nil {
		t.Fatal("out-of-range exit accepted")
	}
	if _, _, err := nlp.SelfSpeculativeDecode(model, []int{model.Config.Vocab}, 1, 0, 1, nlp.Greedy()); err == nil {
		t.Fatal("invalid token accepted")
	}
	if _, err := model.ForwardExit(backend.NewContext(), validPrompt, -1); err == nil {
		t.Fatal("ForwardExit accepted negative layer")
	}
	if _, err := (*nlp.GPT)(nil).ForwardExit(backend.NewContext(), validPrompt, 0); err == nil {
		t.Fatal("ForwardExit accepted nil model")
	}

	zero, stats, err := nlp.SelfSpeculativeDecode(model, validPrompt, 0, 0, 0, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(zero) != len(validPrompt) || stats.Proposed != 0 {
		t.Fatalf("maxNew=0 returned len=%d stats=%+v", len(zero), stats)
	}

	// One context slot remains: the target-only fallback must fill it, not stop one
	// token early as a speculative window that insists on a bonus slot would.
	nearFull := make([]int, model.Config.Ctx-1)
	for i := range nearFull {
		nearFull[i] = tokens[i%len(tokens)]
	}
	out, stats, err := nlp.SelfSpeculativeDecode(model, nearFull, 3, 0, 0, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != model.Config.Ctx || stats.Proposed != 0 {
		t.Fatalf("final-slot fallback len=%d stats=%+v", len(out), stats)
	}
}
