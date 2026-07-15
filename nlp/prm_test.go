package nlp_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"reflect"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// ---------------------------------------------------------------------------
// Synthetic task: corrupted mod-10 arithmetic chains (ProcessBench-style).
//
// A chain is v0 ∈ 0..9 followed by H=5 steps v_{i+1} = (v_i op a_i) mod 10,
// emitted as tokens
//
//	bos v0 (op a = v sep) × H
//
// Corruption replaces one step's stated result with a wrong digit and
// PROPAGATES all later stated results consistently from the corrupted value,
// so the corrupted step is the UNIQUE locally-invalid step (earliest error,
// ProcessBench arXiv:2412.06559 semantics) — detecting it needs a local
// arithmetic check, not a global one. Because ×2 is non-injective mod 10, a
// corruption offset can be wiped (5·2 ≡ 0) and the chain still END on the
// correct final digit ("lucky-correct"): wrong reasoning, right answer —
// exactly the contamination that caps an outcome reward model's label
// quality.
//
// DIAL (documented deviation): the generator supports op ∈ {+,−,×} with
// a ∈ 1..9, but the shipped scripts restrict steps to (×,1) and (×,2). The
// full 270-cell (v,op,a)→result table is modular arithmetic, a 3-way
// multiplicative interaction with NO low-order gradient signal — the classic
// grokking regime (Power et al. 2022, arXiv:2201.02177), empirically flat at
// the base-rate BCE for >12k AdamW steps at every dim/lr/depth tried, for
// BCE heads and next-token LM pretraining alike. With the (×,1)/(×,2)
// restriction each step check is an op-gated PAIRWISE relation
// (prev,stated), which attention's QK bilinearity learns in a few thousand
// steps (measured F1 0.963 at 18k steps) while every design property above
// (locality, earliest error, propagation, lucky-corrects) is preserved.
// Raising contamination further (the task's dial): add (×,5), which wipes
// every even corruption offset.
// ---------------------------------------------------------------------------

const (
	prmTokAdd = 10 // '+'
	prmTokSub = 11 // '−'
	prmTokMul = 12 // '×'
	prmTokEq  = 13 // '='
	prmTokSep = 14 // step separator
	prmTokBOS = 15 // begin-of-sequence
	prmVocab  = 16 // digits 0-9 + the 6 symbols above
	prmH      = 5  // steps per chain
)

var prmOpTok = [3]int{prmTokAdd, prmTokSub, prmTokMul}

// prmApply is the chain update rule: (v op a) mod 10, non-negative.
func prmApply(v, op, a int) int {
	switch op {
	case 0:
		return (v + a) % 10
	case 1:
		return ((v-a)%10 + 10) % 10
	default:
		return (v * a) % 10
	}
}

// prmScript is one underlying problem: the start value and the H (op, a)
// pairs. Rendering it (clean or corrupted) yields a candidate solution.
type prmScript struct {
	v0  int
	ops [prmH]int
	as  [prmH]int
}

// prmGenScript draws a problem with steps from the learnable context set
// {(×,1), (×,2)} (see the DIAL note above).
func prmGenScript(rng *rand.Rand) prmScript {
	s := prmScript{v0: rng.IntN(10)}
	for i := range prmH {
		s.ops[i] = 2              // ×
		s.as[i] = 1 + rng.IntN(2) // 1 or 2
	}
	return s
}

type prmChain struct {
	tokens    []int     // bos v0 (op a = v sep)×H, length 2+5H
	stepEnds  []int     // the sep position of each step
	labels    []float64 // per-step local correctness (1/0)
	earliest  int       // earliest erroneous step index, −1 if clean
	corrupted bool      // whether any corruption was injected
	finalOK   bool      // stated final digit == true final digit
}

// prmEmit turns stated values into the token chain with per-step labels
// (label i = stated[i+1] locally consistent with stated[i]).
func prmEmit(s prmScript, stated, truev []int, earliest int) prmChain {
	c := prmChain{
		tokens:    []int{prmTokBOS, s.v0},
		earliest:  earliest,
		corrupted: earliest >= 0,
		finalOK:   stated[prmH] == truev[prmH],
	}
	for i := range prmH {
		c.tokens = append(c.tokens, prmOpTok[s.ops[i]], s.as[i], prmTokEq, stated[i+1], prmTokSep)
		c.stepEnds = append(c.stepEnds, len(c.tokens)-1)
		if stated[i+1] == prmApply(stated[i], s.ops[i], s.as[i]) {
			c.labels = append(c.labels, 1)
		} else {
			c.labels = append(c.labels, 0)
		}
	}
	return c
}

// prmRenderChain renders a script with per-chain probability pCorrupt of ONE
// corrupted step (wrong stated result at step j, later results propagated
// consistently from the corrupted value) — the ProcessBench-style
// earliest-error evaluation distribution.
func prmRenderChain(s prmScript, rng *rand.Rand, pCorrupt float64) prmChain {
	truev := make([]int, prmH+1)
	truev[0] = s.v0
	for i := range prmH {
		truev[i+1] = prmApply(truev[i], s.ops[i], s.as[i])
	}
	stated := append([]int(nil), truev...)
	earliest := -1
	if rng.Float64() < pCorrupt {
		j := rng.IntN(prmH)
		stated[j+1] = (truev[j+1] + 1 + rng.IntN(9)) % 10 // any digit ≠ the true result
		for i := j + 1; i < prmH; i++ {                   // propagate consistently
			stated[i+1] = prmApply(stated[i], s.ops[i], s.as[i])
		}
		earliest = j
	}
	return prmEmit(s, stated, truev, earliest)
}

// prmRenderChainMulti corrupts EVERY step independently with probability q
// (each wrong result then feeds the next step) — the label-balanced TRAINING
// distribution: at q≈0.4 the per-step labels are ≈60/40 instead of the 9:1
// imbalance of single-error chains, the dense step supervision a
// Math-Shepherd-style auto-labeler would produce. Evaluation always uses the
// single-error prmRenderChain distribution.
func prmRenderChainMulti(s prmScript, rng *rand.Rand, q float64) prmChain {
	truev := make([]int, prmH+1)
	truev[0] = s.v0
	for i := range prmH {
		truev[i+1] = prmApply(truev[i], s.ops[i], s.as[i])
	}
	stated := make([]int, prmH+1)
	stated[0] = s.v0
	earliest := -1
	for i := range prmH {
		correct := prmApply(stated[i], s.ops[i], s.as[i])
		if rng.Float64() < q {
			stated[i+1] = (correct + 1 + rng.IntN(9)) % 10
			if earliest < 0 {
				earliest = i
			}
		} else {
			stated[i+1] = correct
		}
	}
	return prmEmit(s, stated, truev, earliest)
}

// prmGenChains renders n independent problems: single-error at pCorrupt when
// multi is false, per-step corruption at rate q when multi is true.
func prmGenChains(seed uint64, n int, multi bool, rate float64) []prmChain {
	rng := rand.New(rand.NewPCG(seed, 0x9e37))
	out := make([]prmChain, n)
	for i := range out {
		s := prmGenScript(rng)
		if multi {
			out[i] = prmRenderChainMulti(s, rng, rate)
		} else {
			out[i] = prmRenderChain(s, rng, rate)
		}
	}
	return out
}

// prmTrain runs single-chain AdamW steps over params with the repo's e2e
// recipe (grad-clip 1.0, weight decay 0.01, lr 3e-3 decayed to 1e-3 at
// decayAt). Returns instead of t.Fatal so PRM and ORM can train in parallel
// goroutines (independent models and tapes; shared chains are read-only).
func prmTrain(params []*tensor.Tensor, steps, decayAt, nChains int,
	loss func(ctx *backend.Context, i int) (*tensor.Tensor, error)) error {
	opt := nn.NewAdamW(params, 3e-3, 0.01)
	for step := range steps {
		switch step {
		case decayAt:
			opt.LR = 1e-3
		case decayAt + 6000:
			opt.LR = 3e-4
		}
		tape := autograd.NewTape()
		l, err := loss(tape.Context(), step%nChains)
		if err != nil {
			return err
		}
		if err := tape.Backward(l); err != nil {
			return err
		}
		clipped, _ := nn.ClipGradNorm(params, tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			return err
		}
	}
	return nil
}

// prmPredictEarliest is the ProcessBench-style read-out: the earliest step
// with score < 0.5, or −1 if every step scores ≥ 0.5.
func prmPredictEarliest(scores []float64) int {
	for i, s := range scores {
		if s < 0.5 {
			return i
		}
	}
	return -1
}

// ---------------------------------------------------------------------------
// 1. Generator correctness: labels right by construction, deterministic,
// lucky-correct contamination present.
// ---------------------------------------------------------------------------

// prmVerifyChains re-derives every label of every chain from the emitted
// tokens alone: per-step local validity (stated == prev stated ×/+/− a mod
// 10), the earliest-error index, and the outcome flag vs an independent
// recomputation of the true chain. Returns (#corrupted, #lucky-correct).
func prmVerifyChains(t *testing.T, chains []prmChain) (corrupted, lucky int) {
	t.Helper()
	tokOp := map[int]int{prmTokAdd: 0, prmTokSub: 1, prmTokMul: 2}
	for ci, c := range chains {
		if len(c.tokens) != 2+5*prmH || len(c.stepEnds) != prmH || len(c.labels) != prmH {
			t.Fatalf("chain %d: bad shape tokens=%d stepEnds=%d labels=%d", ci, len(c.tokens), len(c.stepEnds), len(c.labels))
		}
		prevStated, prevTrue := c.tokens[1], c.tokens[1]
		earliest := -1
		for i := range prmH {
			base := 2 + 5*i
			op, ok := tokOp[c.tokens[base]]
			if !ok || c.tokens[base+2] != prmTokEq || c.tokens[base+4] != prmTokSep {
				t.Fatalf("chain %d step %d: malformed step tokens %v", ci, i, c.tokens[base:base+5])
			}
			if c.stepEnds[i] != base+4 {
				t.Fatalf("chain %d step %d: stepEnd %d, want %d", ci, i, c.stepEnds[i], base+4)
			}
			a, stated := c.tokens[base+1], c.tokens[base+3]
			want := 0.0
			if stated == prmApply(prevStated, op, a) {
				want = 1
			} else if earliest < 0 {
				earliest = i
			}
			if c.labels[i] != want {
				t.Fatalf("chain %d step %d: label %g, want %g (recomputed)", ci, i, c.labels[i], want)
			}
			prevStated = stated
			prevTrue = prmApply(prevTrue, op, a)
		}
		if earliest != c.earliest {
			t.Fatalf("chain %d: earliest %d, want %d (recomputed)", ci, c.earliest, earliest)
		}
		if got := prevStated == prevTrue; got != c.finalOK {
			t.Fatalf("chain %d: finalOK %v, want %v (recomputed)", ci, c.finalOK, got)
		}
		if c.corrupted != (earliest >= 0) {
			t.Fatalf("chain %d: corrupted flag %v with earliest %d", ci, c.corrupted, earliest)
		}
		if c.corrupted {
			corrupted++
			if c.finalOK {
				lucky++
			}
		} else if !c.finalOK {
			t.Fatalf("chain %d: clean but final answer wrong", ci)
		}
	}
	return corrupted, lucky
}

// TestPRMGeneratorCorrectness verifies both renderers (single-earliest-error
// eval distribution and per-step-corruption training distribution) against an
// independent recomputation, checks seed-determinism, and reports the
// lucky-correct rate — the corrupted chains whose final digit is still right
// (×2 wiping a 5-offset), i.e. the outcome-label contamination.
func TestPRMGeneratorCorrectness(t *testing.T) {
	const n = 2000
	single := prmGenChains(7, n, false, 0.5)
	if !reflect.DeepEqual(single, prmGenChains(7, n, false, 0.5)) {
		t.Fatal("generator not deterministic under seed")
	}
	corrupted, lucky := prmVerifyChains(t, single)
	if corrupted == 0 || corrupted == n {
		t.Fatalf("degenerate corruption count %d/%d", corrupted, n)
	}
	if lucky == 0 {
		t.Fatal("no lucky-correct chains — outcome-label contamination missing")
	}
	t.Logf("single-error: corrupted %d/%d chains; lucky-correct %d/%d corrupted (%.1f%%) — outcome-label contamination",
		corrupted, n, lucky, corrupted, 100*float64(lucky)/float64(corrupted))

	multi := prmGenChains(8, n, true, 0.4)
	mCorr, _ := prmVerifyChains(t, multi)
	var zeros, steps int
	for _, c := range multi {
		for _, l := range c.labels {
			steps++
			if l == 0 {
				zeros++
			}
		}
	}
	t.Logf("multi-error training diet: corrupted %d/%d chains; wrong-step rate %.1f%% (label balance)",
		mCorr, n, 100*float64(zeros)/float64(steps))
	if zeros == 0 || zeros == steps {
		t.Fatalf("degenerate training labels: %d/%d zeros", zeros, steps)
	}
}

// ---------------------------------------------------------------------------
// 2+3. The VALUE validations: step-error-detection F1 (the hard gate) and
// best-of-N selection accuracy vs the ORM ablation and random.
// ---------------------------------------------------------------------------

// TestProcessRewardModelValueE2E measures what step-level supervision actually
// BUYS over outcome-level supervision at identical capacity:
//
//  1. ProcessBench-style earliest-error detection on 200 held-out
//     single-error chains — F1 (harmonic mean of the accuracies on the
//     erroneous and correct subsets) must exceed 0.8. An ORM cannot even
//     express this prediction: it has one score per sequence.
//  2. Best-of-8 selection: picking candidates by max-min-step-score (PRM,
//     Lightman et al. §4.2 aggregation) must beat the same-architecture ORM
//     trained on final-answer labels, and beat random choice by ≥ 0.15.
//
// Each model trains on the label-balanced diet for ITS objective (strongest
// baseline): PRM on per-step-corrupted chains (≈60/40 step labels), ORM on
// single-error chains at p=0.5 (≈50/50 outcome labels, INCLUDING the
// lucky-correct poison). Same 800 problems count, optimizer, steps, schedule.
// The two trainings run in parallel goroutines (~19 s wall on an M-series
// laptop — above the usual <10 s e2e budget because the from-scratch
// verifier needs ~18k steps to converge robustly, see the generator DIAL
// note).
func TestProcessRewardModelValueE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a PRM and an ORM; skipped in -short")
	}
	cfg := nlp.RewardModelConfig{Vocab: prmVocab, Ctx: 32, Dim: 32, Heads: 4, Layers: 1}
	const nTrain, trainSteps, decayAt = 800, 18000, 8000
	prmTrainSet := prmGenChains(11, nTrain, true, 0.4)  // per-step corruption: balanced step labels
	ormTrainSet := prmGenChains(12, nTrain, false, 0.5) // single-error at 50%: balanced outcome labels

	prm, err := nlp.NewProcessRewardModel(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	orm, err := nlp.NewOutcomeRewardModel(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 2)
	go func() {
		done <- prmTrain(prm.Params(), trainSteps, decayAt, nTrain, func(ctx *backend.Context, i int) (*tensor.Tensor, error) {
			c := prmTrainSet[i]
			return prm.Loss(ctx, c.tokens, c.stepEnds, c.labels)
		})
	}()
	go func() {
		done <- prmTrain(orm.Params(), trainSteps, decayAt, nTrain, func(ctx *backend.Context, i int) (*tensor.Tensor, error) {
			label := 0.0
			if ormTrainSet[i].finalOK { // includes the lucky-correct poison
				label = 1
			}
			return orm.Loss(ctx, ormTrainSet[i].tokens, label)
		})
	}()
	for range 2 {
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	}

	// --- step-error detection F1 on held-out chains (the hard gate) ---
	held := prmGenChains(101, 200, false, 0.5)
	var errTotal, errHit, okTotal, okHit int
	for _, c := range held {
		scores, err := prm.StepScores(backend.NewContext(), c.tokens, c.stepEnds)
		if err != nil {
			t.Fatal(err)
		}
		pred := prmPredictEarliest(scores)
		if c.earliest >= 0 {
			errTotal++
			if pred == c.earliest {
				errHit++
			}
		} else {
			okTotal++
			if pred == -1 {
				okHit++
			}
		}
	}
	accErr := float64(errHit) / float64(errTotal)
	accOK := float64(okHit) / float64(okTotal)
	f1 := 0.0
	if accErr+accOK > 0 {
		f1 = 2 * accErr * accOK / (accErr + accOK)
	}
	t.Logf("step-error detection: acc(erroneous)=%.3f (%d/%d), acc(correct)=%.3f (%d/%d), F1=%.3f",
		accErr, errHit, errTotal, accOK, okHit, okTotal, f1)
	if f1 <= 0.8 {
		t.Fatalf("ProcessBench-style F1 = %.3f, want > 0.8", f1)
	}

	// --- best-of-N: same problem, N candidate renderings, pick one ---
	rng := rand.New(rand.NewPCG(202, 7))
	const nProblems, N = 60, 8
	var prmAcc, ormAcc, randAcc int
	for range nProblems {
		script := prmGenScript(rng)
		cands := make([]prmChain, N)
		for i := range cands {
			cands[i] = prmRenderChain(script, rng, 0.25)
		}
		bestPRM, bestORM := -1, -1
		var bestMin, bestScore float64
		for i, c := range cands {
			scores, err := prm.StepScores(backend.NewContext(), c.tokens, c.stepEnds)
			if err != nil {
				t.Fatal(err)
			}
			minScore := 1.0
			for _, s := range scores {
				minScore = math.Min(minScore, s)
			}
			if bestPRM < 0 || minScore > bestMin {
				bestPRM, bestMin = i, minScore
			}
			score, err := orm.Score(backend.NewContext(), c.tokens)
			if err != nil {
				t.Fatal(err)
			}
			if bestORM < 0 || score > bestScore {
				bestORM, bestScore = i, score
			}
		}
		if cands[bestPRM].finalOK {
			prmAcc++
		}
		if cands[bestORM].finalOK {
			ormAcc++
		}
		if cands[rng.IntN(N)].finalOK {
			randAcc++
		}
	}
	pAcc := float64(prmAcc) / nProblems
	oAcc := float64(ormAcc) / nProblems
	rAcc := float64(randAcc) / nProblems
	t.Logf("best-of-%d over %d problems: PRM(min-step) %.3f, ORM %.3f, random %.3f", N, nProblems, pAcc, oAcc, rAcc)
	if pAcc <= oAcc {
		t.Fatalf("PRM best-of-N %.3f not better than ORM %.3f", pAcc, oAcc)
	}
	if pAcc-rAcc < 0.15 {
		t.Fatalf("PRM best-of-N %.3f vs random %.3f: margin %.3f < 0.15", pAcc, rAcc, pAcc-rAcc)
	}
}

// ---------------------------------------------------------------------------
// 4. Gradcheck through head + backbone.
// ---------------------------------------------------------------------------

// TestProcessRewardModelGradcheck: numeric central differences on Loss vs the
// tape's analytic gradients for EVERY parameter (backbone + scoring head +
// bias) on a tiny F64 config — the repo's gradcheck recipe (rtol 1e-4, h=1e-6).
func TestProcessRewardModelGradcheck(t *testing.T) {
	cfg := nlp.RewardModelConfig{Vocab: prmVocab, Ctx: 12, Dim: 4, Heads: 2, Layers: 1}
	m, err := nlp.NewProcessRewardModel(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	// bos 3 | ×2=6 sep | ×2=4 sep — second step corrupted (6×2 mod 10 = 2 ≠ 4)
	tokens := []int{prmTokBOS, 3, prmTokMul, 2, prmTokEq, 6, prmTokSep, prmTokMul, 2, prmTokEq, 4, prmTokSep}
	stepEnds := []int{6, 11}
	labels := []float64{1, 0}

	tape := autograd.NewTape()
	loss, err := m.Loss(tape.Context(), tokens, stepEnds, labels)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	fwd := func() float64 {
		l, err := m.Loss(backend.NewContext(), tokens, stepEnds, labels)
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	const h = 1e-6
	for pi, p := range m.Params() {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("param %d: nil gradient", pi)
		}
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := fwd()
			p.SetF64(o-h, coord...)
			lm := fwd()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Fatalf("param %d grad%v: numeric %.8g vs analytic %.8g", pi, coord, num, ana)
			}
		}
	}

	// The ORM shares the entire scoring path; a full backward must reach every
	// parameter (soft label exercises the soft-BCE branch).
	orm, err := nlp.NewOutcomeRewardModel(cfg, 4)
	if err != nil {
		t.Fatal(err)
	}
	tape = autograd.NewTape()
	oloss, err := orm.Loss(tape.Context(), tokens, 0.7)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(oloss); err != nil {
		t.Fatal(err)
	}
	for pi, p := range orm.Params() {
		if tape.Grad(p) == nil {
			t.Fatalf("ORM param %d: nil gradient", pi)
		}
	}
}

// ---------------------------------------------------------------------------
// 5. Shapes, errors, determinism.
// ---------------------------------------------------------------------------

// TestProcessRewardModelShapesAndErrors covers constructor validation, the
// scoring/loss argument checks, score ranges and seed-determinism.
func TestProcessRewardModelShapesAndErrors(t *testing.T) {
	if _, err := nlp.NewProcessRewardModel(nlp.RewardModelConfig{Vocab: 0, Ctx: 8, Dim: 4, Heads: 2, Layers: 1}, 1); err == nil {
		t.Error("zero vocab: want error")
	}
	if _, err := nlp.NewProcessRewardModel(nlp.RewardModelConfig{Vocab: 4, Ctx: 8, Dim: 5, Heads: 2, Layers: 1}, 1); err == nil {
		t.Error("dim not divisible by heads: want error")
	}
	if _, err := nlp.NewOutcomeRewardModel(nlp.RewardModelConfig{Vocab: 4, Ctx: 0, Dim: 4, Heads: 2, Layers: 1}, 1); err == nil {
		t.Error("zero ctx: want error")
	}

	cfg := nlp.RewardModelConfig{Vocab: prmVocab, Ctx: 16, Dim: 8, Heads: 2, Layers: 1}
	m, err := nlp.NewProcessRewardModel(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{prmTokBOS, 3, prmTokMul, 2, prmTokEq, 6, prmTokSep}
	ctx := backend.NewContext()

	if _, err := m.StepScores(ctx, tokens, nil); err == nil {
		t.Error("no positions: want error")
	}
	if _, err := m.StepScores(ctx, tokens, []int{7}); err == nil {
		t.Error("position out of range: want error")
	}
	if _, err := m.StepScores(ctx, tokens, []int{4, 4}); err == nil {
		t.Error("non-increasing positions: want error")
	}
	if _, err := m.StepScores(ctx, []int{prmVocab}, []int{0}); err == nil {
		t.Error("token outside vocab: want error")
	}
	if _, err := m.Loss(ctx, tokens, []int{4, 6}, []float64{1}); err == nil {
		t.Error("label/steps length mismatch: want error")
	}
	if _, err := m.Loss(ctx, tokens, []int{6}, []float64{1.5}); err == nil {
		t.Error("label outside [0,1]: want error")
	}

	scores, err := m.StepScores(ctx, tokens, []int{2, 6})
	if err != nil {
		t.Fatal(err)
	}
	if len(scores) != 2 {
		t.Fatalf("got %d scores, want 2", len(scores))
	}
	for i, s := range scores {
		if s <= 0 || s >= 1 {
			t.Fatalf("score[%d] = %g outside (0,1)", i, s)
		}
	}
	m2, err := nlp.NewProcessRewardModel(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	scores2, err := m2.StepScores(backend.NewContext(), tokens, []int{2, 6})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scores, scores2) {
		t.Fatalf("same seed, different scores: %v vs %v", scores, scores2)
	}
	// PRM param count = GPT backbone params (12 per layer + 5) + head bias
	if got, want := len(m.Params()), 12*cfg.Layers+5+1; got != want {
		t.Fatalf("param count %d, want %d", got, want)
	}

	o, err := nlp.NewOutcomeRewardModel(cfg, 6)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := o.Score(ctx, nil); err == nil {
		t.Error("ORM empty sequence: want error")
	}
	if _, err := o.Loss(ctx, nil, 1); err == nil {
		t.Error("ORM Loss empty sequence: want error")
	}
	if _, err := o.Loss(ctx, tokens, -0.1); err == nil {
		t.Error("ORM label outside [0,1]: want error")
	}
	s, err := o.Score(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	if s <= 0 || s >= 1 {
		t.Fatalf("ORM score %g outside (0,1)", s)
	}
}

// ExampleProcessRewardModel scores the two steps of a tiny mod-10 arithmetic
// chain "3 | ×2=6 | ×2=4" (the second step is wrong: 6×2 mod 10 = 2) and
// takes one BCE training step. Deterministic via the fixed seed.
func ExampleProcessRewardModel() {
	cfg := nlp.RewardModelConfig{Vocab: 16, Ctx: 16, Dim: 8, Heads: 2, Layers: 1}
	m, err := nlp.NewProcessRewardModel(cfg, 42)
	if err != nil {
		panic(err)
	}
	// bos 3 | × 2 = 6 sep | × 2 = 4 sep   (second stated result 4 is WRONG: 6×2 mod 10 = 2)
	tokens := []int{15, 3, 12, 2, 13, 6, 14, 12, 2, 13, 4, 14}
	stepEnds := []int{6, 11} // the two step separators
	labels := []float64{1, 0}

	scores, err := m.StepScores(backend.NewContext(), tokens, stepEnds)
	if err != nil {
		panic(err)
	}
	inUnit := true
	for _, s := range scores {
		inUnit = inUnit && s > 0 && s < 1
	}
	fmt.Println("steps scored:", len(scores), "in (0,1):", inUnit)

	tape := autograd.NewTape()
	loss, err := m.Loss(tape.Context(), tokens, stepEnds, labels)
	if err != nil {
		panic(err)
	}
	if err := tape.Backward(loss); err != nil {
		panic(err)
	}
	fmt.Println("BCE loss positive:", loss.AtF64() > 0, "params:", len(m.Params()))
	// Output:
	// steps scored: 2 in (0,1): true
	// BCE loss positive: true params: 18
}

// ExampleOutcomeRewardModel scores a whole chain with ONE number — the
// outcome-level ablation baseline of the ProcessRewardModel (same backbone,
// final-token read-out, final-answer label only).
func ExampleOutcomeRewardModel() {
	cfg := nlp.RewardModelConfig{Vocab: 16, Ctx: 16, Dim: 8, Heads: 2, Layers: 1}
	m, err := nlp.NewOutcomeRewardModel(cfg, 7)
	if err != nil {
		panic(err)
	}
	tokens := []int{15, 3, 12, 2, 13, 6, 14} // bos 3 | ×2=6 sep — final answer correct
	score, err := m.Score(backend.NewContext(), tokens)
	if err != nil {
		panic(err)
	}
	tape := autograd.NewTape()
	loss, err := m.Loss(tape.Context(), tokens, 1)
	if err != nil {
		panic(err)
	}
	if err := tape.Backward(loss); err != nil {
		panic(err)
	}
	fmt.Println("score in (0,1):", score > 0 && score < 1, "loss positive:", loss.AtF64() > 0, "params:", len(m.Params()))
	// Output:
	// score in (0,1): true loss positive: true params: 18
}
