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

// H-hop pointer chasing — the synthetic task that MEASURES Coconut's value as
// latent computation DEPTH. A random function f over chaseNodes node symbols
// (every node has exactly one successor) is given in the prompt as its edge
// list in fixed source order, then a start node and an '=' query token; the
// answer is f^H(start), ONE token. Computing it needs H SEQUENTIAL lookups: a
// depth-2 transformer can do at most ~2 per forward pass (≤4 hops even with
// pointer doubling, arXiv:2505.12514's two-layer construction), so H=7 is
// STRUCTURALLY unsolvable in a single forward (K=0) but solvable with K=6
// continuous thoughts — sequential depth (K+1)·layers. The ground-truth chain
// for the curriculum is the hop sequence x1 > x2 > … > f^H(start), one token
// per step (c=1): the H−1=6 intermediate hops are the chain steps the
// curriculum replaces with thoughts, the final hop is the answer segment (see
// chaseSteps for why the answer must not double as the last chain step).
// Chance accuracy = 1/chaseNodes ≈ 0.06.
const (
	chaseNodes = 16         // node symbols, token ids 0..15
	chaseEQ    = chaseNodes // the '=' query token
	chaseVocab = chaseNodes + 1
	chaseHops  = 7 // H: hops per query — structurally out of reach for 2 layers at K=0
)

// chasePuzzle is one pointer-chasing instance: a random successor function,
// a start node, and the ground-truth hop chain ending in the answer.
type chasePuzzle struct {
	succ  []int // succ[i] = f(i): the edge list in fixed source order 0..15
	start int   // the query's start node
	chain []int // x_i = f^i(start), i=1..H; chain[H-1] is the answer
}

// chaseKey fingerprints the (graph, start) pair for train/held-out disjointness.
func chaseKey(succ []int, start int) string { return fmt.Sprint(succ, start) }

// chasePrompt renders the puzzle as tokens: the 16 successor tokens (edge list
// in fixed source order), the start node, and '='. Length chaseNodes+2 = 18.
func chasePrompt(p chasePuzzle) []int {
	return append(append(append([]int{}, p.succ...), p.start), chaseEQ)
}

// chaseSteps is the curriculum's ground-truth chain-of-thought: one
// single-token reasoning step per INTERMEDIATE hop, x1..x_{H−1} (c=1). The
// final hop x_H is the answer segment (chaseAnswer), not a chain step, so
// that at EVERY curriculum stage the first supervised token after the thought
// block is one hop from the block's last input — text token or thought alike.
// (If x_H also appeared as the last chain step, the fully-latent stage would
// need the contradictory rule "repeat the node this thought encodes" on the
// same thought-input row type that every other stage trains to HOP — that
// format conflict empirically collapses the last stage.)
func chaseSteps(p chasePuzzle) [][]int {
	steps := make([][]int, len(p.chain)-1)
	for i, x := range p.chain[:len(p.chain)-1] {
		steps[i] = []int{x}
	}
	return steps
}

// chaseAnswer is the puzzle's answer segment: the final hop, one token.
func chaseAnswer(p chasePuzzle) []int { return []int{p.chain[len(p.chain)-1]} }

// chasePuzzles deterministically samples n puzzles with hops-long chains,
// skipping any (graph, start) pair whose key is in exclude or already sampled
// — so a held-out set built with exclude=trainKeys is provably disjoint.
// Returns the puzzles and the set of their keys.
func chasePuzzles(seed uint64, n, hops int, exclude map[string]bool) ([]chasePuzzle, map[string]bool) {
	rng := rand.New(rand.NewPCG(seed, 0xc0c0))
	keys := map[string]bool{}
	var out []chasePuzzle
	for len(out) < n {
		succ := make([]int, chaseNodes)
		for i := range succ {
			succ[i] = rng.IntN(chaseNodes)
		}
		start := rng.IntN(chaseNodes)
		key := chaseKey(succ, start)
		if exclude[key] || keys[key] {
			continue
		}
		keys[key] = true
		chain := make([]int, hops)
		cur := start
		for i := range chain {
			cur = succ[cur]
			chain[i] = cur
		}
		out = append(out, chasePuzzle{succ: succ, start: start, chain: chain})
	}
	return out, keys
}

// Generator correctness: chains really are f^i(start), sampling is
// deterministic, and held-out puzzles are disjoint from training ones.
func TestCoconutChaseGenerator(t *testing.T) {
	a, _ := chasePuzzles(7, 32, chaseHops, nil)
	b, _ := chasePuzzles(7, 32, chaseHops, nil)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("generator not deterministic for identical seeds")
	}
	for pi, p := range a {
		cur := p.start
		for i := 0; i < chaseHops; i++ {
			cur = p.succ[cur] // independent recomputation of f^(i+1)(start)
			if p.chain[i] != cur {
				t.Fatalf("puzzle %d: chain[%d] = %d, want f^%d(start) = %d", pi, i, p.chain[i], i+1, cur)
			}
		}
		if got := chaseAnswer(p)[0]; got != cur {
			t.Fatalf("puzzle %d: answer %d, want f^%d(start) = %d", pi, got, chaseHops, cur)
		}
		if prompt := chasePrompt(p); len(prompt) != chaseNodes+2 || prompt[len(prompt)-1] != chaseEQ {
			t.Fatalf("puzzle %d: malformed prompt %v", pi, prompt)
		}
	}
	train, trainKeys := chasePuzzles(1, 64, chaseHops, nil)
	held, heldKeys := chasePuzzles(2, 32, chaseHops, trainKeys)
	for k := range heldKeys {
		if trainKeys[k] {
			t.Fatalf("held-out puzzle %q also in the training set", k)
		}
	}
	_ = train
	_ = held
}

// chaseTrain runs the Coconut curriculum: stages 0..maxStage, stepsPerStage
// examples each, with the optimizer state RESET at every stage switch (a fresh
// AdamW per stage — the caller-side reset CurriculumLoss documents, per the
// paper). noCoT trains the K=0 ablation instead: question → answer directly
// (no chain text, no thoughts) on the SAME total step/optimizer budget.
func chaseTrain(t *testing.T, m *nlp.Coconut, puzzles []chasePuzzle, maxStage, stepsPerStage int, noCoT bool) {
	t.Helper()
	rng := rand.New(rand.NewPCG(42, 0xd1ce))
	step := 0
	for stage := 0; stage <= maxStage; stage++ {
		opt := nn.NewAdamW(m.Params(), 3e-3, 0.01)
		var sum float64
		var cnt, n int
		for range stepsPerStage {
			p := puzzles[step%len(puzzles)]
			step++
			// replay: half the steps train the current stage, half a uniform
			// earlier stage, so the prefix computation the thoughts depend on
			// stays anchored while the new depth trains (stabilizes the
			// curriculum against forgetting).
			s := stage
			if !noCoT && stage > 0 && rng.Float64() < 0.5 {
				s = rng.IntN(stage)
			}
			tape := autograd.NewTape()
			var loss *tensor.Tensor
			var err error
			if noCoT {
				loss, err = m.CurriculumLoss(tape.Context(), chasePrompt(p), nil, chaseAnswer(p), 0)
			} else {
				loss, err = m.CurriculumLoss(tape.Context(), chasePrompt(p), chaseSteps(p), chaseAnswer(p), s)
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			if s == stage && stepsPerStage-cnt <= 200 {
				sum += loss.AtF64()
				n++
			}
			cnt++
			clipped, _ := nn.ClipGradNorm(m.Params(), tape.Grad, 1.0)
			if err := opt.Step(clipped); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("stage %d (noCoT=%v): mean current-stage loss over last 200 steps %.4f", stage, noCoT, sum/float64(n))
	}
}

// chaseAccuracy is greedy-decode exact-match accuracy with k thoughts.
func chaseAccuracy(t *testing.T, m *nlp.Coconut, puzzles []chasePuzzle, k int) float64 {
	t.Helper()
	correct := 0
	for _, p := range puzzles {
		out, err := m.Generate(backend.NewContext(), chasePrompt(p), k, 1)
		if err != nil {
			t.Fatal(err)
		}
		if out[0] == chaseAnswer(p)[0] {
			correct++
		}
	}
	return float64(correct) / float64(len(puzzles))
}

// THE VALUE ASSERTION — the measurable structural depth gap, plus the
// pause-token control. Three models with IDENTICAL architecture, parameter
// count (up to the one [1,dim] pause row), optimizer and step budget on the
// H=7 pointer-chasing task that a 2-layer transformer cannot solve in one
// forward pass:
//
//	A) Coconut curriculum → K=6 continuous thoughts at inference,
//	B) K=0 no-CoT ablation (question → answer directly),
//	P) pause-token control: K=6 learned CONSTANT rows, no hidden feedback.
//
// K = H−1 = 6: one thought per intermediate hop; the answer prediction from
// the last thought performs the 7th hop itself (see chaseSteps).
//
// Asserted: acc(A) − acc(B) ≥ 0.4 (the gap is STRUCTURAL — K=0 is depth-
// bounded below the 7 sequential lookups the task needs, so if A's training
// converges the gap must show) and acc(P) < acc(A) − 0.2 (the paper's own
// pause baseline: extra positions without recurrent feedback must NOT close
// the gap, isolating hidden-state feedback as the source of the value).
func TestCoconutLatentDepthValue(t *testing.T) {
	if testing.Short() {
		t.Skip("trains three Coconut models through the curriculum; skipped in -short")
	}
	const (
		nTrain        = 512
		nHeld         = 64
		stepsPerStage = 3000
		maxStage      = chaseHops - 1 // = K = 6 continuous thoughts
	)
	train, trainKeys := chasePuzzles(1, nTrain, chaseHops, nil)
	held, _ := chasePuzzles(2, nHeld, chaseHops, trainKeys)
	cfg := nlp.CoconutConfig{Vocab: chaseVocab, Ctx: 32, Dim: 32, Heads: 4, Layers: 2, Eps: 1e-5}

	coco, err := nlp.NewCoconut(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	chaseTrain(t, coco, train, maxStage, stepsPerStage, false)
	accCoco := chaseAccuracy(t, coco, held, maxStage)

	noCoT, err := nlp.NewCoconut(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	chaseTrain(t, noCoT, train, maxStage, stepsPerStage, true)
	accNoCoT := chaseAccuracy(t, noCoT, held, 0)

	pause, err := nlp.NewCoconut(cfg, 1, nlp.WithCoconutPause())
	if err != nil {
		t.Fatal(err)
	}
	chaseTrain(t, pause, train, maxStage, stepsPerStage, false)
	accPause := chaseAccuracy(t, pause, held, maxStage)

	t.Logf("held-out exact match (chance %.3f): coconut K=%d %.3f | no-CoT K=0 %.3f | pause K=%d %.3f",
		1.0/chaseNodes, maxStage, accCoco, accNoCoT, maxStage, accPause)
	if gap := accCoco - accNoCoT; gap < 0.4 {
		t.Fatalf("latent depth gap %.3f < 0.4 (coconut %.3f, no-CoT %.3f) — continuous thoughts added no measurable value",
			gap, accCoco, accNoCoT)
	}
	if accPause >= accCoco-0.2 {
		t.Fatalf("pause control %.3f not below coconut %.3f − 0.2 — the gain is not attributable to hidden-state feedback",
			accPause, accCoco)
	}
}

// Collapse: with no markers and K=0 thoughts, ForwardWithThoughts is the plain
// GPT forward on the prompt — same weights, same ops, logits equal to 1e-10.
func TestCoconutK0CollapsesToGPT(t *testing.T) {
	cfg := nlp.CoconutConfig{Vocab: 11, Ctx: 8, Dim: 8, Heads: 2, Layers: 2, Eps: 1e-5}
	m, err := nlp.NewCoconut(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{3, 1, 4, 1, 5}
	a, err := m.ForwardWithThoughts(backend.NewContext(), prompt, 0)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.GPT.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	if !a.Shape().Equal(b.Shape()) {
		t.Fatalf("shape %v vs GPT %v", a.Shape(), b.Shape())
	}
	for i := range len(prompt) {
		for v := range cfg.Vocab {
			if d := math.Abs(a.AtF64(i, v) - b.AtF64(i, v)); d > 1e-10 {
				t.Fatalf("logits[%d,%d] differ by %g (> 1e-10)", i, v, d)
			}
		}
	}
}

// Numeric gradcheck THROUGH the latent-thought feedback loop: central
// differences on CurriculumLoss vs the tape's analytic gradients for EVERY
// parameter (the repo's gradcheck recipe: F64, rtol 1e-4 at h=1e-6), at stage 1
// (one thought + remaining chain text) and stage 2 (two chained feedback
// steps, loss only past the thought block) — confirming BPTT through the
// thoughts, including into the <bot>/<eot> markers.
func TestCoconutGradcheck(t *testing.T) {
	cfg := nlp.CoconutConfig{Vocab: 5, Ctx: 12, Dim: 4, Heads: 2, Layers: 1, Eps: 1e-5}
	m, err := nlp.NewCoconut(cfg, 11, nlp.WithCoconutMarkers())
	if err != nil {
		t.Fatal(err)
	}
	question := []int{0, 1}
	chain := [][]int{{2}, {3}}
	answer := []int{4}
	for _, stage := range []int{1, 2} {
		loss := func(ctx *backend.Context) *tensor.Tensor {
			l, err := m.CurriculumLoss(ctx, question, chain, answer, stage)
			if err != nil {
				t.Fatal(err)
			}
			return l
		}
		tape := autograd.NewTape()
		if err := tape.Backward(loss(tape.Context())); err != nil {
			t.Fatal(err)
		}
		const h = 1e-6
		for pi, p := range m.Params() {
			g := tape.Grad(p)
			if g == nil {
				t.Fatalf("stage %d param %d: nil gradient", stage, pi)
			}
			for idx := range p.Numel() {
				coord := tensor.Unravel(idx, p.Shape())
				o := p.AtF64(coord...)
				p.SetF64(o+h, coord...)
				lp := loss(backend.NewContext()).AtF64()
				p.SetF64(o-h, coord...)
				lm := loss(backend.NewContext()).AtF64()
				p.SetF64(o, coord...)
				num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
				if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
					t.Fatalf("stage %d param %d grad%v: numeric %.8g vs analytic %.8g", stage, pi, coord, num, ana)
				}
			}
		}
	}
	t.Logf("gradcheck passed for all %d params through 1- and 2-thought feedback", len(m.Params()))
}

// Constructor validation, thought-block shapes, and error paths.
func TestCoconutValidationAndShapes(t *testing.T) {
	ok := nlp.CoconutConfig{Vocab: 7, Ctx: 16, Dim: 8, Heads: 2, Layers: 1}
	for name, mut := range map[string]func(*nlp.CoconutConfig){
		"zero vocab":     func(c *nlp.CoconutConfig) { c.Vocab = 0 },
		"zero ctx":       func(c *nlp.CoconutConfig) { c.Ctx = 0 },
		"zero dim":       func(c *nlp.CoconutConfig) { c.Dim = 0 },
		"zero layers":    func(c *nlp.CoconutConfig) { c.Layers = 0 },
		"dim%heads":      func(c *nlp.CoconutConfig) { c.Heads = 3 },
		"negative heads": func(c *nlp.CoconutConfig) { c.Heads = -2 },
	} {
		cfg := ok
		mut(&cfg)
		if _, err := nlp.NewCoconut(cfg, 1); err == nil {
			t.Errorf("%s: want error, got nil", name)
		}
	}

	m, err := nlp.NewCoconut(ok, 1, nlp.WithCoconutMarkers())
	if err != nil {
		t.Fatal(err)
	}
	// markers add <bot>+<eot>: rows = len(prompt) + 1 + k + 1
	logits, err := m.ForwardWithThoughts(backend.NewContext(), []int{0, 1, 2}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !logits.Shape().Equal(tensor.Shape{7, ok.Vocab}) {
		t.Fatalf("marker logits shape %v, want (7, %d)", logits.Shape(), ok.Vocab)
	}
	plain, err := nlp.NewCoconut(ok, 1)
	if err != nil {
		t.Fatal(err)
	}
	if logits, err = plain.ForwardWithThoughts(backend.NewContext(), []int{0, 1, 2}, 2); err != nil {
		t.Fatal(err)
	}
	if !logits.Shape().Equal(tensor.Shape{5, ok.Vocab}) {
		t.Fatalf("plain logits shape %v, want (5, %d)", logits.Shape(), ok.Vocab)
	}
	pause, err := nlp.NewCoconut(ok, 1, nlp.WithCoconutPause())
	if err != nil {
		t.Fatal(err)
	}
	if logits, err = pause.ForwardWithThoughts(backend.NewContext(), []int{0, 1, 2}, 2); err != nil {
		t.Fatal(err)
	}
	if !logits.Shape().Equal(tensor.Shape{5, ok.Vocab}) {
		t.Fatalf("pause logits shape %v, want (5, %d)", logits.Shape(), ok.Vocab)
	}

	ctx := backend.NewContext()
	if _, err := m.ForwardWithThoughts(ctx, []int{0, 1}, -1); err == nil {
		t.Error("negative k: want error")
	}
	if _, err := m.ForwardWithThoughts(ctx, nil, 1); err == nil {
		t.Error("empty prompt: want error")
	}
	if _, err := m.ForwardWithThoughts(ctx, []int{0, 99}, 1); err == nil {
		t.Error("token outside vocab: want error")
	}
	if _, err := m.ForwardWithThoughts(ctx, []int{0, 1, 2}, 32); err == nil {
		t.Error("context overflow via thoughts: want error")
	}
	if _, err := m.CurriculumLoss(ctx, []int{0}, [][]int{{1}}, []int{2}, 2); err == nil {
		t.Error("stage beyond chain: want error")
	}
	if _, err := m.CurriculumLoss(ctx, []int{0}, [][]int{{1}}, []int{2}, -1); err == nil {
		t.Error("negative stage: want error")
	}
	if _, err := m.CurriculumLoss(ctx, []int{0}, [][]int{{1}}, nil, 0); err == nil {
		t.Error("empty answer: want error")
	}
	if _, err := m.CurriculumLoss(ctx, []int{0}, [][]int{{}}, []int{2}, 0); err == nil {
		t.Error("empty chain step: want error")
	}
	if _, err := m.CurriculumLoss(ctx, []int{0}, nil, []int{99}, 0); err == nil {
		t.Error("answer outside vocab: want error")
	}
	if _, err := m.Generate(ctx, []int{0, 1}, 1, 0); err == nil {
		t.Error("maxNew 0: want error")
	}
}

// ExampleCoconut shows the full continuous-thought loop on a toy model: build,
// forward the prompt with two latent thoughts, score one curriculum-stage loss,
// and decode greedily after the thought block. Everything is seeded, so the
// output is deterministic.
func ExampleCoconut() {
	cfg := nlp.CoconutConfig{Vocab: 7, Ctx: 16, Dim: 8, Heads: 2, Layers: 1}
	m, _ := nlp.NewCoconut(cfg, 1, nlp.WithCoconutMarkers())
	fmt.Println("params:", len(m.Params()))

	// prompt + <bot> + 2 continuous thoughts + <eot> → logits per position
	logits, _ := m.ForwardWithThoughts(backend.NewContext(), []int{0, 1, 2}, 2)
	fmt.Println("logits:", logits.Shape())

	// curriculum stage 1: first chain step replaced by a thought, loss on the
	// remaining step and the answer only
	loss, _ := m.CurriculumLoss(backend.NewContext(), []int{0, 1, 2}, [][]int{{3}, {4}}, []int{5}, 1)
	fmt.Println("stage-1 loss > 0:", loss.AtF64() > 0)

	// after 2 latent thoughts, decode 3 answer tokens greedily
	out, _ := m.Generate(backend.NewContext(), []int{0, 1, 2}, 2, 3)
	fmt.Println("generated:", len(out))
	// Output:
	// params: 19
	// logits: (7, 7)
	// stage-1 loss > 0: true
	// generated: 3
}
