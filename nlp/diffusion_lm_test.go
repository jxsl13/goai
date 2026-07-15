package nlp

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// tinyDiffusionLM builds the small fixed-geometry model the unit tests share.
func tinyDiffusionLM(t *testing.T, cfg DiffusionLMConfig, seed uint64) *DiffusionLM {
	t.Helper()
	m, err := NewDiffusionLM(cfg, seed)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// Constructor + shapes: config validation, Forward [L]→[L,Vocab+1] (the +1 is
// the [MASK] row), Params present with the GPT layout count, MaskID accepted
// by Forward.
func TestDiffusionLMShapes(t *testing.T) {
	bad := []DiffusionLMConfig{
		{Vocab: 0, Ctx: 4, Dim: 4, Heads: 2, Layers: 1},  // no vocab
		{Vocab: 3, Ctx: 0, Dim: 4, Heads: 2, Layers: 1},  // no ctx
		{Vocab: 3, Ctx: 4, Dim: 0, Heads: 2, Layers: 1},  // no dim
		{Vocab: 3, Ctx: 4, Dim: 4, Heads: 0, Layers: 1},  // no heads
		{Vocab: 3, Ctx: 4, Dim: 4, Heads: 3, Layers: 1},  // heads ∤ dim
		{Vocab: 3, Ctx: 4, Dim: 4, Heads: 2, Layers: 0},  // no layers
		{Vocab: 3, Ctx: 4, Dim: 4, Heads: 2, Layers: -1}, // negative layers
	}
	for i, cfg := range bad {
		if _, err := NewDiffusionLM(cfg, 1); err == nil {
			t.Errorf("bad config %d (%+v): want error", i, cfg)
		}
	}

	cfg := DiffusionLMConfig{Vocab: 5, Ctx: 8, Dim: 8, Heads: 2, Layers: 2}
	m := tinyDiffusionLM(t, cfg, 1)
	if m.MaskID() != cfg.Vocab {
		t.Fatalf("MaskID = %d, want %d", m.MaskID(), cfg.Vocab)
	}
	if m.Config.Eps != 1e-5 {
		t.Errorf("Eps default = %g, want 1e-5", m.Config.Eps)
	}
	for _, b := range m.GPT.Blocks {
		if b.Attn.Causal {
			t.Fatal("block attention is causal; the diffusion backbone must be bidirectional")
		}
	}

	toks := []int{0, 3, m.MaskID(), 2, 1, m.MaskID()} // [MASK] must be a legal input id
	logits, err := m.Forward(backend.NewContext(), toks)
	if err != nil {
		t.Fatal(err)
	}
	if sh := logits.Shape(); sh[0] != len(toks) || sh[1] != cfg.Vocab+1 {
		t.Fatalf("Forward shape %v, want [%d %d]", sh, len(toks), cfg.Vocab+1)
	}
	if want := 2 + 12*cfg.Layers + 3; len(m.Params()) != want {
		t.Fatalf("Params() = %d tensors, want %d", len(m.Params()), want)
	}
	if _, err := m.Forward(backend.NewContext(), []int{m.MaskID() + 1}); err == nil {
		t.Error("token past [MASK]: want error")
	}
	if _, err := m.Forward(backend.NewContext(), make([]int, cfg.Ctx+1)); err == nil {
		t.Error("sequence past ctx: want error")
	}
}

// DiffusionMask statistics: E[masked fraction] ≈ t, absorbing state only (every
// selected position becomes exactly maskID, all others untouched), positions
// consistent with the corruption, determinism under a fixed seed, and the t=1 /
// t=0 boundaries.
func TestDiffusionMaskStats(t *testing.T) {
	const (
		vocab  = 7
		maskID = vocab
		L      = 200
		draws  = 200
		level  = 0.3
	)
	tokens := make([]int, L)
	rng := rand.New(rand.NewPCG(1, 2))
	for i := range tokens {
		tokens[i] = rng.IntN(vocab)
	}

	total := 0
	for range draws {
		corrupted, positions := DiffusionMask(tokens, level, maskID, rng)
		total += len(positions)
		inPos := map[int]bool{}
		for _, p := range positions {
			inPos[p] = true
		}
		for i := range tokens {
			switch {
			case inPos[i] && corrupted[i] != maskID:
				t.Fatalf("position %d selected but corrupted to %d, not [MASK]=%d (absorbing state)", i, corrupted[i], maskID)
			case !inPos[i] && corrupted[i] != tokens[i]:
				t.Fatalf("position %d not selected but changed %d→%d", i, tokens[i], corrupted[i])
			}
		}
	}
	frac := float64(total) / float64(L*draws)
	if math.Abs(frac-level) > 0.01 { // n=40000 Bernoulli(0.3): sd ≈ 0.0023, 0.01 ≈ 4.4σ
		t.Fatalf("masked fraction %.4f, want ≈ %.2f", frac, level)
	}

	// determinism: identical seed → identical corruption
	a1, p1 := DiffusionMask(tokens, level, maskID, rand.New(rand.NewPCG(9, 9)))
	a2, p2 := DiffusionMask(tokens, level, maskID, rand.New(rand.NewPCG(9, 9)))
	if len(p1) != len(p2) {
		t.Fatalf("determinism: %d vs %d positions", len(p1), len(p2))
	}
	for i := range a1 {
		if a1[i] != a2[i] {
			t.Fatalf("determinism: corrupted[%d] %d vs %d", i, a1[i], a2[i])
		}
	}

	// boundaries: t=1 masks everything, t=0 masks nothing
	if _, p := DiffusionMask(tokens, 1, maskID, rng); len(p) != L {
		t.Errorf("t=1 masked %d of %d", len(p), L)
	}
	if _, p := DiffusionMask(tokens, 0, maskID, rng); len(p) != 0 {
		t.Errorf("t=0 masked %d, want 0", len(p))
	}
}

// MaskedLoss matches a hand-computed reference — softmax cross-entropy over the
// masked rows of the model's own logits, scaled by m/(L·t) — and its gradient
// reaches the backbone (nonzero TokEmb and Head grads through the masked rows).
func TestDiffusionMaskedLossReference(t *testing.T) {
	cfg := DiffusionLMConfig{Vocab: 5, Ctx: 6, Dim: 8, Heads: 2, Layers: 1}
	m := tinyDiffusionLM(t, cfg, 3)
	mask := m.MaskID()
	original := []int{0, 1, 2, 3, 4, 1}
	corrupted := []int{0, mask, 2, 3, mask, 1} // masked positions 1 and 4
	const level = 0.4

	// reference: stable log-softmax CE per masked row, mean over rows, × m/(L·t)
	logits, err := m.Forward(backend.NewContext(), corrupted)
	if err != nil {
		t.Fatal(err)
	}
	positions := []int{1, 4}
	var sum float64
	for _, p := range positions {
		mx := math.Inf(-1)
		for c := 0; c < cfg.Vocab+1; c++ {
			mx = math.Max(mx, logits.AtF64(p, c))
		}
		var se float64
		for c := 0; c < cfg.Vocab+1; c++ {
			se += math.Exp(logits.AtF64(p, c) - mx)
		}
		sum += mx + math.Log(se) - logits.AtF64(p, original[p])
	}
	mCount, L := float64(len(positions)), float64(len(corrupted))
	want := (sum / mCount) * (mCount / (L * level))

	loss, err := m.MaskedLoss(backend.NewContext(), corrupted, original, level)
	if err != nil {
		t.Fatal(err)
	}
	if got := loss.AtF64(); math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
		t.Fatalf("MaskedLoss %.12g, want %.12g", got, want)
	}

	// gradient flow: backward reaches TokEmb and Head with nonzero grads
	tape := autograd.NewTape()
	tl, err := m.MaskedLoss(tape.Context(), corrupted, original, level)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(tl); err != nil {
		t.Fatal(err)
	}
	for _, pr := range []struct {
		name string
		w    *tensor.Tensor
	}{{"TokEmb", m.GPT.TokEmb}, {"Head", m.GPT.Head}} {
		g := tape.Grad(pr.w)
		if g == nil {
			t.Fatalf("%s: nil gradient", pr.name)
		}
		nonzero := false
		for i := 0; i < g.Numel() && !nonzero; i++ {
			nonzero = g.AtF64(tensor.Unravel(i, g.Shape())...) != 0
		}
		if !nonzero {
			t.Fatalf("%s: all-zero gradient — loss not wired to the backbone", pr.name)
		}
	}

	// error cases
	if _, err := m.MaskedLoss(backend.NewContext(), corrupted, original[:3], level); err == nil {
		t.Error("length mismatch: want error")
	}
	if _, err := m.MaskedLoss(backend.NewContext(), corrupted, original, 0); err == nil {
		t.Error("t=0: want error")
	}
	if _, err := m.MaskedLoss(backend.NewContext(), corrupted, original, 1.5); err == nil {
		t.Error("t>1: want error")
	}
	if _, err := m.MaskedLoss(backend.NewContext(), original, original, level); err == nil {
		t.Error("no masked positions: want error")
	}
	badOrig := append([]int(nil), original...)
	badOrig[1] = mask // original may not contain [MASK]
	if _, err := m.MaskedLoss(backend.NewContext(), corrupted, badOrig, level); err == nil {
		t.Error("original containing [MASK]: want error")
	}
}

// Loss draws its own t and mask: scalar, positive, deterministic under a fixed
// rng seed, and it survives tiny sequences (redrawing empty masks).
func TestDiffusionLoss(t *testing.T) {
	cfg := DiffusionLMConfig{Vocab: 5, Ctx: 6, Dim: 8, Heads: 2, Layers: 1}
	m := tinyDiffusionLM(t, cfg, 3)
	toks := []int{0, 1, 2, 3, 4, 1}

	l1, err := m.Loss(backend.NewContext(), toks, rand.New(rand.NewPCG(5, 6)))
	if err != nil {
		t.Fatal(err)
	}
	l2, err := m.Loss(backend.NewContext(), toks, rand.New(rand.NewPCG(5, 6)))
	if err != nil {
		t.Fatal(err)
	}
	if l1.Ndim() != 0 {
		t.Fatalf("loss rank %d, want scalar", l1.Ndim())
	}
	if l1.AtF64() <= 0 {
		t.Fatalf("loss %g, want > 0", l1.AtF64())
	}
	if l1.AtF64() != l2.AtF64() {
		t.Fatalf("same seed, different loss: %g vs %g", l1.AtF64(), l2.AtF64())
	}
	// single-token sequence: many draws mask nothing; Loss must redraw, not fail
	if _, err := m.Loss(backend.NewContext(), []int{2}, rand.New(rand.NewPCG(7, 8))); err != nil {
		t.Fatalf("length-1 sequence: %v", err)
	}
	if _, err := m.Loss(backend.NewContext(), nil, rand.New(rand.NewPCG(1, 1))); err == nil {
		t.Error("empty sequence: want error")
	}
	if _, err := m.Loss(backend.NewContext(), toks, nil); err == nil {
		t.Error("nil rng: want error")
	}
}

// Numeric gradcheck through EVERY model parameter on a tiny config: central
// differences on MaskedLoss vs the tape's analytic gradients (the repo's
// gradcheck recipe, rtol 1e-4 at h=1e-6 in F64).
func TestDiffusionLMGradcheck(t *testing.T) {
	cfg := DiffusionLMConfig{Vocab: 3, Ctx: 4, Dim: 4, Heads: 2, Layers: 1}
	m := tinyDiffusionLM(t, cfg, 11)
	mask := m.MaskID()
	original := []int{0, 1, 2, 1}
	corrupted := []int{mask, 1, mask, 1}
	const level = 0.5

	tape := autograd.NewTape()
	loss, err := m.MaskedLoss(tape.Context(), corrupted, original, level)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	fwd := func() float64 {
		l, err := m.MaskedLoss(backend.NewContext(), corrupted, original, level)
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
}

// Sampler unit: steps=1 fills every position in one shot; multi-step
// confidence remasking keeps exactly round(L·(1−(s+1)/steps)) positions masked
// after each step; error cases reject bad arguments.
func TestDiffusionGenerateSchedule(t *testing.T) {
	cfg := DiffusionLMConfig{Vocab: 5, Ctx: 12, Dim: 8, Heads: 2, Layers: 1}
	m := tinyDiffusionLM(t, cfg, 17)

	// steps=1: single parallel shot, nothing masked, all ids in [0,Vocab)
	out, err := m.DiffusionGenerate(6, 1, Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 6 {
		t.Fatalf("length %d, want 6", len(out))
	}
	for i, tok := range out {
		if tok < 0 || tok >= cfg.Vocab {
			t.Fatalf("out[%d] = %d outside real vocab [0,%d) — [MASK] must never be emitted", i, tok, cfg.Vocab)
		}
	}

	// schedule: L=10, steps=4 → masked counts after each step 8, 5, 3, 0
	const length, steps = 10, 4
	var got []int
	_, err = m.diffusionGenerate(length, steps, NewSampler(3, WithTemperature(0.9)), func(step int, seq []int) {
		n := 0
		for _, tok := range seq {
			if tok == m.MaskID() {
				n++
			}
		}
		got = append(got, n)
	})
	if err != nil {
		t.Fatal(err)
	}
	want := make([]int, steps)
	for s := range steps {
		want[s] = int(math.Round(float64(length) * (1 - float64(s+1)/float64(steps))))
	}
	if len(got) != len(want) {
		t.Fatalf("observed %d steps, want %d", len(got), len(want))
	}
	for s := range want {
		if got[s] != want[s] {
			t.Fatalf("masked count after step %d = %d, want %d (schedule %v vs %v)", s, got[s], want[s], got, want)
		}
	}

	// determinism: same sampler seed → same sequence
	g1, err := m.DiffusionGenerate(8, 4, NewSampler(21))
	if err != nil {
		t.Fatal(err)
	}
	g2, err := m.DiffusionGenerate(8, 4, NewSampler(21))
	if err != nil {
		t.Fatal(err)
	}
	for i := range g1 {
		if g1[i] != g2[i] {
			t.Fatalf("determinism: g1[%d]=%d g2[%d]=%d", i, g1[i], i, g2[i])
		}
	}

	// error cases
	if _, err := m.DiffusionGenerate(0, 4, Greedy()); err == nil {
		t.Error("length 0: want error")
	}
	if _, err := m.DiffusionGenerate(cfg.Ctx+1, 4, Greedy()); err == nil {
		t.Error("length > ctx: want error")
	}
	if _, err := m.DiffusionGenerate(4, 0, Greedy()); err == nil {
		t.Error("steps 0: want error")
	}
	if _, err := m.DiffusionGenerate(4, 4, nil); err == nil {
		t.Error("nil sampler: want error")
	}
}
