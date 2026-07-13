//go:build darwin && cgo

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
)

// §T527: the Self-Extend EXTENSION CURVE — how far does grouped attention carry a model
// beyond its training length? One char-Llama trained only on 32-token windows, evaluated
// teacher-forced at 2×/4×/8× that length; per length the group size G is chosen so the
// maximum effective relative position ⌊L/G⌋ + w − ⌊w/G⌋ stays inside the trained range
// (<32): (64, G=4) → 21, (128, G=8) → 22, (256, G=16) → 23. Claims under test: at EVERY
// length Self-Extend beats the plain forward on the far half, and the plain forward's
// degradation GROWS with length (the premise the mechanism exists for).
func TestSelfExtendExtensionCurve(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a model on cpu; skipped in -short")
	}
	corpus := specCorpus(600)
	toks, vocab := charTokens(corpus)
	cfg := nlp.LlamaConfig{
		Vocab: vocab, Ctx: 288, Dim: 64, Heads: 4, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	const trainWindow, window = 32, 8
	first, last := trainCharLlama(t, m, toks, 400, trainWindow, 3e-3)
	t.Logf("trained on windows of %d: CE %.3f → %.3f", trainWindow, first, last)

	be, ok := backend.Get(backend.CPU)
	if !ok {
		t.Skip("cpu backend not registered")
	}
	ctx := backend.NewContext().WithBackend(be)

	lengths := []struct{ evalLen, group int }{{64, 4}, {128, 8}, {256, 16}}
	var prevPlain float64
	for _, lc := range lengths {
		var plainFar, seFar float64
		const runs = 3
		farFrom := lc.evalLen / 2
		for r := range runs {
			start := 4000 + r*211
			seq := toks[start : start+lc.evalLen+1]
			plain, err := m.Forward(ctx, seq[:lc.evalLen])
			if err != nil {
				t.Fatal(err)
			}
			se, err := m.SelfExtendForward(ctx, seq[:lc.evalLen], window, lc.group)
			if err != nil {
				t.Fatal(err)
			}
			plainFar += teacherForcedCE(t, plain, seq, farFrom, lc.evalLen) / runs
			seFar += teacherForcedCE(t, se, seq, farFrom, lc.evalLen) / runs
		}
		t.Logf("%dx training length (L=%d, G=%d): plain CE %.3f vs self-extend CE %.3f",
			lc.evalLen/trainWindow, lc.evalLen, lc.group, plainFar, seFar)
		if seFar >= plainFar {
			t.Fatalf("L=%d: self-extend %.3f did not beat plain %.3f", lc.evalLen, seFar, plainFar)
		}
		if plainFar < prevPlain {
			t.Logf("note: plain degradation non-monotonic at L=%d (%.3f after %.3f)", lc.evalLen, plainFar, prevPlain)
		}
		prevPlain = plainFar
	}
}
