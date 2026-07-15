package nlp_test

import (
	"math"
	"sort"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// End-to-end Byte Latent Transformer proof on the in-repo grammar corpus
// (diffusionGrammar, diffusion_lm_e2e_test.go — the corpus is ASCII, so chars
// ARE bytes and the model is genuinely tokenizer-free). The paper's two-phase
// recipe: (1) train the small entropy patcher on its own next-byte CE and
// FREEZE it, calibrating the boundary threshold to the median entropy; (2)
// train the main BLT on next-byte CE under the frozen entropy patches. The
// eval CE on a fixed window must at least HALVE from the random-init start
// (the char-LM e2e bar, nn/newblocks_lm_e2e_test.go), and greedy generation
// with ONLINE entropy patching must produce a valid grammar window.
func TestBLTGrammarE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("trains an entropy patcher + BLT; skipped in -short")
	}
	toks, _, alphabet := diffusionGrammar()
	text := make([]byte, len(toks))
	for i, tk := range toks {
		text[i] = alphabet[tk]
	}
	const window = 48

	// Phase 1: train the entropy patcher (separately, its own byte-CE).
	pcfg := nlp.EntropyPatcherConfig{Ctx: window, Dim: 16, Heads: 2, Layers: 1, Eps: 1e-5, MaxPatch: 8}
	patcher, err := nlp.NewEntropyPatcher(pcfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	patcherLoss := func() float64 {
		l, err := patcher.Loss(backend.NewContext(), text[:window])
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	pFirst := patcherLoss()
	pOpt := nn.NewAdamW(patcher.Params(), 3e-3, 0.01)
	for step := range 300 {
		start := (step * 89) % (len(text) - window - 1)
		tape := autograd.NewTape()
		loss, err := patcher.Loss(tape.Context(), text[start:start+window])
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(patcher.Params(), tape.Grad, 1.0)
		if err := pOpt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	pLast := patcherLoss()
	t.Logf("entropy patcher next-byte CE %.3f → %.3f", pFirst, pLast)
	if pLast >= pFirst {
		t.Fatalf("entropy patcher did not train: CE %.3f → %.3f", pFirst, pLast)
	}

	// Calibrate the boundary threshold to the median trained entropy: about
	// half the positions become boundaries (avg patch length ≈ 2). The patcher
	// is FROZEN from here on — its boundaries are non-differentiable inputs.
	entropies, err := patcher.Entropies(backend.NewContext(), text[:window])
	if err != nil {
		t.Fatal(err)
	}
	finite := append([]float64(nil), entropies[1:]...)
	sort.Float64s(finite)
	patcher.Threshold = finite[len(finite)/2]
	t.Logf("calibrated entropy threshold: %.3f nats (median)", patcher.Threshold)

	// Phase 2: train the main BLT under the frozen patcher's boundaries.
	cfg := nlp.BLTConfig{Ctx: window, Dim: 32, Heads: 4, EncLayers: 1, LatentLayers: 2, DecLayers: 1,
		Window: 8, Eps: 1e-5}
	m, err := nlp.NewBLT(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	m.Patcher = patcher

	patchCache := map[int][]int{}
	patchesAt := func(start int) []int {
		if lens, ok := patchCache[start]; ok {
			return lens
		}
		lens, err := patcher.Patch(backend.NewContext(), text[start:start+window])
		if err != nil {
			t.Fatal(err)
		}
		patchCache[start] = lens
		return lens
	}
	evalLoss := func() float64 {
		l, err := m.Loss(backend.NewContext(), text[:window], patchesAt(0))
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	first := evalLoss()

	opt := nn.NewAdamW(m.Params(), 3e-3, 0.01)
	for step := range 600 {
		start := (step * 89) % (len(text) - window - 1)
		tape := autograd.NewTape()
		loss, err := m.Loss(tape.Context(), text[start:start+window], patchesAt(start))
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(m.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	last := evalLoss()
	t.Logf("BLT next-byte eval CE %.3f → %.3f (bar %.3f)", first, last, first*0.5)
	if last >= first*0.5 {
		t.Fatalf("BLT did not train: eval CE %.3f → %.3f (bar %.3f)", first, last, first*0.5)
	}
	if math.IsNaN(last) {
		t.Fatal("eval CE is NaN")
	}

	// Generation with ONLINE entropy patching: greedy-extend a 6-byte grammar
	// prefix by 10 bytes; the 16-byte result must be a valid grammar window
	// (the diffusion-LM e2e bar, diffusionGrammarWindows).
	const genLen = 16
	prompt := text[:6]
	out, err := m.Generate(prompt, genLen-len(prompt), nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != genLen {
		t.Fatalf("generated %d bytes, want %d", len(out), genLen)
	}
	t.Logf("BLT generation: %q", string(out))
	if !diffusionGrammarWindows(genLen)[string(out)] {
		t.Fatalf("generated %q is not a valid %d-byte grammar window", string(out), genLen)
	}
}
