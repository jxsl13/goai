package nlp_test

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// diffusionGrammar rebuilds the deterministic grammar corpus of the char-LM
// e2e suite (nn/newblocks_lm_e2e_test.go grammarCorpus — same seed and
// construction): "SUBJ VERB OBJ. " sentences over a 15-char vocabulary,
// tokenized by first appearance. Returns the token stream, the vocab size and
// the id→byte table for decoding generations.
func diffusionGrammar() (toks []int, vocab int, alphabet []byte) {
	rng := rand.New(rand.NewPCG(2, 0x3a3))
	subjects := []string{"the cat", "a dog"}
	verbs := []string{" sees ", " fears "}
	objects := []string{"a star", "the sun"}
	var text []byte
	for range 400 {
		text = append(text, subjects[rng.IntN(2)]...)
		text = append(text, verbs[rng.IntN(2)]...)
		text = append(text, objects[rng.IntN(2)]...)
		text = append(text, '.', ' ')
	}
	seen := map[byte]int{}
	for _, b := range text {
		if _, ok := seen[b]; !ok {
			seen[b] = len(seen)
		}
	}
	toks = make([]int, len(text))
	for i, b := range text {
		toks[i] = seen[b]
	}
	alphabet = make([]byte, len(seen))
	for b, id := range seen {
		alphabet[id] = b
	}
	return toks, len(seen), alphabet
}

// diffusionGrammarWindows returns every valid n-char window of the grammar
// language: all substrings of length n over every two-sentence concatenation
// (n is shorter than the shortest sentence, so a window spans at most one
// sentence boundary — two sentences cover every possible phase).
func diffusionGrammarWindows(n int) map[string]bool {
	subjects := []string{"the cat", "a dog"}
	verbs := []string{" sees ", " fears "}
	objects := []string{"a star", "the sun"}
	var sentences []string
	for _, s := range subjects {
		for _, v := range verbs {
			for _, o := range objects {
				sentences = append(sentences, s+v+o+". ")
			}
		}
	}
	windows := map[string]bool{}
	for _, a := range sentences {
		for _, b := range sentences {
			text := a + b
			for i := 0; i+n <= len(text); i++ {
				windows[text[i:i+n]] = true
			}
		}
	}
	return windows
}

// End-to-end masked-diffusion proof on the in-repo grammar corpus, mirroring
// the char-LM e2e bar (nn/newblocks_lm_e2e_test.go): the fixed-mask evaluation
// loss must at least HALVE from the random-init start, and iterative unmasking
// (DiffusionGenerate, 16 steps) must produce a VALID grammar window with no
// [MASK] left. The eval snapshot masks every even position at t=0.5, so its
// m/(L·t) weight is exactly 1 and the number is a plain masked cross-entropy —
// deterministic across runs (fixed seeds everywhere).
func TestDiffusionLMGrammarE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a masked-diffusion LM; skipped in -short")
	}
	toks, vocab, alphabet := diffusionGrammar()
	const window = 32
	cfg := nlp.DiffusionLMConfig{Vocab: vocab, Ctx: window, Dim: 32, Heads: 4, Layers: 2, Eps: 1e-5}
	model, err := nlp.NewDiffusionLM(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}

	// fixed eval snapshot: window at offset 0, even positions masked, t=0.5
	evalOrig := append([]int(nil), toks[:window]...)
	evalCorr := append([]int(nil), evalOrig...)
	for i := 0; i < window; i += 2 {
		evalCorr[i] = model.MaskID()
	}
	evalLoss := func() float64 {
		l, err := model.MaskedLoss(backend.NewContext(), evalCorr, evalOrig, 0.5)
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	first := evalLoss()

	opt := nn.NewAdamW(model.Params(), 3e-3, 0.01)
	rng := rand.New(rand.NewPCG(3, 7))
	for step := range 20000 {
		start := (step * 89) % (len(toks) - window - 1)
		seq := toks[start : start+window]
		tape := autograd.NewTape()
		loss, err := model.Loss(tape.Context(), seq, rng)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(model.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	last := evalLoss()
	t.Logf("masked-diffusion eval CE %.3f → %.3f", first, last)
	if last >= first*0.5 {
		t.Fatalf("diffusion LM did not train: eval CE %.3f → %.3f (bar %.3f)", first, last, first*0.5)
	}

	// generation: 16 tokens by iterative unmasking over 16 steps, greedy fills
	const genLen = 16
	gen, err := model.DiffusionGenerate(genLen, 16, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(gen) != genLen {
		t.Fatalf("generated %d tokens, want %d", len(gen), genLen)
	}
	var sb strings.Builder
	for i, tok := range gen {
		if tok < 0 || tok >= vocab {
			t.Fatalf("gen[%d] = %d outside vocab [0,%d) — [MASK] or garbage emitted", i, tok, vocab)
		}
		sb.WriteByte(alphabet[tok])
	}
	text := sb.String()
	t.Logf("diffusion generation: %q", text)
	if !diffusionGrammarWindows(genLen)[text] {
		t.Fatalf("generated %q is not a valid %d-char grammar window", text, genLen)
	}
}

// ExampleDiffusionLM shows the full masked-diffusion loop on a toy model:
// build, forward a partially masked sequence, score the masked weighted
// cross-entropy (both the explicit-mask and the draw-your-own variants), and
// generate by iterative unmasking. Everything is seeded, so the output is
// deterministic.
func ExampleDiffusionLM() {
	cfg := nlp.DiffusionLMConfig{Vocab: 5, Ctx: 8, Dim: 8, Heads: 2, Layers: 1}
	m, _ := nlp.NewDiffusionLM(cfg, 1)
	fmt.Println("maskID:", m.MaskID(), "params:", len(m.Params()))

	// forward a corrupted sequence: logits over vocab+1 ids (incl. [MASK])
	corrupted := []int{0, m.MaskID(), 2, m.MaskID()}
	logits, _ := m.Forward(backend.NewContext(), corrupted)
	fmt.Println("logits:", logits.Shape())

	// masked weighted CE at masking level t=0.5 against the clean sequence
	original := []int{0, 1, 2, 3}
	loss, _ := m.MaskedLoss(backend.NewContext(), corrupted, original, 0.5)
	fmt.Println("masked loss > 0:", loss.AtF64() > 0)

	// one Monte-Carlo draw of the training objective (t and mask from rng)
	rng := rand.New(rand.NewPCG(7, 9))
	mc, _ := m.Loss(backend.NewContext(), original, rng)
	fmt.Println("mc loss > 0:", mc.AtF64() > 0)

	// iterative unmasking: 6 tokens in 3 steps, greedy fills
	gen, _ := m.DiffusionGenerate(6, 3, nlp.Greedy())
	valid := true
	for _, tok := range gen {
		valid = valid && tok >= 0 && tok < cfg.Vocab
	}
	fmt.Println("generated:", len(gen), "all real tokens:", valid)
	// Output:
	// maskID: 5 params: 17
	// logits: (4, 6)
	// masked loss > 0: true
	// mc loss > 0: true
	// generated: 6 all real tokens: true
}
