//go:build darwin && cgo

package llamagpu_test

import (
	"math/rand/v2"
	"sort"
	"testing"
	"time"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T434: speculative decoding measured with REAL trained models. Every earlier speculative test
// used random weights (0% or 100% acceptance by construction, §T419/§T420); the projected speedups
// (1.95×@50%, 2.65×@80%) came from the cost model. Here we TRAIN a target and a smaller draft on
// the same character-level corpus with the library's own training loop, so the draft genuinely
// approximates the target — the regime speculative decoding is designed for — and then measure the
// actual acceptance rate and end-to-end speedup on the batched decoders.

// specCorpus builds a deterministic character corpus from a tiny grammar: structured enough that a
// 1-layer draft learns most of it, with PCG-chosen branch points where a small model and a large
// one can disagree.
func specCorpus(sentences int) string {
	subjects := []string{"the cat", "the dog", "a bird", "the fox", "a mouse", "the owl"}
	verbs := []string{" sees ", " chases ", " finds ", " likes ", " fears "}
	objects := []string{"the ball", "the tree", "a fish", "the moon", "a star", "the sun"}
	rng := rand.New(rand.NewPCG(42, 0xda7a))
	var b []byte
	for range sentences {
		b = append(b, subjects[rng.IntN(len(subjects))]...)
		b = append(b, verbs[rng.IntN(len(verbs))]...)
		b = append(b, objects[rng.IntN(len(objects))]...)
		b = append(b, '.', ' ')
	}
	return string(b)
}

// charTokens maps the corpus to token ids over its sorted unique characters.
func charTokens(corpus string) ([]int, int) {
	set := map[rune]bool{}
	for _, r := range corpus {
		set[r] = true
	}
	chars := make([]rune, 0, len(set))
	for r := range set {
		chars = append(chars, r)
	}
	sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })
	id := map[rune]int{}
	for i, r := range chars {
		id[r] = i
	}
	toks := make([]int, 0, len(corpus))
	for _, r := range corpus {
		toks = append(toks, id[r])
	}
	return toks, len(chars)
}

// trainableGPT builds a GPT with a PCG-random init (the deterministic-but-structured randGPT init
// is fine for kernel tests but poor for optimization).
func trainableGPT(t *testing.T, cfg nlp.GPTConfig, seed uint64) *nlp.GPT {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 0x1717))
	fill := func(scale float64, shape ...int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape(shape))
		d := x.Storage().F32()
		for i := range d {
			d[i] = float32(rng.NormFloat64() * scale)
		}
		return x
	}
	ones := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		d := x.Storage().F32()
		for i := range d {
			d[i] = 1
		}
		return x
	}
	zeros := func(n int) *tensor.Tensor { return tensor.New(tensor.F32, tensor.Shape{n}) }
	ffn := 4 * cfg.Dim
	ts := map[string]*tensor.Tensor{
		"tok_emb":   fill(0.02, cfg.Vocab, cfg.Dim),
		"pos_emb":   fill(0.02, cfg.Ctx, cfg.Dim),
		"head":      fill(0.02, cfg.Dim, cfg.Vocab),
		"lnf.gamma": ones(cfg.Dim),
		"lnf.beta":  zeros(cfg.Dim),
	}
	for l := range cfg.Layers {
		if l >= 10 {
			t.Fatalf("trainableGPT supports <10 layers")
		}
		p := func(x string) string { return "blocks." + string(rune('0'+l)) + "." + x }
		ts[p("ln1.gamma")] = ones(cfg.Dim)
		ts[p("ln1.beta")] = zeros(cfg.Dim)
		ts[p("attn.wq")] = fill(0.02, cfg.Dim, cfg.Dim)
		ts[p("attn.wk")] = fill(0.02, cfg.Dim, cfg.Dim)
		ts[p("attn.wv")] = fill(0.02, cfg.Dim, cfg.Dim)
		ts[p("attn.wo")] = fill(0.02, cfg.Dim, cfg.Dim)
		ts[p("ln2.gamma")] = ones(cfg.Dim)
		ts[p("ln2.beta")] = zeros(cfg.Dim)
		ts[p("ffn.w1")] = fill(0.02, cfg.Dim, ffn)
		ts[p("ffn.b1")] = zeros(ffn)
		ts[p("ffn.w2")] = fill(0.02, ffn, cfg.Dim)
		ts[p("ffn.b2")] = zeros(cfg.Dim)
	}
	m, err := nlp.FromSafetensors(cfg, ts)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// trainCharGPT runs next-character training over sliding corpus windows on the given backend and
// requires the loss to drop well below the uniform baseline (the model actually learned).
func trainCharGPT(t *testing.T, m *nlp.GPT, be backend.Backend, corpus []int, steps, window int, lr float64) (first, last float64) {
	t.Helper()
	opt := nn.NewAdamW(m.Params(), lr, 0.01)
	targets := tensor.New(tensor.F32, tensor.Shape{window})
	for step := range steps {
		start := (step * 97) % (len(corpus) - window - 1)
		tokens := corpus[start : start+window]
		for i := range window {
			targets.SetF64(float64(corpus[start+1+i]), i)
		}
		tape := autograd.NewTapeOn(be)
		ctx := tape.Context()
		logits, err := m.Forward(ctx, tokens)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(m.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	if last >= first*0.5 {
		t.Fatalf("model did not train: CE %.4f → %.4f", first, last)
	}
	return first, last
}

// TestSpeculativeWithTrainedModels is the §T434 measurement: train target (3 layers, dim 96) and
// draft (1 layer, dim 48) on the same corpus, verify the greedy speculative output is
// token-for-token the target's own greedy output (losslessness with REAL models), and measure the
// real acceptance rate and end-to-end speedup vs plain batched Generate.
func TestSpeculativeWithTrainedModels(t *testing.T) {
	if testing.Short() {
		t.Skip("trains two models; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("metal not registered")
	}

	corpus := specCorpus(600)
	toks, vocab := charTokens(corpus)
	t.Logf("corpus: %d chars, vocab %d", len(toks), vocab)

	tcfg := nlp.GPTConfig{Vocab: vocab, Ctx: 256, Dim: 96, Heads: 4, Layers: 3, Eps: 1e-5}
	dcfg := nlp.GPTConfig{Vocab: vocab, Ctx: 256, Dim: 48, Heads: 2, Layers: 1, Eps: 1e-5}

	const window = 128
	tm := trainableGPT(t, tcfg, 1)
	s := time.Now()
	tf, tl := trainCharGPT(t, tm, be, toks, 240, window, 3e-3)
	t.Logf("target trained (240 steps, %.1fs): CE %.3f → %.3f", time.Since(s).Seconds(), tf, tl)

	dm := trainableGPT(t, dcfg, 2)
	s = time.Now()
	df, dl := trainCharGPT(t, dm, be, toks, 240, window, 3e-3)
	t.Logf("draft trained (240 steps, %.1fs): CE %.3f → %.3f", time.Since(s).Seconds(), df, dl)

	target, err := llamagpu.NewGPT(tm)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Release()
	draft, err := llamagpu.NewGPT(dm)
	if err != nil {
		t.Fatal(err)
	}
	defer draft.Release()

	prompt := toks[:24]
	const maxNew = 96

	// losslessness with real models: greedy speculative == greedy plain, whatever the acceptance.
	got, stats, err := llamagpu.SpeculativeGenerate(target, draft, prompt, maxNew, 4, nlp.Greedy(), 7)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := llamagpu.NewGPT(tm)
	if err != nil {
		t.Fatal(err)
	}
	defer plain.Release()
	want, err := plain.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("length: speculative %d vs plain %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]: speculative %d vs plain %d", i, got[i], want[i])
		}
	}
	if stats.Proposed == 0 {
		t.Fatal("no draft tokens proposed")
	}
	acc := stats.AcceptanceRate()
	if acc < 0.15 {
		t.Fatalf("trained draft acceptance %.0f%% — draft did not learn the target's distribution", 100*acc)
	}

	// end-to-end timing, medians of 3 (§V22: same-session A/B on the identical workload).
	median := func(f func() error) float64 {
		var ms [3]float64
		for i := range ms {
			s := time.Now()
			if err := f(); err != nil {
				t.Fatal(err)
			}
			ms[i] = time.Since(s).Seconds() * 1e3
		}
		sort.Float64s(ms[:])
		return ms[1]
	}
	tSpec := median(func() error {
		_, _, err := llamagpu.SpeculativeGenerate(target, draft, prompt, maxNew, 4, nlp.Greedy(), 7)
		return err
	})
	tPlain := median(func() error {
		_, err := plain.Generate(prompt, maxNew, nlp.Greedy())
		return err
	})
	t.Logf("REAL trained models: acceptance %.0f%% (proposed %d, accepted %d)", 100*acc, stats.Proposed, stats.Accepted)
	t.Logf("plain %.0f ms (%.0f tok/s) vs speculative %.0f ms (%.0f tok/s) → speedup %.2fx",
		tPlain, float64(maxNew)/tPlain*1e3, tSpec, float64(maxNew)/tSpec*1e3, tPlain/tSpec)
}
