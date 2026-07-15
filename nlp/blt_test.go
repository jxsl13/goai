package nlp_test

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// ByteEntropy matches hand-computed references on fixed logit rows.
func TestBLTByteEntropy(t *testing.T) {
	cases := []struct {
		name   string
		logits []float64
		want   float64
	}{
		// uniform over 2: p = [1/2, 1/2], H = ln 2
		{"uniform2", []float64{0, 0}, 0.6931471805599453},
		// uniform over 4: H = ln 4
		{"uniform4", []float64{1, 1, 1, 1}, 1.3862943611198906},
		// logits [2,0]: p = e²/(e²+1) = 0.8807970779778823, q = 0.11920292202211756
		// H = −(p·ln p + q·ln q) = ln(1+e⁻²) + 2q = 0.12692801104297249 + 0.23840584404423512
		{"skewed", []float64{2, 0}, 0.3653338550872076},
		// shift invariance: softmax([5,3]) == softmax([2,0])
		{"shifted", []float64{5, 3}, 0.3653338550872076},
	}
	for _, tc := range cases {
		if got := nlp.ByteEntropy(tc.logits); math.Abs(got-tc.want) > 1e-9 {
			t.Errorf("%s: ByteEntropy(%v) = %.15g, want %.15g", tc.name, tc.logits, got, tc.want)
		}
	}
	// a one-hot distribution has (numerically) zero entropy
	if got := nlp.ByteEntropy([]float64{100, 0, 0}); got > 1e-9 {
		t.Errorf("one-hot: entropy %.3g, want ~0", got)
	}
}

// The global-threshold boundary rule produces the expected patch lengths on
// fixed entropy sequences, including the MaxPatch cap and the +Inf start.
func TestBLTPatchLengths(t *testing.T) {
	inf := math.Inf(1)
	cases := []struct {
		name      string
		entropies []float64
		threshold float64
		maxPatch  int
		want      []int
	}{
		{"boundaries above threshold", []float64{inf, 0.1, 0.9, 0.2, 0.95, 0.1}, 0.5, 0, []int{2, 2, 2}},
		{"no interior boundary", []float64{inf, 0.1, 0.2, 0.1}, 0.5, 0, []int{4}},
		{"all boundaries", []float64{inf, 0.9, 0.9, 0.9}, 0.5, 0, []int{1, 1, 1, 1}},
		{"max patch cap", []float64{inf, 0.1, 0.1, 0.1, 0.1, 0.1}, 0.5, 3, []int{3, 3}},
		{"single byte", []float64{inf}, 0.5, 0, []int{1}},
		{"empty", nil, 0.5, 0, nil},
	}
	for _, tc := range cases {
		got := nlp.PatchLengths(tc.entropies, tc.threshold, tc.maxPatch)
		if len(got) != len(tc.want) {
			t.Errorf("%s: PatchLengths = %v, want %v", tc.name, got, tc.want)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("%s: PatchLengths = %v, want %v", tc.name, got, tc.want)
				break
			}
		}
	}
}

// The entropy patcher trains: its own next-byte cross-entropy drops on a tiny
// repetitive corpus, and its Patch output is a valid segmentation.
func TestBLTEntropyPatcherTrains(t *testing.T) {
	corpus := bytes.Repeat([]byte("the cat sees a star. "), 12)
	const window = 21
	cfg := nlp.EntropyPatcherConfig{Ctx: window, Dim: 12, Heads: 2, Layers: 1, Eps: 1e-5, MaxPatch: 6}
	p, err := nlp.NewEntropyPatcher(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	eval := corpus[:window]
	evalLoss := func() float64 {
		l, err := p.Loss(backend.NewContext(), eval)
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	first := evalLoss()

	opt := nn.NewAdamW(p.Params(), 3e-3, 0.01)
	for step := range 120 {
		start := (step * 7) % (len(corpus) - window - 1)
		tape := autograd.NewTape()
		loss, err := p.Loss(tape.Context(), corpus[start:start+window])
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(p.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	last := evalLoss()
	t.Logf("entropy patcher next-byte CE %.3f → %.3f", first, last)
	if last >= first {
		t.Fatalf("entropy patcher did not train: CE %.3f → %.3f", first, last)
	}

	// Patch returns a valid frozen segmentation under the trained model.
	p.Threshold = 2.0
	lens, err := p.Patch(backend.NewContext(), eval)
	if err != nil {
		t.Fatal(err)
	}
	sum := 0
	for _, l := range lens {
		if l < 1 || l > cfg.MaxPatch {
			t.Fatalf("patch length %d outside [1,%d] in %v", l, cfg.MaxPatch, lens)
		}
		sum += l
	}
	if sum != len(eval) {
		t.Fatalf("patch lengths %v sum to %d, want %d", lens, sum, len(eval))
	}
	t.Logf("patches over %d bytes: %v", len(eval), lens)
}

// The collapse pin (ADR-0025): with stride-1 patches (every byte its own patch),
// no encoder/decoder byte blocks, and a zeroed PoolWo (identity pooling — the
// mean over a singleton span is the byte state itself and the residual
// cross-attention refinement vanishes), the latent path IS a plain byte-GPT.
// A GPT assembled from the SAME live weight tensors must produce identical
// latent-path logits (same f64 op sequence; 1e-10 relative, expected exact).
func TestBLTCollapseToByteGPT(t *testing.T) {
	cfg := nlp.BLTConfig{Ctx: 8, Dim: 8, Heads: 2, EncLayers: 0, LatentLayers: 2, DecLayers: 0, Eps: 1e-5}
	m, err := nlp.NewBLT(cfg, 42)
	if err != nil {
		t.Fatal(err)
	}
	// identity pooling: zero the pooling output projection so p = p0 = h exactly
	for i := range cfg.Dim {
		for j := range cfg.Dim {
			m.PoolWo.SetF64(0, i, j)
		}
	}
	data := []byte("byte gpt")
	ones := make([]int, len(data))
	for i := range ones {
		ones[i] = 1
	}

	ctx := backend.NewContext()
	got, err := m.ForwardLatent(ctx, data, ones)
	if err != nil {
		t.Fatal(err)
	}

	g := &nlp.GPT{
		Config: nlp.GPTConfig{Vocab: 256, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads,
			Layers: cfg.LatentLayers, Eps: cfg.Eps},
		TokEmb: m.ByteEmb, PosEmb: m.PosEmb, Blocks: m.LatentBlocks, LNf: m.LatentLN, Head: m.Head,
	}
	toks := make([]int, len(data))
	for i, b := range data {
		toks[i] = int(b)
	}
	want, err := g.Forward(ctx, toks)
	if err != nil {
		t.Fatal(err)
	}

	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("shape %v vs GPT %v", got.Shape(), want.Shape())
	}
	for i := range data {
		for v := range 256 {
			a, b := got.AtF64(i, v), want.AtF64(i, v)
			if math.Abs(a-b) > 1e-10*math.Max(1, math.Abs(b)) {
				t.Fatalf("logits[%d,%d]: BLT latent %.14g vs byte-GPT %.14g", i, v, a, b)
			}
		}
	}
	t.Logf("stride-1 BLT latent path matches plain byte-GPT on all %d×256 logits", len(data))
}

// The span-masked pooling attends ONLY its assigned bytes: perturbing a byte of
// patch 2 leaves the latent outputs of patches 0 and 1 bit-identical (the
// −1e30 mask underflows to exactly zero weight; with EncLayers=0 there is no
// byte-side mixing, and the causal latent keeps patch 2 out of rows 0–1), while
// row 2 must change. Gradients flow through the pooling into the byte reps.
func TestBLTPoolingMask(t *testing.T) {
	cfg := nlp.BLTConfig{Ctx: 6, Dim: 4, Heads: 2, EncLayers: 0, LatentLayers: 1, DecLayers: 0, Eps: 1e-5}
	m, err := nlp.NewBLT(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	data := []byte{10, 11, 12, 13, 14, 15} // distinct bytes → distinct embedding rows
	patches := []int{2, 2, 2}

	base, err := m.ForwardLatent(backend.NewContext(), data, patches)
	if err != nil {
		t.Fatal(err)
	}
	// perturb the embedding of byte 14 = data[4], the first byte of patch 2
	orig := m.ByteEmb.AtF64(14, 0)
	m.ByteEmb.SetF64(orig+5, 14, 0)
	pert, err := m.ForwardLatent(backend.NewContext(), data, patches)
	if err != nil {
		t.Fatal(err)
	}
	m.ByteEmb.SetF64(orig, 14, 0)

	for p := range 2 { // patches 0 and 1: off-span byte must have exactly zero influence
		for v := range 256 {
			if base.AtF64(p, v) != pert.AtF64(p, v) {
				t.Fatalf("latent row %d changed by an off-span byte: %.14g vs %.14g (mask leak)",
					p, base.AtF64(p, v), pert.AtF64(p, v))
			}
		}
	}
	changed := false
	for v := range 256 {
		if base.AtF64(2, v) != pert.AtF64(2, v) {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("latent row 2 ignored its own span byte — pooling is not attending its patch")
	}

	// gradients flow: byte reps (ByteEmb rows of the input bytes) and all four
	// pooling projections receive nonzero gradient through the masked pooling.
	tape := autograd.NewTape()
	loss, err := m.Loss(tape.Context(), data, patches)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	gEmb := tape.Grad(m.ByteEmb)
	if gEmb == nil {
		t.Fatal("nil gradient for ByteEmb")
	}
	for _, b := range data[:5] { // bytes whose states feed the next-byte loss
		var norm float64
		for j := range cfg.Dim {
			norm += math.Abs(gEmb.AtF64(int(b), j))
		}
		if norm == 0 {
			t.Fatalf("zero gradient for ByteEmb row %d — no flow into the byte reps", b)
		}
	}
	for name, w := range map[string]*tensor.Tensor{
		"PoolWq": m.PoolWq, "PoolWk": m.PoolWk, "PoolWv": m.PoolWv, "PoolWo": m.PoolWo,
	} {
		g := tape.Grad(w)
		if g == nil {
			t.Fatalf("nil gradient for %s", name)
		}
		var norm float64
		for i := range cfg.Dim {
			for j := range cfg.Dim {
				norm += math.Abs(g.AtF64(i, j))
			}
		}
		if norm == 0 {
			t.Fatalf("zero gradient for %s — pooling cross-attention off the tape", name)
		}
	}
	t.Log("pooling attends only its span (exact invariance) and gradients flow to byte reps + pool weights")
}

// Full numeric gradcheck through EVERY main-model parameter (encoder, pooling,
// latent, decoder cross-attention, decoder, head, hash tables — the patcher is
// frozen and not part of Params) on a tiny config: central differences at
// h=1e-6 vs the tape's analytic gradients, rtol 1e-4, F64 — the repo's
// gradcheck recipe (cla_test.go). Multi-byte patches exercise the span mask,
// the strictly-earlier decoder mask, and the patch-0 gate.
func TestBLTGradcheck(t *testing.T) {
	cfg := nlp.BLTConfig{Ctx: 6, Dim: 4, Heads: 2, EncLayers: 1, LatentLayers: 1, DecLayers: 1,
		Window: 2, HashVocab: 5, HashNGrams: []int{2}, Eps: 1e-5}
	m, err := nlp.NewBLT(cfg, 11)
	if err != nil {
		t.Fatal(err)
	}
	data := []byte{3, 1, 4, 1, 5, 9}
	patches := []int{2, 3, 1}

	loss := func(ctx *backend.Context) *tensor.Tensor {
		l, err := m.Loss(ctx, data, patches)
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
			t.Fatalf("param %d: nil gradient", pi)
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
				t.Fatalf("param %d grad%v: numeric %.8g vs analytic %.8g", pi, coord, num, ana)
			}
		}
	}
	t.Logf("gradcheck passed for all %d main-model params (patcher frozen by construction)", len(m.Params()))
}

// NewBLT and NewEntropyPatcher reject bad geometry.
func TestBLTConfigValidation(t *testing.T) {
	ok := nlp.BLTConfig{Ctx: 8, Dim: 8, Heads: 2, EncLayers: 1, LatentLayers: 1, DecLayers: 1}
	for name, mut := range map[string]func(*nlp.BLTConfig){
		"zero ctx":          func(c *nlp.BLTConfig) { c.Ctx = 0 },
		"zero dim":          func(c *nlp.BLTConfig) { c.Dim = 0 },
		"dim%heads":         func(c *nlp.BLTConfig) { c.Heads = 3 },
		"negative heads":    func(c *nlp.BLTConfig) { c.Heads = -2 },
		"zero latent":       func(c *nlp.BLTConfig) { c.LatentLayers = 0 },
		"negative enc":      func(c *nlp.BLTConfig) { c.EncLayers = -1 },
		"negative dec":      func(c *nlp.BLTConfig) { c.DecLayers = -1 },
		"negative window":   func(c *nlp.BLTConfig) { c.Window = -1 },
		"bad hash ngram":    func(c *nlp.BLTConfig) { c.HashVocab = 8; c.HashNGrams = []int{0} },
		"bad patcher heads": func(c *nlp.BLTConfig) { c.PatcherLayers = 1; c.PatcherDim = 8; c.PatcherHeads = 3 },
	} {
		cfg := ok
		mut(&cfg)
		if _, err := nlp.NewBLT(cfg, 1); err == nil {
			t.Errorf("%s: want error, got nil", name)
		}
	}
	if _, err := nlp.NewBLT(ok, 1); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if _, err := nlp.NewEntropyPatcher(nlp.EntropyPatcherConfig{Ctx: 8, Dim: 8, Heads: 3, Layers: 1}, 1); err == nil {
		t.Error("patcher dim%heads: want error, got nil")
	}

	// runtime validation: bad patch lengths are rejected
	m, err := nlp.NewBLT(ok, 1)
	if err != nil {
		t.Fatal(err)
	}
	for name, lens := range map[string][]int{
		"empty":     {},
		"wrong sum": {1, 1},
		"zero len":  {0, 4},
	} {
		if _, err := m.Forward(backend.NewContext(), []byte("abcd"), lens); err == nil {
			t.Errorf("patch lens %s: want error, got nil", name)
		}
	}
}

// ExampleBLT builds a small Byte Latent Transformer with its entropy patcher,
// segments a byte string into patches, forwards it (byte logits and the
// patch-level latent view), scores the next-byte loss, and generates a few
// bytes with online entropy patching. Everything is seeded, so the output is
// deterministic.
func ExampleBLT() {
	cfg := nlp.BLTConfig{
		Ctx: 16, Dim: 8, Heads: 2, EncLayers: 1, LatentLayers: 1, DecLayers: 1,
		PatcherDim: 8, PatcherHeads: 2, PatcherLayers: 1,
		EntropyThreshold: 3, MaxPatchLen: 4,
	}
	m, _ := nlp.NewBLT(cfg, 1)
	fmt.Println("params:", len(m.Params())) // patcher excluded (frozen)

	data := []byte("hello wo")
	patches, _ := m.Patcher.Patch(backend.NewContext(), data)
	sum := 0
	for _, l := range patches {
		sum += l
	}
	fmt.Println("patch lengths sum:", sum)

	logits, _ := m.Forward(backend.NewContext(), data, patches)
	fmt.Println("byte logits:", logits.Shape())

	latent, _ := m.ForwardLatent(backend.NewContext(), data, patches)
	fmt.Println("latent rows == patches:", latent.Shape()[0] == len(patches))

	loss, _ := m.Loss(backend.NewContext(), data, patches)
	fmt.Println("loss > 0:", loss.AtF64() > 0)

	out, _ := m.Generate(data, 4, nlp.Greedy())
	fmt.Println("generated:", len(out))
	// Output:
	// params: 51
	// patch lengths sum: 8
	// byte logits: (8, 256)
	// latent rows == patches: true
	// loss > 0: true
	// generated: 12
}

// ExampleEntropyPatcher builds the small next-byte entropy model, reads its
// per-position entropy signal, segments a byte string with the
// global-threshold rule, and scores its own training loss. Everything is
// seeded, so the output is deterministic.
func ExampleEntropyPatcher() {
	cfg := nlp.EntropyPatcherConfig{Ctx: 16, Dim: 8, Heads: 2, Layers: 1, Threshold: 3, MaxPatch: 4}
	p, _ := nlp.NewEntropyPatcher(cfg, 1)
	fmt.Println("params:", len(p.Params()))

	data := []byte("abcabcab")
	entropies, _ := p.Entropies(backend.NewContext(), data)
	fmt.Println("entropies:", len(entropies), "start is +Inf:", math.IsInf(entropies[0], 1))

	lens, _ := p.Patch(backend.NewContext(), data)
	sum := 0
	for _, l := range lens {
		sum += l
	}
	fmt.Println("patch lengths sum:", sum)

	loss, _ := p.Loss(backend.NewContext(), data)
	fmt.Println("loss > 0:", loss.AtF64() > 0)
	// Output:
	// params: 17
	// entropies: 8 start is +Inf: true
	// patch lengths sum: 8
	// loss > 0: true
}
