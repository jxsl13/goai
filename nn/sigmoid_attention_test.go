package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1: SigmoidAttention maps [T,dim]→[T,dim]; Params exposes 8 tensors
// (Q,K,V,O × W,B) without a learnable bias, 9 with one.
func TestSigmoidAttentionShapes(t *testing.T) {
	const dim, heads, seq = 8, 2, 5
	s, err := nn.NewSigmoidAttention(tensor.F64, dim, heads, 7)
	if err != nil {
		t.Fatal(err)
	}
	if s.HeadDim != dim/heads {
		t.Errorf("HeadDim=%d, want %d", s.HeadDim, dim/heads)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := randMat(rng, seq, dim)
	out, err := s.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{seq, dim}) {
		t.Fatalf("output shape %v, want [%d %d]", out.Shape(), seq, dim)
	}
	if got, want := len(s.Params()), 8; got != want {
		t.Errorf("Params count %d, want %d (no learnable bias)", got, want)
	}
	sl, err := nn.NewSigmoidAttention(tensor.F64, dim, heads, 7, nn.WithLearnableSigmoidBias())
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(sl.Params()), 9; got != want {
		t.Errorf("learnable-bias Params count %d, want %d", got, want)
	}
	if sl.Bias == nil || !sl.Bias.Shape().Equal(tensor.Shape{1, 1}) {
		t.Errorf("learnable Bias tensor missing or wrong shape: %v", sl.Bias)
	}
}

// §V16: the constructor rejects a dim not divisible by heads and non-positive
// dim/heads; Forward rejects a wrong input width and a wrong rank.
func TestSigmoidAttentionErrors(t *testing.T) {
	if _, err := nn.NewSigmoidAttention(tensor.F64, 7, 2, 1); err == nil {
		t.Error("dim%heads != 0: want error")
	}
	if _, err := nn.NewSigmoidAttention(tensor.F64, 0, 1, 1); err == nil {
		t.Error("dim=0: want error")
	}
	if _, err := nn.NewSigmoidAttention(tensor.F64, 8, 0, 1); err == nil {
		t.Error("heads=0: want error")
	}
	if _, err := nn.NewSigmoidAttention(tensor.F64, 8, -2, 1); err == nil {
		t.Error("negative heads: want error")
	}
	s, _ := nn.NewSigmoidAttention(tensor.F64, 8, 2, 1)
	if _, err := s.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 4})); err == nil {
		t.Error("wrong input width: want error")
	}
	if _, err := s.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 8})); err == nil {
		t.Error("rank-3 input: want error")
	}
}

// sigmoidGradcheck compares every analytic gradient of every parameter of s
// against central finite differences of the summed forward output (F64 only).
// Returns the entry count and max relative error for the report.
func sigmoidGradcheck(t *testing.T, s *nn.SigmoidAttention, x *tensor.Tensor) (int, float64) {
	t.Helper()
	const h = 1e-6
	tape := autograd.NewTape()
	out, err := s.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	sumLoss := func() float64 {
		o, err := s.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var acc float64
		for i := range o.Numel() {
			acc += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return acc
	}
	var entries int
	var maxRel float64
	for pi, p := range s.Params() {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("param %d: nil grad — no gradient reached this parameter", pi)
		}
		var nz bool
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := sumLoss()
			p.SetF64(o-h, coord...)
			lm := sumLoss()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			entries++
			if math.Abs(ana) > 1e-9 {
				nz = true
			}
			rel := math.Abs(num-ana) / math.Max(1, math.Abs(num))
			if rel > maxRel {
				maxRel = rel
			}
			if rel > 1e-4 {
				t.Errorf("param %d grad%v: numeric %.8g vs analytic %.8g", pi, coord, num, ana)
			}
		}
		if !nz {
			t.Errorf("param %d: all analytic grads zero — parameter not exercised", pi)
		}
	}
	return entries, maxRel
}

// §V16 (4) GRADCHECK: central finite differences confirm gradients reach EVERY
// parameter — Q/K/V/O projections AND the learnable bias — through OpSigmoid and
// the value aggregation, in both the bidirectional and causal masks.
func TestSigmoidAttentionGradcheck(t *testing.T) {
	const dim, heads, seq = 4, 2, 3
	rng := rand.New(rand.NewPCG(29, 7))
	x := randMat(rng, seq, dim)

	s, err := nn.NewSigmoidAttention(tensor.F64, dim, heads, 3, nn.WithLearnableSigmoidBias())
	if err != nil {
		t.Fatal(err)
	}
	n, rel := sigmoidGradcheck(t, s, x)
	t.Logf("bidirectional (learnable bias) gradcheck: %d entries, max rel err %.3g", n, rel)

	sc, err := nn.NewSigmoidAttention(tensor.F64, dim, heads, 5,
		nn.WithSigmoidAttnCausal(), nn.WithLearnableSigmoidBias(), nn.WithSigmoidAttnBias(-math.Log(seq)))
	if err != nil {
		t.Fatal(err)
	}
	n, rel = sigmoidGradcheck(t, sc, x)
	t.Logf("causal (learnable bias) gradcheck: %d entries, max rel err %.3g", n, rel)
}

// §V16 (3) CAUSAL CORRECTNESS: with WithSigmoidAttnCausal, output row i must
// depend only on input rows ≤ i. Perturbing a strictly-future input row (and,
// separately, a future VALUE) leaves earlier output rows bit-identical; the
// bidirectional layer, by contrast, changes.
func TestSigmoidAttentionCausal(t *testing.T) {
	const dim, heads, seq = 6, 3, 5
	rng := rand.New(rand.NewPCG(41, 13))
	x := randMat(rng, seq, dim)
	causal, err := nn.NewSigmoidAttention(tensor.F64, dim, heads, 9, nn.WithSigmoidAttnCausal())
	if err != nil {
		t.Fatal(err)
	}
	base, err := causal.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	// Perturb the LAST input row (position seq-1): no earlier output may move.
	x2 := randMat(rng, seq, dim)
	for i := range seq {
		for j := range dim {
			if i < seq-1 {
				x2.SetF64(x.AtF64(i, j), i, j)
			}
		}
	}
	pert, err := causal.Forward(backend.NewContext(), x2)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < seq-1; i++ {
		for j := range dim {
			if d := math.Abs(base.AtF64(i, j) - pert.AtF64(i, j)); d > 1e-12 {
				t.Errorf("causal leak: row %d col %d moved %.3g when only a future input changed", i, j, d)
			}
		}
	}
	// Sanity: the bidirectional layer DOES see the future — earlier rows change.
	bidir, err := nn.NewSigmoidAttention(tensor.F64, dim, heads, 9)
	if err != nil {
		t.Fatal(err)
	}
	b1, _ := bidir.Forward(backend.NewContext(), x)
	b2, _ := bidir.Forward(backend.NewContext(), x2)
	var moved bool
	for j := range dim {
		if math.Abs(b1.AtF64(0, j)-b2.AtF64(0, j)) > 1e-9 {
			moved = true
		}
	}
	if !moved {
		t.Error("bidirectional row 0 unchanged by a future-row edit — mask leaked the other way")
	}
}

// §V16 (6) SANITY: SigmoidAttention.Forward is deterministic — same weights,
// same input, bit-identical output across calls.
func TestSigmoidAttentionDeterminism(t *testing.T) {
	const dim, heads, seq = 8, 2, 4
	rng := rand.New(rand.NewPCG(2, 3))
	x := randMat(rng, seq, dim)
	s, err := nn.NewSigmoidAttention(tensor.F64, dim, heads, 7, nn.WithSigmoidAttnCausal(), nn.WithLearnableSigmoidBias())
	if err != nil {
		t.Fatal(err)
	}
	a, err := s.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	b, err := s.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range seq {
		for j := range dim {
			if a.AtF64(i, j) != b.AtF64(i, j) {
				t.Fatalf("non-deterministic at (%d,%d): %v vs %v", i, j, a.AtF64(i, j), b.AtF64(i, j))
			}
		}
	}
}

// §V16 (default bias): with no bias option the per-forward bias is −log(T). Pin
// it by comparing the default layer to one built WithSigmoidAttnBias(−log T) on
// the same weights (seed) — outputs must be bit-identical at that sequence
// length, and DIFFER from a fixed-bias-0 layer.
func TestSigmoidAttentionDefaultBias(t *testing.T) {
	const dim, heads, seq, seed = 8, 2, 5, 31
	rng := rand.New(rand.NewPCG(9, 4))
	x := randMat(rng, seq, dim)
	def, err := nn.NewSigmoidAttention(tensor.F64, dim, heads, seed)
	if err != nil {
		t.Fatal(err)
	}
	pinned, err := nn.NewSigmoidAttention(tensor.F64, dim, heads, seed, nn.WithSigmoidAttnBias(-math.Log(seq)))
	if err != nil {
		t.Fatal(err)
	}
	zero, err := nn.NewSigmoidAttention(tensor.F64, dim, heads, seed, nn.WithSigmoidAttnBias(0))
	if err != nil {
		t.Fatal(err)
	}
	od, _ := def.Forward(backend.NewContext(), x)
	op, _ := pinned.Forward(backend.NewContext(), x)
	oz, _ := zero.Forward(backend.NewContext(), x)
	var maxSame, maxDiff float64
	for i := range seq {
		for j := range dim {
			if d := math.Abs(od.AtF64(i, j) - op.AtF64(i, j)); d > maxSame {
				maxSame = d
			}
			if d := math.Abs(od.AtF64(i, j) - oz.AtF64(i, j)); d > maxDiff {
				maxDiff = d
			}
		}
	}
	if maxSame != 0 {
		t.Errorf("default bias != −log(T): max diff %.3g, want bit-exact 0", maxSame)
	}
	if maxDiff < 1e-6 {
		t.Errorf("default (−log T) indistinguishable from bias 0 (max diff %.3g) — the −log(n) term did nothing", maxDiff)
	}
	t.Logf("default bias = −log(T): matches pinned @0, differs from bias-0 by %.3g", maxDiff)
}

// §V16 (5) THE VALUE: a tiny causal char-LM whose only mixing layer is a
// SigmoidAttention block (learnable bias) must LEARN — its cross-entropy at
// least HALVES from the uniform-init start over a few hundred steps. This proves
// sigmoid attention is a working, trainable drop-in, not just a correct forward.
func TestSigmoidAttentionLearnsToyTask(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a small sigmoid-attention LM; skipped in -short")
	}
	// Deterministic grammar corpus ("SUBJ VERB OBJ. ") over a small vocab.
	rng := rand.New(rand.NewPCG(2, 0x51a))
	subjects := []string{"the cat", "a dog"}
	verbs := []string{" sees ", " fears "}
	objects := []string{"a star", "the sun"}
	var text []byte
	for range 300 {
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
	vocab := len(seen)
	toks := make([]int, len(text))
	for i, b := range text {
		toks[i] = seen[b]
	}

	const (
		dim    = 32
		window = 48
		steps  = 300
	)
	emb := tensor.New(tensor.F64, tensor.Shape{vocab, dim})
	nn.XavierUniform(emb, vocab, dim, 1)
	attn, err := nn.NewSigmoidAttention(tensor.F64, dim, 4, 5,
		nn.WithSigmoidAttnCausal(), nn.WithLearnableSigmoidBias(), nn.WithSigmoidAttnBias(-math.Log(window)))
	if err != nil {
		t.Fatal(err)
	}
	norm := nn.NewRMSNorm(tensor.F64, dim)
	finalNorm := nn.NewRMSNorm(tensor.F64, dim)
	head := nn.NewLinear(tensor.F64, dim, vocab, 2)
	params := []*tensor.Tensor{emb}
	params = append(params, norm.Params()...)
	params = append(params, attn.Params()...)
	params = append(params, finalNorm.Params()...)
	params = append(params, head.Params()...)

	onehot := func(seq []int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{len(seq), vocab})
		for i, tk := range seq {
			x.SetF64(1, i, tk)
		}
		return x
	}
	forward := func(ctx *backend.Context, seq []int) (*tensor.Tensor, error) {
		x, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{onehot(seq), emb}, nil)
		if err != nil {
			return nil, err
		}
		h := x[0]
		n, err := norm.Forward(ctx, h)
		if err != nil {
			return nil, err
		}
		a, err := attn.Forward(ctx, n)
		if err != nil {
			return nil, err
		}
		sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{h, a}, nil) // residual
		if err != nil {
			return nil, err
		}
		fn, err := finalNorm.Forward(ctx, sum[0])
		if err != nil {
			return nil, err
		}
		return head.Forward(ctx, fn)
	}

	opt := nn.NewAdamW(params, 3e-3, 0.01)
	targets := tensor.New(tensor.F64, tensor.Shape{window})
	var first, last float64
	for step := range steps {
		start := (step * 89) % (len(toks) - window - 1)
		seq := toks[start : start+window]
		for i := range window {
			targets.SetF64(float64(toks[start+1+i]), i)
		}
		tape := autograd.NewTape()
		logits, err := forward(tape.Context(), seq)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(tape.Context(), logits, targets)
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
		clipped, _ := nn.ClipGradNorm(params, tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("sigmoid-attention char-LM: CE %.3f → %.3f (uniform = %.3f)", first, last, math.Log(float64(vocab)))
	if last >= first*0.5 {
		t.Fatalf("sigmoid-attention LM did not train (%.3f → %.3f)", first, last)
	}
}

func ExampleSigmoidAttention() {
	// A tiny causal 2-head sigmoid-attention layer: each attention weight is an
	// element-wise σ(score + bias) — NO softmax normalization across keys. The
	// default bias is −log(T) (T = sequence length), the paper's principled
	// scale-stabilizing choice (arXiv:2409.04431).
	s, _ := nn.NewSigmoidAttention(tensor.F64, 4, 2, 1, nn.WithSigmoidAttnCausal())
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range 3 {
		x.SetF64(1, i, i) // arbitrary non-zero input
	}
	out, _ := s.Forward(backend.NewContext(), x)
	fmt.Printf("%v params=%d\n", out.Shape(), len(s.Params()))
	// Output: (3, 4) params=8
}
