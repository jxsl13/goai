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

// §V16 tier-1: StickBreakingAttention maps [T,dim]→[T,dim] and exposes every
// learnable tensor: 4 Linears × (W,B) = 8, plus one bias scalar per head when the
// bias option is on.
func TestStickBreakingAttentionShapes(t *testing.T) {
	const dim, heads, seq = 8, 2, 5
	s, err := nn.NewStickBreakingAttention(tensor.F64, dim, heads, 7)
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
		t.Errorf("Params count %d, want %d", got, want)
	}
	sb, err := nn.NewStickBreakingAttention(tensor.F64, dim, heads, 7, nn.WithStickBreakingBias())
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(sb.Params()), 8+heads; got != want {
		t.Errorf("Params count with bias %d, want %d", got, want)
	}
}

// §V16.7 sanity: the constructor rejects a dim not divisible by heads and
// non-positive dim/heads; Forward rejects a wrong input width and a wrong rank.
func TestStickBreakingAttentionErrors(t *testing.T) {
	if _, err := nn.NewStickBreakingAttention(tensor.F64, 7, 2, 1); err == nil {
		t.Error("dim%heads != 0: want error")
	}
	if _, err := nn.NewStickBreakingAttention(tensor.F64, 0, 1, 1); err == nil {
		t.Error("dim=0: want error")
	}
	if _, err := nn.NewStickBreakingAttention(tensor.F64, 8, 0, 1); err == nil {
		t.Error("heads=0: want error")
	}
	if _, err := nn.NewStickBreakingAttention(tensor.F64, 8, -2, 1); err == nil {
		t.Error("negative heads: want error")
	}
	s, _ := nn.NewStickBreakingAttention(tensor.F64, 8, 2, 1)
	if _, err := s.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 4})); err == nil {
		t.Error("wrong input width: want error")
	}
	if _, err := s.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 8})); err == nil {
		t.Error("rank-3 input: want error")
	}
}

// sbGradcheck compares every analytic gradient of every parameter of s against
// central finite differences of the summed forward output (F64 only). Returns the
// entry count and max relative error for the report.
func sbGradcheck(t *testing.T, s *nn.StickBreakingAttention, x *tensor.Tensor) (int, float64) {
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
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad — no gradient reached this parameter", name)
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
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
		if !nz {
			t.Errorf("%s: all analytic grads zero — parameter not exercised", name)
		}
	}
	for _, p := range s.Params() {
		check("param", p)
	}
	return entries, maxRel
}

// §V16.5 GRADCHECK (anchor): central finite differences confirm gradients reach
// EVERY parameter — Q/K/V/O projections and the per-head bias — through the
// log-space stick-breaking (softplus/neg/cumsum/exp/mask) and the value
// aggregation.
func TestStickBreakingAttentionGradcheck(t *testing.T) {
	const dim, heads, seq = 4, 2, 4
	rng := rand.New(rand.NewPCG(29, 7))
	x := randMat(rng, seq, dim)
	s, err := nn.NewStickBreakingAttention(tensor.F64, dim, heads, 3)
	if err != nil {
		t.Fatal(err)
	}
	n, rel := sbGradcheck(t, s, x)
	t.Logf("no-bias gradcheck: %d entries, max rel err %.3g", n, rel)

	sb, err := nn.NewStickBreakingAttention(tensor.F64, dim, heads, 5, nn.WithStickBreakingBias())
	if err != nil {
		t.Fatal(err)
	}
	// Nudge the biases off zero so their gradient is exercised on a live value.
	for _, p := range sb.Params()[8:] {
		p.SetF64(0.3, 0, 0)
	}
	n, rel = sbGradcheck(t, sb, x)
	t.Logf("with-bias gradcheck: %d entries, max rel err %.3g", n, rel)
}

// §V16.4 CAUSAL CORRECTNESS (anchor): output row i depends only on input rows ≤ i.
// Perturbing a strictly-future input row leaves earlier output rows bit-identical.
func TestStickBreakingAttentionCausal(t *testing.T) {
	const dim, heads, seq = 6, 3, 5
	rng := rand.New(rand.NewPCG(41, 13))
	x := randMat(rng, seq, dim)
	s, err := nn.NewStickBreakingAttention(tensor.F64, dim, heads, 9)
	if err != nil {
		t.Fatal(err)
	}
	base, err := s.Forward(backend.NewContext(), x)
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
	pert, err := s.Forward(backend.NewContext(), x2)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < seq-1; i++ {
		for j := range dim {
			if d := math.Abs(base.AtF64(i, j) - pert.AtF64(i, j)); d > 1e-12 {
				t.Errorf("causal leak: row %d col %d moved %.3g when only future input changed", i, j, d)
			}
		}
	}
}

// §V16.7 sanity: Forward is deterministic — same weights, same input, bit-identical
// output across calls.
func TestStickBreakingAttentionDeterminism(t *testing.T) {
	const dim, heads, seq = 8, 2, 4
	rng := rand.New(rand.NewPCG(2, 3))
	x := randMat(rng, seq, dim)
	s, err := nn.NewStickBreakingAttention(tensor.F64, dim, heads, 7, nn.WithStickBreakingBias())
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

// sbMSE builds a differentiable mean-squared-error scalar between out and target.
func sbMSE(ctx *backend.Context, out, target *tensor.Tensor) (*tensor.Tensor, error) {
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	diff, err := ex(backend.OpSub, nil, out, target)
	if err != nil {
		return nil, err
	}
	sq, err := ex(backend.OpMul, nil, diff, diff)
	if err != nil {
		return nil, err
	}
	axes := []int{0, 1}
	return ex(backend.OpSum, backend.ReduceAttrs{Axes: axes}, sq)
}

// sbRecencyTarget builds the recency-weighted target y[i] = Σ_{j≤i} w_{i-j} x[j]
// with a fixed geometric decay w_d = β0·(1−β0)^d — EXACTLY the allocation a
// stick-breaking layer with a constant break probability β0 produces. This is the
// pattern stick-breaking's ordering-induced recency can represent WITHOUT any
// position encoding, and that a content-only softmax cannot.
func sbRecencyTarget(x *tensor.Tensor, beta0 float64) *tensor.Tensor {
	t := x.Shape()[0]
	d := x.Shape()[1]
	y := tensor.New(tensor.F64, tensor.Shape{t, d})
	for i := range t {
		for c := range d {
			var acc float64
			for j := 0; j <= i; j++ {
				w := beta0 * math.Pow(1-beta0, float64(i-j))
				acc += w * x.AtF64(j, c)
			}
			y.SetF64(acc, i, c)
		}
	}
	return y
}

// §V16.6 THE VALUE (anchor): a recency task that exploits stick-breaking's
// ordering-induced recency bias and length generalization WITHOUT position
// encoding. The target is a fixed geometric recency-weighted average of past
// tokens (see sbRecencyTarget). Stick-breaking, whose weights ARE an ordered
// geometric allocation, learns it and — because the rule is position-encoding-free
// and identical at every length — generalizes from short training sequences to
// LONGER test sequences. A causal softmax-family baseline (SoftpickAttention),
// which selects keys by content with no notion of order, cannot localize by
// position on content-random inputs and stays high. We report the numbers and
// assert stick-breaking both learns and beats the baseline.
func TestStickBreakingAttentionRecencyValue(t *testing.T) {
	const dim, heads = 8, 2
	const trainLen, testLen = 6, 16
	const steps = 400
	const beta0 = 0.6
	rng := rand.New(rand.NewPCG(2024, 17))

	newBatch := func(seqLen int) (*tensor.Tensor, *tensor.Tensor) {
		x := randMat(rng, seqLen, dim) // pure content, NO positional signal
		return x, sbRecencyTarget(x, beta0)
	}

	// Stick-breaking (with a learnable per-head break bias so it can tune β→β0).
	sb, err := nn.NewStickBreakingAttention(tensor.F64, dim, heads, 3, nn.WithStickBreakingBias())
	if err != nil {
		t.Fatal(err)
	}
	// Softmax-family baseline: causal SoftpickAttention (content-based, no order).
	base, err := nn.NewSoftpickAttention(tensor.F64, dim, heads, 3, nn.WithSoftpickCausal())
	if err != nil {
		t.Fatal(err)
	}

	train := func(fwd func(*backend.Context, *tensor.Tensor) (*tensor.Tensor, error), params []*tensor.Tensor) (float64, float64) {
		opt := nn.NewAdamW(params, 5e-3, 0)
		var first, last float64
		for step := range steps {
			x, y := newBatch(trainLen)
			tape := autograd.NewTape()
			out, err := fwd(tape.Context(), x)
			if err != nil {
				t.Fatal(err)
			}
			loss, err := sbMSE(tape.Context(), out, y)
			if err != nil {
				t.Fatal(err)
			}
			lv := loss.AtF64() / float64(trainLen*dim)
			if step == 0 {
				first = lv
			}
			last = lv
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
		}
		return first, last
	}

	// Held-out evaluation at a LONGER length (length generalization).
	eval := func(fwd func(*backend.Context, *tensor.Tensor) (*tensor.Tensor, error), seqLen int) float64 {
		var acc float64
		const n = 16
		for range n {
			x, y := newBatch(seqLen)
			out, err := fwd(backend.NewContext(), x)
			if err != nil {
				t.Fatal(err)
			}
			for i := range seqLen {
				for c := range dim {
					d := out.AtF64(i, c) - y.AtF64(i, c)
					acc += d * d
				}
			}
		}
		return acc / float64(n*seqLen*dim)
	}

	sbFirst, sbLast := train(sb.Forward, sb.Params())
	baseFirst, baseLast := train(base.Forward, base.Params())

	sbTrainEval := eval(sb.Forward, trainLen)
	sbTestEval := eval(sb.Forward, testLen)
	baseTestEval := eval(base.Forward, testLen)

	t.Logf("stick-breaking train MSE: %.4e → %.4e", sbFirst, sbLast)
	t.Logf("softpick  baseline  MSE: %.4e → %.4e", baseFirst, baseLast)
	t.Logf("stick-breaking eval MSE: len=%d %.4e, len=%d %.4e (length generalization)", trainLen, sbTrainEval, testLen, sbTestEval)
	t.Logf("softpick  baseline eval: len=%d %.4e", testLen, baseTestEval)

	if sbLast >= sbFirst*0.5 {
		t.Errorf("stick-breaking did not learn: MSE %.4e → %.4e (want ≥2× drop)", sbFirst, sbLast)
	}
	if sbTestEval >= baseTestEval {
		t.Errorf("stick-breaking (%.4e) did not beat softmax baseline (%.4e) at len=%d", sbTestEval, baseTestEval, testLen)
	}
	// Length generalization: longer-sequence MSE must stay in the same ballpark as
	// the training length (no positional blow-up).
	if sbTestEval > 5*sbTrainEval+1e-6 {
		t.Errorf("stick-breaking failed to generalize to len=%d: %.4e vs train-len %.4e", testLen, sbTestEval, sbTrainEval)
	}
}

func ExampleStickBreakingAttention() {
	// A tiny stick-breaking attention layer: causal 2-head attention whose weights
	// are a stick-breaking allocation from the most recent key backward, so a query
	// keeps an inherent recency bias with NO position encoding.
	s, _ := nn.NewStickBreakingAttention(tensor.F64, 4, 2, 1)
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range 3 {
		x.SetF64(1, i, i) // arbitrary non-zero input
	}
	out, _ := s.Forward(backend.NewContext(), x)
	fmt.Printf("%v params=%d\n", out.Shape(), len(s.Params()))
	// Output: (3, 4) params=8
}
