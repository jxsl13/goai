package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// selExec runs a single-output op or fails the test — a terse helper for the
// hand-built reference computations below.
func selExec(t *testing.T, ctx *backend.Context, op backend.Op, at backend.Attrs, in ...*tensor.Tensor) *tensor.Tensor {
	t.Helper()
	out, err := backend.Execute(ctx, op, in, at)
	if err != nil {
		t.Fatalf("op %v: %v", op, err)
	}
	return out[0]
}

// selRandMat builds a random [r,c] F64 matrix in [-1,1).
func selRandMat(rng *rand.Rand, r, c int) *tensor.Tensor {
	m := tensor.New(tensor.F64, tensor.Shape{r, c})
	for i := range r {
		for j := range c {
			m.SetF64(2*rng.Float64()-1, i, j)
		}
	}
	return m
}

// selSumAll reduces every element of x to a scalar through ctx (differentiable).
func selSumAll(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	axes := make([]int, x.Ndim())
	for i := range axes {
		axes[i] = i
	}
	o, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{x}, backend.ReduceAttrs{Axes: axes})
	if err != nil {
		return nil, err
	}
	return o[0], nil
}

// setIdentity overwrites l.W with the identity and zeros l.B, turning a Linear
// into a pass-through so a test can inject known head-0 logits.
func selSetIdentity(l *Linear) {
	n := l.W.Shape()[0]
	for i := range n {
		for j := range l.W.Shape()[1] {
			v := 0.0
			if i == j {
				v = 1
			}
			l.W.SetF64(v, i, j)
		}
	}
	for j := range l.B.Shape()[0] {
		l.B.SetF64(0, j)
	}
}

// §V16 tier-1: SelectiveAttention maps [T,dim]→[T,dim] and exposes every
// learnable tensor: 4 Linears × (W,B) = 8 (Q, K, V, O projections). Selection
// adds no parameters.
func TestSelectiveAttentionShapes(t *testing.T) {
	const dim, heads, seq = 8, 2, 5
	s, err := NewSelectiveAttention(tensor.F64, dim, heads, 7)
	if err != nil {
		t.Fatal(err)
	}
	if s.HeadDim != dim/heads {
		t.Errorf("HeadDim=%d, want %d", s.HeadDim, dim/heads)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := selRandMat(rng, seq, dim)
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
}

// §V16: the constructor rejects a dim not divisible by heads and non-positive
// dim/heads; Forward rejects a wrong input width and a wrong rank.
func TestSelectiveAttentionErrors(t *testing.T) {
	if _, err := NewSelectiveAttention(tensor.F64, 7, 2, 1); err == nil {
		t.Error("dim%heads != 0: want error")
	}
	if _, err := NewSelectiveAttention(tensor.F64, 0, 1, 1); err == nil {
		t.Error("dim=0: want error")
	}
	if _, err := NewSelectiveAttention(tensor.F64, 8, 0, 1); err == nil {
		t.Error("heads=0: want error")
	}
	if _, err := NewSelectiveAttention(tensor.F64, 8, -2, 1); err == nil {
		t.Error("negative heads: want error")
	}
	s, _ := NewSelectiveAttention(tensor.F64, 8, 2, 1)
	if _, err := s.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 4})); err == nil {
		t.Error("wrong input width: want error")
	}
	if _, err := s.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 8})); err == nil {
		t.Error("rank-3 input: want error")
	}
}

// selStandardRef computes ordinary causal multi-head softmax attention from s's
// OWN projections (NO selection term) — the baseline the disabled layer must
// reduce to.
func selStandardRef(t *testing.T, s *SelectiveAttention, x *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	ctx := backend.NewContext()
	seq := x.Shape()[0]
	q := selExec(t, ctx, backend.OpMatMul, nil, x, s.QProj.W)
	q = selExec(t, ctx, backend.OpAddBias, nil, q, s.QProj.B)
	k := selExec(t, ctx, backend.OpMatMul, nil, x, s.KProj.W)
	k = selExec(t, ctx, backend.OpAddBias, nil, k, s.KProj.B)
	v := selExec(t, ctx, backend.OpMatMul, nil, x, s.VProj.W)
	v = selExec(t, ctx, backend.OpAddBias, nil, v, s.VProj.B)
	inv := scalarTensor(x.Dtype(), 1/math.Sqrt(float64(s.HeadDim)))
	mask := qkCausalMask(x.Dtype(), seq, seq)
	heads := make([]*tensor.Tensor, s.Heads)
	for h := range s.Heads {
		lo, hi := h*s.HeadDim, (h+1)*s.HeadDim
		qh := selExec(t, ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, q)
		kh := selExec(t, ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, k)
		vh := selExec(t, ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, v)
		khT := selExec(t, ctx, backend.OpTranspose, nil, kh)
		sc := selExec(t, ctx, backend.OpMatMul, nil, qh, khT)
		sc = selExec(t, ctx, backend.OpMul, nil, sc, inv)
		sc = selExec(t, ctx, backend.OpAdd, nil, sc, mask)
		w := selExec(t, ctx, backend.OpSoftmax, nil, sc)
		heads[h] = selExec(t, ctx, backend.OpMatMul, nil, w, vh)
	}
	cat := selExec(t, ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...)
	out := selExec(t, ctx, backend.OpMatMul, nil, cat, s.OProj.W)
	return selExec(t, ctx, backend.OpAddBias, nil, out, s.OProj.B)
}

// §V16.1 COLLAPSE: with selection disabled (F scaled to 0) the layer is
// bit-identical to standard causal softmax attention — pinning that selective
// attention adds ONLY the −F term.
func TestSelectiveCollapseToStandard(t *testing.T) {
	const dim, heads, seq = 8, 4, 6
	rng := rand.New(rand.NewPCG(11, 22))
	x := selRandMat(rng, seq, dim)
	s, err := NewSelectiveAttention(tensor.F64, dim, heads, 5, WithSelectionDisabled())
	if err != nil {
		t.Fatal(err)
	}
	got, err := s.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	want := selStandardRef(t, s, x)
	var maxAbs float64
	for i := range seq {
		for j := range dim {
			if d := math.Abs(got.AtF64(i, j) - want.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	if maxAbs > 1e-10 {
		t.Fatalf("disabled layer differs from standard attention: max abs %.3g", maxAbs)
	}
	t.Logf("collapse: disabled ≡ standard attention, max abs diff %.3g", maxAbs)

	// Sanity: the ENABLED layer actually differs (selection does something).
	on, err := NewSelectiveAttention(tensor.F64, dim, heads, 5) // same seed, selection ON
	if err != nil {
		t.Fatal(err)
	}
	onOut, _ := on.Forward(backend.NewContext(), x)
	var moved bool
	for i := range seq {
		for j := range dim {
			if math.Abs(onOut.AtF64(i, j)-want.AtF64(i, j)) > 1e-6 {
				moved = true
			}
		}
	}
	if !moved {
		t.Error("selection ON produced the standard-attention output — the −F term did nothing")
	}
}

// §V16.3 CUMSUM: F = cumsum(S)−S over the query axis (axis 0) equals the
// strictly-lower causal cumulative sum, checked against a hand-written example.
func TestSelectiveCumsumHandExample(t *testing.T) {
	ctx := backend.NewContext()
	// An arbitrary 4×4 selection-score matrix S.
	sVals := [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
		{13, 14, 15, 16},
	}
	S := tensor.New(tensor.F64, tensor.Shape{4, 4})
	for i := range 4 {
		for j := range 4 {
			S.SetF64(sVals[i][j], i, j)
		}
	}
	cum := selExec(t, ctx, backend.OpCumsum, backend.CumsumAttrs{Axis: 0}, S)
	F := selExec(t, ctx, backend.OpSub, nil, cum, S) // strictly-below cumsum
	// Hand F[i,j] = Σ_{i'<i} S[i',j].
	want := [][]float64{
		{0, 0, 0, 0},     // no earlier query
		{1, 2, 3, 4},     // S row0
		{6, 8, 10, 12},   // S row0+row1
		{15, 18, 21, 24}, // rows 0..2
	}
	for i := range 4 {
		for j := range 4 {
			if d := math.Abs(F.AtF64(i, j) - want[i][j]); d > 1e-10 {
				t.Errorf("F[%d,%d]=%v want %v", i, j, F.AtF64(i, j), want[i][j])
			}
		}
	}
}

// §V16.3 (end-to-end): the layer's internal selection bias F equals the
// documented pipeline S=strictlower(ReLU(L₀)), F=cumsum(S)−S recomputed from the
// same head-0 logits — tying the wired forward to the formula, with identity
// projections so L₀ is exactly (x·xᵀ)/√d.
func TestSelectiveBiasMatchesFormula(t *testing.T) {
	const dim, seq = 5, 6
	s, err := NewSelectiveAttention(tensor.F64, dim, 1, 3) // 1 head → head 0 IS the whole logit
	if err != nil {
		t.Fatal(err)
	}
	selSetIdentity(s.QProj)
	selSetIdentity(s.KProj)
	rng := rand.New(rand.NewPCG(7, 9))
	x := selRandMat(rng, seq, dim)
	_, _, fBias, err := s.forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	// Recompute F by hand from L₀ = (x·xᵀ)/√dim.
	inv := 1 / math.Sqrt(float64(dim))
	l0 := make([][]float64, seq)
	for i := range seq {
		l0[i] = make([]float64, seq)
		for j := range seq {
			var dot float64
			for c := range dim {
				dot += x.AtF64(i, c) * x.AtF64(j, c)
			}
			l0[i][j] = dot * inv
		}
	}
	want := tensor.New(tensor.F64, tensor.Shape{seq, seq})
	for i := range seq {
		for j := range seq {
			var acc float64
			for ip := 0; ip < i; ip++ { // strictly earlier queries
				if j < ip { // strictly-lower mask on S (j<ip), diagonal/future excluded
					acc += math.Max(0, l0[ip][j]) // ReLU
				}
			}
			want.SetF64(acc, i, j)
		}
	}
	for i := range seq {
		for j := range seq {
			if d := math.Abs(fBias.AtF64(i, j) - want.AtF64(i, j)); d > 1e-10 {
				t.Errorf("fBias[%d,%d]=%v want %v", i, j, fBias.AtF64(i, j), want.AtF64(i, j))
			}
		}
	}
}

// §V16.2 MASKING + CAUSALITY. With identity Q/K projections (head 0 logit =
// x·xᵀ/√d) a crafted early query strongly selects one key out; later queries'
// attention weight on that key must collapse toward 0 relative to the
// selection-off baseline. And F is causal: perturbing the input at query row r
// leaves the selection bias for rows ≤ r bit-unchanged.
func TestSelectiveMaskingAndCausality(t *testing.T) {
	const dim, seq = 4, 6
	// Craft x so query 1 has a HUGE dot with key 0 (strong selection of key 0),
	// while the other tokens are mild and non-aligned with key 0.
	x := tensor.New(tensor.F64, tensor.Shape{seq, dim})
	// key 0 lives along axis 0; query 1 aligns strongly with it.
	x.SetF64(6, 0, 0) // key 0 direction (large)
	x.SetF64(6, 1, 0) // query 1 aligns with key 0 ⇒ L₀[1,0] large ⇒ S[1,0] large
	for i := 2; i < seq; i++ {
		x.SetF64(1, i, i%dim) // later tokens: small, spread across other axes
	}

	build := func(disabled bool) *SelectiveAttention {
		var opts []SelectiveAttentionOption
		if disabled {
			opts = append(opts, WithSelectionDisabled())
		}
		s, err := NewSelectiveAttention(tensor.F64, dim, 1, 4, opts...)
		if err != nil {
			t.Fatal(err)
		}
		selSetIdentity(s.QProj)
		selSetIdentity(s.KProj)
		return s
	}
	on, off := build(false), build(true)
	_, wOn, fBias, err := on.forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	_, wOff, _, err := off.forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	// Selection must actually fire: F on key 0 for a late query is large.
	if fBias.AtF64(seq-1, 0) < 1 {
		t.Fatalf("selection did not accumulate on key 0: F[last,0]=%.3g", fBias.AtF64(seq-1, 0))
	}
	// Later queries (i≥2) put far LESS weight on key 0 with selection ON, and that
	// weight is driven toward ~0.
	for i := 2; i < seq; i++ {
		on0, off0 := wOn[0].AtF64(i, 0), wOff[0].AtF64(i, 0)
		if on0 >= off0 {
			t.Errorf("query %d: selection did not reduce weight on key 0 (on=%.4g off=%.4g)", i, on0, off0)
		}
		if on0 > 1e-3 {
			t.Errorf("query %d: key 0 not masked out, weight still %.4g", i, on0)
		}
	}

	// CAUSALITY: perturb the input at row r; F for rows ≤ r is unchanged.
	const r = seq - 2
	x2 := tensor.New(tensor.F64, tensor.Shape{seq, dim})
	for i := range seq {
		for j := range dim {
			x2.SetF64(x.AtF64(i, j), i, j)
		}
	}
	for j := range dim {
		x2.SetF64(x.AtF64(r, j)+0.7, r, j) // change ONLY row r
	}
	_, _, fBias2, err := on.forward(backend.NewContext(), x2)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i <= r; i++ {
		for j := range seq {
			if d := math.Abs(fBias.AtF64(i, j) - fBias2.AtF64(i, j)); d > 1e-10 {
				t.Errorf("causality violated: F[%d,%d] moved %.3g when only row %d changed", i, j, d, r)
			}
		}
	}
	// Sanity: a LATER row of F did move (the perturbation is not inert).
	var moved bool
	for j := range seq {
		if math.Abs(fBias.AtF64(seq-1, j)-fBias2.AtF64(seq-1, j)) > 1e-9 {
			moved = true
		}
	}
	if !moved {
		t.Error("perturbing row r left all later F rows unchanged — test is vacuous")
	}
}

// selGradcheck compares every analytic gradient of every parameter against
// central finite differences of the summed forward output (F64 only). Returns
// the entry count and max relative error.
func selGradcheck(t *testing.T, s *SelectiveAttention, x *tensor.Tensor) (int, float64) {
	t.Helper()
	const h = 1e-6
	tape := autograd.NewTape()
	out, err := s.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := selSumAll(tape.Context(), out)
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
	check("QProj.W", s.QProj.W)
	check("QProj.B", s.QProj.B)
	check("KProj.W", s.KProj.W)
	check("KProj.B", s.KProj.B)
	check("VProj.W", s.VProj.W)
	check("VProj.B", s.VProj.B)
	check("OProj.W", s.OProj.W)
	check("OProj.B", s.OProj.B)
	return entries, maxRel
}

// §V16.4 GRADCHECK: central finite differences confirm gradients reach EVERY
// parameter — including the Q/K of head 0 that feed the selection path (ReLU →
// strict-lower mask → cumsum → −F).
func TestSelectiveAttentionGradcheck(t *testing.T) {
	const dim, heads, seq = 4, 2, 4
	rng := rand.New(rand.NewPCG(29, 7))
	x := selRandMat(rng, seq, dim)
	s, err := NewSelectiveAttention(tensor.F64, dim, heads, 3) // selection ON
	if err != nil {
		t.Fatal(err)
	}
	n, rel := selGradcheck(t, s, x)
	t.Logf("selective gradcheck: %d entries, max rel err %.3g", n, rel)
	if rel > 1e-4 {
		t.Errorf("max rel err %.3g exceeds 1e-4", rel)
	}
}

// §V16.6 sanity: Forward is deterministic — same weights, same input,
// bit-identical output across calls.
func TestSelectiveAttentionDeterminism(t *testing.T) {
	const dim, heads, seq = 8, 2, 4
	rng := rand.New(rand.NewPCG(2, 3))
	x := selRandMat(rng, seq, dim)
	s, err := NewSelectiveAttention(tensor.F64, dim, heads, 7)
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

// §V16.5 THE VALUE: a small causal "latest-value recall" task with distractors.
// Each sequence is a stream over a tiny vocab: token 0 is a distractor (noise),
// tokens 1..V are values; the target at every position is the MOST RECENT value
// seen so far. Later values should mask earlier stale ones — the paper's setting.
// We train an embed → SelectiveAttention → head model for a few hundred steps and
// report the selective loss against an identical selection-OFF baseline (same
// init, same data). The invariant is that loss DECREASES; beating the baseline is
// reported as the headline result.
func TestSelectiveHelpsRecallTask(t *testing.T) {
	if testing.Short() {
		t.Skip("training loop")
	}
	const (
		vocab = 4 // 0 = noise/distractor, 1..3 = values
		dim   = 16
		heads = 2
		seq   = 10
		nSeq  = 24
		steps = 300
	)
	rng := rand.New(rand.NewPCG(2024, 703))

	// Fixed dataset: one-hot inputs [seq,vocab] and integer targets [seq].
	inputs := make([]*tensor.Tensor, nSeq)
	targets := make([]*tensor.Tensor, nSeq)
	for n := range nSeq {
		oh := tensor.New(tensor.F64, tensor.Shape{seq, vocab})
		tg := tensor.New(tensor.F64, tensor.Shape{seq})
		last := 0
		for i := range seq {
			var tok int
			if rng.Float64() < 0.45 {
				tok = 0 // distractor
			} else {
				tok = 1 + rng.IntN(vocab-1) // a value
				last = tok
			}
			oh.SetF64(1, i, tok)
			tg.SetF64(float64(last), i)
		}
		inputs[n], targets[n] = oh, tg
	}

	// Train one configuration; returns (firstLoss, lastLoss).
	train := func(disabled bool) (float64, float64) {
		embed := NewLinear(tensor.F64, vocab, dim, 100)
		head := NewLinear(tensor.F64, dim, vocab, 200)
		var opts []SelectiveAttentionOption
		if disabled {
			opts = append(opts, WithSelectionDisabled())
		}
		attn, err := NewSelectiveAttention(tensor.F64, dim, heads, 300, opts...)
		if err != nil {
			t.Fatal(err)
		}
		var params []*tensor.Tensor
		params = append(params, embed.Params()...)
		params = append(params, attn.Params()...)
		params = append(params, head.Params()...)
		opt := NewAdamW(params, 5e-3, 0.0)

		var first, last float64
		for step := range steps {
			tape := autograd.NewTape()
			ctx := tape.Context()
			logitsRows := make([]*tensor.Tensor, nSeq)
			for n := range nSeq {
				e, err := embed.Forward(ctx, inputs[n])
				if err != nil {
					t.Fatal(err)
				}
				a, err := attn.Forward(ctx, e)
				if err != nil {
					t.Fatal(err)
				}
				lg, err := head.Forward(ctx, a)
				if err != nil {
					t.Fatal(err)
				}
				logitsRows[n] = lg
			}
			allLogits, err := backend.Execute(ctx, backend.OpConcat, logitsRows, backend.ConcatAttrs{Axis: 0})
			if err != nil {
				t.Fatal(err)
			}
			allTargets, err := backend.Execute(ctx, backend.OpConcat, targets, backend.ConcatAttrs{Axis: 0})
			if err != nil {
				t.Fatal(err)
			}
			loss, err := CrossEntropy(ctx, allLogits[0], allTargets[0])
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
			if err := opt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
		}
		return first, last
	}

	selFirst, selLast := train(false)
	baseFirst, baseLast := train(true)
	t.Logf("selective: %.4f → %.4f", selFirst, selLast)
	t.Logf("baseline : %.4f → %.4f", baseFirst, baseLast)
	if selLast >= selFirst {
		t.Errorf("selective loss did not decrease: %.4f → %.4f", selFirst, selLast)
	}
	if selLast < baseLast {
		t.Logf("selective BEATS the selection-off baseline (%.4f < %.4f)", selLast, baseLast)
	} else {
		t.Logf("selective did not beat baseline this run (%.4f vs %.4f)", selLast, baseLast)
	}
}

func ExampleSelectiveAttention() {
	// A tiny selective causal 2-head attention layer. Selection is ON by default:
	// each token can vote (via head 0's rectified logits, accumulated over earlier
	// queries) to mask an earlier key out of future attention (arXiv:2410.02703).
	s, _ := NewSelectiveAttention(tensor.F64, 4, 2, 1)
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range 3 {
		x.SetF64(1, i, i) // arbitrary non-zero input
	}
	out, _ := s.Forward(backend.NewContext(), x)
	fmt.Printf("%v params=%d\n", out.Shape(), len(s.Params()))
	// Output: (3, 4) params=8
}
