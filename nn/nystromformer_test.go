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

// nystrom-prefixed test helpers (this is an internal `package nn` test, so it
// cannot see the `nn_test` helpers randMat/maxAbsDiff/sumAll — §avoid collisions).
func nystromRandMat(rng *rand.Rand, r, c int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{r, c})
	for i := range r {
		for j := range c {
			t.SetF64(rng.NormFloat64(), i, j)
		}
	}
	return t
}

func nystromMaxAbsDiff(a, b *tensor.Tensor) float64 {
	var m float64
	for i := range a.Numel() {
		coord := tensor.Unravel(i, a.Shape())
		if d := math.Abs(a.AtF64(coord...) - b.AtF64(coord...)); d > m {
			m = d
		}
	}
	return m
}

func nystromSumAll(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	axes := make([]int, x.Ndim())
	for i := range axes {
		axes[i] = i
	}
	out, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{x}, backend.ReduceAttrs{Axes: axes})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// nystromExec is a single-output backend.Execute adapter matching execFn, so the
// white-box tests can drive the free-standing pseudo-inverse and reference math.
func nystromExec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// nystromRefOutput recomputes EXACT bidirectional softmax attention through the
// layer's own Q/K/V/O projections: per head S=softmax(QₕKₕᵀ/√d), Oₕ=S·Vₕ, concat,
// then OProj. This is the O(L²) ground truth the Nyström forward approximates.
func nystromRefOutput(t *testing.T, n *NystromAttention, x *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	ctx := backend.NewContext()
	must := func(v *tensor.Tensor, err error) *tensor.Tensor {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	q := must(n.QProj.Forward(ctx, x))
	k := must(n.KProj.Forward(ctx, x))
	v := must(n.VProj.Forward(ctx, x))
	invSqrtD := scalarTensor(x.Dtype(), 1/math.Sqrt(float64(n.HeadDim)))
	heads := make([]*tensor.Tensor, n.Heads)
	for h := range n.Heads {
		lo, hi := h*n.HeadDim, (h+1)*n.HeadDim
		sl := func(m *tensor.Tensor) *tensor.Tensor {
			return must(nystromExec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, m))
		}
		qh, kh, vh := sl(q), sl(k), sl(v)
		khT := must(nystromExec(ctx, backend.OpTranspose, nil, kh))
		sc := must(nystromExec(ctx, backend.OpMatMul, nil, qh, khT))
		sc = must(nystromExec(ctx, backend.OpMul, nil, sc, invSqrtD))
		w := must(nystromExec(ctx, backend.OpSoftmax, nil, sc))
		heads[h] = must(nystromExec(ctx, backend.OpMatMul, nil, w, vh))
	}
	concat := must(nystromExec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...))
	return must(n.OProj.Forward(ctx, concat))
}

// nystromRelDiff returns the relative Frobenius error ‖a−b‖_F / ‖b‖_F.
func nystromRelDiff(a, b *tensor.Tensor) float64 {
	var num, den float64
	for i := range a.Numel() {
		coord := tensor.Unravel(i, a.Shape())
		d := a.AtF64(coord...) - b.AtF64(coord...)
		num += d * d
		bv := b.AtF64(coord...)
		den += bv * bv
	}
	return math.Sqrt(num) / math.Sqrt(den)
}

// §V16 (1) FULL-LANDMARK COLLAPSE — the anchor. With m=L and identity landmarks
// (Q̃=Q, K̃=K) the three softmax blocks are all softmax(QKᵀ/√d), so
// F₁·Ã⁺·F₃ = softmax(QKᵀ)·pinv(softmax(QKᵀ))·softmax(QKᵀ) = softmax(QKᵀ) and the
// layer equals exact bidirectional softmax attention. The residual is bounded only
// by the iterative pseudo-inverse; more iters tighten it.
func TestNystromFullLandmarkCollapse(t *testing.T) {
	const dim, heads, L = 8, 2, 6
	rng := rand.New(rand.NewPCG(11, 22))
	x := nystromRandMat(rng, L, dim)

	for _, iters := range []int{6, 20, 40} {
		n, err := NewNystromAttention(tensor.F64, dim, L,
			WithNystromHeads(heads), WithNystromSeed(7),
			WithNystromIdentityLandmarks(), WithNystromPinvIters(iters))
		if err != nil {
			t.Fatal(err)
		}
		got, err := n.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		ref := nystromRefOutput(t, n, x)
		d := nystromMaxAbsDiff(got, ref)
		t.Logf("full-landmark collapse: pinvIters=%d  max|nyström − softmax-attn| = %.3g", iters, d)
		if iters >= 40 && d > 1e-4 {
			t.Errorf("collapse residual %.3g > 1e-4 at %d pinv iters", d, iters)
		}
	}
}

// §V16 (2) PSEUDO-INVERSE correctness — the iterative Z satisfies the defining
// Moore–Penrose identities AZA≈A and ZAZ≈Z on a realistic row-stochastic kernel
// (softmax of a random symmetric matrix, like the layer's Ã), tighter with iters.
func TestNystromPinvProperties(t *testing.T) {
	const m = 5
	rng := rand.New(rand.NewPCG(3, 9))
	// A = softmax(sym) — row-stochastic and well-conditioned, mimicking Ã.
	sym := tensor.New(tensor.F64, tensor.Shape{m, m})
	for i := range m {
		for j := i; j < m; j++ {
			v := rng.NormFloat64()
			sym.SetF64(v, i, j)
			sym.SetF64(v, j, i)
		}
	}
	ctx := backend.NewContext()
	aOut, err := backend.Execute(ctx, backend.OpSoftmax, []*tensor.Tensor{sym}, nil)
	if err != nil {
		t.Fatal(err)
	}
	a := aOut[0]

	prev := math.Inf(1)
	for _, iters := range []int{2, 6, 16, 32} {
		z, err := nystromPinv(ctx, nystromExec, a, iters)
		if err != nil {
			t.Fatal(err)
		}
		az := mustMat(t, ctx, backend.OpMatMul, a, z)
		aza := mustMat(t, ctx, backend.OpMatMul, az, a)
		za := mustMat(t, ctx, backend.OpMatMul, z, a)
		zaz := mustMat(t, ctx, backend.OpMatMul, za, z)
		e1 := nystromMaxAbsDiff(aza, a)
		e2 := nystromMaxAbsDiff(zaz, z)
		t.Logf("pinv iters=%d: max|AZA−A|=%.3g  max|ZAZ−Z|=%.3g", iters, e1, e2)
		if iters == 32 {
			if e1 > 1e-6 {
				t.Errorf("AZA−A residual %.3g > 1e-6 at 32 iters", e1)
			}
			if e2 > 1e-6 {
				t.Errorf("ZAZ−Z residual %.3g > 1e-6 at 32 iters", e2)
			}
		}
		if e1 > prev+1e-9 {
			t.Errorf("AZA−A residual grew with iters: %.3g → %.3g", prev, e1)
		}
		prev = e1
	}
}

func mustMat(t *testing.T, ctx *backend.Context, op backend.Op, a, b *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	out, err := backend.Execute(ctx, op, []*tensor.Tensor{a, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return out[0]
}

// §V16 (3) LANDMARK SHAPES / complexity — the three softmax blocks are [L,m],
// [m,m], [m,L]; NONE is the [L,L] full matrix, so m<L genuinely avoids the L²
// softmax.
func TestNystromBlockShapes(t *testing.T) {
	const dim, heads, L, m = 8, 2, 12, 3
	rng := rand.New(rand.NewPCG(5, 6))
	x := nystromRandMat(rng, L, dim)
	n, err := NewNystromAttention(tensor.F64, dim, m, WithNystromHeads(heads), WithNystromSeed(1))
	if err != nil {
		t.Fatal(err)
	}
	_, dbg, err := n.forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	want := func(got tensor.Shape, r, c int, name string) {
		if got.Ndim() != 2 || got[0] != r || got[1] != c {
			t.Errorf("%s block shape = %v, want [%d,%d]", name, got, r, c)
		}
	}
	want(dbg.b1Shape, L, m, "F₁")
	want(dbg.midShape, m, m, "Ã")
	want(dbg.b3Shape, m, L, "F₃")
	if !(m < L) {
		t.Fatalf("test misconfigured: need m<L, got m=%d L=%d", m, L)
	}
	t.Logf("blocks F₁=%v Ã=%v F₃=%v — no [%d,%d] matrix formed (m=%d ≪ L=%d)", dbg.b1Shape, dbg.midShape, dbg.b3Shape, L, L, m, L)
}

// §V16 (4) GRADCHECK — central finite differences confirm gradients reach every
// projection (Wq,Wk,Wv,Wo) through the three softmax blocks AND the unrolled
// pseudo-inverse.
func TestNystromGradcheck(t *testing.T) {
	const dim, heads, L, m = 4, 2, 4, 2
	const h = 1e-6
	rng := rand.New(rand.NewPCG(31, 17))
	x := nystromRandMat(rng, L, dim)
	n, err := NewNystromAttention(tensor.F64, dim, m, WithNystromHeads(heads), WithNystromSeed(4), WithNystromPinvIters(6))
	if err != nil {
		t.Fatal(err)
	}

	tape := autograd.NewTape()
	out, err := n.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := nystromSumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	sumLoss := func() float64 {
		o, err := n.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var acc float64
		for i := range o.Numel() {
			acc += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return acc
	}

	// The FOUR projection WEIGHTS must be exercised (nonzero grad). The Q/V/O
	// biases also carry gradient, but the KEY bias is a genuine NULL DIRECTION of
	// softmax attention: adding a constant vector to every key shifts all logits in
	// a row by the same Qᵢ·b, which cancels in the softmax over the key axis (true
	// in all three Nyström blocks). So ∂O/∂b_k = 0 exactly — the finite difference
	// confirms it (its numeric grad is ~0 too), and we assert it rather than flag
	// it as "unexercised".
	weight := map[*tensor.Tensor]bool{n.QProj.W: true, n.KProj.W: true, n.VProj.W: true, n.OProj.W: true}
	var entries int
	var maxRel float64
	for pi, p := range n.Params() {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("param %d: nil grad — no gradient reached this projection", pi)
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
			if rel > 1e-3 {
				t.Errorf("param %d grad%v: numeric %.8g vs analytic %.8g (rel %.3g)", pi, coord, num, ana, rel)
			}
		}
		if weight[p] && !nz {
			t.Errorf("weight param %d: all analytic grads zero — projection not exercised", pi)
		}
		if p == n.KProj.B && nz {
			t.Errorf("key bias grad is nonzero — expected the softmax null direction (∂O/∂b_k=0)")
		}
	}
	t.Logf("gradcheck: %d entries, max rel err %.3g (6 pinv iters); key bias confirmed a softmax null direction", entries, maxRel)
}

// §V16 (5a) THE VALUE — APPROXIMATION improves as m→L: relative error of the
// Nyström output vs exact softmax attention shrinks with more landmarks.
func TestNystromApproximationImprovesWithM(t *testing.T) {
	const dim, heads, L = 16, 2, 64
	rng := rand.New(rand.NewPCG(101, 202))
	x := nystromRandMat(rng, L, dim)

	var errs []float64
	ms := []int{L / 8, L / 4, L / 2}
	for _, m := range ms {
		n, err := NewNystromAttention(tensor.F64, dim, m, WithNystromHeads(heads), WithNystromSeed(2), WithNystromPinvIters(6))
		if err != nil {
			t.Fatal(err)
		}
		got, err := n.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		ref := nystromRefOutput(t, n, x)
		e := nystromRelDiff(got, ref)
		errs = append(errs, e)
		t.Logf("approximation: m=%d (L/%d)  relerr vs full attention = %.3g", m, L/m, e)
	}
	if errs[len(errs)-1] >= errs[0] {
		t.Errorf("approximation did not improve m=%d→%d: relerr %.3g → %.3g", ms[0], ms[len(ms)-1], errs[0], errs[len(errs)-1])
	}
}

// §V16 (5b) THE VALUE — LEARNS: a small bidirectional task (every position must
// predict the token sitting at position 0, which only whole-sequence attention can
// gather) trains for a few hundred steps and the loss decreases.
func TestNystromLearns(t *testing.T) {
	const (
		vocab = 5
		dim   = 16
		heads = 2
		seq   = 12
		m     = 4
		nSeq  = 16
		steps = 250
	)
	rng := rand.New(rand.NewPCG(777, 888))
	inputs := make([]*tensor.Tensor, nSeq)
	targets := make([]*tensor.Tensor, nSeq)
	for s := range nSeq {
		oh := tensor.New(tensor.F64, tensor.Shape{seq, vocab})
		tg := tensor.New(tensor.F64, tensor.Shape{seq})
		toks := make([]int, seq)
		for i := range seq {
			toks[i] = rng.IntN(vocab)
			oh.SetF64(1, i, toks[i])
		}
		for i := range seq {
			tg.SetF64(float64(toks[0]), i) // bidirectional: copy position 0 everywhere
		}
		inputs[s], targets[s] = oh, tg
	}

	embed := NewLinear(tensor.F64, vocab, dim, 100)
	head := NewLinear(tensor.F64, dim, vocab, 200)
	attn, err := NewNystromAttention(tensor.F64, dim, m, WithNystromHeads(heads), WithNystromSeed(300))
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
		for s := range nSeq {
			e, err := embed.Forward(ctx, inputs[s])
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
			logitsRows[s] = lg
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
	t.Logf("nyström learns: loss %.4f → %.4f over %d steps", first, last, steps)
	if last >= first {
		t.Errorf("loss did not decrease: %.4f → %.4f", first, last)
	}
}

// §V16 (6) sanity — construction guards, determinism and Forward shape errors.
func TestNystromSanity(t *testing.T) {
	// heads must divide dim.
	if _, err := NewNystromAttention(tensor.F64, 6, 2, WithNystromHeads(4)); err == nil {
		t.Error("expected error: dim 6 not divisible by 4 heads")
	}
	// numLandmarks ≥ 1.
	if _, err := NewNystromAttention(tensor.F64, 8, 0); err == nil {
		t.Error("expected error: numLandmarks 0")
	}
	// dModel > 0.
	if _, err := NewNystromAttention(tensor.F64, 0, 1); err == nil {
		t.Error("expected error: dModel 0")
	}
	// pinvIters ≥ 1.
	if _, err := NewNystromAttention(tensor.F64, 8, 2, WithNystromPinvIters(0)); err == nil {
		t.Error("expected error: pinvIters 0")
	}

	rng := rand.New(rand.NewPCG(44, 55))
	x := nystromRandMat(rng, 10, 8)

	// m > L errors at Forward.
	nBig, _ := NewNystromAttention(tensor.F64, 8, 20, WithNystromHeads(2))
	if _, err := nBig.Forward(backend.NewContext(), x); err == nil {
		t.Error("expected error: numLandmarks 20 > sequence length 10")
	}
	// identity landmarks require L==m.
	nId, _ := NewNystromAttention(tensor.F64, 8, 4, WithNystromIdentityLandmarks())
	if _, err := nId.Forward(backend.NewContext(), x); err == nil {
		t.Error("expected error: identity landmarks with L(10)!=m(4)")
	}
	// wrong feature width.
	nOK, _ := NewNystromAttention(tensor.F64, 8, 4, WithNystromHeads(2))
	bad := nystromRandMat(rng, 10, 7)
	if _, err := nOK.Forward(backend.NewContext(), bad); err == nil {
		t.Error("expected error: x width 7 != dim 8")
	}

	// determinism: same seed → identical output.
	a1, _ := NewNystromAttention(tensor.F64, 8, 4, WithNystromHeads(2), WithNystromSeed(9))
	a2, _ := NewNystromAttention(tensor.F64, 8, 4, WithNystromHeads(2), WithNystromSeed(9))
	o1, err := a1.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	o2, err := a2.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if d := nystromMaxAbsDiff(o1, o2); d != 0 {
		t.Errorf("same seed not deterministic: max diff %.3g", d)
	}
	// output preserves [L,Dim].
	if o1.Shape().Ndim() != 2 || o1.Shape()[0] != 10 || o1.Shape()[1] != 8 {
		t.Errorf("output shape %v, want [10,8]", o1.Shape())
	}
}

func ExampleNystromAttention() {
	// A tiny 2-head Nyström attention over model width 4 with 2 landmarks. Forward
	// approximates the L×L softmax attention with three small [L,m]/[m,m]/[m,L]
	// softmax blocks bridged by an iterative pseudo-inverse — O(L·m), bidirectional.
	n, _ := NewNystromAttention(tensor.F64, 4, 2, WithNystromHeads(2))
	x := tensor.New(tensor.F64, tensor.Shape{6, 4})
	for i := range 6 {
		x.SetF64(1, i, i%4) // arbitrary non-zero input
	}
	out, _ := n.Forward(backend.NewContext(), x)
	fmt.Printf("%v heads=%d landmarks=%d params=%d\n", out.Shape(), n.Heads, n.NumLandmarks, len(n.Params()))
	// Output: (6, 4) heads=2 landmarks=2 params=8
}
