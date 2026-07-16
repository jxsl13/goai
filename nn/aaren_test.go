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

// aarenMatmul is a pure-Go row-major matmul C[i,j] = Σ_k A[i,k]·B[k,j], used to
// build an INDEPENDENT softmax-attention reference (no shared op graph with the
// layer under test).
func aarenMatmul(a, b *tensor.Tensor) *tensor.Tensor {
	m, kd := a.Shape()[0], a.Shape()[1]
	n := b.Shape()[1]
	c := tensor.New(tensor.F64, tensor.Shape{m, n})
	for i := range m {
		for j := range n {
			var acc float64
			for k := range kd {
				acc += a.AtF64(i, k) * b.AtF64(k, j)
			}
			c.SetF64(acc, i, j)
		}
	}
	return c
}

// aarenSoftmaxRef computes standard multi-head scaled-dot-product attention in
// pure Go f64 — o_i = softmax_j(q_i·k_j/√d_head)·v_j, causal (j≤i) or full — the
// exact function Aaren claims to reformulate. Independent of the layer's op graph.
func aarenSoftmaxRef(a *nn.Aaren, x *tensor.Tensor, causal bool) *tensor.Tensor {
	t := x.Shape()[0]
	q := aarenMatmul(x, a.Wq)
	k := aarenMatmul(x, a.Wk)
	v := aarenMatmul(x, a.Wv)
	scale := 1 / math.Sqrt(float64(a.HeadDim))
	concat := tensor.New(tensor.F64, tensor.Shape{t, a.DModel})
	for h := range a.Heads {
		base := h * a.HeadDim
		for i := range t {
			hi := i
			if !causal {
				hi = t - 1
			}
			// stable softmax over keys 0..hi
			mmax := math.Inf(-1)
			sc := make([]float64, hi+1)
			for j := 0; j <= hi; j++ {
				var s float64
				for c := range a.HeadDim {
					s += q.AtF64(i, base+c) * k.AtF64(j, base+c)
				}
				s *= scale
				sc[j] = s
				if s > mmax {
					mmax = s
				}
			}
			var denom float64
			for j := 0; j <= hi; j++ {
				sc[j] = math.Exp(sc[j] - mmax)
				denom += sc[j]
			}
			for c := range a.HeadDim {
				var acc float64
				for j := 0; j <= hi; j++ {
					acc += sc[j] * v.AtF64(j, base+c)
				}
				concat.SetF64(acc/denom, i, base+c)
			}
		}
	}
	return aarenMatmul(concat, a.Wo)
}

func aarenRow(x *tensor.Tensor, i int) *tensor.Tensor {
	c := x.Shape()[1]
	r := tensor.New(tensor.F64, tensor.Shape{1, c})
	for j := range c {
		r.SetF64(x.AtF64(i, j), 0, j)
	}
	return r
}

// §V16 (7) SANITY: Aaren maps [T,DModel]→[T,DModel]; Params exposes the four
// projection weights; HeadDim = DModel/Heads.
func TestAarenShapes(t *testing.T) {
	const dim, heads, seq = 8, 2, 5
	a, err := nn.NewAaren(tensor.F64, dim, nn.WithAarenHeads(heads), nn.WithAarenSeed(7))
	if err != nil {
		t.Fatal(err)
	}
	if a.HeadDim != dim/heads {
		t.Errorf("HeadDim=%d, want %d", a.HeadDim, dim/heads)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := randMat(rng, seq, dim)
	out, err := a.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{seq, dim}) {
		t.Fatalf("output shape %v, want [%d %d]", out.Shape(), seq, dim)
	}
	if got, want := len(a.Params()), 4; got != want {
		t.Errorf("Params count %d, want %d (Wq,Wk,Wv,Wo)", got, want)
	}
}

// §V16 (7) SANITY: the constructor rejects a non-positive dim and a dim not
// divisible by heads; Forward rejects a wrong width and a wrong rank.
func TestAarenErrors(t *testing.T) {
	if _, err := nn.NewAaren(tensor.F64, 0); err == nil {
		t.Error("dim=0: want error")
	}
	if _, err := nn.NewAaren(tensor.F64, 7, nn.WithAarenHeads(2)); err == nil {
		t.Error("dim%heads != 0: want error")
	}
	if _, err := nn.NewAaren(tensor.F64, 8, nn.WithAarenHeads(0)); err == nil {
		t.Error("heads=0: want error")
	}
	a, _ := nn.NewAaren(tensor.F64, 8, nn.WithAarenHeads(2))
	if _, err := a.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 4})); err == nil {
		t.Error("wrong input width: want error")
	}
	if _, err := a.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 8})); err == nil {
		t.Error("rank-3 input: want error")
	}
}

// §V16 (1) THE ANCHOR — Aaren's parallel Forward is NUMERICALLY IDENTICAL to
// standard causal (and bidirectional) softmax attention, computed by an
// independent pure-Go reference. This is the paper's core claim: a reformulation,
// not an approximation. Asserted bit-close @1e-10.
func TestAarenEqualsSoftmaxAttention(t *testing.T) {
	const dim, heads, seq = 12, 3, 9
	rng := rand.New(rand.NewPCG(17, 5))
	x := randMat(rng, seq, dim)

	for _, tc := range []struct {
		name   string
		causal bool
		opts   []nn.AarenOption
	}{
		{"causal", true, []nn.AarenOption{nn.WithAarenHeads(heads), nn.WithAarenSeed(3)}},
		{"bidirectional", false, []nn.AarenOption{nn.WithAarenHeads(heads), nn.WithAarenSeed(3), nn.WithAarenBidirectional()}},
	} {
		a, err := nn.NewAaren(tensor.F64, dim, tc.opts...)
		if err != nil {
			t.Fatal(err)
		}
		got, err := a.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		want := aarenSoftmaxRef(a, x, tc.causal)
		if d := maxAbsDiff(got, want); d > 1e-10 {
			t.Errorf("%s: Aaren ≠ softmax attention, max abs diff %.3g (want ≤1e-10)", tc.name, d)
		} else {
			t.Logf("%s: Aaren ≡ softmax attention, max abs diff %.3g", tc.name, d)
		}
	}
}

// §V16 (2) PARALLEL ≡ RECURRENT DUALITY — feeding tokens one at a time through
// StepRecurrent (fixed query = position i, folding keys 0..i) reproduces Forward's
// per-position outputs @1e-10, the RNN-update correctness.
func TestAarenParallelRecurrentDuality(t *testing.T) {
	const dim, heads, seq = 8, 2, 7
	rng := rand.New(rand.NewPCG(23, 11))
	x := randMat(rng, seq, dim)
	a, err := nn.NewAaren(tensor.F64, dim, nn.WithAarenHeads(heads), nn.WithAarenSeed(9))
	if err != nil {
		t.Fatal(err)
	}
	parallel, err := a.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	var maxD float64
	for i := range seq {
		st, err := a.NewAarenState(backend.NewContext(), aarenRow(x, i)) // fixed query = x_i
		if err != nil {
			t.Fatal(err)
		}
		var o *tensor.Tensor
		for j := 0; j <= i; j++ { // fold causal prefix one token at a time
			if o, err = a.StepRecurrent(st, aarenRow(x, j)); err != nil {
				t.Fatal(err)
			}
		}
		for c := range dim {
			if d := math.Abs(o.AtF64(0, c) - parallel.AtF64(i, c)); d > maxD {
				maxD = d
			}
		}
	}
	if maxD > 1e-10 {
		t.Fatalf("duality broken: recurrent vs parallel max abs diff %.3g (want ≤1e-10)", maxD)
	}
	t.Logf("parallel ≡ recurrent (per-position), max abs diff %.3g", maxD)
}

// §V16 (3) STABILITY — with scores in the many-hundreds a naive Σe^s overflows to
// +Inf, yet the running-max stabilisation keeps Aaren finite and still equal to
// stable softmax attention @1e-8, in BOTH the parallel and recurrent forms.
func TestAarenStability(t *testing.T) {
	const dim, heads, seq = 4, 1, 6
	rng := rand.New(rand.NewPCG(31, 3))
	x := randMat(rng, seq, dim)
	for i := range seq { // blow the scores up: |x| ~ 30 ⇒ per-head scores in the thousands
		for j := range dim {
			x.SetF64(x.AtF64(i, j)*30, i, j)
		}
	}
	a, err := nn.NewAaren(tensor.F64, dim, nn.WithAarenHeads(heads), nn.WithAarenSeed(2))
	if err != nil {
		t.Fatal(err)
	}
	// Confirm the naive (non-stabilised) denominator really overflows for some row.
	q := aarenMatmul(x, a.Wq)
	k := aarenMatmul(x, a.Wk)
	scale := 1 / math.Sqrt(float64(a.HeadDim))
	var naiveOverflow bool
	for i := range seq {
		var denom float64
		for j := 0; j <= i; j++ {
			var s float64
			for c := range dim {
				s += q.AtF64(i, c) * k.AtF64(j, c)
			}
			denom += math.Exp(s * scale) // NO max subtraction
		}
		if math.IsInf(denom, 1) {
			naiveOverflow = true
		}
	}
	if !naiveOverflow {
		t.Fatal("test setup: naive Σe^s did not overflow — scores not large enough to exercise stability")
	}
	// Aaren must stay finite and equal to the stable reference.
	out, err := a.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range seq {
		for j := range dim {
			if v := out.AtF64(i, j); math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("Forward non-finite at (%d,%d): %v", i, j, v)
			}
		}
	}
	if d := maxAbsDiff(out, aarenSoftmaxRef(a, x, true)); d > 1e-8 {
		t.Errorf("stable Forward ≠ softmax attention under large scores: max abs diff %.3g", d)
	}
	// Recurrent form stays finite and matches too.
	st, err := a.NewAarenState(backend.NewContext(), aarenRow(x, seq-1))
	if err != nil {
		t.Fatal(err)
	}
	var o *tensor.Tensor
	for j := range seq {
		if o, err = a.StepRecurrent(st, aarenRow(x, j)); err != nil {
			t.Fatal(err)
		}
	}
	for j := range dim {
		if v := o.AtF64(0, j); math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("StepRecurrent non-finite at %d: %v", j, v)
		}
	}
	ref := aarenSoftmaxRef(a, x, true)
	for j := range dim {
		if d := math.Abs(o.AtF64(0, j) - ref.AtF64(seq-1, j)); d > 1e-8 {
			t.Errorf("recurrent last row ≠ stable softmax under large scores: diff %.3g", d)
		}
	}
	t.Log("stable under score magnitudes that overflow a naive Σe^s (parallel + recurrent)")
}

// §V16 (4) CAUSALITY — o_i is independent of any k_j,v_j with j>i. Perturbing a
// strictly-future input row leaves every earlier output row bit-unchanged
// (<1e-12); the bidirectional layer, by contrast, does move.
func TestAarenCausality(t *testing.T) {
	const dim, heads, seq = 6, 3, 5
	rng := rand.New(rand.NewPCG(41, 13))
	x := randMat(rng, seq, dim)
	a, err := nn.NewAaren(tensor.F64, dim, nn.WithAarenHeads(heads), nn.WithAarenSeed(4))
	if err != nil {
		t.Fatal(err)
	}
	base, err := a.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	// Perturb only the LAST input row (a strictly-future key/value for every i<seq-1).
	x2 := randMat(rng, seq, dim)
	for i := range seq {
		for j := range dim {
			if i < seq-1 {
				x2.SetF64(x.AtF64(i, j), i, j)
			}
		}
	}
	pert, err := a.Forward(backend.NewContext(), x2)
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
	// Sanity: the bidirectional layer DOES see the future.
	b, err := nn.NewAaren(tensor.F64, dim, nn.WithAarenHeads(heads), nn.WithAarenSeed(4), nn.WithAarenBidirectional())
	if err != nil {
		t.Fatal(err)
	}
	b1, _ := b.Forward(backend.NewContext(), x)
	b2, _ := b.Forward(backend.NewContext(), x2)
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

// aarenGradcheck compares every analytic gradient of Wq,Wk,Wv,Wo against central
// finite differences of the summed forward output. Returns entry count and max
// relative error.
func aarenGradcheck(t *testing.T, a *nn.Aaren, x *tensor.Tensor) (int, float64) {
	t.Helper()
	const h = 1e-6
	tape := autograd.NewTape()
	out, err := a.Forward(tape.Context(), x)
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
		o, err := a.Forward(backend.NewContext(), x)
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
	for pi, p := range a.Params() {
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
			if rel > 1e-4 {
				t.Errorf("param %d grad%v: numeric %.8g vs analytic %.8g", pi, coord, num, ana)
			}
		}
		if !nz {
			t.Errorf("param %d: all analytic grads zero — projection not exercised", pi)
		}
	}
	return entries, maxRel
}

// §V16 (5) GRADCHECK — central finite differences confirm gradients reach EVERY
// projection (Wq,Wk,Wv,Wo) through the cumulative-softmax path, causal and
// bidirectional, max rel err ~1e-4 or better.
func TestAarenGradcheck(t *testing.T) {
	const dim, heads, seq = 4, 2, 3
	rng := rand.New(rand.NewPCG(29, 7))
	x := randMat(rng, seq, dim)

	ca, err := nn.NewAaren(tensor.F64, dim, nn.WithAarenHeads(heads), nn.WithAarenSeed(3))
	if err != nil {
		t.Fatal(err)
	}
	n, rel := aarenGradcheck(t, ca, x)
	t.Logf("causal gradcheck: %d entries, max rel err %.3g", n, rel)

	ba, err := nn.NewAaren(tensor.F64, dim, nn.WithAarenHeads(heads), nn.WithAarenSeed(5), nn.WithAarenBidirectional())
	if err != nil {
		t.Fatal(err)
	}
	n, rel = aarenGradcheck(t, ba, x)
	t.Logf("bidirectional gradcheck: %d entries, max rel err %.3g", n, rel)
}

// §V16 (6a) THE VALUE — O(1) STREAMING: a fixed query's AarenState stays fixed
// size no matter how many tokens are streamed (Heads scalars + Heads·HeadDim
// vector), and streaming a long sequence reproduces full softmax attention @1e-10.
func TestAarenO1Streaming(t *testing.T) {
	const dim, heads = 8, 2
	rng := rand.New(rand.NewPCG(51, 17))
	a, err := nn.NewAaren(tensor.F64, dim, nn.WithAarenHeads(heads), nn.WithAarenSeed(6))
	if err != nil {
		t.Fatal(err)
	}
	qx := randMat(rng, 1, dim) // one FIXED query
	st, err := a.NewAarenState(backend.NewContext(), qx)
	if err != nil {
		t.Fatal(err)
	}
	wantM, wantN, wantD := len(st.M), len(st.N), len(st.D)
	if wantM != heads || wantN != heads*a.HeadDim || wantD != heads {
		t.Fatalf("state sizes M=%d N=%d D=%d, want %d/%d/%d", wantM, wantN, wantD, heads, heads*a.HeadDim, heads)
	}

	const L = 400
	stream := randMat(rng, L, dim)
	var last *tensor.Tensor
	for j := range L {
		if last, err = a.StepRecurrent(st, aarenRow(stream, j)); err != nil {
			t.Fatal(err)
		}
		// The state NEVER grows with the number of folded tokens — the O(1) claim.
		if len(st.M) != wantM || len(st.N) != wantN || len(st.D) != wantD {
			t.Fatalf("state grew after %d tokens: M=%d N=%d D=%d", j+1, len(st.M), len(st.N), len(st.D))
		}
	}

	// Streaming L keys with the fixed query equals full softmax attention of that
	// query over all L keys/values: o = Σ softmax(q·k_j/√d)·v_j (per head).
	q := aarenMatmul(qx, a.Wq)
	k := aarenMatmul(stream, a.Wk)
	v := aarenMatmul(stream, a.Wv)
	scale := 1 / math.Sqrt(float64(a.HeadDim))
	concat := tensor.New(tensor.F64, tensor.Shape{1, dim})
	for hh := range heads {
		base := hh * a.HeadDim
		mmax := math.Inf(-1)
		sc := make([]float64, L)
		for j := range L {
			var s float64
			for c := range a.HeadDim {
				s += q.AtF64(0, base+c) * k.AtF64(j, base+c)
			}
			s *= scale
			sc[j] = s
			if s > mmax {
				mmax = s
			}
		}
		var denom float64
		for j := range L {
			sc[j] = math.Exp(sc[j] - mmax)
			denom += sc[j]
		}
		for c := range a.HeadDim {
			var acc float64
			for j := range L {
				acc += sc[j] * v.AtF64(j, base+c)
			}
			concat.SetF64(acc/denom, 0, base+c)
		}
	}
	want := aarenMatmul(concat, a.Wo)
	if d := maxAbsDiff(last, want); d > 1e-10 {
		t.Fatalf("streaming %d keys ≠ full attention: max abs diff %.3g", L, d)
	}
	t.Logf("O(1) state (M=%d,N=%d,D=%d) held across %d streamed tokens; matches full attention @%.1e",
		wantM, wantN, wantD, L, maxAbsDiff(last, want))
}

// §V16 (7) SANITY: Forward is deterministic — same weights and input give a
// bit-identical output across calls.
func TestAarenDeterminism(t *testing.T) {
	const dim, heads, seq = 8, 2, 4
	rng := rand.New(rand.NewPCG(2, 3))
	x := randMat(rng, seq, dim)
	a, err := nn.NewAaren(tensor.F64, dim, nn.WithAarenHeads(heads), nn.WithAarenSeed(7))
	if err != nil {
		t.Fatal(err)
	}
	p, _ := a.Forward(backend.NewContext(), x)
	q, _ := a.Forward(backend.NewContext(), x)
	for i := range seq {
		for j := range dim {
			if p.AtF64(i, j) != q.AtF64(i, j) {
				t.Fatalf("non-deterministic at (%d,%d): %v vs %v", i, j, p.AtF64(i, j), q.AtF64(i, j))
			}
		}
	}
}

// §V16 (6b) THE VALUE — LEARNS: a tiny causal char-LM whose only mixing layer is
// an Aaren block must LEARN — cross-entropy at least HALVES over a few hundred
// steps, proving Aaren is a trainable drop-in, not just a correct forward.
func TestAarenLearnsToyTask(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a small Aaren LM; skipped in -short")
	}
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
	attn, err := nn.NewAaren(tensor.F64, dim, nn.WithAarenHeads(4), nn.WithAarenSeed(5))
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
		hh := x[0]
		n, err := norm.Forward(ctx, hh)
		if err != nil {
			return nil, err
		}
		at, err := attn.Forward(ctx, n)
		if err != nil {
			return nil, err
		}
		sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{hh, at}, nil) // residual
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
	t.Logf("Aaren char-LM: CE %.3f → %.3f (uniform = %.3f)", first, last, math.Log(float64(vocab)))
	if last >= first*0.5 {
		t.Fatalf("Aaren LM did not train (%.3f → %.3f)", first, last)
	}
}

func ExampleAaren() {
	// A tiny causal 2-head Aaren layer. Forward is the parallel/prefix form —
	// numerically identical to standard causal softmax attention (arXiv:2405.13956,
	// a reformulation, not an approximation). StepRecurrent folds one token at a
	// time into an O(1) state for streaming/decode.
	a, _ := nn.NewAaren(tensor.F64, 4, nn.WithAarenHeads(2))
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range 3 {
		x.SetF64(1, i, i) // arbitrary non-zero input
	}
	out, _ := a.Forward(backend.NewContext(), x)

	// Streaming: fix the query to row 0, fold the 3 tokens one at a time.
	query := tensor.New(tensor.F64, tensor.Shape{1, 4})
	query.SetF64(1, 0, 0)
	st, _ := a.NewAarenState(backend.NewContext(), query)
	var o *tensor.Tensor
	for i := range 3 {
		row := tensor.New(tensor.F64, tensor.Shape{1, 4})
		row.SetF64(1, 0, i)
		o, _ = a.StepRecurrent(st, row)
	}
	fmt.Printf("%v heads=%d params=%d step=%v\n", out.Shape(), a.Heads, len(a.Params()), o.Shape())
	// Output: (3, 4) heads=2 params=4 step=(1, 4)
}
