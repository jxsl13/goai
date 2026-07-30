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

// copeExec runs a single-output op or fails the test.
func copeExec(t *testing.T, ctx *backend.Context, op backend.Op, at backend.Attrs, in ...*tensor.Tensor) *tensor.Tensor {
	t.Helper()
	out, err := backend.Execute(ctx, op, in, at)
	if err != nil {
		t.Fatalf("op %v: %v", op, err)
	}
	return out[0]
}

// copeRandMat builds a random [r,c] F64 matrix in [-1,1).
func copeRandMat(rng *rand.Rand, r, c int) *tensor.Tensor {
	m := tensor.New(tensor.F64, tensor.Shape{r, c})
	for i := range r {
		for j := range c {
			m.SetF64(2*rng.Float64()-1, i, j)
		}
	}
	return m
}

// copeSumAll reduces every element of x to a scalar through ctx (differentiable).
func copeSumAll(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
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

// copeMaxAbs returns the max |a−b| over two equal-shaped rank-2 tensors.
func copeMaxAbs(a, b *tensor.Tensor) float64 {
	var m float64
	r, c := a.Shape()[0], a.Shape()[1]
	for i := range r {
		for j := range c {
			if d := math.Abs(a.AtF64(i, j) - b.AtF64(i, j)); d > m {
				m = d
			}
		}
	}
	return m
}

// TestCoPEFusedBitExactVsDispatch pins the inference gather path (ctx.Recorder == nil,
// copeGatherBias) bit-identical to the differentiable one-hot einsum path (a recording
// tape context): the position-term interpolation is a plain gather of z rows, so on
// finite inputs the two must agree to the last bit.
func TestCoPEFusedBitExactVsDispatch(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for _, tc := range []struct{ dim, heads, seq, maxPos int }{
		{16, 4, 24, 8}, {32, 4, 40, 32}, {8, 2, 13, 3},
	} {
		c, err := NewCoPEAttention(tensor.F64, tc.dim, tc.heads, 3, WithCoPEMaxPos(tc.maxPos))
		if err != nil {
			t.Fatal(err)
		}
		x := copeRandMat(rng, tc.seq, tc.dim)
		fused, err := c.Forward(backend.NewContext(), x) // Recorder == nil → copeGatherBias
		if err != nil {
			t.Fatal(err)
		}
		disp, err := c.Forward(autograd.NewTape().Context(), x) // Recorder != nil → one-hot einsum
		if err != nil {
			t.Fatal(err)
		}
		if m := copeMaxAbs(fused, disp); m != 0 {
			t.Fatalf("dim=%d heads=%d seq=%d maxPos=%d: fused gather vs einsum differ by %g (want bit-exact 0)",
				tc.dim, tc.heads, tc.seq, tc.maxPos, m)
		}
	}
}

// §V16 tier-1: CoPEAttention maps [T,dim]→[T,dim] and exposes every learnable
// tensor: 4 Linears × (W,B) = 8, plus one position table per head.
func TestCoPEAttentionShapes(t *testing.T) {
	const dim, heads, seq = 8, 2, 5
	c, err := NewCoPEAttention(tensor.F64, dim, heads, 7, WithCoPEMaxPos(6))
	if err != nil {
		t.Fatal(err)
	}
	if c.HeadDim != dim/heads {
		t.Errorf("HeadDim=%d, want %d", c.HeadDim, dim/heads)
	}
	if len(c.PosEmb) != heads {
		t.Fatalf("PosEmb count %d, want %d", len(c.PosEmb), heads)
	}
	if !c.PosEmb[0].Shape().Equal(tensor.Shape{7, dim / heads}) {
		t.Errorf("PosEmb[0] shape %v, want [7 %d]", c.PosEmb[0].Shape(), dim/heads)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := copeRandMat(rng, seq, dim)
	out, err := c.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{seq, dim}) {
		t.Fatalf("output shape %v, want [%d %d]", out.Shape(), seq, dim)
	}
	if got, want := len(c.Params()), 8+heads; got != want {
		t.Errorf("Params count %d, want %d", got, want)
	}
}

// §V16.6: the constructor rejects a dim not divisible by heads, non-positive
// dim/heads, and MaxPos < 1; Forward rejects a wrong input width and rank.
func TestCoPEAttentionErrors(t *testing.T) {
	if _, err := NewCoPEAttention(tensor.F64, 7, 2, 1); err == nil {
		t.Error("dim%heads != 0: want error")
	}
	if _, err := NewCoPEAttention(tensor.F64, 0, 1, 1); err == nil {
		t.Error("dim=0: want error")
	}
	if _, err := NewCoPEAttention(tensor.F64, 8, 0, 1); err == nil {
		t.Error("heads=0: want error")
	}
	if _, err := NewCoPEAttention(tensor.F64, 8, -2, 1); err == nil {
		t.Error("negative heads: want error")
	}
	if _, err := NewCoPEAttention(tensor.F64, 8, 2, 1, WithCoPEMaxPos(0)); err == nil {
		t.Error("MaxPos=0: want error")
	}
	c, _ := NewCoPEAttention(tensor.F64, 8, 2, 1)
	if _, err := c.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 4})); err == nil {
		t.Error("wrong input width: want error")
	}
	if _, err := c.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 8})); err == nil {
		t.Error("rank-3 input: want error")
	}
}

// copeRelPosRef computes ordinary causal multi-head attention with a STANDARD
// relative-position bias qᵢ·Eₕ[i−j] (i−j clamped to MaxPos) added to the content
// logit — the hand-built baseline the gates-forced-to-1 CoPE layer must reduce to.
func copeRelPosRef(t *testing.T, c *CoPEAttention, x *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	ctx := backend.NewContext()
	seq := x.Shape()[0]
	q := copeExec(t, ctx, backend.OpAddBias, nil, copeExec(t, ctx, backend.OpMatMul, nil, x, c.QProj.W), c.QProj.B)
	k := copeExec(t, ctx, backend.OpAddBias, nil, copeExec(t, ctx, backend.OpMatMul, nil, x, c.KProj.W), c.KProj.B)
	v := copeExec(t, ctx, backend.OpAddBias, nil, copeExec(t, ctx, backend.OpMatMul, nil, x, c.VProj.W), c.VProj.B)
	mask := qkCausalMask(x.Dtype(), seq, seq)
	heads := make([]*tensor.Tensor, c.Heads)
	for h := range c.Heads {
		lo, hi := h*c.HeadDim, (h+1)*c.HeadDim
		qh := copeExec(t, ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, q)
		kh := copeExec(t, ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, k)
		vh := copeExec(t, ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, v)
		khT := copeExec(t, ctx, backend.OpTranspose, nil, kh)
		sc := copeExec(t, ctx, backend.OpMatMul, nil, qh, khT) // content logit qᵢ·kⱼ
		// Relative-position bias: bias[i,j] = qᵢ · Eₕ[clamp(i−j,0,MaxPos)].
		bias := tensor.New(x.Dtype(), tensor.Shape{seq, seq})
		for i := range seq {
			for j := range seq {
				rel := i - j
				if rel < 0 {
					rel = 0
				}
				if rel > c.MaxPos {
					rel = c.MaxPos
				}
				var dot float64
				for d := range c.HeadDim {
					dot += qh.AtF64(i, d) * c.PosEmb[h].AtF64(rel, d)
				}
				bias.SetF64(dot, i, j)
			}
		}
		sc = copeExec(t, ctx, backend.OpAdd, nil, sc, bias)
		sc = copeExec(t, ctx, backend.OpAdd, nil, sc, mask)
		w := copeExec(t, ctx, backend.OpSoftmax, nil, sc)
		heads[h] = copeExec(t, ctx, backend.OpMatMul, nil, w, vh)
	}
	cat := copeExec(t, ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...)
	return copeExec(t, ctx, backend.OpAddBias, nil, copeExec(t, ctx, backend.OpMatMul, nil, cat, c.OProj.W), c.OProj.B)
}

// §V16.1 COLLAPSE: with every gate forced to 1 the contextual position pᵢⱼ becomes
// the integer relative index i−j, so CoPE reduces BIT-EXACT to a standard
// relative-position bias attention qᵢ·E[i−j]. Pins "CoPE = gated generalization of
// relative position". seq ≤ MaxPos+1 so no clamping ambiguity.
func TestCoPECollapseToRelativePosition(t *testing.T) {
	const dim, heads, seq, maxPos = 8, 2, 6, 8
	rng := rand.New(rand.NewPCG(11, 22))
	x := copeRandMat(rng, seq, dim)
	c, err := NewCoPEAttention(tensor.F64, dim, heads, 5, WithCoPEMaxPos(maxPos), WithCoPEGatesOne())
	if err != nil {
		t.Fatal(err)
	}
	got, err := c.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	want := copeRelPosRef(t, c, x)
	if m := copeMaxAbs(got, want); m > 1e-10 {
		t.Fatalf("gates=1 CoPE differs from relative-PE attention: max abs %.3g", m)
	} else {
		t.Logf("collapse: gates=1 CoPE ≡ relative-PE attention, max abs diff %.3g", m)
	}

	// Sanity: with real (content) gates the layer actually differs.
	on, err := NewCoPEAttention(tensor.F64, dim, heads, 5, WithCoPEMaxPos(maxPos)) // same seed, gates ON
	if err != nil {
		t.Fatal(err)
	}
	onOut, _ := on.Forward(backend.NewContext(), x)
	if copeMaxAbs(onOut, want) < 1e-6 {
		t.Error("content gates produced the relative-PE output — gating did nothing")
	}
}

// §V16.2 GATE→POSITION: p = reverse cumsum of gates matches a hand example @1e-10,
// pᵢᵢ = 0, and future j > i is 0.
func TestCoPEContextualPosHandExample(t *testing.T) {
	ctx := backend.NewContext()
	// A causally-masked gate matrix (k ≤ i nonzero, future already zeroed).
	g := [][]float64{
		{0.5, 0, 0, 0},
		{0.2, 0.9, 0, 0},
		{0.1, 0.4, 0.7, 0},
		{0.3, 0.6, 0.8, 0.5},
	}
	G := tensor.New(tensor.F64, tensor.Shape{4, 4})
	for i := range 4 {
		for j := range 4 {
			G.SetF64(g[i][j], i, j)
		}
	}
	p, err := copeContextualPos(ctx, G)
	if err != nil {
		t.Fatal(err)
	}
	// Hand pᵢⱼ = Σ_{k=j+1}^{i} gᵢₖ.
	want := [][]float64{
		{0, 0, 0, 0},
		{0.9, 0, 0, 0},     // g11
		{1.1, 0.7, 0, 0},   // g11+g12 ; g12
		{1.9, 1.3, 0.5, 0}, // g11+g12+g13 ; g12+g13 ; g13
	}
	for i := range 4 {
		for j := range 4 {
			if d := math.Abs(p.AtF64(i, j) - want[i][j]); d > 1e-10 {
				t.Errorf("p[%d,%d]=%v want %v", i, j, p.AtF64(i, j), want[i][j])
			}
		}
	}
	for i := range 4 {
		if p.AtF64(i, i) != 0 {
			t.Errorf("p[%d,%d]=%v, want 0 (a key is adjacent to itself)", i, i, p.AtF64(i, i))
		}
	}
}

// §V16.2 CAUSALITY: contextual position for query rows i is a backward count over
// keys k ≤ i, so perturbing the input at a LATER position m > i (which only feeds
// keys at m, masked out of rows i < m) leaves p for every row i < m bit-unchanged.
func TestCoPEPositionCausality(t *testing.T) {
	const dim, heads, seq, maxPos = 6, 2, 6, 8
	rng := rand.New(rand.NewPCG(3, 4))
	x := copeRandMat(rng, seq, dim)
	c, err := NewCoPEAttention(tensor.F64, dim, heads, 9, WithCoPEMaxPos(maxPos))
	if err != nil {
		t.Fatal(err)
	}
	_, pos, err := c.forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	const m = seq - 2 // perturb this row
	x2 := tensor.New(tensor.F64, tensor.Shape{seq, dim})
	for i := range seq {
		for j := range dim {
			x2.SetF64(x.AtF64(i, j), i, j)
		}
	}
	for j := range dim {
		x2.SetF64(x.AtF64(m, j)+0.6, m, j) // change ONLY row m
	}
	_, pos2, err := c.forward(backend.NewContext(), x2)
	if err != nil {
		t.Fatal(err)
	}
	for h := range heads {
		for i := 0; i < m; i++ { // rows strictly before the perturbation
			for j := range seq {
				if d := math.Abs(pos[h].AtF64(i, j) - pos2[h].AtF64(i, j)); d > 1e-10 {
					t.Errorf("head %d causality violated: p[%d,%d] moved %.3g when only row %d changed", h, i, j, d, m)
				}
			}
		}
	}
	// Sanity: a row AT/after the perturbation did move.
	var moved bool
	for j := range seq {
		if math.Abs(pos[0].AtF64(seq-1, j)-pos2[0].AtF64(seq-1, j)) > 1e-9 {
			moved = true
		}
	}
	if !moved {
		t.Error("perturbing row m left all later p unchanged — test is vacuous")
	}
}

// §V16.3 INTERPOLATION: through the REAL forward path (Zₕ einsum with the one-hots
// from copePosIndices), qᵢ·e[p] returns the exact table row for integer p and the
// linear blend 0.7·E[2]+0.3·E[3] for p=2.3; clamping at MaxPos returns E[MaxPos].
func TestCoPEInterpolation(t *testing.T) {
	const maxPos, headDim = 4, 3
	rows := maxPos + 1
	// A known position table E and a query that reads dimension 0 (q = e_0), so
	// qᵢ·e[p] = e[p][0] and we can read off the interpolation directly.
	E := tensor.New(tensor.F64, tensor.Shape{rows, headDim})
	for r := range rows {
		for d := range headDim {
			E.SetF64(float64(10*r+d), r, d) // E[r][0] = 10r
		}
	}
	q := tensor.New(tensor.F64, tensor.Shape{1, headDim})
	q.SetF64(1, 0, 0) // q = e_0
	ctx := backend.NewContext()
	ehT := copeExec(t, ctx, backend.OpTranspose, nil, E)
	z := copeExec(t, ctx, backend.OpMatMul, nil, q, ehT) // z[0][m] = q·E[m] = E[m][0] = 10m

	// Hand-crafted single-row position matrix p (shape [1,1]) fed through the exact
	// forward interpolation to get qᵢ·e[p].
	interp := func(pv float64) float64 {
		p := tensor.New(tensor.F64, tensor.Shape{1, 1})
		p.SetF64(pv, 0, 0)
		floorP, floorPlus1P, ohLo, ohHi := copePosIndices(p, maxPos)
		wHi := copeExec(t, ctx, backend.OpSub, nil, p, floorP)
		wLo := copeExec(t, ctx, backend.OpSub, nil, floorPlus1P, p)
		biasLo := copeExec(t, ctx, backend.OpEinsum, backend.EinsumAttrs{Spec: "ijm,im->ij"}, ohLo, z)
		biasHi := copeExec(t, ctx, backend.OpEinsum, backend.EinsumAttrs{Spec: "ijm,im->ij"}, ohHi, z)
		lo := copeExec(t, ctx, backend.OpMul, nil, wLo, biasLo)
		hi := copeExec(t, ctx, backend.OpMul, nil, wHi, biasHi)
		return copeExec(t, ctx, backend.OpAdd, nil, lo, hi).AtF64(0, 0)
	}

	cases := []struct {
		p, want float64
		name    string
	}{
		{0, 0, "integer 0 → E[0][0]=0"},
		{2, 20, "integer 2 → E[2][0]=20"},
		{2.3, 0.7*20 + 0.3*30, "2.3 → 0.7·E[2]+0.3·E[3]"},
		{3.75, 0.25*30 + 0.75*40, "3.75 → 0.25·E[3]+0.75·E[4]"},
		{float64(maxPos), 10 * maxPos, "clamp: MaxPos → E[MaxPos][0]"},
	}
	for _, cs := range cases {
		if got := interp(cs.p); math.Abs(got-cs.want) > 1e-10 {
			t.Errorf("%s: interp(%v)=%v want %v", cs.name, cs.p, got, cs.want)
		}
	}
}

// copeGradcheck compares every analytic gradient against central finite differences
// of the summed forward output. Returns entry count and max relative error.
func copeGradcheck(t *testing.T, c *CoPEAttention, x *tensor.Tensor) (int, float64) {
	t.Helper()
	const h = 1e-6
	tape := autograd.NewTape()
	out, err := c.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := copeSumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	sumLoss := func() float64 {
		o, err := c.Forward(backend.NewContext(), x)
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
	check("QProj.W", c.QProj.W)
	check("QProj.B", c.QProj.B)
	check("KProj.W", c.KProj.W)
	check("KProj.B", c.KProj.B)
	check("VProj.W", c.VProj.W)
	check("VProj.B", c.VProj.B)
	check("OProj.W", c.OProj.W)
	check("OProj.B", c.OProj.B)
	for h := range c.Heads {
		check(fmt.Sprintf("PosEmb[%d]", h), c.PosEmb[h])
	}
	return entries, maxRel
}

// §V16.4 GRADCHECK: central finite differences confirm gradients reach EVERY
// parameter — the four projections (Q/K feed both the content logit and, via the
// gates, the contextual position) AND every per-head position table E (via the
// interpolated qᵢ·e[p] term).
func TestCoPEAttentionGradcheck(t *testing.T) {
	const dim, heads, seq, maxPos = 4, 2, 4, 5
	rng := rand.New(rand.NewPCG(29, 7))
	// Small-magnitude input so the fractional positions stay clear of integer kinks
	// (the interpolation is piecewise-linear; central differences are exact away
	// from integer p).
	x := copeRandMat(rng, seq, dim)
	for i := range seq {
		for j := range dim {
			x.SetF64(0.3*x.AtF64(i, j), i, j)
		}
	}
	c, err := NewCoPEAttention(tensor.F64, dim, heads, 3, WithCoPEMaxPos(maxPos))
	if err != nil {
		t.Fatal(err)
	}
	n, rel := copeGradcheck(t, c, x)
	t.Logf("CoPE gradcheck: %d entries, max rel err %.3g", n, rel)
	if rel > 1e-4 {
		t.Errorf("max rel err %.3g exceeds 1e-4", rel)
	}
}

// §V16.6 sanity: Forward is deterministic — same weights, same input, bit-identical
// output across calls.
func TestCoPEAttentionDeterminism(t *testing.T) {
	const dim, heads, seq = 8, 2, 4
	rng := rand.New(rand.NewPCG(2, 3))
	x := copeRandMat(rng, seq, dim)
	c, err := NewCoPEAttention(tensor.F64, dim, heads, 7)
	if err != nil {
		t.Fatal(err)
	}
	a, err := c.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	b, err := c.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if m := copeMaxAbs(a, b); m != 0 {
		t.Fatalf("non-deterministic: max abs diff %v", m)
	}
}

// §V16.5 THE VALUE (paper's headline): a counting/selective task where token-INDEX
// position fails but CoPE's content-gated position works. The stream mixes NOISE
// tokens (0) and VALUE tokens (1..V); the target at each position is the SECOND
// most-recent value token, skipping any noise between ("2 value-tokens back").
//
// This defeats the two things content and absolute position can do: pure CONTENT
// attention can spot value tokens but cannot isolate "the 2nd one back" (all values
// look alike — that needs COUNTING), and a learned-ABSOLUTE-position baseline cannot
// pin it because "2 values back" sits at a data-dependent absolute distance (however
// many noise tokens intervene). CoPE gates noise to ~0 and counts only values, so
// "contextual position 2" is exactly the target. We train a CoPE model and an
// absolute-PE baseline (same data/op budget) and report both; CoPE should learn it
// (high accuracy) and beat the baseline.
func TestCoPEHelpsSkipNoiseRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("training loop")
	}
	const (
		vocab  = 5 // 0 = noise, 1..4 = values
		dim    = 24
		heads  = 2
		seq    = 14
		nTrain = 32
		nEval  = 24
		steps  = 300
		maxPos = 14
	)

	// A held-out eval set is essential: with enough capacity BOTH models memorize
	// the training sequences, so the discriminator is GENERALIZATION to unseen noise
	// patterns — CoPE learns the noise-invariant counting rule, absolute PE overfits.
	// One-hot inputs [seq,vocab] and integer targets [seq] = the SECOND most-recent
	// VALUE token (skip noise); 0 until two values have been seen.
	makeData := func(n int, rng *rand.Rand) (ins, tgs []*tensor.Tensor) {
		ins, tgs = make([]*tensor.Tensor, n), make([]*tensor.Tensor, n)
		for s := range n {
			oh := tensor.New(tensor.F64, tensor.Shape{seq, vocab})
			tg := tensor.New(tensor.F64, tensor.Shape{seq})
			last, secondLast := 0, 0 // most-recent and 2nd-most-recent value so far
			for i := range seq {
				var tok int
				if rng.Float64() < 0.4 {
					tok = 0 // noise / distractor
				} else {
					tok = 1 + rng.IntN(vocab-1) // a value
				}
				oh.SetF64(1, i, tok)
				tg.SetF64(float64(secondLast), i) // target BEFORE consuming this token's value
				if tok != 0 {
					secondLast, last = last, tok
				}
			}
			ins[s], tgs[s] = oh, tg
		}
		return ins, tgs
	}
	trainRng := rand.New(rand.NewPCG(2405, 18719))
	evalRng := rand.New(rand.NewPCG(999, 12345)) // disjoint stream ⇒ unseen sequences
	inputs, targets := makeData(nTrain, trainRng)
	evalInputs, evalTargets := makeData(nEval, evalRng)
	nSeq := nTrain

	allTargets, err := backend.Execute(backend.NewContext(), backend.OpConcat, targets, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		t.Fatal(err)
	}

	// held-out accuracy of a model's per-position argmax against the eval targets.
	accuracy := func(fwd func(ctx *backend.Context, oh *tensor.Tensor) *tensor.Tensor) float64 {
		ctx := backend.NewContext()
		var correct, total int
		for n := range nEval {
			lg := fwd(ctx, evalInputs[n])
			for i := range seq {
				best, bestv := 0, math.Inf(-1)
				for v := range vocab {
					if p := lg.AtF64(i, v); p > bestv {
						best, bestv = v, p
					}
				}
				if float64(best) == evalTargets[n].AtF64(i) {
					correct++
				}
				total++
			}
		}
		return float64(correct) / float64(total)
	}

	// ---- CoPE model: embed → CoPEAttention → head ----
	trainCoPE := func() (float64, float64, float64) {
		embed := NewLinear(tensor.F64, vocab, dim, 100)
		head := NewLinear(tensor.F64, dim, vocab, 200)
		attn, err := NewCoPEAttention(tensor.F64, dim, heads, 300, WithCoPEMaxPos(maxPos))
		if err != nil {
			t.Fatal(err)
		}
		var params []*tensor.Tensor
		params = append(params, embed.Params()...)
		params = append(params, attn.Params()...)
		params = append(params, head.Params()...)
		opt := NewAdamW(params, 5e-3, 0.0)
		fwd := func(ctx *backend.Context, oh *tensor.Tensor) *tensor.Tensor {
			e, err := embed.Forward(ctx, oh)
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
			return lg
		}
		var first, last float64
		for step := range steps {
			tape := autograd.NewTape()
			ctx := tape.Context()
			rows := make([]*tensor.Tensor, nSeq)
			for n := range nSeq {
				rows[n] = fwd(ctx, inputs[n])
			}
			allLogits, err := backend.Execute(ctx, backend.OpConcat, rows, backend.ConcatAttrs{Axis: 0})
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
		return first, last, accuracy(fwd)
	}

	// ---- Absolute-PE baseline: (embed + learned absolute position) → plain causal
	// attention → head. Same op budget, token-index position instead of gated count.
	trainAbs := func() (float64, float64, float64) {
		embed := NewLinear(tensor.F64, vocab, dim, 100)
		head := NewLinear(tensor.F64, dim, vocab, 200)
		posEmb := tensor.New(tensor.F64, tensor.Shape{seq, dim}) // learned absolute position
		XavierUniform(posEmb, seq, dim, 111)
		qp := NewLinear(tensor.F64, dim, dim, 300)
		kp := NewLinear(tensor.F64, dim, dim, 301)
		vp := NewLinear(tensor.F64, dim, dim, 302)
		op := NewLinear(tensor.F64, dim, dim, 303)
		headDim := dim / heads
		var params []*tensor.Tensor
		params = append(params, embed.Params()...)
		params = append(params, posEmb)
		for _, l := range []*Linear{qp, kp, vp, op} {
			params = append(params, l.Params()...)
		}
		params = append(params, head.Params()...)
		opt := NewAdamW(params, 5e-3, 0.0)
		ex := func(ctx *backend.Context, o backend.Op, at backend.Attrs, in ...*tensor.Tensor) *tensor.Tensor {
			out, err := backend.Execute(ctx, o, in, at)
			if err != nil {
				t.Fatal(err)
			}
			return out[0]
		}
		mask := qkCausalMask(tensor.F64, seq, seq)
		inv := scalarTensor(tensor.F64, 1/math.Sqrt(float64(headDim)))
		fwd := func(ctx *backend.Context, oh *tensor.Tensor) *tensor.Tensor {
			e, err := embed.Forward(ctx, oh)
			if err != nil {
				t.Fatal(err)
			}
			e = ex(ctx, backend.OpAdd, nil, e, posEmb) // + absolute position
			q, _ := qp.Forward(ctx, e)
			k, _ := kp.Forward(ctx, e)
			v, _ := vp.Forward(ctx, e)
			hs := make([]*tensor.Tensor, heads)
			for h := range heads {
				lo, hi := h*headDim, (h+1)*headDim
				qh := ex(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, q)
				kh := ex(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, k)
				vh := ex(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, v)
				sc := ex(ctx, backend.OpMatMul, nil, qh, ex(ctx, backend.OpTranspose, nil, kh))
				sc = ex(ctx, backend.OpMul, nil, sc, inv)
				sc = ex(ctx, backend.OpAdd, nil, sc, mask)
				w := ex(ctx, backend.OpSoftmax, nil, sc)
				hs[h] = ex(ctx, backend.OpMatMul, nil, w, vh)
			}
			cat := ex(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, hs...)
			a, _ := op.Forward(ctx, cat)
			lg, _ := head.Forward(ctx, a)
			return lg
		}
		var first, last float64
		for step := range steps {
			tape := autograd.NewTape()
			ctx := tape.Context()
			rows := make([]*tensor.Tensor, nSeq)
			for n := range nSeq {
				rows[n] = fwd(ctx, inputs[n])
			}
			allLogits := ex(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, rows...)
			loss, err := CrossEntropy(ctx, allLogits, allTargets[0])
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
		return first, last, accuracy(fwd)
	}

	cFirst, cLast, cAcc := trainCoPE()
	aFirst, aLast, aAcc := trainAbs()
	t.Logf("CoPE     : train loss %.4f → %.4f, held-out accuracy %.3f", cFirst, cLast, cAcc)
	t.Logf("abs-PE   : train loss %.4f → %.4f, held-out accuracy %.3f", aFirst, aLast, aAcc)
	if cLast >= cFirst {
		t.Errorf("CoPE loss did not decrease: %.4f → %.4f", cFirst, cLast)
	}
	if cAcc < 0.85 {
		t.Errorf("CoPE did not generalize the skip-noise counting task: held-out accuracy %.3f < 0.85", cAcc)
	}
	if cAcc > aAcc+0.05 {
		t.Logf("CoPE BEATS the absolute-PE baseline out of sample (%.3f > %.3f) — the content-gated count generalizes, token-index position overfits", cAcc, aAcc)
	} else {
		t.Errorf("CoPE did not beat absolute-PE baseline out of sample (%.3f vs %.3f)", cAcc, aAcc)
	}
}

func ExampleCoPEAttention() {
	// A tiny Contextual-Position-Encoding causal 2-head attention layer. Position is
	// a content-gated count (σ(qᵢ·kⱼ) accumulated backward), not the token index
	// (arXiv:2405.18719).
	c, _ := NewCoPEAttention(tensor.F64, 4, 2, 1, WithCoPEMaxPos(8))
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	for i := range 3 {
		x.SetF64(1, i, i) // arbitrary non-zero input
	}
	out, _ := c.Forward(backend.NewContext(), x)
	fmt.Printf("%v params=%d\n", out.Shape(), len(c.Params()))
	// Output: (3, 4) params=10
}
