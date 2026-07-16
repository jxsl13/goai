package vision

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// White-box tests (package vision) for the anchors that need the raw layer weights: shape-flow +
// hand-trace, zero-residual identity, transpose/mixing-axes, and gradcheck.

// ---- small numeric helpers (mixer-prefixed to avoid collisions with the CNN/ViT tests) ----

func mixerReadMat(t *tensor.Tensor) [][]float64 {
	sh := t.Shape()
	out := make([][]float64, sh[0])
	for i := range out {
		out[i] = make([]float64, sh[1])
		for j := range out[i] {
			out[i][j] = t.AtF64(i, j)
		}
	}
	return out
}

func mixerReadVec(t *tensor.Tensor) []float64 {
	n := t.Numel()
	out := make([]float64, n)
	for i := range out {
		out[i] = t.AtF64(i)
	}
	return out
}

// mixerLinear computes y = x·W + b with W [in,out], b [out] (the nn.Linear layout).
func mixerLinear(x, w [][]float64, b []float64) [][]float64 {
	in := len(w)
	out := len(w[0])
	y := make([][]float64, len(x))
	for i := range x {
		y[i] = make([]float64, out)
		for k := range out {
			s := b[k]
			for j := range in {
				s += x[i][j] * w[j][k]
			}
			y[i][k] = s
		}
	}
	return y
}

func mixerTranspose(x [][]float64) [][]float64 {
	r, c := len(x), len(x[0])
	y := make([][]float64, c)
	for i := range y {
		y[i] = make([]float64, r)
		for j := range x {
			y[i][j] = x[j][i]
		}
	}
	return y
}

// mixerGELU is the exact GELU x·Φ(x) matching backend.OpGELU.
func mixerGELU(x [][]float64) [][]float64 {
	y := make([][]float64, len(x))
	for i := range x {
		y[i] = make([]float64, len(x[i]))
		for j := range x[i] {
			v := x[i][j]
			y[i][j] = v * 0.5 * (1 + math.Erf(v/math.Sqrt2))
		}
	}
	return y
}

// mixerLayerNorm normalizes each row over the last axis (torch semantics: biased variance, eps
// inside the sqrt) with learnable gamma/beta — matching nn.LayerNorm.
func mixerLayerNorm(x [][]float64, gamma, beta []float64, eps float64) [][]float64 {
	y := make([][]float64, len(x))
	for i := range x {
		d := len(x[i])
		var mean float64
		for _, v := range x[i] {
			mean += v
		}
		mean /= float64(d)
		var variance float64
		for _, v := range x[i] {
			variance += (v - mean) * (v - mean)
		}
		variance /= float64(d)
		inv := 1 / math.Sqrt(variance+eps)
		y[i] = make([]float64, d)
		for j := range x[i] {
			y[i][j] = (x[i][j]-mean)*inv*gamma[j] + beta[j]
		}
	}
	return y
}

func mixerAdd(a, b [][]float64) [][]float64 {
	y := make([][]float64, len(a))
	for i := range a {
		y[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			y[i][j] = a[i][j] + b[i][j]
		}
	}
	return y
}

// mixerRefLayer reproduces one Mixer layer on X [S,C] from the raw weights of block bi. When
// tokenMix is false the token-mixing branch is skipped (used to show channel-mixing keeps patches
// independent).
func mixerRefLayer(m *MLPMixer, bi int, x [][]float64, tokenMix bool) [][]float64 {
	blk := m.Layers[bi]
	if tokenMix {
		tn := mixerLayerNorm(x, mixerReadVec(blk.tokenNorm.Gamma), mixerReadVec(blk.tokenNorm.Beta), blk.tokenNorm.Eps)
		u := mixerTranspose(tn)                                                        // [C,S]
		u = mixerLinear(u, mixerReadMat(blk.tokenFC1.W), mixerReadVec(blk.tokenFC1.B)) // [C,Dt]
		u = mixerGELU(u)
		u = mixerLinear(u, mixerReadMat(blk.tokenFC2.W), mixerReadVec(blk.tokenFC2.B)) // [C,S]
		x = mixerAdd(x, mixerTranspose(u))                                             // [S,C]
	}
	cn := mixerLayerNorm(x, mixerReadVec(blk.channelNorm.Gamma), mixerReadVec(blk.channelNorm.Beta), blk.channelNorm.Eps)
	v := mixerLinear(cn, mixerReadMat(blk.channelFC1.W), mixerReadVec(blk.channelFC1.B)) // [S,Dc]
	v = mixerGELU(v)
	v = mixerLinear(v, mixerReadMat(blk.channelFC2.W), mixerReadVec(blk.channelFC2.B)) // [S,C]
	return mixerAdd(x, v)
}

// mixerRefForward reproduces the whole model (from the patch embedding through the head) by hand.
func mixerRefForward(m *MLPMixer, ctx *backend.Context, img *tensor.Tensor) []float64 {
	emb, err := m.embedding(ctx, img)
	if err != nil {
		panic(err)
	}
	x := mixerReadMat(emb)
	for bi := range m.Layers {
		x = mixerRefLayer(m, bi, x, true)
	}
	x = mixerLayerNorm(x, mixerReadVec(m.Norm.Gamma), mixerReadVec(m.Norm.Beta), m.Norm.Eps)
	c := len(x[0])
	pooled := make([]float64, c)
	for _, row := range x {
		for j := range row {
			pooled[j] += row[j]
		}
	}
	for j := range pooled {
		pooled[j] /= float64(len(x))
	}
	return mixerLinear([][]float64{pooled}, mixerReadMat(m.Head.W), mixerReadVec(m.Head.B))[0]
}

// mixerImage builds one deterministic F64 [C,H,W] image.
func mixerImage(c, hw, seed int) *tensor.Tensor {
	img := tensor.New(tensor.F64, tensor.Shape{c, hw, hw})
	for ci := range c {
		for i := range hw {
			for j := range hw {
				img.SetF64(math.Sin(float64(seed*7+ci*13+i*3+j*5)), ci, i, j)
			}
		}
	}
	return img
}

// §V16.1 SHAPE/FLOW + hand-trace: patch embedding is [S,C], every layer preserves [S,C], the head
// is [1,classes]; a full hand-traced forward matches the model @1e-10.
func TestMixerShapeFlowAndHandTrace(t *testing.T) {
	m, err := NewMLPMixer(1, 8, 3, 7,
		WithMixerDtype(tensor.F64), WithMixerPatch(4), WithMixerHidden(6),
		WithMixerLayers(2), WithMixerTokenDim(5), WithMixerChannelDim(7))
	if err != nil {
		t.Fatal(err)
	}
	img := mixerImage(1, 8, 1)
	ctx := backend.NewContext()
	// S = (8/4)^2 = 4 patches, C = 6.
	emb, err := m.embedding(ctx, img)
	if err != nil {
		t.Fatal(err)
	}
	if !emb.Shape().Equal(tensor.Shape{4, 6}) {
		t.Fatalf("embedding shape %v, want [4 6]", emb.Shape())
	}
	feat, err := m.Features(ctx, img)
	if err != nil {
		t.Fatal(err)
	}
	if !feat.Shape().Equal(tensor.Shape{4, 6}) {
		t.Fatalf("features shape %v (every layer must preserve [S,C]), want [4 6]", feat.Shape())
	}
	logits, err := m.Forward(ctx, img)
	if err != nil {
		t.Fatal(err)
	}
	if !logits.Shape().Equal(tensor.Shape{1, 3}) {
		t.Fatalf("logits shape %v, want [1 3]", logits.Shape())
	}
	want := mixerRefForward(m, ctx, img)
	for c := range 3 {
		if got := logits.AtF64(0, c); math.Abs(got-want[c]) > 1e-10 {
			t.Fatalf("logit[%d] = %.15g, hand-trace = %.15g (Δ %.2e > 1e-10)", c, got, want[c], math.Abs(got-want[c]))
		}
	}
	batch := tensor.New(tensor.F64, tensor.Shape{2, 1, 8, 8})
	bl, err := m.Forward(ctx, batch)
	if err != nil {
		t.Fatal(err)
	}
	if !bl.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("batch logits %v, want [2 3]", bl.Shape())
	}
}

// §V16.2 RESIDUAL/IDENTITY: with the token- and channel-MLP second layers zero-initialized every
// Mixer layer is the identity, so the N-layer stack passes the (final-normed) patch embedding
// through unchanged @1e-10 — pinning the residual structure.
func TestMixerZeroResidualIsIdentity(t *testing.T) {
	m, err := NewMLPMixer(1, 8, 3, 7,
		WithMixerDtype(tensor.F64), WithMixerHidden(6), WithMixerLayers(3), WithMixerZeroResidual())
	if err != nil {
		t.Fatal(err)
	}
	img := mixerImage(1, 8, 2)
	ctx := backend.NewContext()
	emb, err := m.embedding(ctx, img)
	if err != nil {
		t.Fatal(err)
	}
	feat, err := m.Features(ctx, img)
	if err != nil {
		t.Fatal(err)
	}
	wantNorm := mixerLayerNorm(mixerReadMat(emb), mixerReadVec(m.Norm.Gamma), mixerReadVec(m.Norm.Beta), m.Norm.Eps)
	fm := mixerReadMat(feat)
	for i := range fm {
		for j := range fm[i] {
			if math.Abs(fm[i][j]-wantNorm[i][j]) > 1e-10 {
				t.Fatalf("layer not identity at [%d,%d]: %.15g vs %.15g", i, j, fm[i][j], wantNorm[i][j])
			}
		}
	}
}

// §V16.3 TRANSPOSE/MIXING-AXES: token-mixing genuinely mixes ACROSS patches (perturbing one
// patch's pixels changes a DIFFERENT patch's features), while the channel-mixing path keeps
// patches independent (perturbing one patch leaves the others unchanged).
func TestMixerMixingAxes(t *testing.T) {
	m, err := NewMLPMixer(1, 8, 3, 7,
		WithMixerDtype(tensor.F64), WithMixerPatch(4), WithMixerHidden(6),
		WithMixerLayers(1), WithMixerTokenDim(5), WithMixerChannelDim(7))
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	imgA := mixerImage(1, 8, 3)
	imgB := mixerImage(1, 8, 3)
	// Perturb only patch (0,0): a pixel in [0..4)x[0..4) → touches patch index 0 only in the embedding.
	imgB.SetF64(imgB.AtF64(0, 0, 0)+1.0, 0, 0, 0)

	// (a) Real model, token-mixing ON: the patch-0 change must reach patch 1's features.
	fA, err := m.Features(ctx, imgA)
	if err != nil {
		t.Fatal(err)
	}
	fB, err := m.Features(ctx, imgB)
	if err != nil {
		t.Fatal(err)
	}
	var crossDelta float64
	for j := range 6 {
		crossDelta += math.Abs(fA.AtF64(1, j) - fB.AtF64(1, j)) // patch row 1 (a DIFFERENT patch)
	}
	if crossDelta < 1e-6 {
		t.Fatalf("token-mixing did not propagate across patches: row-1 delta %.2e (expected non-vacuous)", crossDelta)
	}

	// (b) Channel-mixing path in isolation (token branch skipped): perturbing patch 0 leaves the
	// OTHER patch rows untouched — the channel MLP is applied per-patch. The reference layer matches
	// the model @1e-10 (TestMixerShapeFlowAndHandTrace), so it runs the same channel-mixing math.
	embA, _ := m.embedding(ctx, imgA)
	embB, _ := m.embedding(ctx, imgB)
	yA := mixerRefLayer(m, 0, mixerReadMat(embA), false)
	yB := mixerRefLayer(m, 0, mixerReadMat(embB), false)
	var otherRowDelta, ownRowDelta float64
	for j := range 6 {
		otherRowDelta += math.Abs(yA[1][j] - yB[1][j]) // patch 1: must be unchanged
		ownRowDelta += math.Abs(yA[0][j] - yB[0][j])   // patch 0: must change
	}
	if otherRowDelta > 1e-12 {
		t.Fatalf("channel-mixing coupled patches: row-1 delta %.2e (must be ~0 — patches independent)", otherRowDelta)
	}
	if ownRowDelta < 1e-6 {
		t.Fatalf("channel-mixing ignored the perturbed patch: row-0 delta %.2e (should change)", ownRowDelta)
	}
}

// §V16.4 GRADCHECK: finite-difference the loss w.r.t. parameters spanning the patch embedding,
// both MLPs of a layer, the norms and the head; analytic vs numeric within ~1e-4.
func TestMixerGradCheck(t *testing.T) {
	m, err := NewMLPMixer(1, 8, 2, 11,
		WithMixerDtype(tensor.F64), WithMixerPatch(4), WithMixerHidden(4),
		WithMixerLayers(1), WithMixerTokenDim(3), WithMixerChannelDim(5))
	if err != nil {
		t.Fatal(err)
	}
	img := mixerImage(1, 8, 5)
	targets := tensor.New(tensor.F64, tensor.Shape{1})
	targets.SetF64(1, 0)

	loss := func() float64 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, err := m.Forward(ctx, img)
		if err != nil {
			t.Fatal(err)
		}
		l, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}

	tape := autograd.NewTape()
	ctx := tape.Context()
	logits, err := m.Forward(ctx, img)
	if err != nil {
		t.Fatal(err)
	}
	l, err := nn.CrossEntropy(ctx, logits, targets)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(l); err != nil {
		t.Fatal(err)
	}

	blk := m.Layers[0]
	params := []*tensor.Tensor{
		m.Embed.W, m.Embed.B,
		blk.tokenNorm.Gamma, blk.tokenFC1.W, blk.tokenFC2.W,
		blk.channelFC1.W, blk.channelFC2.W,
		m.Norm.Gamma, m.Head.W, m.Head.B,
	}
	const eps = 1e-6
	var maxRel float64
	for _, p := range params {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("nil grad for a parameter (shape %v) — not connected", p.Shape())
		}
		n := p.Numel()
		for _, idx := range []int{0, n / 2, n - 1} {
			coord := tensor.Unravel(idx, p.Shape())
			orig := p.AtF64(coord...)
			p.SetF64(orig+eps, coord...)
			lp := loss()
			p.SetF64(orig-eps, coord...)
			lm := loss()
			p.SetF64(orig, coord...)
			num := (lp - lm) / (2 * eps)
			ana := g.AtF64(coord...)
			rel := math.Abs(num-ana) / (1 + math.Max(math.Abs(num), math.Abs(ana)))
			if rel > maxRel {
				maxRel = rel
			}
		}
	}
	if maxRel > 1e-4 {
		t.Fatalf("gradcheck max rel err %.2e > 1e-4", maxRel)
	}
	t.Logf("gradcheck max rel err %.2e", maxRel)
}
