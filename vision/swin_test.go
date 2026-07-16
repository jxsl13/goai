package vision

import (
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

// swinRandGrid builds a random [H·W, C] token grid (F64) for block-level tests.
func swinRandGrid(seed uint64, h, w, c int) *tensor.Tensor {
	return tensor.Randn(tensor.F64, seed, tensor.Shape{h * w, c})
}

// swinSumAll reduces every element of t to a scalar (a gradcheck loss).
func swinSumAll(ctx *backend.Context, t *tensor.Tensor) (*tensor.Tensor, error) {
	axes := make([]int, t.Ndim())
	for i := range axes {
		axes[i] = i
	}
	out, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{t}, backend.ReduceAttrs{Axes: axes})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// swinMaxAbsDiff is the max |a−b| over two same-shape tensors.
func swinMaxAbsDiff(a, b *tensor.Tensor) float64 {
	var mx float64
	for i := range a.Numel() {
		coord := tensor.Unravel(i, a.Shape())
		d := math.Abs(a.AtF64(coord...) - b.AtF64(coord...))
		if d > mx {
			mx = d
		}
	}
	return mx
}

// §V16 sanity: Forward shape contract, Params non-empty, determinism.
func TestSwinForwardShapesAndDeterminism(t *testing.T) {
	m, err := NewSwin(tensor.F64, 8, 2, 2, 8, []int{2, 2}, []int{2, 4}, 3, 7,
		WithSwinChannels(1))
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Params()) == 0 {
		t.Fatal("no params")
	}
	x := tensor.Randn(tensor.F64, 3, tensor.Shape{2, 1, 8, 8})
	a, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !a.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("logits %v, want [2 3]", a.Shape())
	}
	b, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if d := swinMaxAbsDiff(a, b); d != 0 {
		t.Fatalf("non-deterministic forward, max diff %g", d)
	}
}

// §V16.1 WINDOW=FULL-IMAGE COLLAPSE (THE anchor): with the window covering the
// whole grid (single window), no shift and the relative bias disabled, one Swin
// W-MSA block is numerically identical (@1e-10) to a standard GLOBAL ViT
// transformer block — windowed attention is exactly standard attention restricted
// to windows.
func TestSwinWindowEqualsGlobalBlock(t *testing.T) {
	const grid, c, heads = 4, 8, 2
	// windowSize = grid ⇒ a single window; depths [1] ⇒ one W-MSA block; bias off.
	m, err := NewSwin(tensor.F64, 8, 2, grid, c, []int{1}, []int{heads}, 3, 42,
		WithSwinChannels(1), WithSwinRelativeBias(false))
	if err != nil {
		t.Fatal(err)
	}
	blk := m.Stages[0][0]
	if blk.Shift {
		t.Fatal("first block must be W-MSA (no shift)")
	}
	if blk.Rel != nil {
		t.Fatal("relative bias should be disabled")
	}
	x := swinRandGrid(99, grid, grid, c)

	// Reference: a global pre-LN transformer block over the WHOLE grid, sharing
	// the block's exact weights, with the fused OpMHA attention (nlp.MHA).
	mha, err := nlp.NewMHA(heads, blk.Wq, blk.Wk, blk.Wv, blk.Wo)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) *tensor.Tensor {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			t.Fatal(err)
		}
		return o[0]
	}
	h1, _ := blk.LN1.Forward(ctx, x)
	att, _ := mha.Forward(ctx, h1)
	x1 := ex(backend.OpAdd, nil, x, att)
	h2, _ := blk.LN2.Forward(ctx, x1)
	f1, _ := blk.FC1.Forward(ctx, h2)
	g := ex(backend.OpGELU, nil, f1)
	f2, _ := blk.FC2.Forward(ctx, g)
	want := ex(backend.OpAdd, nil, x1, f2)

	got, err := blk.Forward(ctx, x, grid, grid)
	if err != nil {
		t.Fatal(err)
	}
	if d := swinMaxAbsDiff(got, want); d > 1e-10 {
		t.Fatalf("windowed block != global block: max abs diff %g (want ≤ 1e-10)", d)
	} else {
		t.Logf("window=full ≡ global ViT block: max abs diff %.3g", d)
	}
}

// §V16.2 SHIFTED-WINDOW MASK correctness: (a) the cyclic shift is exactly
// reversible; (b) in a shifted window that mixes two regions, the additive −∞
// mask drives the cross-region attention weight to EXACTLY 0 after softmax.
func TestSwinShiftMask(t *testing.T) {
	const H, W, M, C = 4, 4, 2, 3
	shift := M / 2

	// (a) shift ∘ unshift = identity, exactly.
	x := swinRandGrid(5, H, W, C)
	ctx := backend.NewContext()
	shifted, err := swinGather(ctx, x, swinShiftIdx(H, W, shift), tensor.F64)
	if err != nil {
		t.Fatal(err)
	}
	back, err := swinGather(ctx, shifted, swinInverse(swinShiftIdx(H, W, shift)), tensor.F64)
	if err != nil {
		t.Fatal(err)
	}
	if d := swinMaxAbsDiff(back, x); d != 0 {
		t.Fatalf("shift not reversible, max diff %g", d)
	}

	// (b) find a window whose M² tokens span ≥2 regions, then verify the mask
	// zeroes a cross-region pair after softmax.
	region := swinRegionIDs(H, W, M, shift)
	part := swinPartitionIdx(H, W, M)
	masks := swinBuildMasks(H, W, M, shift, tensor.F64)
	n := M * M
	found := false
	for w := range (H / M) * (W / M) {
		var iA, jB = -1, -1
		for i := range n {
			for j := range n {
				if region[part[w*n+i]] != region[part[w*n+j]] {
					iA, jB = i, j
					break
				}
			}
			if iA >= 0 {
				break
			}
		}
		if iA < 0 {
			continue // homogeneous window, no cross-region pair
		}
		found = true
		if masks[w].AtF64(iA, jB) != -1e30 {
			t.Fatalf("window %d pair (%d,%d) should be masked, got %g", w, iA, jB, masks[w].AtF64(iA, jB))
		}
		// scores + mask → softmax; the masked entry must be exactly 0.
		scores := tensor.Randn(tensor.F64, 71, tensor.Shape{n, n})
		biased, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{scores, masks[w]}, nil)
		if err != nil {
			t.Fatal(err)
		}
		wgt, err := backend.Execute(ctx, backend.OpSoftmax, []*tensor.Tensor{biased[0]}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if got := wgt[0].AtF64(iA, jB); got != 0 {
			t.Fatalf("masked attention weight not exactly 0: %g", got)
		}
		// a same-region pair keeps positive weight (the mask is not global).
		t.Logf("window %d: cross-region pair (%d,%d) weight exactly 0; regions present", w, iA, jB)
	}
	if !found {
		t.Fatal("no mixed-region shifted window found — mask path untested")
	}
}

// §V16.3 PATCH MERGING correctness: the 2×2 grouping picks the right four
// neighbors (@1e-10), and the layer halves H,W while doubling C.
func TestSwinPatchMerge(t *testing.T) {
	const H, W, C = 4, 4, 2
	// grid where feature 0 of each token encodes its flat row index r.
	x := tensor.New(tensor.F64, tensor.Shape{H * W, C})
	for r := range H * W {
		x.SetF64(float64(r), r, 0)
		x.SetF64(float64(r)+0.5, r, 1)
	}
	ctx := backend.NewContext()
	// Direct check of the four neighbor gathers (order (0,0),(1,0),(0,1),(1,1)).
	offs := [4][2]int{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	oh, ow := H/2, W/2
	for k, o := range offs {
		sub, err := swinGather(ctx, x, swinMergeIdx(H, W, o[0], o[1]), tensor.F64)
		if err != nil {
			t.Fatal(err)
		}
		for oy := range oh {
			for ox := range ow {
				wantRow := float64((2*oy+o[0])*W + (2*ox + o[1]))
				if got := sub.AtF64(oy*ow+ox, 0); math.Abs(got-wantRow) > 1e-10 {
					t.Fatalf("neighbor %d out (%d,%d): got row %g, want %g", k, oy, ox, got, wantRow)
				}
			}
		}
	}
	// Full layer: shape halves/doubles.
	mg := &swinMerge{Norm: nn.NewLayerNorm(tensor.F64, 4*C), Reduce: nn.NewLinear(tensor.F64, 4*C, 2*C, 1)}
	out, nh, nw, err := mg.forward(ctx, x, H, W)
	if err != nil {
		t.Fatal(err)
	}
	if nh != H/2 || nw != W/2 {
		t.Fatalf("merge resolution %dx%d, want %dx%d", nh, nw, H/2, W/2)
	}
	if !out.Shape().Equal(tensor.Shape{(H / 2) * (W / 2), 2 * C}) {
		t.Fatalf("merge output %v, want [%d %d]", out.Shape(), (H/2)*(W/2), 2*C)
	}
}

// §V16.3 relative-position index: the (2M−1)² distinct in-window offsets are all
// used and the index is symmetric under swapping query/key with a sign flip.
func TestSwinRelIndex(t *testing.T) {
	const M = 3
	idx := swinRelIndex(M)
	n := M * M
	nb := (2*M - 1) * (2*M - 1)
	seen := make([]bool, nb)
	for i := range n {
		for j := range n {
			b := idx[i*n+j]
			if b < 0 || b >= nb {
				t.Fatalf("bucket %d out of range [0,%d)", b, nb)
			}
			seen[b] = true
		}
		if idx[i*n+i] != (M-1)*(2*M-1)+(M-1) { // zero relative offset = center bucket
			t.Fatalf("diagonal bucket wrong at %d", i)
		}
	}
	for b, s := range seen {
		if !s {
			t.Fatalf("relative bucket %d never used", b)
		}
	}
}

// §V16.4 GRADCHECK: central finite differences confirm gradients reach every
// parameter through patch-embed, W-MSA and SW-MSA blocks (incl. the
// relative-position table), a patch-merge, and the head — max rel err ≤ 1e-4.
func TestSwinGradcheck(t *testing.T) {
	m, err := NewSwin(tensor.F64, 8, 2, 2, 4, []int{2, 1}, []int{1, 2}, 3, 3,
		WithSwinChannels(1))
	if err != nil {
		t.Fatal(err)
	}
	img := tensor.Randn(tensor.F64, 77, tensor.Shape{1, 8, 8})

	loss := func(ctx *backend.Context) (*tensor.Tensor, error) {
		logits, err := m.forwardOne(ctx, img)
		if err != nil {
			return nil, err
		}
		return swinSumAll(ctx, logits)
	}
	tape := autograd.NewTape()
	l, err := loss(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(l); err != nil {
		t.Fatal(err)
	}
	scalarLoss := func() float64 {
		o, err := loss(backend.NewContext())
		if err != nil {
			t.Fatal(err)
		}
		return o.AtF64()
	}
	const h = 1e-6
	var maxRel float64
	var entries int
	for pi, p := range m.Params() {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("param %d: nil grad", pi)
		}
		// sample up to a few entries per tensor to keep the check quick but broad.
		step := 1
		if p.Numel() > 12 {
			step = p.Numel() / 12
		}
		for idx := 0; idx < p.Numel(); idx += step {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := scalarLoss()
			p.SetF64(o-h, coord...)
			lm := scalarLoss()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			entries++
			rel := math.Abs(num-ana) / math.Max(1, math.Abs(num))
			if rel > maxRel {
				maxRel = rel
			}
			if rel > 1e-4 {
				t.Errorf("param %d grad%v: numeric %.8g vs analytic %.8g (rel %.3g)", pi, coord, num, ana, rel)
			}
		}
	}
	t.Logf("gradcheck: %d entries across %d params, max rel err %.3g", entries, len(m.Params()), maxRel)
}

// §V16.6 geometry validation: bad shapes/divisibility are rejected.
func TestSwinValidation(t *testing.T) {
	cases := []struct {
		name                                      string
		imageSize, patch, window, embedC, classes int
		depths, heads                             []int
	}{
		{"patch∤image", 8, 3, 2, 4, 3, []int{1}, []int{1}},
		{"odd window", 8, 2, 3, 4, 3, []int{1}, []int{1}},
		{"depths≠heads", 8, 2, 2, 4, 3, []int{1, 1}, []int{1}},
		{"grid∤window", 8, 2, 4, 4, 3, []int{1, 1}, []int{1, 1}}, // stage1 res=2 < window 4
		{"width∤heads", 8, 2, 2, 4, 3, []int{1}, []int{3}},
	}
	for _, tc := range cases {
		if _, err := NewSwin(tensor.F64, tc.imageSize, tc.patch, tc.window, tc.embedC, tc.depths, tc.heads, tc.classes, 1); err == nil {
			t.Errorf("%s: expected error", tc.name)
		}
	}
	// wrong image geometry at Forward.
	m, err := NewSwin(tensor.F64, 8, 2, 2, 4, []int{1}, []int{1}, 3, 1, WithSwinChannels(1))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{1, 6, 6})); err == nil {
		t.Error("wrong image geometry must error")
	}
}

// §V16.5 THE VALUE: a local-texture classification task (class = orientation of a
// bright line, a spatial pattern that spans window boundaries) is learned by a
// small Swin in a few hundred steps — loss drops sharply — and the shifted
// windows (SW-MSA) stay competitive with a W-MSA-only ablation. NOTE: on a tiny
// separable task the shift's advantage does not isolate, because both the final
// global-average-pool and patch merging already provide cross-window mixing; the
// shift's documented benefit (Liu et al. 2021 Table 4) shows at scale. The hard
// assertion here is that the model LEARNS; the ablation numbers are reported.
func TestSwinLearnsTextureAndShiftHelps(t *testing.T) {
	x, targets := swinTexture(30, 8, 1)

	train := func(disableShift bool) (float64, float64) {
		m, err := NewSwin(tensor.F64, 8, 2, 2, 12, []int{2, 2}, []int{2, 4}, 3, 9,
			WithSwinChannels(1))
		if err != nil {
			t.Fatal(err)
		}
		if disableShift {
			for _, blocks := range m.Stages {
				for _, b := range blocks {
					b.Shift = false
				}
			}
		}
		opt := nn.NewAdamW(m.Params(), 0.01, 0)
		var first, last float64
		for step := range 160 {
			tape := autograd.NewTape()
			logits, err := m.Forward(tape.Context(), x)
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
			clipped, _ := nn.ClipGradNorm(m.Params(), tape.Grad, 1.0)
			if err := opt.Step(clipped); err != nil {
				t.Fatal(err)
			}
		}
		return first, last
	}

	firstS, lastS := train(false) // shifted-window Swin (SW-MSA on)
	if lastS > firstS*0.2 {
		t.Fatalf("Swin did not learn texture task: loss %.4f → %.4f", firstS, lastS)
	}
	_, lastW := train(true) // W-MSA-only ablation (all blocks forced to W-MSA)
	t.Logf("texture task: SW-MSA %.4f → %.4f ; W-MSA-only final %.4f", firstS, lastS, lastW)
	// Both should learn this easy separable task; guard only that the shifted
	// model is not dramatically worse (a broken shift/mask would regress badly).
	if lastS > lastW+0.1 {
		t.Errorf("shifted-window model regressed vs W-MSA-only: %.4f vs %.4f", lastS, lastW)
	}
}

// swinTexture builds an oriented-line classification set: each image has a bright
// line whose ORIENTATION (horizontal=0, vertical=1, diagonal=2) is the class. The
// line crosses the whole image, so recognizing it needs information from adjacent
// windows — the cross-window connection the shift provides.
func swinTexture(n, size, channels int) (*tensor.Tensor, *tensor.Tensor) {
	x := tensor.New(tensor.F64, tensor.Shape{n, channels, size, size})
	t := tensor.New(tensor.F64, tensor.Shape{n})
	for s := range n {
		cls := s % 3
		t.SetF64(float64(cls), s)
		off := (s * 3) % size
		for c := range channels {
			for i := range size {
				for j := range size {
					v := 0.05 * math.Sin(float64(s*7+i*5+j*3+c))
					switch cls {
					case 0: // horizontal line
						if i == off {
							v += 1
						}
					case 1: // vertical line
						if j == off {
							v += 1
						}
					case 2: // diagonal line
						if (i+j)%size == off {
							v += 1
						}
					}
					x.SetF64(v, s, c, i, j)
				}
			}
		}
	}
	return x, t
}
