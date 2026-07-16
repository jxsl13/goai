package vision

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// maeImage builds one deterministic, spatially structured [C,size,size] F64
// image (smooth gradients + a diagonal ridge) so masked patches are inferable
// from visible context. mae-prefixed to avoid clashing with vit_/mixer_ helpers.
func maeImage(channels, size int, seed float64) *tensor.Tensor {
	img := tensor.New(tensor.F64, tensor.Shape{channels, size, size})
	for c := range channels {
		for y := range size {
			for x := range size {
				v := math.Sin(0.6*float64(x)+seed) + math.Cos(0.5*float64(y)-seed) +
					0.3*float64((x+y)%size) + 0.1*float64(c)
				img.SetF64(v, c, y, x)
			}
		}
	}
	return img
}

// TestMAEMaskingPartition — §V16.1: exactly round(maskRatio·S) patches masked and
// the rest kept; keep/mask is a valid partition (disjoint, covers all S);
// deterministic given the seed; the encoder genuinely runs on keepCount patches.
func TestMAEMaskingPartition(t *testing.T) {
	const size, patch = 8, 2 // S = 16
	m, err := NewMAE(1, size, 7,
		WithMAEDtype(tensor.F64), WithMAEPatch(patch),
		WithMAEDim(8), WithMAEHeads(2), WithMAEEncoderDepth(1),
		WithMAEDecoderDim(8), WithMAEDecoderHeads(2), WithMAEDecoderDepth(1),
		WithMAEMaskRatio(0.75))
	if err != nil {
		t.Fatal(err)
	}
	s := m.numPatches()
	if s != 16 {
		t.Fatalf("S = %d, want 16", s)
	}
	keep, masked := m.Mask()
	wantMasked := int(math.Round(0.75 * float64(s))) // 12
	if len(masked) != wantMasked {
		t.Fatalf("masked %d, want %d", len(masked), wantMasked)
	}
	if len(keep) != s-wantMasked {
		t.Fatalf("keep %d, want %d", len(keep), s-wantMasked)
	}
	// Valid partition: disjoint union covers 0..S-1 exactly once.
	seen := make([]int, s)
	for _, i := range append(append([]int{}, keep...), masked...) {
		if i < 0 || i >= s {
			t.Fatalf("index %d out of range", i)
		}
		seen[i]++
	}
	for i, c := range seen {
		if c != 1 {
			t.Fatalf("patch %d covered %d times, want exactly 1", i, c)
		}
	}
	// Determinism: same seed → identical mask.
	m2, _ := NewMAE(1, size, 7, WithMAEDtype(tensor.F64), WithMAEPatch(patch),
		WithMAEDim(8), WithMAEHeads(2), WithMAEEncoderDepth(1),
		WithMAEDecoderDim(8), WithMAEDecoderHeads(2), WithMAEDecoderDepth(1),
		WithMAEMaskRatio(0.75))
	k2, mk2 := m2.Mask()
	if fmt.Sprint(keep) != fmt.Sprint(k2) || fmt.Sprint(masked) != fmt.Sprint(mk2) {
		t.Fatalf("mask not deterministic for a fixed seed")
	}
	// A different seed should (with overwhelming probability) differ.
	m3, _ := NewMAE(1, size, 99, WithMAEDtype(tensor.F64), WithMAEPatch(patch),
		WithMAEDim(8), WithMAEHeads(2), WithMAEEncoderDepth(1),
		WithMAEDecoderDim(8), WithMAEDecoderHeads(2), WithMAEDecoderDepth(1),
		WithMAEMaskRatio(0.75))
	if k3, _ := m3.Mask(); fmt.Sprint(keep) == fmt.Sprint(k3) {
		t.Fatalf("different seed produced identical mask")
	}
	// The DEFINING efficiency: the encoder input length equals keepCount, not S.
	latent, ek, _, err := m.Encode(backend.NewContext(), maeImage(1, size, 0.3))
	if err != nil {
		t.Fatal(err)
	}
	if latent.Shape()[0] != len(keep) {
		t.Fatalf("encoder latent length %d, want keepCount %d (must be < S=%d)", latent.Shape()[0], len(keep), s)
	}
	if latent.Shape()[0] >= s {
		t.Fatalf("encoder processed %d patches — MAE must encode ONLY the %d kept, not all %d", latent.Shape()[0], len(keep), s)
	}
	if len(ek) != len(keep) {
		t.Fatalf("Encode keep length %d != Mask keep %d", len(ek), len(keep))
	}
}

// TestMAEUnshuffleOrder — §V16.2: the unshuffle restores ORIGINAL patch order.
// White-box test of the production unshuffleRows: tag proj row k with value k and
// the mask token with -1; assert output row `orig` holds k for kept positions and
// -1 for masked ones (a patch masked at position i lands at output row i).
func TestMAEUnshuffleOrder(t *testing.T) {
	const s, d = 6, 3
	keep := []int{4, 1, 5}   // arbitrary shuffled kept order
	masked := []int{0, 2, 3} // the complement
	proj := tensor.New(tensor.F64, tensor.Shape{len(keep), d})
	for k := range keep {
		for j := range d {
			proj.SetF64(float64(k), k, j) // row k tagged with value k
		}
	}
	maskTok := tensor.New(tensor.F64, tensor.Shape{1, d})
	for j := range d {
		maskTok.SetF64(-1, 0, j)
	}
	out, err := unshuffleRows(backend.NewContext(), proj, maskTok, keep, masked, s)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{s, d}) {
		t.Fatalf("unshuffled shape %v, want [%d %d]", out.Shape(), s, d)
	}
	inv := map[int]int{} // original position -> keep-order index
	for k, orig := range keep {
		inv[orig] = k
	}
	for orig := range s {
		want := -1.0 // masked slot
		if k, ok := inv[orig]; ok {
			want = float64(k) // kept slot carries its keep-order row
		}
		if got := out.AtF64(orig, 0); got != want {
			t.Fatalf("row %d = %g, want %g — unshuffle placed a patch in the wrong slot", orig, got, want)
		}
	}

	// And the full Reconstruct output has one prediction per patch, in order,
	// reshaping back to the image shape.
	m := mustMAE(t)
	pred, _, _, err := m.Reconstruct(backend.NewContext(), maeImage(1, 8, 0.2))
	if err != nil {
		t.Fatal(err)
	}
	patchDim := 1 * m.Patch * m.Patch
	if !pred.Shape().Equal(tensor.Shape{m.numPatches(), patchDim}) {
		t.Fatalf("pred shape %v, want [S=%d, patchDim=%d]", pred.Shape(), m.numPatches(), patchDim)
	}
	if _, err := backend.Execute(backend.NewContext(), backend.OpReshape, []*tensor.Tensor{pred.Contiguous()},
		backend.ReshapeAttrs{Shape: tensor.Shape{m.numPatches() * patchDim}}); err != nil {
		t.Fatalf("pred does not reshape back to the image element count: %v", err)
	}
}

// TestMAELossOnMaskedOnly — §V16.3: the loss depends ONLY on masked patches.
// Perturbing a VISIBLE patch's target pixels leaves the loss unchanged; perturbing
// a MASKED patch's target changes it. loss ≥ 0, and = 0 when pred == target.
func TestMAELossOnMaskedOnly(t *testing.T) {
	m := mustMAE(t)
	img := maeImage(1, 8, 0.4)
	pred, err := m.patchify(img) // stand-in prediction (a fixed constant tensor)
	if err != nil {
		t.Fatal(err)
	}
	keep, masked := m.Mask()

	base := func(target *tensor.Tensor) float64 {
		l, err := m.maskedMSE(backend.NewContext(), pred, target, masked)
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}

	// pred == target on all patches → loss exactly 0, and ≥ 0.
	tgt0, _ := m.patchify(img)
	if l := base(tgt0); l != 0 {
		t.Fatalf("loss with pred==target = %g, want 0", l)
	}

	// Perturb a VISIBLE (kept) patch's target: loss must NOT change.
	tgtV, _ := m.patchify(img)
	tgtV.SetF64(tgtV.AtF64(keep[0], 0)+5, keep[0], 0)
	if l := base(tgtV); l != 0 {
		t.Fatalf("perturbing a VISIBLE patch changed the loss to %g — loss must ignore kept patches", l)
	}

	// Perturb a MASKED patch's target: loss must change (increase from 0).
	tgtM, _ := m.patchify(img)
	tgtM.SetF64(tgtM.AtF64(masked[0], 0)+5, masked[0], 0)
	if l := base(tgtM); !(l > 0) {
		t.Fatalf("perturbing a MASKED patch left loss at %g — masked patches must drive the loss", l)
	}
}

// TestMAEGradcheck — §V16.4: finite-difference gradcheck through the encoder,
// decoder, [MASK] token and pixel head. Max relative error ~1e-4 or better.
func TestMAEGradcheck(t *testing.T) {
	m, err := NewMAE(1, 6, 3,
		WithMAEDtype(tensor.F64), WithMAEPatch(3), // S = 4
		WithMAEDim(8), WithMAEHeads(2), WithMAEEncoderDepth(1),
		WithMAEDecoderDim(6), WithMAEDecoderHeads(2), WithMAEDecoderDepth(1),
		WithMAEMaskRatio(0.5))
	if err != nil {
		t.Fatal(err)
	}
	img := maeImage(1, 6, 0.15)

	loss := func() float64 {
		l, err := m.MAELoss(backend.NewContext(), img)
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	// Analytic grads via the tape.
	tape := autograd.NewTape()
	l, err := m.MAELoss(tape.Context(), img)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(l); err != nil {
		t.Fatal(err)
	}

	// Named parameters spanning every trainable region of the model.
	targets := map[string]*tensor.Tensor{
		"maskToken":  m.MaskToken,
		"headW":      m.Head.W,
		"encBlockWq": m.EncoderBlocks[0].attn.Wq,
		"encFc1W":    m.EncoderBlocks[0].fc1.W,
		"decBlockWv": m.DecoderBlocks[0].attn.Wv,
		"decEmbedW":  m.DecEmbed.W,
		"encPos":     m.EncPos,
		"decPos":     m.DecPos,
		"embedW":     m.Embed.W,
	}
	const h = 1e-6
	var maxRel float64
	for name, p := range targets {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s received no gradient", name)
		}
		n := p.Numel()
		for _, idx := range []int{0, n / 2, n - 1} { // sample a few elements
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := loss()
			p.SetF64(o-h, coord...)
			lm := loss()
			p.SetF64(o, coord...)
			num := (lp - lm) / (2 * h)
			ana := g.AtF64(coord...)
			rel := math.Abs(num-ana) / math.Max(1, math.Abs(num))
			if rel > maxRel {
				maxRel = rel
			}
			if rel > 1e-4 {
				t.Errorf("%s[%d]: numeric %.8g vs analytic %.8g (rel %.2e)", name, idx, num, ana, rel)
			}
		}
	}
	t.Logf("MAE gradcheck max relative error: %.2e", maxRel)
}

// TestMAEReconstructionLearns — §V16.5 (THE VALUE): trained on a small structured
// image set, the masked-reconstruction MSE decreases substantially — the model
// learns to infer masked patches from visible context.
func TestMAEReconstructionLearns(t *testing.T) {
	m, err := NewMAE(1, 8, 5,
		WithMAEDtype(tensor.F64), WithMAEPatch(2), // S = 16
		WithMAEDim(24), WithMAEHeads(3), WithMAEEncoderDepth(2),
		WithMAEDecoderDim(16), WithMAEDecoderHeads(2), WithMAEDecoderDepth(1),
		WithMAEMaskRatio(0.5))
	if err != nil {
		t.Fatal(err)
	}
	imgs := make([]*tensor.Tensor, 4)
	for i := range imgs {
		imgs[i] = maeImage(1, 8, 0.7*float64(i))
	}
	opt := nn.NewAdamW(m.Params(), 0.01, 0)
	var first, last float64
	curve := []float64{}
	for step := range 300 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		var loss *tensor.Tensor
		for _, img := range imgs {
			l, err := m.MAELoss(ctx, img)
			if err != nil {
				t.Fatal(err)
			}
			if loss == nil {
				loss = l
			} else {
				sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{loss, l}, nil)
				if err != nil {
					t.Fatal(err)
				}
				loss = sum[0]
			}
		}
		lv := loss.AtF64() / float64(len(imgs))
		if step == 0 {
			first = lv
		}
		last = lv
		if step%50 == 0 {
			curve = append(curve, lv)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(m.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("MAE reconstruction MSE curve (every 50 steps): %v → final %.5f", curve, last)
	if last > first*0.5 {
		t.Fatalf("MAE did not learn to reconstruct: MSE %.5f → %.5f (want a substantial drop)", first, last)
	}
}

// TestMAEValidation — §V16.6 sanity: geometry, dims, ratio, and shape errors.
func TestMAEValidation(t *testing.T) {
	cases := []struct {
		name string
		opts []MAEOption
	}{
		{"patch-not-dividing", []MAEOption{WithMAEPatch(3)}}, // 3 ∤ 8
		{"enc-dim-not-div-heads", []MAEOption{WithMAEDim(30), WithMAEHeads(4)}},
		{"dec-dim-not-div-heads", []MAEOption{WithMAEDecoderDim(30), WithMAEDecoderHeads(4)}},
		{"ratio-zero", []MAEOption{WithMAEMaskRatio(0)}},
		{"ratio-one", []MAEOption{WithMAEMaskRatio(1)}},
		{"ratio-above-one", []MAEOption{WithMAEMaskRatio(1.5)}},
	}
	for _, c := range cases {
		if _, err := NewMAE(1, 8, 7, c.opts...); err == nil {
			t.Errorf("%s: expected error, got nil", c.name)
		}
	}
	if _, err := NewMAE(0, 8, 7); err == nil {
		t.Error("channels=0 accepted")
	}
	// A ratio that collapses the kept set to zero must error (keep ≥ 1 required).
	if _, err := NewMAE(1, 4, 7, WithMAEPatch(2), WithMAEMaskRatio(0.95)); err == nil {
		t.Error("ratio leaving zero kept patches accepted")
	}
	// Wrong image geometry at Forward time.
	m := mustMAE(t)
	bad := tensor.New(tensor.F64, tensor.Shape{1, 6, 6})
	if _, _, _, err := m.Reconstruct(backend.NewContext(), bad); err == nil {
		t.Error("wrong image geometry must error")
	}
	// Params covers every stack (non-empty, all non-nil).
	ps := m.Params()
	if len(ps) == 0 {
		t.Fatal("Params empty")
	}
	for i, p := range ps {
		if p == nil {
			t.Fatalf("Params[%d] is nil", i)
		}
	}
}

// mustMAE builds a small F64 MAE over 8×8 single-channel images for the tests.
func mustMAE(t *testing.T) *MAE {
	t.Helper()
	m, err := NewMAE(1, 8, 11,
		WithMAEDtype(tensor.F64), WithMAEPatch(2),
		WithMAEDim(12), WithMAEHeads(2), WithMAEEncoderDepth(1),
		WithMAEDecoderDim(8), WithMAEDecoderHeads(2), WithMAEDecoderDepth(1),
		WithMAEMaskRatio(0.75))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// ExampleMAE pretrains nothing — it shows the masked-autoencoder forward on one
// 8×8 image: the encoder sees only the kept patches, the decoder reconstructs a
// pixel prediction per patch, and MAELoss is the masked-reconstruction MSE.
func ExampleMAE() {
	m, err := NewMAE(1, 8, 7, WithMAEPatch(2), WithMAEMaskRatio(0.75),
		WithMAEDim(16), WithMAEHeads(2), WithMAEEncoderDepth(1),
		WithMAEDecoderDim(8), WithMAEDecoderHeads(2), WithMAEDecoderDepth(1))
	if err != nil {
		panic(err)
	}
	img := tensor.New(tensor.F32, tensor.Shape{1, 8, 8}) // zeros: just the shapes

	// Mask reports the deterministic keep/mask partition for this seed.
	keep, masked := m.Mask()
	// The encoder sees ONLY the kept patches: latent length = keepCount, not S.
	latent, _, _, err := m.Encode(backend.NewContext(), img)
	if err != nil {
		panic(err)
	}
	// Reconstruct predicts one patch per position (in original order); MAELoss is
	// the masked-reconstruction MSE that drives training.
	pred, _, _, err := m.Reconstruct(backend.NewContext(), img)
	if err != nil {
		panic(err)
	}
	loss, err := m.MAELoss(backend.NewContext(), img)
	if err != nil {
		panic(err)
	}
	fmt.Println(latent.Shape(), pred.Shape(), len(keep), len(masked), loss.AtF64() >= 0, len(m.Params()) > 0)
	// Output: (4, 16) (16, 4) 4 12 true true
}
