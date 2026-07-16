package vision

// Degenerate-input robustness (hostile) audit for the vision architectures added
// this session: MLP-Mixer, Swin (windowed + shifted-window mask, relative-position
// bias) and MAE (random masking, masked-only loss). Every test drives an
// adversarial-but-VALID degenerate image or geometry — a shifted-window SW-MSA
// with the −1e30 cross-region mask, a single-patch (1×1 grid) global-average pool,
// an all-zero / all-equal / extreme-magnitude image, and the MAE mask-ratio edges
// — and asserts finite forward output, finite gradients, and either a graceful
// construction error or no panic. Helpers are prefixed vishostile* to stay clear
// of the mixer*/swin*/mae*/hostile* names already in use.
//
// Result: 0 bugs. The −1e30 SW-MSA mask never fully masks a softmax row (a token's
// own diagonal shares its region, so it is always unmasked → the row normalizes
// finitely); the MAE mask-ratio and patch-count edges are rejected at construction
// (NewMAE guarantees ≥1 masked and ≥1 kept patch, so the masked-only MSE is never
// 0/0); and LayerNorm's variance ε keeps all-equal / extreme images finite through
// every stack. These passing tests are the hardening asset.

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// vishostileFinite fails the test if any element of x is NaN or ±Inf.
func vishostileFinite(t *testing.T, name string, x *tensor.Tensor) {
	t.Helper()
	if x == nil {
		return
	}
	for i := 0; i < x.Numel(); i++ {
		v := x.AtF64(tensor.Unravel(i, x.Shape())...)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("%s: non-finite value %v at flat index %d", name, v, i)
		}
	}
}

// vishostileImg builds a [C,size,size] F64 image with every pixel set to fill
// (fill=0 → all-zero, a nonzero constant → all-equal, ±1e6 → extreme magnitude).
func vishostileImg(c, size int, fill float64) *tensor.Tensor {
	img := tensor.New(tensor.F64, tensor.Shape{c, size, size})
	for i := 0; i < img.Numel(); i++ {
		img.SetF64(fill, tensor.Unravel(i, img.Shape())...)
	}
	return img
}

// vishostileSumAll reduces x to a scalar (a differentiable stand-in loss so the
// whole graph gets a backward pass).
func vishostileSumAll(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{x}, backend.ReduceAttrs{})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// vishostileFills is the shared degenerate-image battery.
func vishostileFills() []float64 { return []float64{0, 7, 1e6, -1e6} }

// TestVisHostileSwinShiftedMask exercises the SW-MSA shifted-window path with the
// −1e30 cross-region mask actually engaged (grid 8 > window 4 → two windows per
// axis, half-window shift). A fully-masked softmax row would be 0/0; the diagonal
// keeps every row partly unmasked. Assert finite forward + finite gradients on
// all-zero, all-equal and extreme images (relative-position bias on).
func TestVisHostileSwinShiftedMask(t *testing.T) {
	for _, fill := range vishostileFills() {
		m, err := NewSwin(tensor.F64, 8, 1, 4, 4, []int{2}, []int{2}, 3, 42, WithSwinChannels(3))
		if err != nil {
			t.Fatalf("fill %g: NewSwin: %v", fill, err)
		}
		tape := autograd.NewTape()
		ctx := tape.Context()
		out, err := m.Forward(ctx, vishostileImg(3, 8, fill))
		if err != nil {
			t.Fatalf("fill %g: Forward: %v", fill, err)
		}
		vishostileFinite(t, "swin-fwd", out)
		loss, err := vishostileSumAll(ctx, out)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatalf("fill %g: backward: %v", fill, err)
		}
		for _, p := range m.Params() {
			vishostileFinite(t, "swin-grad", tape.Grad(p))
		}
	}
}

// TestVisHostileSwinSingleWindow covers a single-window grid (grid == window, so
// there is nothing to shift into and no mask) on extreme images — the W-MSA-only
// degenerate geometry.
func TestVisHostileSwinSingleWindow(t *testing.T) {
	for _, fill := range vishostileFills() {
		m, err := NewSwin(tensor.F64, 4, 1, 4, 4, []int{2}, []int{2}, 3, 7, WithSwinChannels(3))
		if err != nil {
			t.Fatalf("fill %g: NewSwin: %v", fill, err)
		}
		tape := autograd.NewTape()
		ctx := tape.Context()
		out, err := m.Forward(ctx, vishostileImg(3, 4, fill))
		if err != nil {
			t.Fatalf("fill %g: Forward: %v", fill, err)
		}
		vishostileFinite(t, "swin1-fwd", out)
		loss, err := vishostileSumAll(ctx, out)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatalf("fill %g: backward: %v", fill, err)
		}
		for _, p := range m.Params() {
			vishostileFinite(t, "swin1-grad", tape.Grad(p))
		}
	}
}

// TestVisHostileMAEReconstruct runs the full MAE reconstruction (masked-only MSE)
// on all-zero, all-equal and extreme images and asserts finite loss + gradients.
func TestVisHostileMAEReconstruct(t *testing.T) {
	for _, fill := range vishostileFills() {
		m, err := NewMAE(3, 8, 1, WithMAEPatch(4), WithMAEDtype(tensor.F64), WithMAEMaskRatio(0.5))
		if err != nil {
			t.Fatalf("fill %g: NewMAE: %v", fill, err)
		}
		tape := autograd.NewTape()
		ctx := tape.Context()
		_, masked, loss, err := m.Reconstruct(ctx, vishostileImg(3, 8, fill))
		if err != nil {
			t.Fatalf("fill %g: Reconstruct: %v", fill, err)
		}
		if len(masked) == 0 {
			t.Fatalf("fill %g: no masked patches (masked-only loss would be 0/0)", fill)
		}
		vishostileFinite(t, "mae-loss", loss)
		if err := tape.Backward(loss); err != nil {
			t.Fatalf("fill %g: backward: %v", fill, err)
		}
		for _, p := range m.Params() {
			vishostileFinite(t, "mae-grad", tape.Grad(p))
		}
	}
}

// TestVisHostileMAEEdgeRatios asserts the mask-ratio / patch-count edges that
// would make the masked-only loss 0/0 are rejected at construction rather than
// producing a NaN: a single-patch image (0 masked or 0 kept), a mask ratio so
// small it masks nothing, and one so large it keeps nothing.
func TestVisHostileMAEEdgeRatios(t *testing.T) {
	if _, err := NewMAE(3, 4, 1, WithMAEPatch(4)); err == nil {
		t.Error("single-patch (s=1) MAE should error, not build")
	}
	if _, err := NewMAE(3, 8, 1, WithMAEPatch(4), WithMAEMaskRatio(0.001)); err == nil {
		t.Error("mask ratio ~0 (nothing masked) should error")
	}
	if _, err := NewMAE(3, 8, 1, WithMAEPatch(4), WithMAEMaskRatio(0.999)); err == nil {
		t.Error("mask ratio ~1 (nothing kept) should error")
	}
}

// TestVisHostileMixerSinglePatch covers the MLP-Mixer single-patch geometry
// (patch == image size → 1×1 grid, S=1): token-mixing collapses to a 1→d→1
// scalar map and the global-average pool is over a single patch. Assert finite
// forward + gradients on extreme images.
func TestVisHostileMixerSinglePatch(t *testing.T) {
	for _, fill := range vishostileFills() {
		m, err := NewMLPMixer(3, 4, 5, 42, WithMixerPatch(4), WithMixerDtype(tensor.F64))
		if err != nil {
			t.Fatalf("fill %g: NewMLPMixer: %v", fill, err)
		}
		tape := autograd.NewTape()
		ctx := tape.Context()
		out, err := m.Forward(ctx, vishostileImg(3, 4, fill))
		if err != nil {
			t.Fatalf("fill %g: Forward: %v", fill, err)
		}
		vishostileFinite(t, "mixer1-fwd", out)
		loss, err := vishostileSumAll(ctx, out)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatalf("fill %g: backward: %v", fill, err)
		}
		for _, p := range m.Params() {
			vishostileFinite(t, "mixer1-grad", tape.Grad(p))
		}
	}
}

// TestVisHostileMixerExtreme covers a multi-patch MLP-Mixer (2×2 grid) on the
// all-zero / all-equal / extreme image battery — LayerNorm's variance ε must keep
// a constant-valued image finite through the token/channel mixing.
func TestVisHostileMixerExtreme(t *testing.T) {
	for _, fill := range vishostileFills() {
		m, err := NewMLPMixer(3, 8, 5, 3, WithMixerPatch(4), WithMixerDtype(tensor.F64))
		if err != nil {
			t.Fatalf("fill %g: NewMLPMixer: %v", fill, err)
		}
		tape := autograd.NewTape()
		ctx := tape.Context()
		out, err := m.Forward(ctx, vishostileImg(3, 8, fill))
		if err != nil {
			t.Fatalf("fill %g: Forward: %v", fill, err)
		}
		vishostileFinite(t, "mixer-fwd", out)
		loss, err := vishostileSumAll(ctx, out)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatalf("fill %g: backward: %v", fill, err)
		}
		for _, p := range m.Params() {
			vishostileFinite(t, "mixer-grad", tape.Grad(p))
		}
	}
}
