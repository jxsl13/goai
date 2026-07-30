package vision

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// MLPMixer is the all-MLP vision architecture of Tolstikhin, Houlsby et al. 2021
// ("MLP-Mixer: An all-MLP Architecture for Vision", arXiv:2105.01601, §C18): an
// image classifier that uses NEITHER convolution NOR attention. The image is cut
// into square patches and each patch is linearly embedded into a table
// X ∈ R^{S×C} (S patches, C channels). Every Mixer layer then alternates two
// MLPs applied along the two axes of that table, each wrapped in a pre-LayerNorm
// residual:
//
//   - Token-mixing MLP — mixes information ACROSS patches. LayerNorm over C,
//     transpose to [C,S], an MLP (Linear S→D_token→S with GELU) acting on the
//     patch axis, transpose back, residual. The same MLP is shared across all C
//     channels, so it learns "which patches talk to which".
//   - Channel-mixing MLP — mixes information ACROSS features WITHIN each patch.
//     LayerNorm over C, an MLP (Linear C→D_channel→C with GELU) acting on the
//     channel axis, residual. This is the ordinary per-token feed-forward block.
//
// After N layers a final LayerNorm, a global average pool over the patches, and
// a Linear head produce the class logits. The two transposes plus the two MLPs
// are the entire model — token-mixing works on the transposed [C,S] so its
// Linear touches the patch dimension, channel-mixing works on [S,C] so its
// Linear touches the feature dimension.
//
// # For the AI professional
//
// Mixer is a pure-MLP token mixer: it replaces ViT's self-attention (see the
// vision.ViT type) with a learned, position-fixed token-mixing MLP. It has no
// query/key/value interaction and no relative positions — the patch grid is
// baked into the token-MLP weights, so unlike attention it does not generalize
// across sequence length. Composed entirely from the verified nn.Linear /
// nn.LayerNorm blocks and the backend transpose/GELU ops, it trains through the
// standard tape → loss → Backward → Step loop and its Params feed any nn
// optimizer.
//
// # For the newcomer
//
// A convolutional network slides small filters over an image; a transformer lets
// tiles "attend" to each other. MLP-Mixer does neither: it chops the image into
// tiles, lays them out as a grid of numbers, and then just repeatedly mixes that
// grid — first down the columns (blending tiles) and then across the rows
// (blending each tile's own features) — using plain fully-connected layers. The
// surprising result of the paper is that this simple recipe classifies images
// competitively.
//
// Further reading: Tolstikhin, Houlsby et al. 2021 (the paper, arXiv:2105.01601);
// Dosovitskiy et al. 2021 (ViT, the attention-based predecessor); Touvron et al.
// 2021, "ResMLP", for a closely related all-MLP design.
type MLPMixer struct {
	Patch  int           // square patch edge length
	Embed  *nn.Linear    // patch embedding [C·p·p → hidden]
	Layers []*mixerBlock // the N token-mix + channel-mix layers
	Norm   *nn.LayerNorm // final LayerNorm before pooling
	Head   *nn.Linear    // classification head [hidden → classes]

	channels, size, hidden int // expected input geometry + embedding width
}

// mixerBlock is one Mixer layer: a token-mixing MLP (mixes across patches) then a
// channel-mixing MLP (mixes across features), each a pre-LayerNorm residual.
type mixerBlock struct {
	tokenNorm              *nn.LayerNorm // LayerNorm over C, before token-mixing
	tokenFC1, tokenFC2     *nn.Linear    // token MLP: S → D_token → S (acts on the patch axis)
	channelNorm            *nn.LayerNorm // LayerNorm over C, before channel-mixing
	channelFC1, channelFC2 *nn.Linear    // channel MLP: C → D_channel → C (acts on the feature axis)
}

// MLPMixerOption configures NewMLPMixer (functional options, §C12).
type MLPMixerOption func(*mixerCfg)

type mixerCfg struct {
	patch, hidden, layers, tokenMLP, channelMLP int
	zeroResidual                                bool
	dtype                                       tensor.Dtype
}

// NOTE on defaults: the Mixer option defaults below are DELIBERATELY TINY — sized so the tests
// and runnable examples train in milliseconds, NOT the research configurations. For reference,
// Mixer-B/16 (Tolstikhin, Houlsby et al. 2021, §C18) uses patch 16, hidden C 768, 12 layers,
// token MLP 384, channel MLP 3072. Scale these up toward those values for real workloads; the
// shapes must stay consistent (patch dividing the image size, all dims > 0).

// WithMixerPatch sets the square patch edge length — the size of the image tiles that become the
// rows of the token table.
//
// In plain terms: how finely the image is chopped up. Smaller patches = more tiles = more detail
// but a larger patch axis S=(size/patch)² for the token-mixing MLP to span; larger patches are
// cheaper but coarser. Boundary behavior — patch must DIVIDE the image size (else NewMLPMixer
// errors); patch = size gives a single patch (S=1, token-mixing becomes a no-op scalar map).
//
// Default 4 (a tiny-image demo value; Mixer-B/16 uses 16, §C18).
func WithMixerPatch(p int) MLPMixerOption { return func(c *mixerCfg) { c.patch = p } }

// WithMixerHidden sets the embedding width C — the number of channels each patch is projected to
// and the width carried through every layer.
//
// In plain terms: how much "room" each tile's representation has; wider = more capacity and more
// compute/parameters. Boundary behavior — must be > 0; the channel-mixing MLP mixes across these
// C features, so too small underfits and too large is wasteful for the task.
//
// Default 32 (a tiny demo value; Mixer-B/16 uses 768, §C18).
func WithMixerHidden(c int) MLPMixerOption { return func(cfg *mixerCfg) { cfg.hidden = c } }

// WithMixerLayers sets the number of Mixer layers (each = one token-mix + one channel-mix block).
//
// In plain terms: how many rounds of mixing — deeper models blend information more thoroughly but
// cost proportionally more compute. Boundary behavior — layers ≥ 1; the residual + LayerNorm in
// every block (both present) keep deep stacks trainable.
//
// Default 2 (a tiny demo value; Mixer-B/16 uses 12, §C18).
func WithMixerLayers(n int) MLPMixerOption { return func(c *mixerCfg) { c.layers = n } }

// WithMixerTokenDim sets the hidden width D_token of the token-mixing MLP — the MLP that mixes
// ACROSS patches.
//
// In plain terms: the size of the little two-layer network applied along the patch axis; wider =
// more capacity to model patch-to-patch relationships. Boundary behavior — must be > 0; this is
// the ONLY place the model mixes patches, so it is what lets a class depend on the global
// arrangement of tiles.
//
// Default 2·hidden (a tiny demo value; Mixer-B/16 uses 384, §C18).
func WithMixerTokenDim(d int) MLPMixerOption { return func(c *mixerCfg) { c.tokenMLP = d } }

// WithMixerChannelDim sets the hidden width D_channel of the channel-mixing MLP — the per-patch
// feed-forward MLP that mixes ACROSS features.
//
// In plain terms: the size of the two-layer network applied to each patch's own C features; this
// is the ordinary transformer-style feed-forward width. Boundary behavior — must be > 0; the
// standard is a multiple of hidden (the 4× expansion ratio).
//
// Default 4·hidden (research-grounded: the near-universal 4× MLP expansion ratio; Mixer-B/16 uses
// 3072 = 4·768, §C18).
func WithMixerChannelDim(d int) MLPMixerOption { return func(c *mixerCfg) { c.channelMLP = d } }

// WithMixerZeroResidual zero-initializes the SECOND linear of both MLPs in every layer, so each
// Mixer layer starts as the exact identity (y = X) and the whole stack passes the patch embedding
// through unchanged.
//
// In plain terms: the standard "zero-init the residual branch" trick — training starts from a
// clean identity map and each layer learns how much to add, which stabilizes deep stacks.
// Boundary behavior: a flag, no numeric extremes; with it set a freshly built model is a pure
// patch-embed → pool → head pipeline until training moves the weights. Default off.
func WithMixerZeroResidual() MLPMixerOption { return func(c *mixerCfg) { c.zeroResidual = true } }

// WithMixerDtype sets the parameter dtype.
//
// In plain terms: the numeric precision of the weights — F32 (default) is the safe, accurate
// choice; F64 doubles memory for extra precision (mainly useful for gradient checks). Boundary
// behavior: a dtype enum, no numeric extremes. Default F32 (the standard training precision).
func WithMixerDtype(d tensor.Dtype) MLPMixerOption { return func(c *mixerCfg) { c.dtype = d } }

// NewMLPMixer builds an MLP-Mixer classifier for [channels, imageSize, imageSize] images and the
// given number of classes. Deterministic via seed. The image size must be a multiple of the patch
// size (all shapes are fixed at construction). Mirrors NewViT's constructor and patch-embedding
// style; the architectural dimensions are functional options (§C12) with tiny demo defaults.
func NewMLPMixer(channels, imageSize, classes int, seed uint64, opts ...MLPMixerOption) (*MLPMixer, error) {
	if channels <= 0 || imageSize <= 0 || classes <= 0 {
		return nil, fmt.Errorf("vision: NewMLPMixer needs channels, imageSize, classes > 0")
	}
	cfg := mixerCfg{patch: 4, hidden: 32, layers: 2, dtype: tensor.F32}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.hidden <= 0 || cfg.layers <= 0 {
		return nil, fmt.Errorf("vision: hidden %d and layers %d must be > 0", cfg.hidden, cfg.layers)
	}
	if cfg.tokenMLP == 0 {
		cfg.tokenMLP = 2 * cfg.hidden
	}
	if cfg.channelMLP == 0 {
		cfg.channelMLP = 4 * cfg.hidden
	}
	if cfg.tokenMLP <= 0 || cfg.channelMLP <= 0 {
		return nil, fmt.Errorf("vision: token MLP %d and channel MLP %d must be > 0", cfg.tokenMLP, cfg.channelMLP)
	}
	if cfg.patch <= 0 || imageSize%cfg.patch != 0 {
		return nil, fmt.Errorf("vision: patch %d must divide image size %d", cfg.patch, imageSize)
	}
	grid := imageSize / cfg.patch
	s := grid * grid // number of patches S
	m := &MLPMixer{Patch: cfg.patch, channels: channels, size: imageSize, hidden: cfg.hidden}
	patchDim := channels * cfg.patch * cfg.patch
	m.Embed = nn.NewLinear(cfg.dtype, patchDim, cfg.hidden, seed)
	for i := range cfg.layers {
		base := seed + 10 + uint64(i)*8
		blk := &mixerBlock{
			tokenNorm:   nn.NewLayerNorm(cfg.dtype, cfg.hidden),
			tokenFC1:    nn.NewLinear(cfg.dtype, s, cfg.tokenMLP, base),
			tokenFC2:    nn.NewLinear(cfg.dtype, cfg.tokenMLP, s, base+1),
			channelNorm: nn.NewLayerNorm(cfg.dtype, cfg.hidden),
			channelFC1:  nn.NewLinear(cfg.dtype, cfg.hidden, cfg.channelMLP, base+2),
			channelFC2:  nn.NewLinear(cfg.dtype, cfg.channelMLP, cfg.hidden, base+3),
		}
		if cfg.zeroResidual {
			zeroParam(blk.tokenFC2.W)
			zeroParam(blk.channelFC2.W)
		}
		m.Layers = append(m.Layers, blk)
	}
	m.Norm = nn.NewLayerNorm(cfg.dtype, cfg.hidden)
	m.Head = nn.NewLinear(cfg.dtype, cfg.hidden, classes, seed+3)
	return m, nil
}

// zeroParam sets every element of a freshly created parameter tensor to zero.
func zeroParam(t *tensor.Tensor) {
	switch t.Dtype() {
	case tensor.F64:
		s := t.Storage().F64()
		for i := range s {
			s[i] = 0
		}
	case tensor.F32:
		s := t.Storage().F32()
		for i := range s {
			s[i] = 0
		}
	}
}

// Params returns every trainable tensor (for the optimizers).
func (m *MLPMixer) Params() []*tensor.Tensor {
	out := m.Embed.Params()
	for _, b := range m.Layers {
		out = append(out, b.tokenNorm.Params()...)
		out = append(out, b.tokenFC1.Params()...)
		out = append(out, b.tokenFC2.Params()...)
		out = append(out, b.channelNorm.Params()...)
		out = append(out, b.channelFC1.Params()...)
		out = append(out, b.channelFC2.Params()...)
	}
	out = append(out, m.Norm.Params()...)
	out = append(out, m.Head.Params()...)
	return out
}

// mixerPatchify flattens one [C,H,W] image into [S, C·p·p] patch rows (row-major over the patch
// grid, channel-major within a patch — the same flattening as ViT.patchify). The image carries no
// gradient, so this runs outside the tape.
func (m *MLPMixer) mixerPatchify(img *tensor.Tensor) (*tensor.Tensor, error) {
	sh := img.Shape()
	if img.Ndim() != 3 || sh[0] != m.channels || sh[1] != m.size || sh[2] != m.size {
		return nil, fmt.Errorf("vision: MLPMixer expects [%d,%d,%d] images, got %v", m.channels, m.size, m.size, sh)
	}
	return patchifyImage(img, m.Embed.W.Dtype(), m.channels, m.size, m.Patch, "MLPMixer")
}

// embedding returns the patch-embedding table X ∈ R^{S×C} for one [C,H,W] image — the input to
// the first Mixer layer, before any mixing. The full forward runs this through the layers,
// LayerNorm, pooling and head.
func (m *MLPMixer) embedding(ctx *backend.Context, img *tensor.Tensor) (*tensor.Tensor, error) {
	patches, err := m.mixerPatchify(img)
	if err != nil {
		return nil, err
	}
	out, err := m.Embed.Forward(ctx, patches)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// Features returns the token table X ∈ R^{S×C} after all N Mixer layers and the final LayerNorm
// for one [C,H,W] image — the representation the head pools and reads. Forward is Features pooled
// over patches through the classification head.
func (m *MLPMixer) Features(ctx *backend.Context, img *tensor.Tensor) (*tensor.Tensor, error) {
	x, err := m.embedding(ctx, img)
	if err != nil {
		return nil, err
	}
	for _, b := range m.Layers {
		if x, err = b.forward(ctx, x); err != nil {
			return nil, err
		}
	}
	return m.Norm.Forward(ctx, x)
}

// Forward computes class logits [batch, classes] for images x [batch, C, H, W] (a single
// [C,H,W] image is also accepted and treated as batch 1).
func (m *MLPMixer) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() == 3 {
		return m.forwardOne(ctx, x)
	}
	if x.Ndim() != 4 {
		return nil, fmt.Errorf("vision: MLPMixer expects [batch,C,H,W] or [C,H,W], got %v", x.Shape())
	}
	// Batched path: run the whole batch as ONE packed [B·P, hidden] token table so every embed / MLP /
	// LayerNorm / head is a single large GEMM instead of B tiny per-image ones (the per-image loop's
	// cost, mirrors the ViT T908 fix). Token-mixing's per-image transpose stays per-image (cheap
	// elementwise) but its two GEMMs batch. Numerically identical to looping forwardOne.
	B := x.Shape()[0]
	patchRows := make([]*tensor.Tensor, B)
	for b := range B {
		img, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{x},
			backend.SliceAttrs{Axis: 0, Start: b, End: b + 1})
		if err != nil {
			return nil, err
		}
		one, err := backend.Execute(ctx, backend.OpReshape, []*tensor.Tensor{img[0]},
			backend.ReshapeAttrs{Shape: tensor.Shape{m.channels, m.size, m.size}})
		if err != nil {
			return nil, err
		}
		if patchRows[b], err = m.mixerPatchify(one[0]); err != nil {
			return nil, err
		}
	}
	allPatches, err := backend.Execute(ctx, backend.OpConcat, patchRows, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		return nil, err
	}
	h, err := m.Embed.Forward(ctx, allPatches[0]) // [B·P, hidden]
	if err != nil {
		return nil, err
	}
	for _, blk := range m.Layers {
		if h, err = blk.forwardBatched(ctx, h, B); err != nil {
			return nil, err
		}
	}
	if h, err = m.Norm.Forward(ctx, h); err != nil {
		return nil, err
	}
	// Global average pool over each image's P patches → [B, hidden], then the head once.
	P := h.Shape()[0] / B
	pooled := make([]*tensor.Tensor, B)
	for b := range B {
		hb, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{h},
			backend.SliceAttrs{Axis: 0, Start: b * P, End: (b + 1) * P})
		if err != nil {
			return nil, err
		}
		pm, err := backend.Execute(ctx, backend.OpMean, []*tensor.Tensor{hb[0]},
			backend.ReduceAttrs{Axes: []int{0}, KeepDims: true})
		if err != nil {
			return nil, err
		}
		pooled[b] = pm[0]
	}
	pool, err := backend.Execute(ctx, backend.OpConcat, pooled, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		return nil, err
	}
	return m.Head.Forward(ctx, pool[0]) // [B, classes]
}

// forwardOne runs one [C,H,W] image to [1, classes] logits: features → global-average-pool over
// patches → head.
func (m *MLPMixer) forwardOne(ctx *backend.Context, img *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := m.Features(ctx, img)
	if err != nil {
		return nil, err
	}
	// Global average pool over the S patches → [1, C].
	pooled, err := backend.Execute(ctx, backend.OpMean, []*tensor.Tensor{h},
		backend.ReduceAttrs{Axes: []int{0}, KeepDims: true})
	if err != nil {
		return nil, err
	}
	return m.Head.Forward(ctx, pooled[0]) // [1, classes]
}

// forwardBatched is mixerBlock.forward over a packed [batch·P, C] batch (P patches per image). Channel
// mixing and both LayerNorms are row-wise, so they run on the packed matrix unchanged. Token-mixing
// mixes ACROSS an image's P patches, so its transpose is done per image (cheap elementwise) but the
// two token MLPs run as single [batch·C, P] GEMMs. Identical to forward per image.
func (b *mixerBlock) forwardBatched(ctx *backend.Context, x *tensor.Tensor, batch int) (*tensor.Tensor, error) {
	P := x.Shape()[0] / batch
	C := x.Shape()[1]
	// Token-mixing: batched norm → per-image transpose to [C,P] → batched token MLP → per-image transpose back.
	h, err := b.tokenNorm.Forward(ctx, x) // [B·P, C]
	if err != nil {
		return nil, err
	}
	tParts := make([]*tensor.Tensor, batch)
	for i := range batch {
		hb, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{h},
			backend.SliceAttrs{Axis: 0, Start: i * P, End: (i + 1) * P})
		if err != nil {
			return nil, err
		}
		ht, err := backend.Execute(ctx, backend.OpTranspose, []*tensor.Tensor{hb[0]}, nil) // [C, P]
		if err != nil {
			return nil, err
		}
		tParts[i] = ht[0]
	}
	hT, err := backend.Execute(ctx, backend.OpConcat, tParts, backend.ConcatAttrs{Axis: 0}) // [B·C, P]
	if err != nil {
		return nil, err
	}
	u, err := b.tokenFC1.Forward(ctx, hT[0]) // [B·C, D_token]
	if err != nil {
		return nil, err
	}
	g, err := backend.Execute(ctx, backend.OpGELU, []*tensor.Tensor{u}, nil)
	if err != nil {
		return nil, err
	}
	u, err = b.tokenFC2.Forward(ctx, g[0]) // [B·C, P]
	if err != nil {
		return nil, err
	}
	bParts := make([]*tensor.Tensor, batch)
	for i := range batch {
		ub, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{u},
			backend.SliceAttrs{Axis: 0, Start: i * C, End: (i + 1) * C})
		if err != nil {
			return nil, err
		}
		ut, err := backend.Execute(ctx, backend.OpTranspose, []*tensor.Tensor{ub[0]}, nil) // [P, C]
		if err != nil {
			return nil, err
		}
		bParts[i] = ut[0]
	}
	tok, err := backend.Execute(ctx, backend.OpConcat, bParts, backend.ConcatAttrs{Axis: 0}) // [B·P, C]
	if err != nil {
		return nil, err
	}
	sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{x, tok[0]}, nil)
	if err != nil {
		return nil, err
	}
	x = sum[0]

	// Channel-mixing: fully batched (row-wise over C).
	h, err = b.channelNorm.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	v, err := b.channelFC1.Forward(ctx, h) // [B·P, D_channel]
	if err != nil {
		return nil, err
	}
	g, err = backend.Execute(ctx, backend.OpGELU, []*tensor.Tensor{v}, nil)
	if err != nil {
		return nil, err
	}
	v, err = b.channelFC2.Forward(ctx, g[0]) // [B·P, C]
	if err != nil {
		return nil, err
	}
	sum, err = backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{x, v}, nil)
	if err != nil {
		return nil, err
	}
	return sum[0], nil
}

// forward runs one Mixer layer on x [S,C]: token-mixing then channel-mixing, each a pre-LayerNorm
// residual. Preserves the [S,C] shape.
func (b *mixerBlock) forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	// Token-mixing: mix ACROSS patches. LayerNorm over C, transpose to [C,S], MLP on the patch
	// axis, transpose back, residual.
	h, err := b.tokenNorm.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	ht, err := backend.Execute(ctx, backend.OpTranspose, []*tensor.Tensor{h}, nil) // [C,S]
	if err != nil {
		return nil, err
	}
	u, err := b.tokenFC1.Forward(ctx, ht[0]) // [C, D_token]
	if err != nil {
		return nil, err
	}
	g, err := backend.Execute(ctx, backend.OpGELU, []*tensor.Tensor{u}, nil)
	if err != nil {
		return nil, err
	}
	u, err = b.tokenFC2.Forward(ctx, g[0]) // [C, S]
	if err != nil {
		return nil, err
	}
	ut, err := backend.Execute(ctx, backend.OpTranspose, []*tensor.Tensor{u}, nil) // [S, C]
	if err != nil {
		return nil, err
	}
	sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{x, ut[0]}, nil)
	if err != nil {
		return nil, err
	}
	x = sum[0]

	// Channel-mixing: mix ACROSS features within each patch. LayerNorm over C, MLP on the feature
	// axis, residual.
	h, err = b.channelNorm.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	v, err := b.channelFC1.Forward(ctx, h) // [S, D_channel]
	if err != nil {
		return nil, err
	}
	g, err = backend.Execute(ctx, backend.OpGELU, []*tensor.Tensor{v}, nil)
	if err != nil {
		return nil, err
	}
	v, err = b.channelFC2.Forward(ctx, g[0]) // [S, C]
	if err != nil {
		return nil, err
	}
	sum, err = backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{x, v}, nil)
	if err != nil {
		return nil, err
	}
	return sum[0], nil
}
