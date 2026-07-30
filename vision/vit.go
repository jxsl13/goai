package vision

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// ViT is the Vision Transformer (Dosovitskiy et al. 2021, "An Image is Worth
// 16x16 Words: Transformers for Image Recognition at Scale", §T590/§R236): an
// image is cut into fixed-size square patches, each patch is flattened and
// linearly embedded, a learnable [class] token is prepended and learned 1D
// position embeddings are added; the token sequence then runs through a stack
// of pre-LayerNorm transformer encoder blocks (non-causal multi-head attention
// + GELU MLP, residual around each), and the classification head reads the
// final [class] token.
//
// In plain terms: instead of sliding small filters over the image the way a
// CNN does, the image is treated like a sentence whose "words" are little
// square tiles — and the same transformer machinery that reads text reads the
// tiles. This is the standard vision architecture behind modern image
// classifiers and the vision encoders of every vision-language model
// (SigLIP-class encoders in LLaVA-style systems, §R236).
//
// Further reading: Dosovitskiy et al. 2021 (the paper); Zhai et al. 2023,
// "Sigmoid Loss for Language Image Pre-Training", for the VLM-encoder lineage;
// Steiner et al. 2021, "How to train your ViT", for the training-recipe
// landscape.
type ViT struct {
	Patch  int            // square patch edge length
	Embed  *nn.Linear     // patch embedding [C·p·p → D]
	Class  *tensor.Tensor // learnable [class] token [1, D]
	Pos    *tensor.Tensor // learned position embeddings [N+1, D]
	Blocks []*vitBlock    // pre-LN encoder stack
	Norm   *nn.LayerNorm  // final LayerNorm before the head
	Head   *nn.Linear     // classification head [D → classes]

	channels, size int // expected input geometry (C, H=W=size)
}

// vitBlock is one pre-LN encoder block: x += MHA(LN1(x)); x += MLP(LN2(x)).
type vitBlock struct {
	ln1, ln2 *nn.LayerNorm
	attn     *nlp.MHA
	fc1, fc2 *nn.Linear
}

// ViTOption configures NewViT (functional options, §C12).
type ViTOption func(*vitCfg)

type vitCfg struct {
	patch, dim, depth, heads, mlp int
	dtype                         tensor.Dtype
}

// NOTE on defaults: the ViT option defaults below are DELIBERATELY TINY — sized so the tests
// and runnable examples train in milliseconds, NOT the research configurations. For reference,
// ViT-Base (Dosovitskiy et al. 2021, §R236) uses patch 16, dim 768, depth 12, heads 12, MLP
// 4·dim. Scale these up toward those values for real workloads; the shapes must stay
// consistent (dim divisible by heads, patch dividing the image size).

// WithViTPatch sets the square patch edge length — the size of the image tiles that become the
// transformer's "words".
//
// In plain terms: how finely the image is chopped up. Smaller patches = more tiles = more
// detail but quadratically more attention compute (the sequence length is (size/patch)²);
// larger patches are cheaper but coarser. Boundary behavior — patch must DIVIDE the image size
// (else NewViT errors); patch = size gives a single token (no spatial structure).
//
// Default 4 (a tiny-image demo value; ViT-Base uses 16, §R236).
func WithViTPatch(p int) ViTOption { return func(c *vitCfg) { c.patch = p } }

// WithViTDim sets the embedding dimension D — the width of each token's vector through the
// encoder.
//
// In plain terms: how much "room" each tile's representation has; wider = more capacity and
// more compute/parameters. Boundary behavior — must be divisible by the head count (each head
// gets D/heads channels); too small underfits, too large is wasteful for the task.
//
// Default 32 (a tiny demo value; ViT-Base uses 768, §R236).
func WithViTDim(d int) ViTOption { return func(c *vitCfg) { c.dim = d } }

// WithViTDepth sets the number of transformer encoder blocks stacked in the model.
//
// In plain terms: how many layers of processing — deeper models capture more abstract features
// but cost proportionally more compute and are harder to train. Boundary behavior — depth 1 is
// a single block (shallow); very deep needs residuals + normalization (both present) to train.
//
// Default 2 (a tiny demo value; ViT-Base uses 12, §R236).
func WithViTDepth(n int) ViTOption { return func(c *vitCfg) { c.depth = n } }

// WithViTHeads sets the number of attention heads per block.
//
// In plain terms: attention is split into this many parallel "views", each attending to a
// different subspace of the D channels. Boundary behavior — heads must DIVIDE dim; 1 head is
// ordinary single attention; more heads give finer specialization but each gets fewer channels
// (D/heads).
//
// Default 4 (a tiny demo value; ViT-Base uses 12, §R236).
func WithViTHeads(h int) ViTOption { return func(c *vitCfg) { c.heads = h } }

// WithViTMLP sets the hidden width of the per-block feed-forward MLP.
//
// In plain terms: the size of the little two-layer network applied to each token after
// attention; wider = more per-token capacity. Boundary behavior — narrower than dim bottlenecks
// the block; the standard is a multiple of dim.
//
// Default 4·dim (research-grounded: the paper's 4× expansion ratio, Dosovitskiy et al. 2021,
// §R236 — the near-universal transformer MLP ratio).
func WithViTMLP(m int) ViTOption { return func(c *vitCfg) { c.mlp = m } }

// WithViTDtype sets the parameter dtype.
//
// In plain terms: the numeric precision of the weights — F32 (default) is the safe, accurate
// choice; F64 doubles memory for extra precision (mainly useful for gradient checks). Boundary
// behavior: a dtype enum, no numeric extremes. Default F32 (the standard training precision).
func WithViTDtype(d tensor.Dtype) ViTOption { return func(c *vitCfg) { c.dtype = d } }

// NewViT builds a ViT classifier for [channels, size, size] images and the given
// number of classes. Deterministic via seed. The image size must be a multiple of
// the patch size; the class-token and position parameters are initialized at
// 0.02·N(0,1) (the paper's truncated-normal scale).
func NewViT(channels, size, classes int, seed uint64, opts ...ViTOption) (*ViT, error) {
	if channels <= 0 || size <= 0 || classes <= 0 {
		return nil, fmt.Errorf("vision: NewViT needs channels, size, classes > 0")
	}
	cfg := vitCfg{patch: 4, dim: 32, depth: 2, heads: 4, dtype: tensor.F32}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.mlp == 0 {
		cfg.mlp = 4 * cfg.dim
	}
	if cfg.patch <= 0 || size%cfg.patch != 0 {
		return nil, fmt.Errorf("vision: patch %d must divide image size %d", cfg.patch, size)
	}
	if cfg.dim%cfg.heads != 0 {
		return nil, fmt.Errorf("vision: dim %d must be divisible by heads %d", cfg.dim, cfg.heads)
	}
	n := (size / cfg.patch) * (size / cfg.patch)
	m := &ViT{Patch: cfg.patch, channels: channels, size: size}
	patchDim := channels * cfg.patch * cfg.patch
	m.Embed = nn.NewLinear(cfg.dtype, patchDim, cfg.dim, seed)
	m.Class = scaleInPlace(tensor.Randn(cfg.dtype, seed+1, tensor.Shape{1, cfg.dim}), 0.02)
	m.Pos = scaleInPlace(tensor.Randn(cfg.dtype, seed+2, tensor.Shape{n + 1, cfg.dim}), 0.02)
	for i := range cfg.depth {
		s := seed + 10 + uint64(i)*8
		wq := scaleInPlace(tensor.Randn(cfg.dtype, s, tensor.Shape{cfg.dim, cfg.dim}), 0.02)
		wk := scaleInPlace(tensor.Randn(cfg.dtype, s+1, tensor.Shape{cfg.dim, cfg.dim}), 0.02)
		wv := scaleInPlace(tensor.Randn(cfg.dtype, s+2, tensor.Shape{cfg.dim, cfg.dim}), 0.02)
		wo := scaleInPlace(tensor.Randn(cfg.dtype, s+3, tensor.Shape{cfg.dim, cfg.dim}), 0.02)
		attn, err := nlp.NewMHA(cfg.heads, wq, wk, wv, wo)
		if err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, &vitBlock{
			ln1:  nn.NewLayerNorm(cfg.dtype, cfg.dim),
			ln2:  nn.NewLayerNorm(cfg.dtype, cfg.dim),
			attn: attn,
			fc1:  nn.NewLinear(cfg.dtype, cfg.dim, cfg.mlp, s+4),
			fc2:  nn.NewLinear(cfg.dtype, cfg.mlp, cfg.dim, s+5),
		})
	}
	m.Norm = nn.NewLayerNorm(cfg.dtype, cfg.dim)
	m.Head = nn.NewLinear(cfg.dtype, cfg.dim, classes, seed+3)
	return m, nil
}

// scaleInPlace multiplies every element of a freshly created parameter tensor.
func scaleInPlace(t *tensor.Tensor, f float64) *tensor.Tensor {
	switch t.Dtype() {
	case tensor.F64:
		for i, v := range t.Storage().F64() {
			t.Storage().F64()[i] = v * f
		}
	case tensor.F32:
		for i, v := range t.Storage().F32() {
			t.Storage().F32()[i] = v * float32(f)
		}
	}
	return t
}

// Params returns every trainable tensor (for the optimizers).
func (m *ViT) Params() []*tensor.Tensor {
	out := []*tensor.Tensor{m.Class, m.Pos}
	out = append(out, m.Embed.Params()...)
	for _, b := range m.Blocks {
		out = append(out, b.ln1.Params()...)
		out = append(out, b.attn.Wq, b.attn.Wk, b.attn.Wv, b.attn.Wo)
		out = append(out, b.ln2.Params()...)
		out = append(out, b.fc1.Params()...)
		out = append(out, b.fc2.Params()...)
	}
	out = append(out, m.Norm.Params()...)
	out = append(out, m.Head.Params()...)
	return out
}

// patchify flattens one [C,H,W] image into [N, C·p·p] patch rows (row-major over
// the patch grid, channel-major within a patch — the paper's flattening). The
// image carries no gradient, so this runs outside the tape.
func (m *ViT) patchify(img *tensor.Tensor) (*tensor.Tensor, error) {
	sh := img.Shape()
	if img.Ndim() != 3 || sh[0] != m.channels || sh[1] != m.size || sh[2] != m.size {
		return nil, fmt.Errorf("vision: ViT expects [%d,%d,%d] images, got %v", m.channels, m.size, m.size, sh)
	}
	return patchifyImage(img, m.Class.Dtype(), m.channels, m.size, m.Patch, "ViT")
}

// makeReader returns a flat-index reader over a contiguous F32/F64 tensor.
func makeReader(t *tensor.Tensor) func(int) float64 {
	if t.Dtype() == tensor.F64 {
		s := t.Storage().F64()
		return func(i int) float64 { return s[i] }
	}
	s := t.Storage().F32()
	return func(i int) float64 { return float64(s[i]) }
}

// Forward computes class logits [batch, classes] for images x [batch, C, H, W]
// (a single [C,H,W] image is also accepted and treated as batch 1).
func (m *ViT) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() == 3 {
		return m.forwardOne(ctx, x)
	}
	if x.Ndim() != 4 {
		return nil, fmt.Errorf("vision: ViT expects [batch,C,H,W] or [C,H,W], got %v", x.Shape())
	}
	// Batched path: run the whole batch through the encoder as ONE packed [B·(N+1), D] sequence set,
	// so every projection/embed/MLP/head is a single large GEMM instead of B tiny per-image ones (the
	// per-image loop's cost, BENCHMARKS.md §7 / T908). Only the core softmax(QKᵀ)V stays per-sequence
	// (MHA.ForwardBatched). Numerically identical to looping forwardOne — every op here is row-wise.
	B := x.Shape()[0]
	S := m.Pos.Shape()[0] // N+1 tokens (class + patches)
	N := S - 1
	// patchify each image (no gradient) and stack into one [B·N, C·p·p] matrix for a single Embed GEMM.
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
		if patchRows[b], err = m.patchify(one[0]); err != nil {
			return nil, err
		}
	}
	allPatches, err := backend.Execute(ctx, backend.OpConcat, patchRows, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		return nil, err
	}
	emb, err := m.Embed.Forward(ctx, allPatches[0]) // [B·N, D]
	if err != nil {
		return nil, err
	}
	// Prepend the [class] token and add position embeddings per image → packed [B·(N+1), D].
	seqs := make([]*tensor.Tensor, B)
	for b := range B {
		pb, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{emb},
			backend.SliceAttrs{Axis: 0, Start: b * N, End: (b + 1) * N})
		if err != nil {
			return nil, err
		}
		cat, err := backend.Execute(ctx, backend.OpConcat, []*tensor.Tensor{m.Class, pb[0]}, backend.ConcatAttrs{Axis: 0})
		if err != nil {
			return nil, err
		}
		wp, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{cat[0], m.Pos}, nil)
		if err != nil {
			return nil, err
		}
		seqs[b] = wp[0]
	}
	packed, err := backend.Execute(ctx, backend.OpConcat, seqs, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		return nil, err
	}
	h := packed[0]
	for _, blk := range m.Blocks {
		if h, err = blk.forwardBatched(ctx, h, B); err != nil {
			return nil, err
		}
	}
	if h, err = m.Norm.Forward(ctx, h); err != nil {
		return nil, err
	}
	// Gather each image's [class] row (row b·S) → [B, D] and run the classification head once.
	clsRows := make([]*tensor.Tensor, B)
	for b := range B {
		cr, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{h},
			backend.SliceAttrs{Axis: 0, Start: b * S, End: b*S + 1})
		if err != nil {
			return nil, err
		}
		clsRows[b] = cr[0]
	}
	cls, err := backend.Execute(ctx, backend.OpConcat, clsRows, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		return nil, err
	}
	return m.Head.Forward(ctx, cls[0]) // [B, classes]
}

// Features returns the encoder's final token representations [N+1, D] for one
// [C,H,W] image — row 0 is the [class] token, rows 1..N the patch tokens, all
// after the final LayerNorm. This is the vision-encoder output a VLM projector
// consumes (LLaVA-style systems take the patch rows, §T592); Forward is
// Features' class row through the classification head.
func (m *ViT) Features(ctx *backend.Context, img *tensor.Tensor) (*tensor.Tensor, error) {
	patches, err := m.patchify(img)
	if err != nil {
		return nil, err
	}
	x, err := m.Embed.Forward(ctx, patches)
	if err != nil {
		return nil, err
	}
	cat, err := backend.Execute(ctx, backend.OpConcat, []*tensor.Tensor{m.Class, x}, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		return nil, err
	}
	seq, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{cat[0], m.Pos}, nil)
	if err != nil {
		return nil, err
	}
	h := seq[0]
	for _, b := range m.Blocks {
		if h, err = b.forward(ctx, h); err != nil {
			return nil, err
		}
	}
	return m.Norm.Forward(ctx, h)
}

// forwardOne runs one [C,H,W] image to [1, classes] logits.
func (m *ViT) forwardOne(ctx *backend.Context, img *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := m.Features(ctx, img)
	if err != nil {
		return nil, err
	}
	cls, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{h},
		backend.SliceAttrs{Axis: 0, Start: 0, End: 1}) // the [class] row
	if err != nil {
		return nil, err
	}
	return m.Head.Forward(ctx, cls[0]) // [1, classes]
}

// forwardBatched is vitBlock.forward over a packed [batch·seq, D] batch: identical op sequence, but
// the attention runs through MHA.ForwardBatched (batched projections, per-sequence SDPA). LN, the MLP
// GEMMs, GELU and the residual adds are all row-wise, so they operate on the packed matrix unchanged.
func (b *vitBlock) forwardBatched(ctx *backend.Context, x *tensor.Tensor, batch int) (*tensor.Tensor, error) {
	h, err := b.ln1.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	a, err := b.attn.ForwardBatched(ctx, h, batch)
	if err != nil {
		return nil, err
	}
	sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{x, a}, nil)
	if err != nil {
		return nil, err
	}
	x = sum[0]
	if h, err = b.ln2.Forward(ctx, x); err != nil {
		return nil, err
	}
	if h, err = b.fc1.Forward(ctx, h); err != nil {
		return nil, err
	}
	g, err := backend.Execute(ctx, backend.OpGELU, []*tensor.Tensor{h}, nil)
	if err != nil {
		return nil, err
	}
	if h, err = b.fc2.Forward(ctx, g[0]); err != nil {
		return nil, err
	}
	sum, err = backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{x, h}, nil)
	if err != nil {
		return nil, err
	}
	return sum[0], nil
}

func (b *vitBlock) forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := b.ln1.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	a, err := b.attn.Forward(ctx, h)
	if err != nil {
		return nil, err
	}
	sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{x, a}, nil)
	if err != nil {
		return nil, err
	}
	x = sum[0]
	if h, err = b.ln2.Forward(ctx, x); err != nil {
		return nil, err
	}
	if h, err = b.fc1.Forward(ctx, h); err != nil {
		return nil, err
	}
	g, err := backend.Execute(ctx, backend.OpGELU, []*tensor.Tensor{h}, nil)
	if err != nil {
		return nil, err
	}
	if h, err = b.fc2.Forward(ctx, g[0]); err != nil {
		return nil, err
	}
	sum, err = backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{x, h}, nil)
	if err != nil {
		return nil, err
	}
	return sum[0], nil
}
