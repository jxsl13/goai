package vision

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// MAE is the Masked Autoencoder (He, Chen, Xie, Li, Dollár & Girshick 2021,
// "Masked Autoencoders Are Scalable Vision Learners", arXiv:2111.06377) — a
// self-supervised vision pre-training method. An image is cut into P×P patches;
// a large random fraction (the paper's default 75%) of the patches is masked
// out, and — crucially — the ViT ENCODER runs over ONLY the small visible
// subset (25% of patches), which is what makes MAE cheap to pre-train. A
// lightweight decoder then re-inserts a shared learned [MASK] token at every
// masked position, UNSHUFFLES the sequence back to the original patch order,
// adds decoder position embeddings, runs a few transformer blocks, and a linear
// head predicts the raw pixels of each patch. Training minimises the mean
// squared error between the predicted and original pixels, computed only on the
// masked patches.
//
// # For the AI professional
//
// MAE is masked-image-modeling, the vision analogue of BERT's masked-language
// modeling: a high mask ratio removes most of the input, an asymmetric
// encoder/decoder pair reconstructs it, and the reconstruction target is the
// pixels themselves rather than a discrete token. The asymmetry is the point —
// the deep encoder (WithMAEDim/WithMAEEncoderDepth, ViT-scale in the paper: dim
// 1024, depth 24 for ViT-Large) processes only the kept tokens, while the
// decoder is deliberately narrow and shallow (WithMAEDecoderDim/
// WithMAEDecoderDepth, the paper's default decoder is ~8 blocks at dim 512) and
// is discarded after pre-training. After training, MAE.Encode yields transfer
// features for fine-tuning downstream heads; the reconstruction objective here
// is the pre-training signal. Distinct from the contrastive/self-distillation
// SSL family (DINO/SimSiam in nn) — those learn invariances between augmented
// views; MAE learns to infer masked content from visible context.
//
// # For the newcomer
//
// Hide three-quarters of an image, then ask a network to paint back the missing
// pieces from the quarter it can still see. To do that well the network has to
// learn what images are actually made of — how a patch of sky, fur, or brick
// tends to continue — and those learned instincts are exactly what transfers to
// real tasks like classification later. MAE is efficient because the big part
// of the network only ever looks at the visible quarter, never the hidden three
// quarters.
//
// Further reading: He et al. 2021 (the paper, arXiv:2111.06377); Devlin et al.
// 2019, "BERT", for the masked-modeling lineage; Dosovitskiy et al. 2021 (ViT,
// the encoder backbone, see vision.ViT).
//
// In plain terms: mask most of the image, encode only what's left, and learn by
// reconstructing the hidden patches — cheap, scalable self-supervised vision
// pre-training.
type MAE struct {
	Patch int // square patch edge length

	Embed         *nn.Linear     // patch embedding [C·p·p → encDim]
	EncPos        *tensor.Tensor // encoder position embeddings [S, encDim]
	EncoderBlocks []*maeBlock    // encoder transformer stack (runs on the kept patches only)
	EncNorm       *nn.LayerNorm  // final encoder LayerNorm

	DecEmbed      *nn.Linear     // latent → decoder width [encDim → decDim]
	MaskToken     *tensor.Tensor // shared learned [MASK] token [1, decDim]
	DecPos        *tensor.Tensor // decoder position embeddings [S, decDim]
	DecoderBlocks []*maeBlock    // decoder transformer stack (runs on the full sequence)
	DecNorm       *nn.LayerNorm  // final decoder LayerNorm
	Head          *nn.Linear     // pixel head [decDim → C·p·p]

	channels, size int     // input geometry (C, H=W=size)
	maskRatio      float64 // fraction of patches masked out
	seed           uint64  // rng seed for the deterministic mask
}

// maeBlock is one pre-LN transformer block (shared shape with the ViT encoder
// block): x += MHA(LN1(x)); x += MLP(LN2(x)).
type maeBlock struct {
	ln1, ln2 *nn.LayerNorm
	attn     *nlp.MHA
	fc1, fc2 *nn.Linear
}

// MAEOption configures NewMAE (functional options, §C12).
type MAEOption func(*maeCfg)

type maeCfg struct {
	patch                      int
	dim, depth, heads          int // encoder width / depth / heads
	decDim, decDepth, decHeads int // decoder width / depth / heads
	mlp                        int // encoder+decoder MLP hidden width (0 → 4·dim of that stack)
	maskRatio                  float64
	dtype                      tensor.Dtype
}

// NOTE on defaults: the option defaults below are DELIBERATELY TINY — sized so the tests and
// runnable examples train in milliseconds, NOT the research configurations. For reference,
// MAE with a ViT-Large encoder (He et al. 2021, arXiv:2111.06377) uses patch 16, encoder dim
// 1024 / depth 24, and a lightweight decoder dim 512 / depth 8. Scale these up toward those
// values for real workloads; the shapes must stay consistent (each stack's dim divisible by
// its head count, patch dividing the image size).

// WithMAEMaskRatio sets the fraction of patches masked out (hidden from the encoder).
//
// In plain terms: how much of the image to hide before asking the model to reconstruct it —
// higher means a harder, more informative pretext task and a cheaper encoder (it sees only the
// 1−ratio that remains). Boundary behavior — must lie strictly in (0,1); round(ratio·S) patches
// are masked and the rest kept, so ratio near 0 masks nothing and near 1 keeps nothing (both
// error once the count collapses below one masked or one kept patch).
//
// Default 0.75 (research-grounded: the paper's headline 75% mask ratio, He et al. 2021,
// arXiv:2111.06377 — high masking is what makes the pretext task non-trivial for images).
func WithMAEMaskRatio(r float64) MAEOption { return func(c *maeCfg) { c.maskRatio = r } }

// WithMAEDim sets the encoder embedding dimension — the width of each token's vector through
// the (deep) encoder.
//
// In plain terms: how much representational room the encoder has; wider = more capacity and
// compute. Boundary behavior — must be divisible by the encoder head count (WithMAEHeads).
//
// Default 32 (a tiny demo value; MAE-Large uses encoder dim 1024, arXiv:2111.06377).
func WithMAEDim(d int) MAEOption { return func(c *maeCfg) { c.dim = d } }

// WithMAEEncoderDepth sets the number of transformer blocks in the ENCODER — the deep half of
// MAE that runs on the visible patches only.
//
// In plain terms: how many layers process the kept patches; deeper = stronger features but more
// compute. Boundary behavior — depth ≥ 1; this is the part kept for downstream transfer.
//
// Default 2 (a tiny demo value; MAE-Large uses encoder depth 24, arXiv:2111.06377).
func WithMAEEncoderDepth(n int) MAEOption { return func(c *maeCfg) { c.depth = n } }

// WithMAEHeads sets the number of attention heads in the encoder blocks.
//
// In plain terms: how many parallel attention subspaces the encoder splits into. Boundary
// behavior — must DIVIDE the encoder dim (WithMAEDim). Default 4 (a tiny demo value).
func WithMAEHeads(h int) MAEOption { return func(c *maeCfg) { c.heads = h } }

// WithMAEDecoderDim sets the DECODER embedding dimension — the width of the lightweight decoder
// that reconstructs pixels.
//
// In plain terms: the decoder is deliberately narrower than the encoder (it is thrown away
// after pre-training), so this is usually smaller than WithMAEDim. Boundary behavior — must be
// divisible by the decoder head count (WithMAEDecoderHeads).
//
// Default 16 (a tiny demo value; the paper's decoder dim is 512, arXiv:2111.06377).
func WithMAEDecoderDim(d int) MAEOption { return func(c *maeCfg) { c.decDim = d } }

// WithMAEDecoderDepth sets the number of transformer blocks in the DECODER — the lightweight
// half that re-inserts mask tokens and predicts pixels.
//
// In plain terms: kept shallow on purpose — the decoder only exists during pre-training and is
// discarded afterwards. Boundary behavior — depth ≥ 1.
//
// Default 1 (a tiny demo value; the paper's decoder depth is 8, arXiv:2111.06377 — still
// "lightweight" relative to the 24-block encoder).
func WithMAEDecoderDepth(n int) MAEOption { return func(c *maeCfg) { c.decDepth = n } }

// WithMAEDecoderHeads sets the number of attention heads in the decoder blocks.
//
// In plain terms: the decoder's parallel attention subspaces. Boundary behavior — must DIVIDE
// the decoder dim (WithMAEDecoderDim). Default 4 (a tiny demo value).
func WithMAEDecoderHeads(h int) MAEOption { return func(c *maeCfg) { c.decHeads = h } }

// WithMAEPatch sets the square patch edge length — the size of the image tiles that become
// tokens.
//
// In plain terms: how finely the image is chopped up; smaller patches = more tokens = more
// detail but more compute. Boundary behavior — patch must DIVIDE the image size (else NewMAE
// errors). Default 4 (a tiny demo value; MAE uses patch 16, arXiv:2111.06377).
func WithMAEPatch(p int) MAEOption { return func(c *maeCfg) { c.patch = p } }

// WithMAEMLP sets the hidden width of the per-block feed-forward MLP (used in BOTH the encoder
// and decoder stacks).
//
// In plain terms: the size of the little two-layer network applied to each token after
// attention. Boundary behavior — narrower than the stack dim bottlenecks the block; the standard
// is a multiple of the dim. Default 4× each stack's own dim (the paper's 4× transformer MLP
// ratio, arXiv:2111.06377).
func WithMAEMLP(m int) MAEOption { return func(c *maeCfg) { c.mlp = m } }

// WithMAEDtype sets the parameter dtype.
//
// In plain terms: the numeric precision of the weights — F32 (default) is the safe, accurate
// choice; F64 doubles memory for extra precision (mainly useful for gradient checks). Boundary
// behavior: a dtype enum, no numeric extremes. Default F32 (the standard training precision).
func WithMAEDtype(d tensor.Dtype) MAEOption { return func(c *maeCfg) { c.dtype = d } }

// NewMAE builds a Masked Autoencoder for [channels, size, size] images. The
// image size must be a multiple of the patch size, each stack's dim must be
// divisible by its head count, and the mask ratio must lie in (0,1) and leave
// at least one masked and one kept patch. The mask is deterministic given seed
// (see WithMAEMaskRatio). All parameters are initialised at 0.02·N(0,1) (the
// ViT truncated-normal scale, mirroring NewViT), and the [MASK] token likewise.
func NewMAE(channels, size int, seed uint64, opts ...MAEOption) (*MAE, error) {
	if channels <= 0 || size <= 0 {
		return nil, fmt.Errorf("vision: NewMAE needs channels, size > 0")
	}
	cfg := maeCfg{
		patch: 4, dim: 32, depth: 2, heads: 4,
		decDim: 16, decDepth: 1, decHeads: 4,
		maskRatio: 0.75, dtype: tensor.F32,
	}
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.patch <= 0 || size%cfg.patch != 0 {
		return nil, fmt.Errorf("vision: patch %d must divide image size %d", cfg.patch, size)
	}
	if cfg.dim <= 0 || cfg.decDim <= 0 {
		return nil, fmt.Errorf("vision: encoder dim %d and decoder dim %d must be > 0", cfg.dim, cfg.decDim)
	}
	if cfg.heads <= 0 || cfg.dim%cfg.heads != 0 {
		return nil, fmt.Errorf("vision: encoder dim %d must be divisible by heads %d", cfg.dim, cfg.heads)
	}
	if cfg.decHeads <= 0 || cfg.decDim%cfg.decHeads != 0 {
		return nil, fmt.Errorf("vision: decoder dim %d must be divisible by heads %d", cfg.decDim, cfg.decHeads)
	}
	if cfg.depth <= 0 || cfg.decDepth <= 0 {
		return nil, fmt.Errorf("vision: encoder depth %d and decoder depth %d must be > 0", cfg.depth, cfg.decDepth)
	}
	if !(cfg.maskRatio > 0 && cfg.maskRatio < 1) {
		return nil, fmt.Errorf("vision: mask ratio %g must be in (0,1)", cfg.maskRatio)
	}
	grid := size / cfg.patch
	s := grid * grid
	keep := s - maeMaskCount(cfg.maskRatio, s)
	if keep < 1 || keep >= s {
		return nil, fmt.Errorf("vision: mask ratio %g leaves %d kept of %d patches — need ≥1 masked and ≥1 kept", cfg.maskRatio, keep, s)
	}
	patchDim := channels * cfg.patch * cfg.patch

	m := &MAE{
		Patch: cfg.patch, channels: channels, size: size,
		maskRatio: cfg.maskRatio, seed: seed,
	}
	m.Embed = nn.NewLinear(cfg.dtype, patchDim, cfg.dim, seed)
	m.EncPos = scaleInPlace(tensor.Randn(cfg.dtype, seed+1, tensor.Shape{s, cfg.dim}), 0.02)
	for i := range cfg.depth {
		blk, err := newMAEBlock(cfg.dtype, cfg.dim, cfg.heads, maeMLP(cfg.mlp, cfg.dim), seed+100+uint64(i)*8)
		if err != nil {
			return nil, err
		}
		m.EncoderBlocks = append(m.EncoderBlocks, blk)
	}
	m.EncNorm = nn.NewLayerNorm(cfg.dtype, cfg.dim)

	m.DecEmbed = nn.NewLinear(cfg.dtype, cfg.dim, cfg.decDim, seed+2)
	m.MaskToken = scaleInPlace(tensor.Randn(cfg.dtype, seed+3, tensor.Shape{1, cfg.decDim}), 0.02)
	m.DecPos = scaleInPlace(tensor.Randn(cfg.dtype, seed+4, tensor.Shape{s, cfg.decDim}), 0.02)
	for i := range cfg.decDepth {
		blk, err := newMAEBlock(cfg.dtype, cfg.decDim, cfg.decHeads, maeMLP(cfg.mlp, cfg.decDim), seed+500+uint64(i)*8)
		if err != nil {
			return nil, err
		}
		m.DecoderBlocks = append(m.DecoderBlocks, blk)
	}
	m.DecNorm = nn.NewLayerNorm(cfg.dtype, cfg.decDim)
	m.Head = nn.NewLinear(cfg.dtype, cfg.decDim, patchDim, seed+5)
	return m, nil
}

// maeMLP resolves the per-block MLP width: explicit override, else 4× the stack's own dim.
func maeMLP(override, dim int) int {
	if override > 0 {
		return override
	}
	return 4 * dim
}

// maeMaskCount is the number of masked patches: round(maskRatio·S) (the paper's rounding).
func maeMaskCount(maskRatio float64, s int) int {
	return int(math.Round(maskRatio * float64(s)))
}

// newMAEBlock builds one pre-LN transformer block at the given width/heads/mlp.
func newMAEBlock(dtype tensor.Dtype, dim, heads, mlp int, seed uint64) (*maeBlock, error) {
	wq := scaleInPlace(tensor.Randn(dtype, seed, tensor.Shape{dim, dim}), 0.02)
	wk := scaleInPlace(tensor.Randn(dtype, seed+1, tensor.Shape{dim, dim}), 0.02)
	wv := scaleInPlace(tensor.Randn(dtype, seed+2, tensor.Shape{dim, dim}), 0.02)
	wo := scaleInPlace(tensor.Randn(dtype, seed+3, tensor.Shape{dim, dim}), 0.02)
	attn, err := nlp.NewMHA(heads, wq, wk, wv, wo)
	if err != nil {
		return nil, err
	}
	return &maeBlock{
		ln1:  nn.NewLayerNorm(dtype, dim),
		ln2:  nn.NewLayerNorm(dtype, dim),
		attn: attn,
		fc1:  nn.NewLinear(dtype, dim, mlp, seed+4),
		fc2:  nn.NewLinear(dtype, mlp, dim, seed+5),
	}, nil
}

func (b *maeBlock) forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
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

// numPatches is S = (size/patch)².
func (m *MAE) numPatches() int {
	g := m.size / m.Patch
	return g * g
}

// Params returns every trainable tensor (for the optimizers) — encoder, decoder,
// the [MASK] token, position embeddings, and the pixel head.
func (m *MAE) Params() []*tensor.Tensor {
	out := []*tensor.Tensor{m.EncPos, m.MaskToken, m.DecPos}
	out = append(out, m.Embed.Params()...)
	for _, b := range m.EncoderBlocks {
		out = append(out, b.params()...)
	}
	out = append(out, m.EncNorm.Params()...)
	out = append(out, m.DecEmbed.Params()...)
	for _, b := range m.DecoderBlocks {
		out = append(out, b.params()...)
	}
	out = append(out, m.DecNorm.Params()...)
	out = append(out, m.Head.Params()...)
	return out
}

func (b *maeBlock) params() []*tensor.Tensor {
	out := b.ln1.Params()
	out = append(out, b.attn.Wq, b.attn.Wk, b.attn.Wv, b.attn.Wo)
	out = append(out, b.ln2.Params()...)
	out = append(out, b.fc1.Params()...)
	out = append(out, b.fc2.Params()...)
	return out
}

// Mask samples the deterministic keep/mask partition for this model's seed and
// patch count: it shuffles the S patch indices and keeps the first S−round(
// maskRatio·S). keep and masked are DISJOINT and together cover every patch
// index 0..S−1 exactly once (a valid partition, §V16.1). The result depends only
// on the model seed, so repeated calls agree. keep[k] is the original patch
// index of the k-th token the encoder sees (in shuffled order); masked lists the
// hidden patches. Exposed so callers can inspect which patches were reconstructed.
func (m *MAE) Mask() (keep, masked []int) {
	s := m.numPatches()
	perm := rand.New(rand.NewSource(int64(m.seed))).Perm(s) //nolint:gosec // deterministic mask, not cryptographic
	keepCount := s - maeMaskCount(m.maskRatio, s)
	keep = append([]int(nil), perm[:keepCount]...)
	masked = append([]int(nil), perm[keepCount:]...)
	return keep, masked
}

// patchify flattens one [C,H,W] image into [S, C·p·p] patch rows (row-major over
// the patch grid, channel-major within a patch — the ViT flattening, matching
// vision.ViT). The image carries no gradient, so this runs outside the tape.
func (m *MAE) patchify(img *tensor.Tensor) (*tensor.Tensor, error) {
	sh := img.Shape()
	if img.Ndim() != 3 || sh[0] != m.channels || sh[1] != m.size || sh[2] != m.size {
		return nil, fmt.Errorf("vision: MAE expects [%d,%d,%d] images, got %v", m.channels, m.size, m.size, sh)
	}
	p := m.Patch
	grid := m.size / p
	n := grid * grid
	read := makeReader(img.Contiguous())
	data := make([]float64, 0, n*m.channels*p*p)
	for py := range grid {
		for px := range grid {
			for c := range m.channels {
				for dy := range p {
					for dx := range p {
						data = append(data, read(((c*m.size)+py*p+dy)*m.size+px*p+dx))
					}
				}
			}
		}
	}
	out := tensor.New(m.EncPos.Dtype(), tensor.Shape{n, m.channels * p * p})
	switch out.Dtype() {
	case tensor.F64:
		copy(out.Storage().F64(), data)
	case tensor.F32:
		dst := out.Storage().F32()
		for i, v := range data {
			dst[i] = float32(v)
		}
	default:
		return nil, fmt.Errorf("vision: MAE supports F32/F64, got %v", out.Dtype())
	}
	return out, nil
}

// unshuffleRows reassembles the full [S,D] decoder input in ORIGINAL patch order
// (§V16.2): kept position keep[k] takes row k of proj[keepCount,D], every masked
// position takes the shared maskToken[1,D]. keep and masked must partition
// 0..S−1. Because a patch masked at original position i lands at output row i,
// the decoder — and thus the pixel head — predicts each patch in its original
// slot. Slice+Concat keep the assembly differentiable (grad routes each row back
// to proj or the mask token).
func unshuffleRows(ctx *backend.Context, proj, maskToken *tensor.Tensor, keep, masked []int, s int) (*tensor.Tensor, error) {
	full := make([]*tensor.Tensor, s)
	for k, orig := range keep {
		row, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{proj},
			backend.SliceAttrs{Axis: 0, Start: k, End: k + 1})
		if err != nil {
			return nil, err
		}
		full[orig] = row[0]
	}
	for _, orig := range masked {
		full[orig] = maskToken // [1, D], the same token at every masked slot
	}
	out, err := backend.Execute(ctx, backend.OpConcat, full, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// gatherRows selects rows idx (in order) from x[N,D] → [len(idx),D] via a
// Slice-per-row + Concat, so gradients scatter back to the originating rows.
func gatherRows(ctx *backend.Context, x *tensor.Tensor, idx []int) (*tensor.Tensor, error) {
	rows := make([]*tensor.Tensor, len(idx))
	for k, i := range idx {
		r, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{x},
			backend.SliceAttrs{Axis: 0, Start: i, End: i + 1})
		if err != nil {
			return nil, err
		}
		rows[k] = r[0]
	}
	out, err := backend.Execute(ctx, backend.OpConcat, rows, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Encode runs the MAE encoder over the VISIBLE patches only and returns the
// encoder latent [keepCount, encDim] together with the keep/mask partition
// (§V16.1). This is the defining efficiency of MAE — the encoder input length is
// keepCount (= S−round(maskRatio·S)), NOT S. It is also the representation a
// downstream fine-tuning head would consume after pre-training. Accepts one
// [C,H,W] image.
func (m *MAE) Encode(ctx *backend.Context, img *tensor.Tensor) (latent *tensor.Tensor, keep, masked []int, err error) {
	patches, err := m.patchify(img)
	if err != nil {
		return nil, nil, nil, err
	}
	emb, err := m.Embed.Forward(ctx, patches) // [S, encDim]
	if err != nil {
		return nil, nil, nil, err
	}
	// Add position embeddings to ALL patches BEFORE masking, so each kept token
	// carries its own position (the paper adds pos-embed pre-masking).
	posd, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{emb, m.EncPos}, nil)
	if err != nil {
		return nil, nil, nil, err
	}
	keep, masked = m.Mask()
	h, err := gatherRows(ctx, posd[0], keep) // [keepCount, encDim] — encoder sees ONLY these
	if err != nil {
		return nil, nil, nil, err
	}
	for _, b := range m.EncoderBlocks {
		if h, err = b.forward(ctx, h); err != nil {
			return nil, nil, nil, err
		}
	}
	latent, err = m.EncNorm.Forward(ctx, h)
	if err != nil {
		return nil, nil, nil, err
	}
	return latent, keep, masked, nil
}

// Reconstruct runs the full MAE forward pass on one [C,H,W] image and returns:
//
//   - pred: per-patch pixel predictions [S, C·p·p], in ORIGINAL patch order — a
//     patch masked at position i has its prediction at row i (§V16.2, the
//     unshuffle restores order).
//   - masked: the indices of the reconstructed (hidden) patches.
//   - loss: the training objective — the mean squared error between pred and the
//     original pixels, averaged over pixels and over the MASKED patches ONLY
//     (§V16.3); visible patches do not contribute. loss ≥ 0, and = 0 exactly
//     when pred equals the original pixels on every masked patch.
//
// The pass: embed+position all patches, encode ONLY the kept subset, project the
// latent to the decoder width, re-insert the shared [MASK] token at every masked
// position while UNSHUFFLING back to original order, add decoder positions, run
// the decoder blocks, and apply the pixel head.
func (m *MAE) Reconstruct(ctx *backend.Context, img *tensor.Tensor) (pred *tensor.Tensor, masked []int, loss *tensor.Tensor, err error) {
	target, err := m.patchify(img) // original pixels [S, patchDim], the reconstruction target (no grad)
	if err != nil {
		return nil, nil, nil, err
	}
	latent, keep, masked, err := m.Encode(ctx, img) // [keepCount, encDim]
	if err != nil {
		return nil, nil, nil, err
	}
	proj, err := m.DecEmbed.Forward(ctx, latent) // [keepCount, decDim]
	if err != nil {
		return nil, nil, nil, err
	}

	// UNSHUFFLE: build the full [S, decDim] sequence in ORIGINAL patch order —
	// kept positions take their projected latent row, masked positions take the
	// shared learned [MASK] token (§V16.2).
	seq, err := unshuffleRows(ctx, proj, m.MaskToken, keep, masked, m.numPatches())
	if err != nil {
		return nil, nil, nil, err
	}
	dec, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{seq, m.DecPos}, nil) // + decoder positions
	if err != nil {
		return nil, nil, nil, err
	}
	h := dec[0]
	for _, b := range m.DecoderBlocks {
		if h, err = b.forward(ctx, h); err != nil {
			return nil, nil, nil, err
		}
	}
	if h, err = m.DecNorm.Forward(ctx, h); err != nil {
		return nil, nil, nil, err
	}
	pred, err = m.Head.Forward(ctx, h) // [S, patchDim] in original order
	if err != nil {
		return nil, nil, nil, err
	}

	loss, err = m.maskedMSE(ctx, pred, target, masked)
	if err != nil {
		return nil, nil, nil, err
	}
	return pred, masked, loss, nil
}

// maskedMSE is the mean squared error between pred and target over the MASKED
// patches only, averaged per pixel (normalized per patch): gather the masked
// rows of both, diff = pred − target, then mean(diff⊙diff). Squaring is OpMul
// (B60: not a scaling op), so the whole loss is differentiable and gradients
// flow only through the masked rows (§V16.3).
func (m *MAE) maskedMSE(ctx *backend.Context, pred, target *tensor.Tensor, masked []int) (*tensor.Tensor, error) {
	predM, err := gatherRows(ctx, pred, masked)
	if err != nil {
		return nil, err
	}
	tgtM, err := gatherRows(ctx, target, masked) // target has no tape node → constant
	if err != nil {
		return nil, err
	}
	diff, err := backend.Execute(ctx, backend.OpSub, []*tensor.Tensor{predM, tgtM}, nil)
	if err != nil {
		return nil, err
	}
	sq, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{diff[0], diff[0]}, nil)
	if err != nil {
		return nil, err
	}
	mean, err := backend.Execute(ctx, backend.OpMean, []*tensor.Tensor{sq[0]}, nil)
	if err != nil {
		return nil, err
	}
	return mean[0], nil
}

// MAELoss is the training objective for one [C,H,W] image: the masked-patch
// reconstruction MSE (§V16.3). It is Reconstruct's loss without the intermediate
// tensors — drive the standard tape → loss → Backward → Step loop with it.
func (m *MAE) MAELoss(ctx *backend.Context, img *tensor.Tensor) (*tensor.Tensor, error) {
	_, _, loss, err := m.Reconstruct(ctx, img)
	return loss, err
}
