package vision

import (
	"fmt"
	"math"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// SwinTransformer is the Swin Transformer (Liu, Lin, Cao, Hu, Wei, Zhang, Lin &
// Guo 2021, "Swin Transformer: Hierarchical Vision Transformer using Shifted
// Windows", arXiv:2103.14030): a hierarchical vision transformer that computes
// self-attention inside LOCAL non-overlapping windows — so its cost grows
// LINEARLY with the image area instead of quadratically like the global-attention
// ViT — and SHIFTS those windows by half a window between consecutive blocks so
// information crosses window boundaries. Between stages a PATCH-MERGING layer
// concatenates each 2×2 neighborhood of tokens and projects it, halving the
// spatial resolution and doubling the channel width to build a feature pyramid
// (the "hierarchical" part — the trait that makes Swin a general vision backbone,
// not just a classifier).
//
// # For the AI professional
//
// The forward pass is: patch-embed the image into an H×W×C token grid (H=W=
// imageSize/patchSize), then run len(depths) stages. A stage's blocks alternate
// W-MSA (regular window multi-head self-attention) and SW-MSA (shifted-window
// attention); each block is the standard pre-LayerNorm transformer block
// x = x + Attn(LN1(x)); x = x + MLP(LN2(x)) with attention RESTRICTED to M×M
// windows and a learned relative-position bias added to the logits. SW-MSA
// cyclically shifts the grid by (M/2, M/2) before windowing so adjacent windows
// mix, and applies an additive −∞ attention mask so the tokens the shift wrapped
// in from the opposite edge do not attend across the wrap. Patch merging between
// stages does the 2×2-concat → LayerNorm → Linear(4C→2C) downsample. A final
// LayerNorm, global-average-pool over the surviving tokens, and Linear head
// produce the class logits. Everything composes the verified nn/nlp primitives
// (Linear, LayerNorm, matmul, softmax, and the differentiable OpEmbed row-gather
// for windowing/shifting), so the standard tape → loss → Backward → Step loop
// trains it and gradients reach the relative-position-bias table.
//
// # For the newcomer
//
// The global ViT lets every patch look at every other patch, which is expensive
// for large images. Swin instead lets a patch look only at its neighbors inside a
// small tile ("window"), which is cheap; to stop the tiles from becoming isolated
// it slides the tile grid halfway every other layer so neighbors that were split
// apart share a tile next time. Periodically it merges each 2×2 block of tiles
// into one — like zooming out — so deeper layers see coarser, larger structures,
// exactly the coarse-to-fine pyramid a convolutional network builds. Build one
// with NewSwin, train it with the loop in the examples, and call Forward for a
// score per class.
//
// Further reading: Liu et al. 2021 (the paper); Dosovitskiy et al. 2021 for the
// global-attention ViT that Swin localizes; Liu et al. 2022, "Swin Transformer
// V2", for the scaled-up successor.
type SwinTransformer struct {
	Patch  int            // square patch edge length (patchSize)
	Window int            // window edge length M
	Embed  *nn.Linear     // patch embedding [C·p·p → embedC]
	Stages [][]*SwinBlock // per-stage block lists (W-MSA/SW-MSA alternating)
	Merges []*swinMerge   // patch-merging layers between stages (len = stages−1)
	Norm   *nn.LayerNorm  // final LayerNorm before the head
	Head   *nn.Linear     // classification head [Cfinal → classes]

	channels, size, grid int          // input geometry: C, H=W=size, token grid edge
	dtype                tensor.Dtype // parameter dtype
}

// SwinBlock is one windowed pre-LayerNorm transformer block:
// x = x + Attn(LN1(x)); x = x + MLP(LN2(x)), where Attn is multi-head
// self-attention computed INDEPENDENTLY within each M×M window (a W-MSA block),
// with a per-head learned relative-position bias added to the logits — and, when
// Shift is set, computed on the grid cyclically shifted by (M/2, M/2) with an
// additive mask that blocks the wrapped-edge tokens from attending across the
// wrap (an SW-MSA block). With a single window covering the whole grid and the
// relative bias disabled, the block is numerically identical to a standard global
// ViT transformer block (§V16.1).
type SwinBlock struct {
	Heads          int            // attention heads
	Dim            int            // channel width C of this stage
	Window         int            // window edge length M
	Shift          bool           // SW-MSA (true) vs W-MSA (false)
	Wq, Wk, Wv, Wo *tensor.Tensor // attention projections, each [C, C]
	LN1, LN2       *nn.LayerNorm  // pre-norms for attention and MLP
	FC1, FC2       *nn.Linear     // MLP: C → mlp → C
	Rel            *swinRelBias   // relative-position bias (nil = disabled)
}

// swinRelBias is the Swin relative-position bias: a learned per-head scalar added
// to every attention logit inside a window, indexed by the relative (row,col)
// offset of the two tokens (Liu et al. 2021 §3.2). The table is [(2M−1)², heads];
// a constant one-hot selector turns the lookup into a differentiable matmul
// against the table (the nn.T5RelativeBias trick), so gradients accumulate into
// each relative-offset entry.
type swinRelBias struct {
	Table  *tensor.Tensor // [(2M−1)², heads], trainable
	oneHot *tensor.Tensor // [M²·M², (2M−1)²], constant selector
	m      int            // window edge M
}

// swinMerge is one patch-merging layer (Liu et al. 2021 §3.1): it groups each 2×2
// neighborhood of the [H,W,C] token grid, concatenates the four tokens' features
// into 4C, LayerNorm-normalizes, and linearly projects 4C→2C — halving H and W
// and doubling the channel width.
type swinMerge struct {
	Norm   *nn.LayerNorm // over the concatenated 4C features
	Reduce *nn.Linear    // 4C → 2C
}

// SwinOption configures NewSwin (functional options, §C12).
type SwinOption func(*swinCfg)

type swinCfg struct {
	mlpRatio int
	channels int
	relBias  bool
}

// WithSwinMLPRatio sets the per-block feed-forward expansion ratio — the MLP
// hidden width is ratio·C for a stage of width C.
//
// In plain terms: the size of the little two-layer network applied to each token
// after attention, relative to the channel width; wider = more per-token
// capacity. Boundary behavior — ratio 1 makes the MLP no wider than the block;
// the standard is a small integer multiple. Default 4 (research-grounded: the
// paper's 4× expansion, Liu et al. 2021 §3.1 — the near-universal transformer MLP
// ratio).
func WithSwinMLPRatio(ratio int) SwinOption { return func(c *swinCfg) { c.mlpRatio = ratio } }

// WithSwinChannels sets the number of input image channels the patch embedding
// expects.
//
// In plain terms: 3 for a color (RGB) image, 1 for grayscale. Boundary behavior —
// must be > 0 and must match the channel count of the images passed to Forward.
// Default 3 (the standard RGB input the paper trains on).
func WithSwinChannels(c int) SwinOption { return func(cfg *swinCfg) { cfg.channels = c } }

// WithSwinRelativeBias toggles the learned relative-position bias added to the
// windowed attention logits.
//
// In plain terms: Swin gives each pair of in-window positions a small learned
// nudge based on how far apart they are (row/column offset), encoding spatial
// structure the way position embeddings do for text. Boundary behavior —
// disabling it (false) makes windowed attention pure content-based dot-product
// attention, so a single-window block collapses EXACTLY to a global ViT block
// (§V16.1). Default true (research-grounded: Liu et al. 2021 §3.2 report it the
// strongest of the position-encoding choices they ablate).
func WithSwinRelativeBias(on bool) SwinOption { return func(c *swinCfg) { c.relBias = on } }

// NewSwin builds a Swin Transformer classifier for square images and the given
// number of classes. windowSize is the attention window edge M; embedC is the
// first stage's channel width (doubled by every patch merge); depths gives the
// number of blocks per stage (len(depths) stages, blocks alternating W-MSA/SW-MSA
// starting with W-MSA), and heads the attention head count per stage
// (len(heads) == len(depths)). The image channel count defaults to 3 (override
// with WithSwinChannels). Deterministic via seed.
//
// Geometry constraints (all validated): imageSize must be a multiple of
// patchSize; the token grid edge (imageSize/patchSize) and every subsequently
// halved stage resolution must be a positive multiple of windowSize and — except
// the last stage — even (so patch merging can halve it); windowSize must be even
// (the shift is M/2); and each stage's channel width (embedC·2^stage) must be
// divisible by that stage's head count. The relative-position table and attention
// projections initialize at 0.02·N(0,1) (the paper's truncated-normal scale);
// patch-merge and MLP weights use Xavier-uniform via nn.NewLinear.
func NewSwin(dtype tensor.Dtype, imageSize, patchSize, windowSize, embedC int, depths, heads []int, classes int, seed uint64, opts ...SwinOption) (*SwinTransformer, error) {
	cfg := swinCfg{mlpRatio: 4, channels: 3, relBias: true}
	for _, o := range opts {
		o(&cfg)
	}
	if imageSize <= 0 || patchSize <= 0 || windowSize <= 0 || embedC <= 0 || classes <= 0 || cfg.channels <= 0 {
		return nil, fmt.Errorf("vision: NewSwin needs imageSize, patchSize, windowSize, embedC, classes, channels > 0")
	}
	if imageSize%patchSize != 0 {
		return nil, fmt.Errorf("vision: patchSize %d must divide imageSize %d", patchSize, imageSize)
	}
	if windowSize%2 != 0 {
		return nil, fmt.Errorf("vision: windowSize %d must be even (shift is M/2)", windowSize)
	}
	if len(depths) == 0 || len(depths) != len(heads) {
		return nil, fmt.Errorf("vision: depths (%d) and heads (%d) must be equal-length and non-empty", len(depths), len(heads))
	}
	if cfg.mlpRatio <= 0 {
		return nil, fmt.Errorf("vision: MLP ratio must be > 0, got %d", cfg.mlpRatio)
	}
	grid := imageSize / patchSize
	for s := range depths {
		if depths[s] <= 0 || heads[s] <= 0 {
			return nil, fmt.Errorf("vision: stage %d needs depth>0 and heads>0", s)
		}
		res := grid >> uint(s)
		c := embedC << uint(s)
		if res <= 0 || res%windowSize != 0 {
			return nil, fmt.Errorf("vision: stage %d resolution %d must be a positive multiple of windowSize %d", s, res, windowSize)
		}
		if s < len(depths)-1 && res%2 != 0 {
			return nil, fmt.Errorf("vision: stage %d resolution %d must be even to patch-merge", s, res)
		}
		if c%heads[s] != 0 {
			return nil, fmt.Errorf("vision: stage %d width %d not divisible by heads %d", s, c, heads[s])
		}
	}

	m := &SwinTransformer{
		Patch: patchSize, Window: windowSize,
		channels: cfg.channels, size: imageSize, grid: grid, dtype: dtype,
	}
	m.Embed = nn.NewLinear(dtype, cfg.channels*patchSize*patchSize, embedC, seed)
	for si := range depths {
		c := embedC << uint(si)
		mlp := cfg.mlpRatio * c
		var blocks []*SwinBlock
		for bi := range depths[si] {
			s := seed + 100*uint64(si) + 10*uint64(bi) + 1
			blk := &SwinBlock{
				Heads: heads[si], Dim: c, Window: windowSize, Shift: bi%2 == 1,
				Wq:  scaleInPlace(tensor.Randn(dtype, s, tensor.Shape{c, c}), 0.02),
				Wk:  scaleInPlace(tensor.Randn(dtype, s+1, tensor.Shape{c, c}), 0.02),
				Wv:  scaleInPlace(tensor.Randn(dtype, s+2, tensor.Shape{c, c}), 0.02),
				Wo:  scaleInPlace(tensor.Randn(dtype, s+3, tensor.Shape{c, c}), 0.02),
				LN1: nn.NewLayerNorm(dtype, c),
				LN2: nn.NewLayerNorm(dtype, c),
				FC1: nn.NewLinear(dtype, c, mlp, s+4),
				FC2: nn.NewLinear(dtype, mlp, c, s+5),
			}
			if cfg.relBias {
				blk.Rel = newSwinRelBias(dtype, windowSize, heads[si], s+6)
			}
			blocks = append(blocks, blk)
		}
		m.Stages = append(m.Stages, blocks)
		if si < len(depths)-1 {
			m.Merges = append(m.Merges, &swinMerge{
				Norm:   nn.NewLayerNorm(dtype, 4*c),
				Reduce: nn.NewLinear(dtype, 4*c, 2*c, seed+1000+uint64(si)),
			})
		}
	}
	cFinal := embedC << uint(len(depths)-1)
	m.Norm = nn.NewLayerNorm(dtype, cFinal)
	m.Head = nn.NewLinear(dtype, cFinal, classes, seed+9999)
	return m, nil
}

// newSwinRelBias builds the relative-position bias for an M×M window with a
// 0.02·N(0,1) table and the constant one-hot selector matrix.
func newSwinRelBias(dtype tensor.Dtype, m, heads int, seed uint64) *swinRelBias {
	nb := (2*m - 1) * (2*m - 1)
	table := scaleInPlace(tensor.Randn(dtype, seed, tensor.Shape{nb, heads}), 0.02)
	relIdx := swinRelIndex(m)
	n := m * m
	oneHot := tensor.New(dtype, tensor.Shape{n * n, nb})
	for p := range n * n {
		oneHot.SetF64(1, p, relIdx[p])
	}
	return &swinRelBias{Table: table, oneHot: oneHot, m: m}
}

// Params returns every trainable tensor (for the optimizers).
func (m *SwinTransformer) Params() []*tensor.Tensor {
	out := append([]*tensor.Tensor{}, m.Embed.Params()...)
	for _, blocks := range m.Stages {
		for _, b := range blocks {
			out = append(out, b.LN1.Params()...)
			out = append(out, b.Wq, b.Wk, b.Wv, b.Wo)
			if b.Rel != nil {
				out = append(out, b.Rel.Table)
			}
			out = append(out, b.LN2.Params()...)
			out = append(out, b.FC1.Params()...)
			out = append(out, b.FC2.Params()...)
		}
	}
	for _, mg := range m.Merges {
		out = append(out, mg.Norm.Params()...)
		out = append(out, mg.Reduce.Params()...)
	}
	out = append(out, m.Norm.Params()...)
	out = append(out, m.Head.Params()...)
	return out
}

// Forward computes class logits [batch, classes] for images x [batch, C, H, W]
// (a single [C,H,W] image is also accepted and treated as batch 1).
func (m *SwinTransformer) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() == 3 {
		return m.forwardOne(ctx, x)
	}
	if x.Ndim() != 4 {
		return nil, fmt.Errorf("vision: Swin expects [batch,C,H,W] or [C,H,W], got %v", x.Shape())
	}
	// Batched path: run the batch as ONE packed [B·N, C] token grid so every projection / MLP / merge /
	// LayerNorm / head is a single large GEMM. The per-image spatial ops (cyclic shift, window
	// partition, patch-merge grouping) stay per-image gathers — cheap index permutations — but the
	// windowed attention runs over all B·numWin windows at once (projections batch). Identical to
	// looping forwardOne (mirrors the ViT/MLPMixer T908 fixes).
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
		if patchRows[b], err = m.swinPatchify(one[0]); err != nil {
			return nil, err
		}
	}
	allPatches, err := backend.Execute(ctx, backend.OpConcat, patchRows, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		return nil, err
	}
	xb, err := m.Embed.Forward(ctx, allPatches[0]) // [B·N, embedC]
	if err != nil {
		return nil, err
	}
	h, w := m.grid, m.grid
	for si, blocks := range m.Stages {
		for _, blk := range blocks {
			if xb, err = blk.forwardBatched(ctx, xb, h, w, B); err != nil {
				return nil, err
			}
		}
		if si < len(m.Merges) {
			if xb, h, w, err = m.Merges[si].forwardBatched(ctx, xb, h, w, B); err != nil {
				return nil, err
			}
		}
	}
	if xb, err = m.Norm.Forward(ctx, xb); err != nil {
		return nil, err
	}
	// Global average pool over each image's remaining tokens → [B, Cfinal], then the head once.
	tok := xb.Shape()[0] / B
	pooled := make([]*tensor.Tensor, B)
	for b := range B {
		xbi, err := backend.Execute(ctx, backend.OpSlice, []*tensor.Tensor{xb},
			backend.SliceAttrs{Axis: 0, Start: b * tok, End: (b + 1) * tok})
		if err != nil {
			return nil, err
		}
		pm, err := backend.Execute(ctx, backend.OpMean, []*tensor.Tensor{xbi[0]},
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

// forwardBatched is SwinBlock.Forward over a packed [batch·N, C] grid. LayerNorms and the MLP are
// row-wise → batch unchanged. The cyclic shift and window partition are per-image gathers (built per
// image, cheap), but their windows are concatenated so windowedAttention projects Q/K/V for all
// batch·numWin windows in single GEMMs. Identical to Forward per image.
func (b *SwinBlock) forwardBatched(ctx *backend.Context, x *tensor.Tensor, h, w, batch int) (*tensor.Tensor, error) {
	m := b.Window
	shift := 0
	if b.Shift && h > m {
		shift = m / 2
	}
	N := h * w
	numWin := (h / m) * (w / m)
	ln1, err := b.LN1.Forward(ctx, x) // [batch·N, C] batched
	if err != nil {
		return nil, err
	}
	shiftIdx := swinShiftIdx(h, w, shift)
	partIdx := swinPartitionIdx(h, w, m)
	// Fuse the per-image slice → shift-gather → partition-gather → concat into ONE
	// batched row-gather over ln1[batch·N, C]. Composing the two permutations
	// (combo[o] = shiftIdx[partIdx[o]]) and baking the i·N image offset into the
	// index picks the identical source row for every output row, so allWin is
	// byte-identical to the per-image pipeline — with ~batch× fewer dispatches.
	combo := partIdx
	if shift > 0 {
		combo = make([]int, len(partIdx))
		for o, pp := range partIdx {
			combo[o] = shiftIdx[pp]
		}
	}
	batchIdx := make([]int, batch*N)
	for i := range batch {
		off := i * N
		for o, c := range combo {
			batchIdx[off+o] = off + c
		}
	}
	allWin, err := swinGather(ctx, ln1, batchIdx, b.dtype()) // [batch·numWin·M², C]
	if err != nil {
		return nil, err
	}
	var masks []*tensor.Tensor
	if shift > 0 {
		masks = swinBuildMasks(h, w, m, shift, b.dtype())
	}
	att, err := b.windowedAttention(ctx, allWin, batch*numWin, m, masks)
	if err != nil {
		return nil, err
	}
	// per-image inverse partition → inverse shift, back to [batch·N, C].
	invPart := swinInverse(partIdx)
	invShift := swinInverse(shiftIdx)
	winRows := numWin * m * m
	// Symmetric inverse fusion: one batched gather over att[batch·winRows, C] with
	// the composed inverse permutation (invCombo[o] = invPart[invShift[o]]) and the
	// i·winRows offset baked in — byte-identical to the per-image inverse pipeline.
	invCombo := invPart
	if shift > 0 {
		invCombo = make([]int, len(invPart))
		for o, sh := range invShift {
			invCombo[o] = invPart[sh]
		}
	}
	outIdx := make([]int, batch*winRows)
	for i := range batch {
		off := i * winRows
		for o, c := range invCombo {
			outIdx[off+o] = off + c
		}
	}
	a, err := swinGather(ctx, att, outIdx, b.dtype()) // [batch·N, C]
	if err != nil {
		return nil, err
	}
	x, err = swinExec2(ctx, backend.OpAdd, nil, x, a)
	if err != nil {
		return nil, err
	}
	// MLP branch — all row-wise, batches over the packed matrix.
	ln2, err := b.LN2.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	f1, err := b.FC1.Forward(ctx, ln2)
	if err != nil {
		return nil, err
	}
	gelu, err := swinExec1a(ctx, backend.OpGELU, nil, f1)
	if err != nil {
		return nil, err
	}
	f2, err := b.FC2.Forward(ctx, gelu)
	if err != nil {
		return nil, err
	}
	return swinExec2(ctx, backend.OpAdd, nil, x, f2)
}

// forwardOne runs one [C,H,W] image to [1, classes] logits: patch-embed → stages
// (windowed blocks + patch merges) → final LayerNorm → global-average-pool →
// head.
func (m *SwinTransformer) forwardOne(ctx *backend.Context, img *tensor.Tensor) (*tensor.Tensor, error) {
	patches, err := m.swinPatchify(img)
	if err != nil {
		return nil, err
	}
	x, err := m.Embed.Forward(ctx, patches) // [N, embedC], N = grid²
	if err != nil {
		return nil, err
	}
	h, w := m.grid, m.grid
	for si, blocks := range m.Stages {
		for _, blk := range blocks {
			if x, err = blk.Forward(ctx, x, h, w); err != nil {
				return nil, err
			}
		}
		if si < len(m.Merges) {
			if x, h, w, err = m.Merges[si].forward(ctx, x, h, w); err != nil {
				return nil, err
			}
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	pooled, err := backend.Execute(ctx, backend.OpMean, []*tensor.Tensor{x},
		backend.ReduceAttrs{Axes: []int{0}, KeepDims: true}) // [1, Cfinal]
	if err != nil {
		return nil, err
	}
	return m.Head.Forward(ctx, pooled[0]) // [1, classes]
}

// swinPatchify flattens one [C,H,W] image into [grid², C·p·p] patch rows
// (row-major over the patch grid, channel-major within a patch — the standard
// flattening; identical to ViT.patchify). The image carries no gradient, so this
// runs outside the tape.
func (m *SwinTransformer) swinPatchify(img *tensor.Tensor) (*tensor.Tensor, error) {
	sh := img.Shape()
	if img.Ndim() != 3 || sh[0] != m.channels || sh[1] != m.size || sh[2] != m.size {
		return nil, fmt.Errorf("vision: Swin expects [%d,%d,%d] images, got %v", m.channels, m.size, m.size, sh)
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
	out := tensor.New(m.dtype, tensor.Shape{n, m.channels * p * p})
	switch out.Dtype() {
	case tensor.F64:
		copy(out.Storage().F64(), data)
	case tensor.F32:
		dst := out.Storage().F32()
		for i, v := range data {
			dst[i] = float32(v)
		}
	default:
		return nil, fmt.Errorf("vision: Swin supports F32/F64, got %v", out.Dtype())
	}
	return out, nil
}

// Forward runs one windowed pre-LN block over a token grid x[H·W, C] laid out
// row-major over the H×W positions (H and W passed explicitly since the grid is
// flattened). It LayerNorms, optionally cyclic-shifts (SW-MSA), partitions into
// M×M windows, runs per-window multi-head attention (with the relative bias and,
// when shifted, the cross-region mask), un-partitions, un-shifts, adds the
// residual, then the pre-norm MLP — returning the updated grid in the same
// [H·W, C] layout. Fully differentiable, so gradients reach the projections, the
// MLP, and the relative-position table.
//
// In plain terms: this is one Swin layer applied to the tile grid — attention
// confined to little windows (plus the half-window slide when Shift is set) —
// exposed on its own so a single block can be inspected or reused.
func (b *SwinBlock) Forward(ctx *backend.Context, x *tensor.Tensor, h, w int) (*tensor.Tensor, error) {
	m := b.Window
	shift := 0
	if b.Shift && h > m { // a single-window grid has nothing to shift into
		shift = m / 2
	}
	ln1, err := b.LN1.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	g := ln1
	if shift > 0 {
		if g, err = swinGather(ctx, g, swinShiftIdx(h, w, shift), b.dtype()); err != nil {
			return nil, err
		}
	}
	partIdx := swinPartitionIdx(h, w, m)
	part, err := swinGather(ctx, g, partIdx, b.dtype())
	if err != nil {
		return nil, err
	}
	numWin := (h / m) * (w / m)
	var masks []*tensor.Tensor
	if shift > 0 {
		masks = swinBuildMasks(h, w, m, shift, b.dtype())
	}
	att, err := b.windowedAttention(ctx, part, numWin, m, masks)
	if err != nil {
		return nil, err
	}
	// reverse the window partition, then the cyclic shift.
	a, err := swinGather(ctx, att, swinInverse(partIdx), b.dtype())
	if err != nil {
		return nil, err
	}
	if shift > 0 {
		if a, err = swinGather(ctx, a, swinInverse(swinShiftIdx(h, w, shift)), b.dtype()); err != nil {
			return nil, err
		}
	}
	x, err = swinExec2(ctx, backend.OpAdd, nil, x, a)
	if err != nil {
		return nil, err
	}
	// MLP branch.
	ln2, err := b.LN2.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	f1, err := b.FC1.Forward(ctx, ln2)
	if err != nil {
		return nil, err
	}
	gelu, err := swinExec1a(ctx, backend.OpGELU, nil, f1)
	if err != nil {
		return nil, err
	}
	f2, err := b.FC2.Forward(ctx, gelu)
	if err != nil {
		return nil, err
	}
	return swinExec2(ctx, backend.OpAdd, nil, x, f2)
}

// dtype reports the block's parameter dtype (from Wq).
func (b *SwinBlock) dtype() tensor.Dtype { return b.Wq.Dtype() }

// windowedAttention runs multi-head scaled-dot-product attention independently
// within each window of xWin[numWin·M², C]: it projects Q,K,V once for all
// windows, then per window and per head computes softmax(QKᵀ/√dₖ + relBias +
// mask)·V, concatenates heads and windows, and applies the output projection.
// masks is nil for W-MSA or a per-window additive [M²,M²] mask for SW-MSA.
// swinF32 and swinF64 read a tensor's typed backing slice, so the fused attention arm below can be
// written once over a type parameter instead of twice per dtype.
func swinF32(t *tensor.Tensor) []float32 { return t.Storage().F32() }
func swinF64(t *tensor.Tensor) []float64 { return t.Storage().F64() }

// swinFusedWindowAttn is the inference arm of windowedAttention: same arithmetic, far fewer
// dispatches.
//
// The per-window per-head loop was about 95% of this package's backend dispatches for about 2% of
// its arithmetic — roughly ten Execute calls on 256-element tensors per (window, head) pair, each
// one an output allocation plus an Attrs interface box. Seven of the ten are pure DATA MOVEMENT or
// elementwise scaling: three column slices, one transpose, the 1/sqrt(dk) scale, the relative-bias
// add, the mask add, and then two levels of Concat to reassemble what the slices took apart. This
// arm does all of that as index arithmetic over the packed storage and keeps only the three calls
// that are real arithmetic: Q·Kᵀ, the softmax, and the weights·V.
//
// BIT-IDENTITY. The movement is bit-identical by construction — no value is computed, only copied.
// The scale/bias/mask chain is bit-identical BECAUSE it is written as three separately rounded
// statements: fused into one expression, arm64 would contract `s*inv + bias` into an FMA and round
// once where the dispatched path rounds twice (§PS3025). The two matmuls and the softmax are the
// backend's own kernels, unchanged, so they cannot drift.
//
// The scratch tensors are reused across iterations. That is safe only because this arm runs with no
// recorder: a taped run keeps its inputs alive for the backward pass, which is why the caller gates
// on ctx.Recorder == nil and falls back to the dispatched path for training.
func swinFusedWindowAttn[T float32 | float64](ctx *backend.Context, b *SwinBlock, q, k, v *tensor.Tensor,
	numWin, n, dk int, inv T, biases, masks []*tensor.Tensor, get func(*tensor.Tensor) []T) (*tensor.Tensor, error) {
	qs, ks, vs := get(q), get(k), get(v)
	out := tensor.New(b.dtype(), tensor.Shape{numWin * n, b.Dim})
	os := get(out)
	qh := tensor.New(b.dtype(), tensor.Shape{n, dk})
	kt := tensor.New(b.dtype(), tensor.Shape{dk, n})
	vh := tensor.New(b.dtype(), tensor.Shape{n, dk})
	qhs, kts, vhs := get(qh), get(kt), get(vh)
	for w := range numWin {
		base := w * n * b.Dim
		var mk []T
		if masks != nil {
			mk = get(masks[w%len(masks)])
		}
		for hh := range b.Heads {
			off := base + hh*dk
			for i := range n {
				row := off + i*b.Dim
				copy(qhs[i*dk:(i+1)*dk], qs[row:row+dk])
				copy(vhs[i*dk:(i+1)*dk], vs[row:row+dk])
				krow := ks[row : row+dk]
				for d := range dk {
					kts[d*n+i] = krow[d] // the transpose, as the write pattern
				}
			}
			sc, err := swinExec2(ctx, backend.OpMatMul, nil, qh, kt) // [n, n]
			if err != nil {
				return nil, err
			}
			ss := get(sc)
			var bh []T
			if biases != nil {
				bh = get(biases[hh])
			}
			for i := range ss {
				// Each step is wrapped in a conversion to the type parameter. Separate STATEMENTS
				// are not enough: the Go spec lets an implementation fuse floating-point operations
				// ACROSS statements, and on arm64 it does — this loop written as three plain
				// assignments contracted into an FMADD and diverged from the dispatched path by one
				// ulp on 66 of 256 logits, which the bit-identity gate caught. Only an explicit
				// conversion forces the intermediate rounding (§PS3025).
				ss[i] = T(ss[i] * inv)
				if bh != nil {
					ss[i] = T(ss[i] + bh[i])
				}
				if mk != nil {
					ss[i] = T(ss[i] + mk[i])
				}
			}
			wgt, err := swinExec1a(ctx, backend.OpSoftmax, nil, sc)
			if err != nil {
				return nil, err
			}
			hd, err := swinExec2(ctx, backend.OpMatMul, nil, wgt, vh)
			if err != nil {
				return nil, err
			}
			hs := get(hd)
			for i := range n { // scattering here IS both levels of Concat, for free
				copy(os[off+i*b.Dim:off+i*b.Dim+dk], hs[i*dk:(i+1)*dk])
			}
		}
	}
	return swinExec2(ctx, backend.OpMatMul, nil, out, b.Wo)
}

func (b *SwinBlock) windowedAttention(ctx *backend.Context, xWin *tensor.Tensor, numWin, m int, masks []*tensor.Tensor) (*tensor.Tensor, error) {
	n := m * m
	dk := b.Dim / b.Heads
	q, err := swinExec2(ctx, backend.OpMatMul, nil, xWin, b.Wq)
	if err != nil {
		return nil, err
	}
	k, err := swinExec2(ctx, backend.OpMatMul, nil, xWin, b.Wk)
	if err != nil {
		return nil, err
	}
	v, err := swinExec2(ctx, backend.OpMatMul, nil, xWin, b.Wv)
	if err != nil {
		return nil, err
	}
	inv := swinScalar(b.dtype(), 1/math.Sqrt(float64(dk)))
	var biases []*tensor.Tensor
	if b.Rel != nil {
		biases = make([]*tensor.Tensor, b.Heads)
		for hh := range b.Heads {
			if biases[hh], err = b.Rel.headBias(ctx, hh); err != nil {
				return nil, err
			}
		}
	}
	// The fused arm computes the same values with three dispatches per (window, head) instead of
	// about ten. It is gated on there being no recorder: a taped run needs every intermediate on the
	// tape for backprop, so training keeps the dispatched path below.
	if ctx.Recorder == nil {
		switch b.dtype() {
		case tensor.F32:
			return swinFusedWindowAttn(ctx, b, q, k, v, numWin, n, dk,
				float32(1/math.Sqrt(float64(dk))), biases, masks, swinF32)
		case tensor.F64:
			return swinFusedWindowAttn(ctx, b, q, k, v, numWin, n, dk,
				1/math.Sqrt(float64(dk)), biases, masks, swinF64)
		}
	}
	rowSlice := func(t *tensor.Tensor, w int) (*tensor.Tensor, error) {
		return swinExec1a(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: w * n, End: (w + 1) * n}, t)
	}
	colSlice := func(t *tensor.Tensor, hh int) (*tensor.Tensor, error) {
		return swinExec1a(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: hh * dk, End: (hh + 1) * dk}, t)
	}
	winOuts := make([]*tensor.Tensor, numWin)
	for w := range numWin {
		qw, err := rowSlice(q, w)
		if err != nil {
			return nil, err
		}
		kw, err := rowSlice(k, w)
		if err != nil {
			return nil, err
		}
		vw, err := rowSlice(v, w)
		if err != nil {
			return nil, err
		}
		heads := make([]*tensor.Tensor, b.Heads)
		for hh := range b.Heads {
			qh, err := colSlice(qw, hh)
			if err != nil {
				return nil, err
			}
			kh, err := colSlice(kw, hh)
			if err != nil {
				return nil, err
			}
			vh, err := colSlice(vw, hh)
			if err != nil {
				return nil, err
			}
			khT, err := swinExec1a(ctx, backend.OpTranspose, nil, kh) // [dk, n]
			if err != nil {
				return nil, err
			}
			sc, err := swinExec2(ctx, backend.OpMatMul, nil, qh, khT) // [n, n]
			if err != nil {
				return nil, err
			}
			if sc, err = swinExec2(ctx, backend.OpMul, nil, sc, inv); err != nil { // B60: scale via OpMul
				return nil, err
			}
			if biases != nil {
				if sc, err = swinExec2(ctx, backend.OpAdd, nil, sc, biases[hh]); err != nil {
					return nil, err
				}
			}
			if masks != nil {
				// masks describe one image's numWin windows; when this runs over B·numWin batched
				// windows the geometry (and mask) repeats every numWin, so index modulo (§batched).
				if sc, err = swinExec2(ctx, backend.OpAdd, nil, sc, masks[w%len(masks)]); err != nil {
					return nil, err
				}
			}
			wgt, err := swinExec1a(ctx, backend.OpSoftmax, nil, sc) // over last axis
			if err != nil {
				return nil, err
			}
			if heads[hh], err = swinExec2(ctx, backend.OpMatMul, nil, wgt, vh); err != nil {
				return nil, err
			}
		}
		if winOuts[w], err = swinExec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...); err != nil {
			return nil, err
		}
	}
	cat, err := swinExec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, winOuts...)
	if err != nil {
		return nil, err
	}
	return swinExec2(ctx, backend.OpMatMul, nil, cat, b.Wo)
}

// headBias returns the [M²,M²] relative-position bias matrix for head hh, built
// as oneHot·Table[:,hh] so it is differentiable w.r.t. the bias table.
func (r *swinRelBias) headBias(ctx *backend.Context, hh int) (*tensor.Tensor, error) {
	n := r.m * r.m
	col, err := swinExec1a(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: hh, End: hh + 1}, r.Table)
	if err != nil {
		return nil, err
	}
	flat, err := swinExec2(ctx, backend.OpMatMul, nil, r.oneHot, col) // [n·n, 1]
	if err != nil {
		return nil, err
	}
	return swinExec1a(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{n, n}}, flat)
}

// forward downsamples the token grid x[H·W, C] to [(H/2)·(W/2), 2C] by
// concatenating each 2×2 neighborhood (order (0,0),(1,0),(0,1),(1,1)) into 4C,
// LayerNorm, and Linear(4C→2C). Returns the new grid and its halved H,W.
func (mg *swinMerge) forward(ctx *backend.Context, x *tensor.Tensor, h, w int) (*tensor.Tensor, int, int, error) {
	dt := x.Dtype()
	subs := make([]*tensor.Tensor, 4)
	offs := [4][2]int{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	for i, o := range offs {
		s, err := swinGather(ctx, x, swinMergeIdx(h, w, o[0], o[1]), dt)
		if err != nil {
			return nil, 0, 0, err
		}
		subs[i] = s
	}
	cat, err := swinExec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, subs...) // [(H/2)(W/2), 4C]
	if err != nil {
		return nil, 0, 0, err
	}
	nrm, err := mg.Norm.Forward(ctx, cat)
	if err != nil {
		return nil, 0, 0, err
	}
	out, err := mg.Reduce.Forward(ctx, nrm)
	if err != nil {
		return nil, 0, 0, err
	}
	return out, h / 2, w / 2, nil
}

// forwardBatched is swinMerge.forward over a packed [batch·(H·W), C] grid. The 2×2 neighborhood
// gather is per-image (a spatial index permutation), but the concatenated [batch·(H/2·W/2), 4C]
// rows run through one batched LayerNorm + Reduce GEMM. Identical to forward per image.
func (mg *swinMerge) forwardBatched(ctx *backend.Context, x *tensor.Tensor, h, w, batch int) (*tensor.Tensor, int, int, error) {
	dt := x.Dtype()
	N := h * w
	offs := [4][2]int{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	idx := make([][]int, 4)
	for i, o := range offs {
		idx[i] = swinMergeIdx(h, w, o[0], o[1])
	}
	cats := make([]*tensor.Tensor, batch)
	for bi := range batch {
		xi, err := swinExec1a(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: bi * N, End: (bi + 1) * N}, x)
		if err != nil {
			return nil, 0, 0, err
		}
		subs := make([]*tensor.Tensor, 4)
		for i := range offs {
			s, err := swinGather(ctx, xi, idx[i], dt)
			if err != nil {
				return nil, 0, 0, err
			}
			subs[i] = s
		}
		if cats[bi], err = swinExec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, subs...); err != nil { // [(H/2·W/2), 4C]
			return nil, 0, 0, err
		}
	}
	cat, err := swinExec1(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, cats...) // [batch·(H/2·W/2), 4C]
	if err != nil {
		return nil, 0, 0, err
	}
	nrm, err := mg.Norm.Forward(ctx, cat)
	if err != nil {
		return nil, 0, 0, err
	}
	out, err := mg.Reduce.Forward(ctx, nrm)
	if err != nil {
		return nil, 0, 0, err
	}
	return out, h / 2, w / 2, nil
}

// --- windowing/index helpers (all pure permutations of grid rows) ---

// swinExec1 runs a single-output op and unwraps the result.
func swinExec1(ctx *backend.Context, op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, in, at)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// swinIns1Pool and swinIns2Pool reuse the input slice backend.Execute takes, for the 1- and
// 2-input ops on the windowed-attention path. Go builds a fresh slice for a variadic pack at
// every call site, and windowedAttention issues eight of them per head per window.
//
// backend.Execute retains its inputs slice ONLY through ctx.Recorder.Record, which fires when a
// tape is attached; with a nil Recorder the slice is dead the instant Execute returns, so it goes
// straight back to the pool. Under a recorder both helpers defer to the variadic form, because
// Record stores that exact slice in the tape node and a pooled one would be overwritten by the
// next op — the same contract nlp's exec2 has carried since T960.
var (
	swinIns1Pool = sync.Pool{New: func() any { s := make([]*tensor.Tensor, 1); return &s }}
	swinIns2Pool = sync.Pool{New: func() any { s := make([]*tensor.Tensor, 2); return &s }}
)

// swinExec1a runs a 1-input op with a pooled input slice when the context is not recording.
func swinExec1a(ctx *backend.Context, op backend.Op, at backend.Attrs, a *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx.Recorder != nil {
		return swinExec1(ctx, op, at, a)
	}
	sp := swinIns1Pool.Get().(*[]*tensor.Tensor)
	s := *sp
	s[0] = a
	out, err := backend.Execute(ctx, op, s, at)
	s[0] = nil
	swinIns1Pool.Put(sp)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// swinExec2 runs a 2-input op with a pooled input slice when the context is not recording.
func swinExec2(ctx *backend.Context, op backend.Op, at backend.Attrs, a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx.Recorder != nil {
		return swinExec1(ctx, op, at, a, b)
	}
	sp := swinIns2Pool.Get().(*[]*tensor.Tensor)
	s := *sp
	s[0], s[1] = a, b
	out, err := backend.Execute(ctx, op, s, at)
	s[0], s[1] = nil, nil
	swinIns2Pool.Put(sp)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// swinGather reorders the rows of t by idx via the differentiable OpEmbed
// row-gather: out[i] = t[idx[i]] (grad scatters back), so window partition,
// cyclic shift, patch-merge grouping, and their inverses are all one gather.
func swinGather(ctx *backend.Context, t *tensor.Tensor, idx []int, dtype tensor.Dtype) (*tensor.Tensor, error) {
	return swinExec2(ctx, backend.OpEmbed, nil, t, swinIdxTensor(dtype, idx))
}

// swinIdxTensor builds a rank-1 float index tensor (OpEmbed reads indices as
// floats, §B12).
func swinIdxTensor(dtype tensor.Dtype, idx []int) *tensor.Tensor {
	t := tensor.New(dtype, tensor.Shape{len(idx)})
	for i, v := range idx {
		t.SetF64(float64(v), i)
	}
	return t
}

// swinScalar makes a [1] tensor holding v (for OpMul broadcast scaling).
func swinScalar(dtype tensor.Dtype, v float64) *tensor.Tensor {
	t := tensor.New(dtype, tensor.Shape{1})
	t.SetF64(v, 0)
	return t
}

// swinPartitionIdx maps window-major output rows (window (by,bx), then row-major
// within the M×M window) to their source grid rows in an H×W grid.
func swinPartitionIdx(h, w, m int) []int {
	wy, wx := h/m, w/m
	out := make([]int, h*w)
	o := 0
	for by := range wy {
		for bx := range wx {
			for dy := range m {
				for dx := range m {
					out[o] = (by*m+dy)*w + (bx*m + dx)
					o++
				}
			}
		}
	}
	return out
}

// swinInverse returns the inverse permutation: inv[perm[o]] = o.
func swinInverse(perm []int) []int {
	inv := make([]int, len(perm))
	for o, s := range perm {
		inv[s] = o
	}
	return inv
}

// swinShiftIdx is the cyclic shift by (s,s): out[y·W+x] = grid[((y+s)%H)·W +
// (x+s)%W] — content moves toward the top-left (torch.roll by −s), so gathering
// with it shifts the grid and gathering with its inverse restores it.
func swinShiftIdx(h, w, s int) []int {
	out := make([]int, h*w)
	for y := range h {
		for x := range w {
			out[y*w+x] = ((y+s)%h)*w + (x+s)%w
		}
	}
	return out
}

// swinRegionIDs labels each position of the shifted H×W grid with the id of the
// original block it came from (the paper's masking scheme): the three row bands
// [0,H−M), [H−M,H−s), [H−s,H) crossed with the same column bands give up to nine
// regions. Two positions with different ids were wrapped together by the shift
// and must not attend to each other.
func swinRegionIDs(h, w, m, s int) []int {
	region := make([]int, h*w)
	hs := [3][2]int{{0, h - m}, {h - m, h - s}, {h - s, h}}
	ws := [3][2]int{{0, w - m}, {w - m, w - s}, {w - s, w}}
	cnt := 0
	for _, hb := range hs {
		for _, wb := range ws {
			for y := hb[0]; y < hb[1]; y++ {
				for x := wb[0]; x < wb[1]; x++ {
					region[y*w+x] = cnt
				}
			}
			cnt++
		}
	}
	return region
}

// swinBuildMasks builds the per-window additive attention mask for SW-MSA: entry
// [i,j] is 0 when the two window tokens share a region id and −1e30 otherwise, so
// softmax gives cross-region pairs weight exactly 0 (the −∞ underflows to 0).
func swinBuildMasks(h, w, m, s int, dtype tensor.Dtype) []*tensor.Tensor {
	region := swinRegionIDs(h, w, m, s)
	part := swinPartitionIdx(h, w, m)
	numWin := (h / m) * (w / m)
	n := m * m
	masks := make([]*tensor.Tensor, numWin)
	for wi := range numWin {
		mask := tensor.New(dtype, tensor.Shape{n, n})
		for i := range n {
			ri := region[part[wi*n+i]]
			for j := range n {
				if ri != region[part[wi*n+j]] {
					mask.SetF64(-1e30, i, j)
				}
			}
		}
		masks[wi] = mask
	}
	return masks
}

// swinMergeIdx maps each patch-merge output cell (oy,ox) to the source grid row
// of its (dy,dx) neighbor in the 2×2 block.
func swinMergeIdx(h, w, dy, dx int) []int {
	oh, ow := h/2, w/2
	out := make([]int, oh*ow)
	o := 0
	for oy := range oh {
		for ox := range ow {
			out[o] = (2*oy+dy)*w + (2*ox + dx)
			o++
		}
	}
	return out
}

// swinRelIndex builds the [M²·M²] flat table of relative-position bucket indices:
// for query i and key j in an M×M window, the bucket is
// (Δrow+M−1)·(2M−1) + (Δcol+M−1), the (2M−1)² distinct in-window offsets.
func swinRelIndex(m int) []int {
	n := m * m
	idx := make([]int, n*n)
	for i := range n {
		iy, ix := i/m, i%m
		for j := range n {
			jy, jx := j/m, j%m
			dy := iy - jy + m - 1
			dx := ix - jx + m - 1
			idx[i*n+j] = dy*(2*m-1) + dx
		}
	}
	return idx
}
