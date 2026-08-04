package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// HymbaBlock is the Hymba hybrid-head block (Dong et al. 2024, NVIDIA, "Hymba: A
// Hybrid-head Architecture for Small Language Models", arXiv:2411.13676 §2.1
// Eq.3, §R245). Where stacked hybrids (e.g. Jamba) alternate whole attention
// layers with whole Mamba layers, Hymba runs BOTH sequence mixers INSIDE one
// layer, in PARALLEL, on the same input — attention heads act as high-resolution
// "snapshot" memory over the context while the SSM (state-space model: a learned
// linear recurrence that summarizes the past into a fixed-size state) heads act
// as an efficient "fading" summary — and fuses the two streams before a single
// shared output projection. For X̃ ∈ [T, dim]:
//
//	Q,K,V, x_ssm, z = split( W_in_proj · X̃ )               (ONE shared input projection)
//	attn = MHA(Q, K, V)                                     (causal multi-head softmax attention)
//	ssm  = SelectiveScan(x_ssm) ⊙ SiLU(z)                   (Mamba-style selective scan + gate)
//	Y    = W_out_proj( β₁ ⊙ norm(attn) + β₂ ⊙ norm(ssm) )   (Eq. 3: β-norm fusion, shared out-proj)
//
// The two pre-fusion streams share the fusion width (attention feature dim ==
// SSM inner dim == dim here) so they can be added. Each branch is normalized
// SEPARATELY before fusing because the raw SSM output magnitudes dominate the
// attention branch (paper §2.1: ‖Y_ssm‖ ≫ ‖Y_attn‖); β₁, β₂ are LEARNABLE
// per-channel rescale vectors (init 1) that let training re-balance the mix.
// The paper's text says only "norm" for the fusion normalizer; this
// implementation uses RMSNorm (root-mean-square normalization, GoAI's default
// norm, matching the Llama/Mamba family Hymba builds on) — a documented
// assumption (§R245), with each branch's γ trainable.
//
// The attention core is the fused OpMHA op (head split/softmax/concat internal,
// VJP-backed) and the SSM branch is the OpConv1D → SiLU → OpSoftplus → OpSSM
// selective-scan chain of MambaBlock — but the block deliberately does NOT call
// nlp.MHA.Forward or MambaBlock.Forward: both bake their own output projection
// into Forward, while Hymba must fuse the raw branch outputs BEFORE its single
// SHARED out-proj. Every op is dispatched and VJP-backed, so the whole block
// trains end to end: gradients reach the shared in-proj, both branches'
// parameters, β₁/β₂, both norms, and the shared out-proj.
//
// Optionally (paper §2.3 Eq.4, off by default) the block prepends m META TOKENS
// — learnable, input-independent embeddings that act as a learned cache
// initialization and attention sink; they are sliced off the output, so the
// caller always sees [T, dim] regardless. Enable with WithHymbaMetaTokens.
type HymbaBlock struct {
	DModel int // model/embedding width (columns of the input X̃)
	DInner int // fusion width shared by both branches (== DModel; attention feature dim and SSM inner dim)
	Heads  int // number of attention heads (DInner must divide by Heads)
	N      int // SSM state size (rows of the per-channel recurrent state)
	DConv  int // depthwise causal-conv kernel width of the SSM branch (Mamba default 4)
	DtRank int // low-rank width of the Δ (step-size) projection, ceil(DModel/16) (Mamba default)

	// InProj is the SHARED input projection [DModel → 5·DInner]: one matmul whose
	// output is sliced into the attention Q, K, V and the SSM input x_ssm and gate
	// z — the paper's [W^Q,W^K,W^V,W^SSM,W^G] fused projection (§R245).
	InProj *Linear

	// ConvW/ConvB are the SSM branch's depthwise causal-conv filters [DInner,
	// DConv] and bias [DInner] (Mamba local mixing before the scan).
	ConvW, ConvB *tensor.Tensor
	// DtLow/BProj/CProj project the conv output to the input-dependent SSM
	// parameters: the low-rank Δ (DInner→DtRank) and the B, C matrices (DInner→N).
	DtLow, BProj, CProj *Linear
	// DtProj maps the low-rank Δ back up (DtRank→DInner); softplus of its output
	// is the per-token step size of the selective scan.
	DtProj *Linear
	// ALog is the log-parameterized state matrix [DInner, N] (A = −exp(ALog) keeps
	// the recurrence stable); Dskip is the per-channel skip connection [DInner].
	ALog, Dskip *tensor.Tensor

	// NormAttn/NormSSM are the SEPARATE per-branch fusion normalizers (Eq. 3
	// "norm"; RMSNorm by assumption, documented above). Normalizing each branch
	// on its own is what stops the large-magnitude SSM stream from drowning out
	// the attention stream.
	NormAttn, NormSSM *RMSNorm
	// Beta1/Beta2 are the learnable per-channel fusion scales β₁, β₂ ∈ [1,
	// DInner] (init 1, broadcast over tokens): training adjusts them to
	// re-balance how much of each branch enters the shared out-proj.
	Beta1, Beta2 *tensor.Tensor

	// OutProj is the single SHARED output projection [DInner → DModel] applied to
	// the fused stream — Hymba's W_out_proj (both branches flow through it).
	OutProj *Linear

	// MetaTokens holds the m learnable meta-token embeddings [m, DModel]
	// prepended to every input (paper §2.3, learned cache-init + attention sink);
	// nil when the option is off (the default). Trainable when present.
	MetaTokens *tensor.Tensor

	metaLen int // requested meta-token count (0 = off)
}

// HymbaOption configures a HymbaBlock (functional-options idiom, §C12).
type HymbaOption func(*HymbaBlock)

// WithHymbaMetaTokens enables m learnable meta tokens (Hymba paper §2.3 Eq.4):
// input-independent embeddings prepended to every sequence before the block and
// sliced off its output, acting as a learned cache initialization and attention
// sink (the paper reports +1.55% reasoning / +2.75% recall with m=128, its
// recommended value — pass 128 to match, §C21). Default 0 = disabled: the block
// runs on the raw sequence, byte-identical to not passing the option at all.
// Small m (1–8) still provides a sink but little learned-memory benefit; m must
// be ≥ 0 (negative rejected by the constructor). Each meta token adds a full
// [DModel] parameter row and lengthens the attention/scan by m positions.
func WithHymbaMetaTokens(m int) HymbaOption { return func(b *HymbaBlock) { b.metaLen = m } }

// NewHymbaBlock builds a Hymba hybrid-head block over model width dim with
// heads attention heads (dim must divide by heads) and SSM state size ssmState
// (Mamba's default is 16). The fusion width is dim for both branches (the
// paper's constraint that the pre-fusion streams share a channel dim); the SSM
// branch uses Mamba's canonical conv width 4 and Δ-rank ceil(dim/16)
// (arXiv:2312.00752 defaults, §C21). All projections are Xavier-uniform, β₁/β₂
// start at 1 (a balanced fusion), and both fusion norms start as identity
// RMSNorms (γ=1). Deterministic via seed.
func NewHymbaBlock(dtype tensor.Dtype, dim, heads, ssmState int, seed uint64, opts ...HymbaOption) (*HymbaBlock, error) {
	if dim <= 0 || ssmState <= 0 {
		return nil, fmt.Errorf("nn: HymbaBlock dim %d and ssmState %d must be positive", dim, ssmState)
	}
	if heads <= 0 || dim%heads != 0 {
		return nil, fmt.Errorf("nn: HymbaBlock dim %d not divisible by heads %d", dim, heads)
	}
	dInner := dim             // fusion width: attention feature dim == SSM inner dim (Eq. 3 add constraint)
	dConv := 4                // Mamba canonical depthwise-conv width (arXiv:2312.00752)
	dtRank := (dim + 15) / 16 // ceil(dim/16), Mamba's Δ low-rank default
	b := &HymbaBlock{
		DModel: dim, DInner: dInner, Heads: heads, N: ssmState, DConv: dConv, DtRank: dtRank,
		InProj:   NewLinear(dtype, dim, 5*dInner, seed),
		DtLow:    NewLinear(dtype, dInner, dtRank, seed+2),
		BProj:    NewLinear(dtype, dInner, ssmState, seed+3),
		CProj:    NewLinear(dtype, dInner, ssmState, seed+4),
		DtProj:   NewLinear(dtype, dtRank, dInner, seed+5),
		OutProj:  NewLinear(dtype, dInner, dim, seed+6),
		NormAttn: NewRMSNorm(dtype, dInner),
		NormSSM:  NewRMSNorm(dtype, dInner),
	}
	b.ConvW = tensor.New(dtype, tensor.Shape{dInner, dConv})
	XavierUniform(b.ConvW, dConv, 1, seed+10)
	b.ConvB = tensor.New(dtype, tensor.Shape{dInner})
	Zeros(b.ConvB)
	b.ALog = tensor.New(dtype, tensor.Shape{dInner, ssmState})
	XavierUniform(b.ALog, dInner, ssmState, seed+11) // A = −exp(A_log) < 0
	b.Dskip = tensor.New(dtype, tensor.Shape{dInner})
	Zeros(b.Dskip)
	ones := func() *tensor.Tensor {
		t := tensor.New(dtype, tensor.Shape{1, dInner})
		for i := range dInner {
			t.SetF64(1, 0, i)
		}
		return t
	}
	b.Beta1, b.Beta2 = ones(), ones() // β init 1: start with an evenly-weighted fusion
	for _, o := range opts {
		o(b)
	}
	if b.metaLen < 0 {
		return nil, fmt.Errorf("nn: HymbaBlock meta-token count %d must be ≥ 0", b.metaLen)
	}
	if b.metaLen > 0 {
		b.MetaTokens = tensor.New(dtype, tensor.Shape{b.metaLen, dim})
		fillUniform(b.MetaTokens, -0.5, 0.5, seed+30) // small random init, like the soft-prompt embeddings
	}
	return b, nil
}

func (b *HymbaBlock) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs the hybrid-head block on x[T, DModel] → [T, DModel]. Both
// sequence mixers see the same shared-projection input in parallel; their
// separately-normalized, β-rescaled outputs are added and pushed through the
// single shared out-proj (Eq. 3). With meta tokens enabled the m learned
// embeddings are prepended before the block and their output rows sliced off,
// so the result stays [T, DModel]. Fully differentiable — gradients reach the
// shared in-proj, the attention path, every SSM parameter, β₁/β₂, both fusion
// norms, the out-proj, and (when enabled) the meta tokens.
func (b *HymbaBlock) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != b.DModel {
		return nil, fmt.Errorf("nn: HymbaBlock expects x [T,%d], got %v", b.DModel, x.Shape())
	}
	t := x.Shape()[0]
	if b.MetaTokens != nil { // §2.3: prepend the learned meta tokens (sliced off below)
		var err error
		//perfscan:ignore PS3024,PS6018 OpConcat fused-op dispatch, negligible vs compute | meta-token concat runs only when enabled, once per forward
		if x, err = b.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 0}, b.MetaTokens, x); err != nil {
			return nil, err
		}
	}
	// ONE shared input projection, sliced into the five per-branch streams
	// [W^Q,W^K,W^V,W^SSM,W^G] (§R245); OpSlice is VJP-backed, so the shared
	// weight accumulates gradients from BOTH branches.
	p, err := b.InProj.Forward(ctx, x) // [L, 5·DInner]
	if err != nil {
		return nil, err
	}
	chunk := func(i int) (*tensor.Tensor, error) {
		//perfscan:ignore PS3024 OpSlice chunk dispatch, VJP-backed fused op negligible
		return b.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: i * b.DInner, End: (i + 1) * b.DInner}, p)
	}
	q, err := chunk(0)
	if err != nil {
		return nil, err
	}
	k, err := chunk(1)
	if err != nil {
		return nil, err
	}
	v, err := chunk(2)
	if err != nil {
		return nil, err
	}
	xs, err := chunk(3)
	if err != nil {
		return nil, err
	}
	z, err := chunk(4)
	if err != nil {
		return nil, err
	}
	// Attention branch: the fused causal multi-head softmax-attention core
	// (head split/concat internal) on the shared projection — NO per-branch
	// out-proj; the raw head concat goes straight to the fusion.
	//perfscan:ignore PS3024 OpMHA fused dispatch, attention-compute-dominated
	attn, err := b.exec(ctx, backend.OpMHA, backend.AttnAttrs{Heads: b.Heads, Causal: true}, q, k, v)
	if err != nil {
		return nil, err
	}
	// SSM branch: the Mamba selective-scan chain on the shared projection —
	// causal depthwise conv, SiLU, input-dependent Δ/B/C, scan, gate. Again no
	// per-branch out-proj.
	ssm, err := b.ssmBranch(ctx, xs, z)
	if err != nil {
		return nil, err
	}
	// Fusion (Eq. 3): per-branch RMSNorm (the SSM magnitudes would otherwise
	// dominate), learnable per-channel β rescale, add, ONE shared out-proj.
	na, err := b.NormAttn.Forward(ctx, attn)
	if err != nil {
		return nil, err
	}
	ns, err := b.NormSSM.Forward(ctx, ssm)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpMul beta1 dispatch, negligible vs matmuls
	if na, err = b.exec(ctx, backend.OpMul, nil, na, b.Beta1); err != nil { // β₁ ⊙ norm(attn), broadcast [1,DInner]
		return nil, err
	}
	//perfscan:ignore PS3024 OpMul beta2 dispatch, negligible vs matmuls
	if ns, err = b.exec(ctx, backend.OpMul, nil, ns, b.Beta2); err != nil { // β₂ ⊙ norm(ssm)
		return nil, err
	}
	//perfscan:ignore PS3024 OpAdd fusion dispatch, negligible
	fused, err := b.exec(ctx, backend.OpAdd, nil, na, ns)
	if err != nil {
		return nil, err
	}
	out, err := b.OutProj.Forward(ctx, fused) // [L, DModel]
	if err != nil {
		return nil, err
	}
	if b.MetaTokens != nil { // drop the meta rows: caller always sees [T, DModel]
		m := b.MetaTokens.Shape()[0]
		//perfscan:ignore PS3024 OpSlice meta-row drop dispatch, once per forward
		return b.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 0, Start: m, End: m + t}, out)
	}
	return out, nil
}

// ssmBranch runs the Mamba-style selective scan on the shared-projection SSM
// input xs and gate z (both [L, DInner]) → [L, DInner]: causal depthwise conv +
// SiLU local mixing, input-dependent Δ = softplus(dt_proj(dt_low(x))) and B, C
// projections, the OpSSM selective scan with A = −exp(A_log) and skip D, then
// the y ⊙ SiLU(z) gate. Identical math to MambaBlock's interior, minus its
// out-proj (Hymba's out-proj is shared with the attention branch).
func (b *HymbaBlock) ssmBranch(ctx *backend.Context, xs, z *tensor.Tensor) (*tensor.Tensor, error) {
	//perfscan:ignore PS3024 OpConv1D fused dispatch, kernel-compute-dominated
	xc, err := b.exec(ctx, backend.OpConv1D, nil, xs, b.ConvW, b.ConvB)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpSiLU fused-op dispatch, negligible
	if xc, err = b.exec(ctx, backend.OpSiLU, nil, xc); err != nil {
		return nil, err
	}
	dtLow, err := b.DtLow.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	dtPre, err := b.DtProj.Forward(ctx, dtLow)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpSoftplus fused-op dispatch, negligible
	delta, err := b.exec(ctx, backend.OpSoftplus, nil, dtPre)
	if err != nil {
		return nil, err
	}
	bMat, err := b.BProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	cMat, err := b.CProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpExp on A_log dispatch, tiny param tensor
	expA, err := b.exec(ctx, backend.OpExp, nil, b.ALog)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpNeg dispatch, tiny param tensor
	a, err := b.exec(ctx, backend.OpNeg, nil, expA)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpSSM fused selective-scan dispatch, scan-compute-dominated
	y, err := b.exec(ctx, backend.OpSSM, nil, xc, delta, a, bMat, cMat, b.Dskip)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpSiLU gate dispatch, negligible
	gate, err := b.exec(ctx, backend.OpSiLU, nil, z)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpMul gate dispatch, negligible vs compute
	return b.exec(ctx, backend.OpMul, nil, y, gate)
}

// Params returns every trainable tensor: the shared in-proj, the SSM branch
// (conv, Δ/B/C projections, A_log, D), both fusion-norm gains, β₁/β₂, the
// shared out-proj, and the meta tokens when enabled. Feed this to an optimizer.
func (b *HymbaBlock) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{b.ConvW, b.ConvB, b.ALog, b.Dskip, b.NormAttn.Gamma, b.NormSSM.Gamma, b.Beta1, b.Beta2}
	for _, l := range []*Linear{b.InProj, b.DtLow, b.BProj, b.CProj, b.DtProj, b.OutProj} {
		ps = append(ps, l.Params()...)
	}
	if b.MetaTokens != nil {
		ps = append(ps, b.MetaTokens)
	}
	return ps
}
