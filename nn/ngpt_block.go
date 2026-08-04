package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// NGPTBlock is the normalized-Transformer block of nGPT (§R243; Loshchilov, Hsieh, Sun & Ginsburg
// 2024, "nGPT: Normalized Transformer with Representation Learning on the Hypersphere",
// arXiv:2410.01131). Every hidden state lives on the unit hypersphere Sᵈ⁻¹, so the block carries NO
// LayerNorm/RMSNorm at all — normalization is folded into the residual update itself. For an input
// X ∈ [L, d_model] with H heads (per-head dim d_k = d_model/H) the two sub-layers are the paper's
// hypersphere "LERP-then-normalize" residual (Eq. 10-11), NOT the usual h ← h + f(h):
//
//	h  = Normalize(X)                                            // enter the sphere
//	h ← Normalize( h + α_A ⊙ (Attn(h) − h) )                     // attention sub-layer (Eq. 10)
//	h ← Normalize( h + α_M ⊙ (MLP(h)  − h) )                     // MLP sub-layer      (Eq. 11)
//
// where α_A, α_M ∈ ℝ^{d_model} are LEARNABLE per-dimension "eigen learning rate" vectors (the
// paper's per-coordinate step size along the sphere), ⊙ is elementwise, and Normalize is
// L2-normalization along d_model. This is a linear interpolation followed by a re-projection to the
// sphere (AXPY then L2-normalize) — deliberately NOT the geodesic SLERP of nn/slerp.go, matching the
// paper's approximation that at small α the LERP and the true great-circle step agree to first order.
//
// The two interior operators are unit-sphere variants of the usual attention and SwiGLU MLP:
//
//   - Attn: after the q,k projections, q and k are L2-normalized along the HEAD dim d_k and scaled by
//     the learnable s_qk (§R243 §2.3), so every logit is a bounded cosine. The softmax scale is then
//     √d_k, NOT the conventional 1/√d_k (the FLIP, §R243 §2.6): because q,k are unit-norm the logits
//     are already bounded in [−1,1], so the temperature is RAISED (multiply, not divide) to give the
//     softmax a usable dynamic range. v is the on-sphere value projection (left unnormalized).
//   - MLP: SwiGLU with learnable interior scales (§R243 §2.4) — the gate branch is scaled by s_u and
//     the value branch by s_v·√d_model before SiLU(gate)⊙value, restoring the variance that the
//     unit-norm inputs would otherwise suppress.
//
// All weight matrices are kept with unit-norm columns (each output feature's weight vector lies on
// the sphere, §R243 §2.1) so a unit-norm input yields bounded, cosine-like pre-activations. Because
// GoAI optimizers have no post-step hook, this normalization is NOT done inside Forward: the training
// loop MUST call NormalizeWeights after every optimizer Step (a caller who forgets it silently trains
// an ordinary — non-nGPT — model). NewNGPTBlock leaves a fresh block already normalized.
//
// The learnable scales (s_qk, s_u, s_v) and eigen-LRs (α_A, α_M) are ordinary Parameters trained by
// backprop; only the weight column-renormalization is the non-gradient op. Every scale is stored at
// the paper's small init magnitude 1/√d_model (§R243 §2.5 "scaling factors": the tiny stored value
// gives the parameter a controlled effective learning rate under a single global LR) and multiplied
// by a fixed compensation constant in Forward so its EFFECTIVE value starts at the intended target
// (s_qk, s_u → 1; s_v·√d_model → √d_model; α → 0.05). This block is hidden→hidden only: the token
// embedding and the s_z-scaled logit head (§R243 §2.2) are the caller's responsibility.
type NGPTBlock struct {
	DModel  int // model/embedding width d_model (the sphere dimension)
	Heads   int // number of attention heads H
	HeadDim int // per-head dim d_k = d_model/H
	Hidden  int // MLP inner width

	Wq []*tensor.Tensor // per-head query projections [d_model, d_k]; columns kept unit-norm
	Wk []*tensor.Tensor // per-head key projections   [d_model, d_k]; columns kept unit-norm
	Wv []*tensor.Tensor // per-head value projections  [d_model, d_k]; columns kept unit-norm
	Wo []*tensor.Tensor // per-head output projections [d_k, d_model]; columns kept unit-norm

	Wgate *tensor.Tensor // MLP gate matrix  [d_model, hidden]; columns kept unit-norm
	Wup   *tensor.Tensor // MLP value matrix [d_model, hidden]; columns kept unit-norm
	Wdown *tensor.Tensor // MLP down matrix  [hidden, d_model]; columns kept unit-norm

	Sqk    []*tensor.Tensor // per-head learnable q/k scale [1, d_k] (§R243 §2.3), effective init 1
	Su     *tensor.Tensor   // learnable MLP gate scale  [1, hidden] (§R243 §2.4), effective init 1
	Sv     *tensor.Tensor   // learnable MLP value scale [1, hidden] (§R243 §2.4), effective init √d_model
	AlphaA *tensor.Tensor   // learnable attention eigen-LR [1, d_model] (§R243 Eq.10), effective init 0.05
	AlphaM *tensor.Tensor   // learnable MLP eigen-LR       [1, d_model] (§R243 Eq.11), effective init 0.05

	Causal bool    // when true, query i attends only to keys j ≤ i
	Eps    float64 // L2-normalization variance floor (default 1e-6)

	sqkComp, suComp, svComp, alphaComp float64 // fixed Forward compensation constants (§R243 §2.5)
	softmaxScale                       float64 // √d_k — the RAISED-temperature softmax scale (§R243 §2.6 FLIP)
}

// NGPTOption configures an NGPTBlock (functional-options idiom, §C12).
type NGPTOption func(*NGPTBlock)

// WithNGPTCausal sets whether the attention sub-layer is causal (default true — the block is meant as
// a decoder layer). Pass false for a bidirectional (encoder) block.
func WithNGPTCausal(causal bool) NGPTOption { return func(b *NGPTBlock) { b.Causal = causal } }

// WithNGPTEps overrides the L2-normalization variance floor (default 1e-6) used by every Normalize in
// the block (the sphere projections and the q/k head normalization). A larger floor trades exactness
// for extra stability when hidden states can collapse toward the origin.
func WithNGPTEps(eps float64) NGPTOption { return func(b *NGPTBlock) { b.Eps = eps } }

// NewNGPTBlock builds an nGPT block over d_model with heads heads (d_model must be divisible by
// heads) and MLP inner width hidden. Weights are Xavier-uniform and then column-normalized so the
// fresh block already satisfies the nGPT unit-sphere weight invariant (§R243 §2.1). All learnable
// scales/eigen-LRs start at the paper's targets (s_qk, s_u → 1; s_v·√d_model → √d_model; α → 0.05).
func NewNGPTBlock(dtype tensor.Dtype, dModel, heads, hidden int, seed uint64, opts ...NGPTOption) (*NGPTBlock, error) {
	if heads <= 0 || dModel%heads != 0 {
		return nil, fmt.Errorf("nn: NGPTBlock d_model %d not divisible by heads %d", dModel, heads)
	}
	if hidden <= 0 {
		return nil, fmt.Errorf("nn: NGPTBlock hidden %d must be positive", hidden)
	}
	hd := dModel / heads
	base := 1 / math.Sqrt(float64(dModel)) // §R243 §2.5 stored-scale magnitude
	b := &NGPTBlock{
		DModel: dModel, Heads: heads, HeadDim: hd, Hidden: hidden,
		Causal: true, Eps: 1e-6,
		// Forward compensations so the EFFECTIVE scale starts at its target (stored value is `base`):
		sqkComp:      1 / base,               // s_qk effective init = base·(1/base)      = 1
		suComp:       1 / base,               // s_u  effective init = base·(1/base)      = 1
		svComp:       float64(dModel),        // s_v·√d effective init = base·d_model      = √d_model
		alphaComp:    0.05 / base,            // α    effective init = base·(0.05/base)   = 0.05
		softmaxScale: math.Sqrt(float64(hd)), // √d_k — the FLIP (§R243 §2.6)
	}
	fill := func(shape tensor.Shape, v float64) *tensor.Tensor {
		t := tensor.New(dtype, shape)
		//perfscan:ignore PS1001 NewNGPTBlock constructor fill closure, one-time model init
		for i := range t.Numel() {
			t.SetF64(v, tensor.Unravel(i, shape)...)
		}
		return t
	}
	var s uint64 = seed
	for range heads {
		//perfscan:ignore PS6016 constructor per-head init loop one-time; Shape-literal alloc resource-only
		wq := tensor.New(dtype, tensor.Shape{dModel, hd})
		XavierUniform(wq, dModel, hd, s+1)
		//perfscan:ignore PS6016 constructor per-head init loop one-time; resource-only alloc
		wk := tensor.New(dtype, tensor.Shape{dModel, hd})
		XavierUniform(wk, dModel, hd, s+2)
		//perfscan:ignore PS6016 constructor per-head init loop one-time; resource-only alloc
		wv := tensor.New(dtype, tensor.Shape{dModel, hd})
		XavierUniform(wv, dModel, hd, s+3)
		//perfscan:ignore PS6016 constructor per-head init loop one-time; resource-only alloc
		wo := tensor.New(dtype, tensor.Shape{hd, dModel})
		XavierUniform(wo, hd, dModel, s+4)
		b.Wq = append(b.Wq, wq)
		b.Wk = append(b.Wk, wk)
		b.Wv = append(b.Wv, wv)
		b.Wo = append(b.Wo, wo)
		//perfscan:ignore PS6016 constructor per-head init loop one-time; resource-only alloc
		b.Sqk = append(b.Sqk, fill(tensor.Shape{1, hd}, base))
		s += 8
	}
	b.Wgate = tensor.New(dtype, tensor.Shape{dModel, hidden})
	XavierUniform(b.Wgate, dModel, hidden, s+1)
	b.Wup = tensor.New(dtype, tensor.Shape{dModel, hidden})
	XavierUniform(b.Wup, dModel, hidden, s+2)
	b.Wdown = tensor.New(dtype, tensor.Shape{hidden, dModel})
	XavierUniform(b.Wdown, hidden, dModel, s+3)
	b.Su = fill(tensor.Shape{1, hidden}, base)
	b.Sv = fill(tensor.Shape{1, hidden}, base)
	b.AlphaA = fill(tensor.Shape{1, dModel}, base)
	b.AlphaM = fill(tensor.Shape{1, dModel}, base)
	for _, o := range opts {
		o(b)
	}
	b.NormalizeWeights() // start on-spec: unit-norm weight columns before any training step
	return b, nil
}

func (b *NGPTBlock) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs the nGPT block on x[L, d_model] → [L, d_model]. The input is first projected onto the
// unit sphere, then the two hypersphere residual sub-layers (attention, MLP) run in sequence; the
// returned hidden state has unit-norm rows (§V16 sphere invariant). Fully differentiable — gradients
// reach every weight matrix, every scale (s_qk, s_u, s_v) and both eigen-LRs (α_A, α_M).
func (b *NGPTBlock) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != b.DModel {
		return nil, fmt.Errorf("nn: NGPTBlock expects x [L,%d], got %v", b.DModel, x.Shape())
	}
	h, err := b.normalize(ctx, x) // enter the sphere
	if err != nil {
		return nil, err
	}
	attn, err := b.attention(ctx, h)
	if err != nil {
		return nil, err
	}
	if h, err = b.sphereResidual(ctx, h, attn, b.AlphaA); err != nil {
		return nil, err
	}
	mlp, err := b.mlp(ctx, h)
	if err != nil {
		return nil, err
	}
	return b.sphereResidual(ctx, h, mlp, b.AlphaM)
}

// sphereResidual is the nGPT LERP-then-normalize update Normalize(h + α ⊙ (sub − h)) (§R243 Eq.10-11):
// a linear interpolation toward the sub-layer output at per-dimension rate α, re-projected onto the
// unit sphere. Deliberately AXPY-style (not the geodesic SLERP helper).
func (b *NGPTBlock) sphereResidual(ctx *backend.Context, h, sub, alpha *tensor.Tensor) (*tensor.Tensor, error) {
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; matmul-dominated forward
	delta, err := b.exec(ctx, backend.OpSub, nil, sub, h) // sub − h
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; matmul-dominated forward
	aEff, err := b.exec(ctx, backend.OpMul, nil, alpha, scalarTensor(alpha.Dtype(), b.alphaComp)) // effective α
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; matmul-dominated forward
	step, err := b.exec(ctx, backend.OpMul, nil, delta, aEff) // α ⊙ (sub − h), broadcast [1,d]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; matmul-dominated forward
	lerp, err := b.exec(ctx, backend.OpAdd, nil, h, step) // h + α ⊙ (sub − h)
	if err != nil {
		return nil, err
	}
	return b.normalize(ctx, lerp)
}

// attention runs the on-sphere multi-head attention sub-layer on h[L,d_model] → [L,d_model]. Each
// head projects then L2-normalizes q,k over d_k, scales by s_qk, forms cosine logits scaled by the
// RAISED √d_k temperature (§R243 §2.6 FLIP), softmaxes, reads v, and projects out; the heads are
// summed. Per-head projections keep every tensor a tracked leaf (no differentiable split/concat).
func (b *NGPTBlock) attention(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	l := h.Shape()[0]
	var mask *tensor.Tensor
	if b.Causal {
		mask = qkCausalMask(h.Dtype(), l, l)
	}
	scale := scalarTensor(h.Dtype(), b.softmaxScale)
	sqkC := scalarTensor(h.Dtype(), b.sqkComp)
	var out *tensor.Tensor
	for hd := range b.Heads {
		//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; per-head matmul-dominated
		q, err := b.exec(ctx, backend.OpMatMul, nil, h, b.Wq[hd])
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; per-head matmul-dominated
		k, err := b.exec(ctx, backend.OpMatMul, nil, h, b.Wk[hd])
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; per-head matmul-dominated
		v, err := b.exec(ctx, backend.OpMatMul, nil, h, b.Wv[hd]) // on-sphere value, left unnormalized
		if err != nil {
			return nil, err
		}
		// L2-normalize q,k over the HEAD dim d_k, then scale by the effective s_qk (§R243 §2.3).
		qn, err := qkL2NormalizeLastAxis(ctx, q, b.Eps)
		if err != nil {
			return nil, err
		}
		kn, err := qkL2NormalizeLastAxis(ctx, k, b.Eps)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; matmul-dominated forward
		sqkEff, err := b.exec(ctx, backend.OpMul, nil, b.Sqk[hd], sqkC) // effective s_qk, [1,d_k]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; matmul-dominated forward
		if qn, err = b.exec(ctx, backend.OpMul, nil, qn, sqkEff); err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; matmul-dominated forward
		if kn, err = b.exec(ctx, backend.OpMul, nil, kn, sqkEff); err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; matmul-dominated forward
		knT, err := b.exec(ctx, backend.OpTranspose, nil, kn) // [d_k, L]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; per-head matmul-dominated
		scores, err := b.exec(ctx, backend.OpMatMul, nil, qn, knT) // cosine logits [L,L]
		if err != nil {
			return nil, err
		}
		// The FLIP: multiply by √d_k (raise temperature), NOT divide by it — the unit-norm q,k make
		// the raw logits bounded, so a raised temperature restores the softmax's dynamic range (§R243
		// §2.6). Using the conventional 1/√d_k here would over-flatten the distribution.
		//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; matmul-dominated forward
		if scores, err = b.exec(ctx, backend.OpMul, nil, scores, scale); err != nil {
			return nil, err
		}
		if b.Causal {
			//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; matmul-dominated forward
			if scores, err = b.exec(ctx, backend.OpAdd, nil, scores, mask); err != nil {
				return nil, err
			}
		}
		//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; matmul-dominated forward
		attn, err := b.exec(ctx, backend.OpSoftmax, nil, scores)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; per-head matmul-dominated
		ctxh, err := b.exec(ctx, backend.OpMatMul, nil, attn, v) // [L, d_k]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; per-head matmul-dominated
		oh, err := b.exec(ctx, backend.OpMatMul, nil, ctxh, b.Wo[hd]) // [L, d_model]
		if err != nil {
			return nil, err
		}
		if out == nil {
			out = oh
			//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; matmul-dominated forward
		} else if out, err = b.exec(ctx, backend.OpAdd, nil, out, oh); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// mlp runs the on-sphere SwiGLU sub-layer on h[L,d_model] → [L,d_model]: SiLU(s_u ⊙ (h·Wgate)) ⊙
// (s_v·√d_model ⊙ (h·Wup)) then ·Wdown. The interior scales restore the variance the unit-norm input
// suppresses (§R243 §2.4).
func (b *NGPTBlock) mlp(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; mlp matmul-dominated
	u, err := b.exec(ctx, backend.OpMatMul, nil, h, b.Wgate)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; mlp matmul-dominated
	suEff, err := b.exec(ctx, backend.OpMul, nil, b.Su, scalarTensor(h.Dtype(), b.suComp)) // effective s_u
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; mlp matmul-dominated
	us, err := b.exec(ctx, backend.OpMul, nil, u, suEff) // broadcast [1,hidden]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; mlp matmul-dominated
	gate, err := b.exec(ctx, backend.OpSiLU, nil, us)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; mlp matmul-dominated
	vv, err := b.exec(ctx, backend.OpMatMul, nil, h, b.Wup)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; mlp matmul-dominated
	svEff, err := b.exec(ctx, backend.OpMul, nil, b.Sv, scalarTensor(h.Dtype(), b.svComp)) // effective s_v·√d
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; mlp matmul-dominated
	vs, err := b.exec(ctx, backend.OpMul, nil, vv, svEff)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; mlp matmul-dominated
	hmid, err := b.exec(ctx, backend.OpMul, nil, gate, vs)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 exec variadic-pack alloc resource-only; mlp matmul-dominated
	return b.exec(ctx, backend.OpMatMul, nil, hmid, b.Wdown)
}

// normalize L2-normalizes h along the last (d_model) axis, projecting it onto the unit hypersphere.
func (b *NGPTBlock) normalize(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	return qkL2NormalizeLastAxis(ctx, h, b.Eps)
}

// NormalizeWeights L2-normalizes every weight matrix so each column (an output feature's weight
// vector over the input axis) has unit norm, placing it on the unit hypersphere (§R243 §2.1). This is
// the plain, in-place, NON-differentiable operation the nGPT training loop MUST run after every
// optimizer Step (GoAI optimizers have no post-step hook) — skipping it silently trains an ordinary
// Transformer. It touches only the weight matrices; the learnable scales and eigen-LRs are updated by
// the optimizer like any other parameter.
func (b *NGPTBlock) NormalizeWeights() {
	all := make([]*tensor.Tensor, 0, 4*b.Heads+3)
	all = append(all, b.Wgate, b.Wup, b.Wdown)
	for hd := range b.Heads {
		all = append(all, b.Wq[hd], b.Wk[hd], b.Wv[hd], b.Wo[hd])
	}
	for _, w := range all {
		normalizeColumns(w)
	}
}

// normalizeColumns rescales each column of a [in,out] matrix to unit L2 norm over the input axis
// (a zero column is left untouched). In-place, pure-f64 — not recorded on any tape.
func normalizeColumns(w *tensor.Tensor) {
	in, out := w.Shape()[0], w.Shape()[1]
	// NormalizeWeights runs this over every weight matrix after each optimizer
	// Step, so for a contiguous, offset-0 w read/write the typed backing slice by
	// flat index (column j at stride `out`) instead of the per-element AtF64/SetF64
	// dispatch. Bit-identical to the general path below (see the slowNormalizeColumns
	// oracle); F16/BF16 fall through.
	if w.IsContiguous() && w.Offset() == 0 {
		switch w.Dtype() {
		// Row-major reblock: the per-column reduction/scale over a row-major buffer strides
		// the reads by `out` (8 KB apart), defeating the prefetcher and vectorization. Walk
		// rows contiguously with an `out`-length sum-of-squares accumulator instead. Bit-
		// identical to the column-major form (and the slowNormalizeColumns oracle): each
		// column j still sums i=0..in-1 in order, and the per-element divide is unchanged.
		case tensor.F64:
			d := w.Storage().F64()
			nrm := make([]float64, out)
			for i := 0; i < in; i++ {
				drow := d[i*out : i*out+out]
				for j, v := range drow {
					//perfscan:ignore PS3017 already-optimized flat reblock fast path; half-bounds-check micro
					nrm[j] += v * v
				}
			}
			for j := range nrm {
				nrm[j] = math.Sqrt(nrm[j])
			}
			for i := 0; i < in; i++ {
				drow := d[i*out : i*out+out]
				for j := range drow {
					if n := nrm[j]; n != 0 {
						drow[j] = drow[j] / n
					}
				}
			}
			return
		case tensor.F32:
			d := w.Storage().F32()
			nrm := make([]float64, out)
			for i := 0; i < in; i++ {
				drow := d[i*out : i*out+out]
				for j, v := range drow {
					fv := float64(v)
					//perfscan:ignore PS3075 deliberate row-major nrm accumulator reblock = the optimized form
					nrm[j] += fv * fv
				}
			}
			for j := range nrm {
				nrm[j] = math.Sqrt(nrm[j])
			}
			for i := 0; i < in; i++ {
				drow := d[i*out : i*out+out]
				for j := range drow {
					if n := nrm[j]; n != 0 {
						drow[j] = float32(float64(drow[j]) / n)
					}
				}
			}
			return
		}
	}
	for j := range out {
		var ss float64
		for i := range in {
			v := w.AtF64(i, j)
			ss += v * v
		}
		n := math.Sqrt(ss)
		if n == 0 {
			continue
		}
		for i := range in {
			w.SetF64(w.AtF64(i, j)/n, i, j)
		}
	}
}

// Params returns every trainable tensor: the weight matrices (per-head Wq/Wk/Wv/Wo and the MLP
// Wgate/Wup/Wdown), the learnable scales (per-head s_qk, s_u, s_v) and the eigen-LRs (α_A, α_M). Feed
// this to an optimizer — and remember to call NormalizeWeights after each Step.
func (b *NGPTBlock) Params() []*tensor.Tensor {
	p := []*tensor.Tensor{b.Wgate, b.Wup, b.Wdown, b.Su, b.Sv, b.AlphaA, b.AlphaM}
	for hd := range b.Heads {
		p = append(p, b.Wq[hd], b.Wk[hd], b.Wv[hd], b.Wo[hd], b.Sqk[hd])
	}
	return p
}
