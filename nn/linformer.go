package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// LinformerAttention is BIDIRECTIONAL multi-head self-attention with LINEAR
// time and memory in the sequence length, from Wang, Li, Khabsa, Fang & Ma
// (Facebook AI 2020, "Linformer: Self-Attention with Linear Complexity",
// arXiv:2006.04768). Standard attention forms the full [L,L] score matrix
// softmax(Q·Kᵀ/√d)·V and so costs O(L²·d) — quadratic in the sequence length L.
// Linformer's insight is that this score matrix is empirically LOW RANK, so the
// KEY and VALUE sequences can be compressed along their LENGTH axis, from L rows
// down to a fixed k ≪ L, with two small LEARNED linear projections E,F ∈ R^{k×L}
// BEFORE attention:
//
//	Q = x·Wq;  K = x·Wk;  V = x·Wv              (weights [dModel,dModel])
//	K̄ = E·K                                     ([k,dModel]: projects L keys → k)
//	V̄ = F·V                                     ([k,dModel]: projects L values → k)
//	per head h: Aₕ = softmax(Qₕ·K̄ₕᵀ/√d_k)·V̄ₕ   (scores [L,k], NOT [L,L]; output [L,d_k])
//	out = concat(A₀..A_{h−1})·Wo
//
// The score matrix is now [L,k] instead of [L,L], and each head costs O(L·k·d_k)
// — LINEAR in L for a fixed k. E and F are single per-layer parameters shared
// across all heads (the paper's default sharing).
//
// # k, the projected length (§C21)
//
// projDim k is THE knob and its whole point is the L²→L·k trade:
//   - k must satisfy 1 ≤ k ≤ L. k = L keeps the full O(L·L) cost with no
//     compression; k ≪ L is the useful regime (the paper reports k = 256
//     competitive at L = 512..4096, i.e. a constant independent of L).
//   - SMALLER k → cheaper (less compute, smaller [L,k] scores) but a coarser
//     low-rank approximation of full attention. LARGER k → closer to full
//     attention but more expensive. The approximation error shrinks as k grows
//     toward L (empirically monotone for a fixed projection family).
//   - The projection is genuinely a compression only for k < L: it maps the L
//     keys/values into k learned "pseudo tokens".
//
// # NON-CAUSAL by construction (read before using in a decoder)
//
// Linformer is an ENCODER / BIDIRECTIONAL mechanism and CANNOT be made causal.
// The length projection E·K mixes ALL L key positions into each of the k
// compressed rows, so pseudo-key a already contains information from future
// positions j > i — there is no coherent way to mask position i from attending
// to j > i once the sequence has been collapsed. This layer therefore applies NO
// causal mask; use it for bidirectional encoders (masked-LM, classification,
// retrieval), not autoregressive decoding. For causal linear attention see other
// families entirely (e.g. RetNet/GLA-style decay, or the Aaren layer in this
// package). Faking a causal mask here would be incorrect, so it is deliberately
// absent.
//
// # Fixed sequence length L (§C21)
//
// E and F have a FIXED L columns baked in at construction (they are [k,L]
// tensors), so this layer only accepts inputs of exactly length L — Forward
// returns an error on any other length. This is inherent to the mechanism, not a
// limitation of the implementation: the learned length projection is tied to the
// specific L it was trained for. (Contrast content-based attention, whose weights
// are length-agnostic.) Build one layer per sequence length you need.
//
// # Why no new kernel is needed
//
// Linformer composes entirely from existing dispatched ops, each with a
// first-order VJP, so the whole layer trains end to end with NO backend change:
// the four projections and the two length projections E·K, F·V are OpMatMul; the
// per-head scores are OpMatMul + OpTranspose scaled by OpMul (B60: the 1/√d_k
// factor may be tiny, so it is a multiply, never an AXPY); the weights are
// OpSoftmax over the k compressed keys; the value aggregation and output
// projection are OpMatMul; heads split/join with OpSlice/OpConcat. Gradients flow
// to Wq/Wk/Wv/Wo AND to the projections E,F through the standard dispatch.
//
// FURTHER READING (§C18): Wang et al. 2020, arXiv:2006.04768 (this layer);
// Vaswani et al. 2017, arXiv:1706.03762 (the O(L²) softmax attention it
// approximates); the MHA block in this repo's nlp package is that quadratic
// baseline.
type LinformerAttention struct {
	Dim     int // model / embedding width (columns of x)
	Heads   int // number of attention heads
	HeadDim int // per-head dimension d_k = Dim/Heads
	SeqLen  int // fixed sequence length L (E,F have L columns)
	ProjDim int // projected length k (E,F have k rows); 1 ≤ k ≤ L

	Wq, Wk, Wv, Wo *tensor.Tensor // projection weights, each [Dim, Dim]
	// E, F are the learned LENGTH projections, each [ProjDim, SeqLen] = [k, L]:
	// K̄ = E·K compresses the L keys to k, V̄ = F·V the L values to k. With
	// WithLinformerShareKV, F is the SAME tensor as E (F == E), so KV share one
	// projection and Params reports E once.
	E, F *tensor.Tensor

	shareKV bool // F == E (shared key/value projection)
}

// LinformerAttentionOption configures a LinformerAttention layer
// (functional-options idiom, §C12).
type LinformerAttentionOption func(*linformerConfig)

type linformerConfig struct {
	heads    int
	seed     uint64
	shareKV  bool
	identity bool
}

// WithLinformerHeads sets the number of attention heads (default 1). Dim must be
// divisible by heads; each head gets HeadDim = Dim/heads. The E,F length
// projections are shared across all heads (the paper's default sharing), so head
// count changes only the score/value split, not the projection parameters.
func WithLinformerHeads(heads int) LinformerAttentionOption {
	return func(c *linformerConfig) { c.heads = heads }
}

// WithLinformerShareKV ties the value projection to the key projection: F = E
// (the same tensor). This is the paper's key-value parameter sharing — it halves
// the projection parameters (Params then reports E once, not E and F separately)
// with, per the paper, negligible quality cost. Default OFF (independent E and
// F). Gradients into E then accumulate contributions from BOTH the K̄ = E·K and
// V̄ = E·V paths.
func WithLinformerShareKV() LinformerAttentionOption {
	return func(c *linformerConfig) { c.shareKV = true }
}

// WithLinformerSeed sets the RNG seed for parameter init (default 0), for
// reproducibility (§V13). The four Xavier-uniform projections use seed..seed+3
// and the length projections E,F use seed+4,seed+5.
func WithLinformerSeed(seed uint64) LinformerAttentionOption {
	return func(c *linformerConfig) { c.seed = seed }
}

// WithLinformerIdentityProj is a TEST SEAM that sets E = F = the L×L identity
// matrix instead of random projections; it REQUIRES projDim k == seqLen L (else
// NewLinformer errors, since identity is only defined square). With identity
// projections K̄ = K and V̄ = V, so the layer collapses EXACTLY to standard
// bidirectional softmax attention softmax(Q·Kᵀ/√d_k)·V — the §V16 anchor pinning
// "Linformer = full attention with a low-rank length projection". Not for
// production use (it defeats the whole point of the compression); it exists to
// verify the collapse.
func WithLinformerIdentityProj() LinformerAttentionOption {
	return func(c *linformerConfig) { c.identity = true }
}

// NewLinformer builds a Linformer attention layer over model width dModel for a
// FIXED sequence length seqLen (= L), compressing the key/value length to projDim
// (= k) via learned projections E,F ∈ R^{k×L}. The four Q/K/V/output projections
// are Xavier-uniform [dModel,dModel] Linears with no bias (Glorot 2010); E,F are
// Xavier-uniform [k,L]. It is bidirectional (non-causal by construction). Options
// set the head count (WithLinformerHeads, default 1), key/value projection
// sharing (WithLinformerShareKV), the seed (WithLinformerSeed) and the
// identity-projection test seam (WithLinformerIdentityProj).
//
// Errors: dModel ≤ 0; heads ≤ 0 or dModel not divisible by heads; seqLen < 1;
// projDim < 1; projDim > seqLen (k must not exceed L — that would be an
// up-projection, never a compression); identity requested with projDim ≠ seqLen.
func NewLinformer(dtype tensor.Dtype, dModel, seqLen, projDim int, opts ...LinformerAttentionOption) (*LinformerAttention, error) {
	cfg := linformerConfig{heads: 1}
	for _, o := range opts {
		o(&cfg)
	}
	if dModel <= 0 {
		return nil, fmt.Errorf("nn: Linformer dModel %d must be positive", dModel)
	}
	if cfg.heads <= 0 || dModel%cfg.heads != 0 {
		return nil, fmt.Errorf("nn: Linformer dModel %d not divisible by heads %d", dModel, cfg.heads)
	}
	if seqLen < 1 {
		return nil, fmt.Errorf("nn: Linformer seqLen %d must be ≥ 1", seqLen)
	}
	if projDim < 1 {
		return nil, fmt.Errorf("nn: Linformer projDim %d must be ≥ 1", projDim)
	}
	if projDim > seqLen {
		return nil, fmt.Errorf("nn: Linformer projDim k=%d must not exceed seqLen L=%d (k≤L)", projDim, seqLen)
	}
	if cfg.identity && projDim != seqLen {
		return nil, fmt.Errorf("nn: Linformer identity projection requires projDim k=%d == seqLen L=%d", projDim, seqLen)
	}
	m := &LinformerAttention{
		Dim:     dModel,
		Heads:   cfg.heads,
		HeadDim: dModel / cfg.heads,
		SeqLen:  seqLen,
		ProjDim: projDim,
		shareKV: cfg.shareKV,
	}
	m.Wq = xavierMat(dtype, dModel, dModel, cfg.seed)
	m.Wk = xavierMat(dtype, dModel, dModel, cfg.seed+1)
	m.Wv = xavierMat(dtype, dModel, dModel, cfg.seed+2)
	m.Wo = xavierMat(dtype, dModel, dModel, cfg.seed+3)
	if cfg.identity {
		m.E = tensor.Eye(dtype, seqLen)
		if cfg.shareKV {
			m.F = m.E
		} else {
			m.F = tensor.Eye(dtype, seqLen)
		}
	} else {
		m.E = xavierMat(dtype, projDim, seqLen, cfg.seed+4)
		if cfg.shareKV {
			m.F = m.E
		} else {
			m.F = xavierMat(dtype, projDim, seqLen, cfg.seed+5)
		}
	}
	return m, nil
}

// xavierMat builds a [rows,cols] Xavier-uniform tensor. For E,F ∈ R^{k×L} the
// fan-in is L (the axis being contracted) and fan-out is k.
func xavierMat(dtype tensor.Dtype, rows, cols int, seed uint64) *tensor.Tensor {
	t := tensor.New(dtype, tensor.Shape{rows, cols})
	XavierUniform(t, cols, rows, seed)
	return t
}

func (m *LinformerAttention) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs Linformer attention on x[L, Dim] → [L, Dim]: the Q/K/V
// projections, the E·K and F·V length projections down to k rows, per-head scaled
// scores Q·K̄ᵀ/√d_k of shape [L,k], the softmax over the k compressed keys, the
// value aggregation, head concatenation, then the output projection. Bidirectional
// (no mask). Fully differentiable — gradients reach Wq/Wk/Wv/Wo and the length
// projections E,F. The input length MUST equal the fixed SeqLen L (E,F have L
// columns); any other length is an error.
func (m *LinformerAttention) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != m.Dim {
		return nil, fmt.Errorf("nn: Linformer expects x [L,%d], got %v", m.Dim, x.Shape())
	}
	if x.Shape()[0] != m.SeqLen {
		return nil, fmt.Errorf("nn: Linformer input length %d != fixed SeqLen %d (E,F are fixed-L)", x.Shape()[0], m.SeqLen)
	}
	//perfscan:ignore PS3024 OpMatMul Q-proj graph dispatch; matmul IS the work, can't fuse without breaking autograd
	q, err := m.exec(ctx, backend.OpMatMul, nil, x, m.Wq) // [L, Dim]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpMatMul K-proj graph dispatch; matmul-dominated, non-fusible
	k, err := m.exec(ctx, backend.OpMatMul, nil, x, m.Wk) // [L, Dim]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpMatMul V-proj graph dispatch; matmul-dominated, non-fusible
	v, err := m.exec(ctx, backend.OpMatMul, nil, x, m.Wv) // [L, Dim]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpMatMul E·K length-projection dispatch; matmul IS the work
	kbar, err := m.exec(ctx, backend.OpMatMul, nil, m.E, k) // E[k,L]·K[L,Dim] = [k, Dim]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpMatMul F·V length-projection dispatch; matmul IS the work
	vbar, err := m.exec(ctx, backend.OpMatMul, nil, m.F, v) // F[k,L]·V[L,Dim] = [k, Dim]
	if err != nil {
		return nil, err
	}
	invSqrtD := scalarTensor(x.Dtype(), 1/math.Sqrt(float64(m.HeadDim)))
	heads := make([]*tensor.Tensor, m.Heads)
	for h := range m.Heads {
		lo, hi := h*m.HeadDim, (h+1)*m.HeadDim
		slice := func(t *tensor.Tensor) (*tensor.Tensor, error) {
			//perfscan:ignore PS3024,PS6018 OpSlice per-head; cheap, matmul-dominated per-head loop (PS4011 false-positive class) | per-head slice closure
			return m.exec(ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: lo, End: hi}, t)
		}
		qh, err := slice(q) // [L, HeadDim]
		if err != nil {
			return nil, err
		}
		kbh, err := slice(kbar) // [k, HeadDim]
		if err != nil {
			return nil, err
		}
		vbh, err := slice(vbar) // [k, HeadDim]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 OpTranspose per-head; cheap, matmul-dominated
		kbhT, err := m.exec(ctx, backend.OpTranspose, nil, kbh) // [HeadDim, k]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 OpMatMul scores; the work, non-fusible dispatch
		sc, err := m.exec(ctx, backend.OpMatMul, nil, qh, kbhT) // [L, k]
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 OpMul scale; cheap elementwise dispatch in matmul-dominated head loop
		if sc, err = m.exec(ctx, backend.OpMul, nil, sc, invSqrtD); err != nil { // B60: OpMul, scale may be tiny
			return nil, err
		}
		//perfscan:ignore PS3024 OpSoftmax; already-fused backend op, matmul-dominated context
		w, err := m.exec(ctx, backend.OpSoftmax, nil, sc) // [L, k], softmax over the k compressed keys
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 OpMatMul value-aggregation; the work, non-fusible
		if heads[h], err = m.exec(ctx, backend.OpMatMul, nil, w, vbh); err != nil { // [L, HeadDim]
			return nil, err
		}
	}
	concat, err := m.exec(ctx, backend.OpConcat, backend.ConcatAttrs{Axis: 1}, heads...) // [L, Dim]
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpMatMul output projection; the work, non-fusible
	return m.exec(ctx, backend.OpMatMul, nil, concat, m.Wo) // [L, Dim]
}

// Params returns every trainable tensor — the Q/K/V/output projection weights and
// the length projection E, plus F when it is NOT shared with E
// (WithLinformerShareKV makes F == E, so it is reported once). Feed this to an
// optimizer. Count: 6 without sharing, 5 with WithLinformerShareKV.
func (m *LinformerAttention) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.Wq, m.Wk, m.Wv, m.Wo, m.E}
	if !m.shareKV {
		ps = append(ps, m.F)
	}
	return ps
}
