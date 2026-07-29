package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// QKNorm implements scaled-cosine query-key normalization (Henry et al. 2020,
// "Query-Key Normalization for Transformers", arXiv:2010.04245, EMNLP Findings).
// The query and key vectors are L2-normalized along head_dim per head (Eq. 2,
// q̂ᵢ = qᵢ/‖qᵢ‖), so each Q̂·K̂ entry is a cosine similarity bounded in [−1,1].
// The usual 1/√d_k factor is then REPLACED by a single learned scalar g (Eq. 4→5,
// attn = softmax(g · Q̂K̂ᵀ)·V), so the pre-softmax logits are bounded in [−g,+g].
// Decoupling logit magnitude from d_k stabilizes attention, especially for
// low-resource and long-context training. Unlike ScaleNorm, QKNorm complements
// LayerNorm/RMSNorm rather than replacing it.
type QKNorm struct {
	G   *tensor.Tensor // learned scalar scale g, shape [1,1] (§V16: Eq. 5)
	Eps float64        // L2-norm variance floor (default 1e-6)
}

// NewQKNorm builds a QKNorm with the learned scale initialized to initG. Use
// QKNormInitG to compute the paper's recommended g₀ from the training sequence
// length.
func NewQKNorm(dtype tensor.Dtype, initG float64) *QKNorm {
	g := tensor.New(dtype, tensor.Shape{1, 1})
	g.SetF64(initG, 0, 0)
	return &QKNorm{G: g, Eps: 1e-6}
}

// QKNormInitG returns Henry et al. 2020's recommended initialization for the
// learned scale g (Eq. 3): g₀ = log₂(L² − L). Per the paper, L is the
// 97.5th-percentile training sequence length (a data-derived value), NOT the raw
// maximum sequence length — pass that percentile here.
func QKNormInitG(percentileSeqLen int) float64 {
	l := float64(percentileSeqLen)
	return math.Log2(l*l - l)
}

// Attention computes single-head scaled-cosine attention. q, k, v are rank-2
// [seq, head_dim] (q may have a different query length than k/v). It returns the
// attended output [seq_q, head_dim_v]. When causal is true, query position i
// attends only to key positions j ≤ i. The whole computation is differentiable
// w.r.t. q, k, v and the learned scale G.
func (qk *QKNorm) Attention(ctx *backend.Context, q, k, v *tensor.Tensor, causal bool) (*tensor.Tensor, error) {
	if q.Ndim() != 2 || k.Ndim() != 2 || v.Ndim() != 2 {
		return nil, fmt.Errorf("nn: QKNorm.Attention wants rank-2 [seq,dim] q,k,v, got %d/%d/%d", q.Ndim(), k.Ndim(), v.Ndim())
	}
	if q.Shape()[1] != k.Shape()[1] {
		return nil, fmt.Errorf("nn: QKNorm.Attention q/k head_dim mismatch: %d vs %d", q.Shape()[1], k.Shape()[1])
	}
	if k.Shape()[0] != v.Shape()[0] {
		return nil, fmt.Errorf("nn: QKNorm.Attention k/v seq mismatch: %d vs %d", k.Shape()[0], v.Shape()[0])
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	qn, err := qkL2NormalizeLastAxis(ctx, q, qk.Eps)
	if err != nil {
		return nil, err
	}
	kn, err := qkL2NormalizeLastAxis(ctx, k, qk.Eps)
	if err != nil {
		return nil, err
	}
	knT, err := ex(backend.OpTranspose, nil, kn) // [head_dim, seq_k]
	if err != nil {
		return nil, err
	}
	cos, err := ex(backend.OpMatMul, nil, qn, knT) // [seq_q, seq_k], entries in [-1,1]
	if err != nil {
		return nil, err
	}
	logits, err := ex(backend.OpMul, nil, cos, qk.G) // g·Q̂K̂ᵀ, broadcast [sq,sk]*[1,1]
	if err != nil {
		return nil, err
	}
	if causal {
		mask := qkCausalMask(logits.Dtype(), q.Shape()[0], k.Shape()[0])
		if logits, err = ex(backend.OpAdd, nil, logits, mask); err != nil {
			return nil, err
		}
	}
	attn, err := ex(backend.OpSoftmax, nil, logits) // row-softmax over the last (key) axis
	if err != nil {
		return nil, err
	}
	return ex(backend.OpMatMul, nil, attn, v) // [seq_q, head_dim_v]
}

// Params returns the learned scale for the optimizer.
func (qk *QKNorm) Params() []*tensor.Tensor { return []*tensor.Tensor{qk.G} }

// qkL2NormalizeLastAxis returns x with each vector along the last axis scaled to
// unit L2 norm (x / √(Σx²+eps)), differentiably.
func qkL2NormalizeLastAxis(ctx *backend.Context, x *tensor.Tensor, eps float64) (*tensor.Tensor, error) {
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	sq, err := ex(backend.OpMul, nil, x, x)
	if err != nil {
		return nil, err
	}
	ss, err := ex(backend.OpSum, backend.ReduceAttrs{Axes: []int{x.Ndim() - 1}, KeepDims: true}, sq)
	if err != nil {
		return nil, err
	}
	epsT := tensor.New(x.Dtype(), tensor.Shape{1, 1})
	epsT.SetF64(eps, 0, 0)
	ssF, err := ex(backend.OpAdd, nil, ss, epsT) // variance floor keeps the √ VJP finite
	if err != nil {
		return nil, err
	}
	norm, err := ex(backend.OpSqrt, nil, ssF)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpDiv, nil, x, norm) // broadcast [...,d]/[...,1]
}

// qkCausalMask builds an additive [sq,sk] mask that is 0 where key j ≤ query i
// (aligned to the end of the key axis) and a large negative constant above it, so
// softmax assigns ~0 weight to future positions.
func qkCausalMask(dtype tensor.Dtype, sq, sk int) *tensor.Tensor {
	m := tensor.New(dtype, tensor.Shape{sq, sk})
	off := sk - sq // right-align when the query is a suffix of the keys
	// Only cells j>i+off get -1e30 (the rest stay 0 from New); write them directly into the
	// contiguous storage, skipping the lower triangle and the per-element SetF64 dispatch.
	// Byte-identical to the SetF64 loop.
	switch dtype {
	case tensor.F64:
		d := m.Storage().F64()
		for i := range sq {
			for j := i + off + 1; j < sk; j++ {
				if j >= 0 {
					d[i*sk+j] = -1e30
				}
			}
		}
	case tensor.F32:
		d := m.Storage().F32()
		for i := range sq {
			for j := i + off + 1; j < sk; j++ {
				if j >= 0 {
					d[i*sk+j] = -1e30
				}
			}
		}
	default:
		for i := range sq {
			for j := range sk {
				if j > i+off {
					m.SetF64(-1e30, i, j)
				}
			}
		}
	}
	return m
}
