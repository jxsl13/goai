package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// DiffAttention implements Differential Attention (Ye, Dong, Zhang, Zhu, Yang,
// Wang, Ma & Wei 2024, "Differential Transformer", arXiv:2410.05258). The query
// and key projections are split into two groups, giving two softmax attention
// maps whose DIFFERENCE — scaled by a learnable λ — is applied to the shared
// values (Eq. 1):
//
//	DiffAttn(X) = (softmax(Q₁K₁ᵀ/√d) − λ·softmax(Q₂K₂ᵀ/√d))·V
//
// Subtracting the two maps cancels the common-mode "attention noise" both share,
// concentrating probability mass on relevant context (analogous to a differential
// amplifier). λ is re-parameterized (Eq. 2) from four learnable d-vectors so it
// stays well-conditioned and centered on a depth-dependent constant λ_init:
//
//	λ = exp(λ_q1·λ_k1) − exp(λ_q2·λ_k2) + λ_init
//
// with λ_init = 0.8 − 0.6·exp(−0.3·(l−1)) for 1-based layer index l (Eq. 3). Per
// the paper each head output is then normalized (per-head RMSNorm, "GroupNorm")
// and scaled by (1−λ_init) before the output projection — apply nn.RMSNorm and
// SublayerScale() to the returned head output for the full sublayer.
type DiffAttention struct {
	LambdaQ1   *tensor.Tensor // learnable [1,d], λ centers on LambdaInit when all four are 0
	LambdaK1   *tensor.Tensor // learnable [1,d]
	LambdaQ2   *tensor.Tensor // learnable [1,d]
	LambdaK2   *tensor.Tensor // learnable [1,d]
	LambdaInit float64        // depth-dependent offset λ_init (Eq. 3)
}

// NewDiffAttention builds a differential-attention head for head dimension d at
// 1-based layer index layer (layer ≥ 1). The four λ-vectors start at 0, so the
// effective λ starts exactly at λ_init = 0.8 − 0.6·exp(−0.3·(layer−1)), matching
// the paper's intent that λ is initialized near this depth-dependent value.
func NewDiffAttention(dtype tensor.Dtype, d, layer int) *DiffAttention {
	if layer < 1 {
		layer = 1
	}
	return &DiffAttention{
		LambdaQ1:   tensor.New(dtype, tensor.Shape{1, d}),
		LambdaK1:   tensor.New(dtype, tensor.Shape{1, d}),
		LambdaQ2:   tensor.New(dtype, tensor.Shape{1, d}),
		LambdaK2:   tensor.New(dtype, tensor.Shape{1, d}),
		LambdaInit: DiffLambdaInit(layer),
	}
}

// DiffLambdaInit returns the paper's depth-dependent λ initializer (Eq. 3):
// λ_init = 0.8 − 0.6·exp(−0.3·(layer−1)) for the 1-based layer index.
func DiffLambdaInit(layer int) float64 {
	return 0.8 - 0.6*math.Exp(-0.3*float64(layer-1))
}

// SublayerScale returns the (1−λ_init) factor the paper applies after the per-head
// GroupNorm, so the caller can complete the sublayer: (1−λ_init)·RMSNorm(out).
func (da *DiffAttention) SublayerScale() float64 { return 1 - da.LambdaInit }

// Attention computes single-head differential attention. q1,k1,q2,k2 are rank-2
// [seq, head_dim] (the two query/key groups); v is [seq, head_dim_v]; all four
// q/k share head_dim. It returns the attended output [seq_q, head_dim_v]. When
// causal is true, query position i attends only to key positions j ≤ i. The whole
// computation is differentiable w.r.t. q1,k1,q2,k2,v and the four λ-vectors.
func (da *DiffAttention) Attention(ctx *backend.Context, q1, k1, q2, k2, v *tensor.Tensor, causal bool) (*tensor.Tensor, error) {
	for _, t := range []*tensor.Tensor{q1, k1, q2, k2, v} {
		if t.Ndim() != 2 {
			return nil, fmt.Errorf("nn: DiffAttention wants rank-2 [seq,dim] inputs, got %v", t.Shape())
		}
	}
	d := q1.Shape()[1]
	if k1.Shape()[1] != d || q2.Shape()[1] != d || k2.Shape()[1] != d {
		return nil, fmt.Errorf("nn: DiffAttention head_dim mismatch across q1/k1/q2/k2")
	}
	if da.LambdaQ1.Shape()[1] != d {
		return nil, fmt.Errorf("nn: DiffAttention λ-vector dim %d != head_dim %d", da.LambdaQ1.Shape()[1], d)
	}
	sk := k1.Shape()[0]
	if k2.Shape()[0] != sk || v.Shape()[0] != sk {
		return nil, fmt.Errorf("nn: DiffAttention key/value seq length mismatch")
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	invSqrtD := scalarTensor(q1.Dtype(), 1/math.Sqrt(float64(d)))
	var mask *tensor.Tensor
	if causal {
		mask = qkCausalMask(q1.Dtype(), q1.Shape()[0], sk)
	}
	// softmax(QᵢKᵢᵀ/√d), with the causal mask applied to the pre-softmax logits.
	softmaxMap := func(q, k *tensor.Tensor) (*tensor.Tensor, error) {
		kt, err := ex(backend.OpTranspose, nil, k)
		if err != nil {
			return nil, err
		}
		s, err := ex(backend.OpMatMul, nil, q, kt)
		if err != nil {
			return nil, err
		}
		if s, err = ex(backend.OpMul, nil, s, invSqrtD); err != nil {
			return nil, err
		}
		if causal {
			if s, err = ex(backend.OpAdd, nil, s, mask); err != nil {
				return nil, err
			}
		}
		return ex(backend.OpSoftmax, nil, s)
	}
	a1, err := softmaxMap(q1, k1)
	if err != nil {
		return nil, err
	}
	a2, err := softmaxMap(q2, k2)
	if err != nil {
		return nil, err
	}
	lambda, err := da.lambda(ctx)
	if err != nil {
		return nil, err
	}
	scaled2, err := ex(backend.OpMul, nil, a2, lambda) // broadcast [sq,sk]*[1,1]
	if err != nil {
		return nil, err
	}
	attn, err := ex(backend.OpSub, nil, a1, scaled2) // softmax₁ − λ·softmax₂
	if err != nil {
		return nil, err
	}
	return ex(backend.OpMatMul, nil, attn, v)
}

// lambda computes the reparameterized scalar λ = exp(λq1·λk1) − exp(λq2·λk2) +
// λ_init as a [1,1] tensor, differentiable w.r.t. the four λ-vectors.
func (da *DiffAttention) lambda(ctx *backend.Context) (*tensor.Tensor, error) {
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, at)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	expDot := func(a, b *tensor.Tensor) (*tensor.Tensor, error) {
		bt, err := ex(backend.OpTranspose, nil, b) // [d,1]
		if err != nil {
			return nil, err
		}
		dot, err := ex(backend.OpMatMul, nil, a, bt) // [1,1] = a·b
		if err != nil {
			return nil, err
		}
		return ex(backend.OpExp, nil, dot)
	}
	e1, err := expDot(da.LambdaQ1, da.LambdaK1)
	if err != nil {
		return nil, err
	}
	e2, err := expDot(da.LambdaQ2, da.LambdaK2)
	if err != nil {
		return nil, err
	}
	diff, err := ex(backend.OpSub, nil, e1, e2)
	if err != nil {
		return nil, err
	}
	return ex(backend.OpAdd, nil, diff, scalarTensor(e1.Dtype(), da.LambdaInit))
}

// Params returns the four learnable λ-vectors for the optimizer.
func (da *DiffAttention) Params() []*tensor.Tensor {
	return []*tensor.Tensor{da.LambdaQ1, da.LambdaK1, da.LambdaQ2, da.LambdaK2}
}

// scalarTensor makes a [1,1] constant that broadcasts against a [rows,cols] tensor.
func scalarTensor(dtype tensor.Dtype, v float64) *tensor.Tensor {
	t := tensor.New(dtype, tensor.Shape{1, 1})
	t.SetF64(v, 0, 0)
	return t
}
