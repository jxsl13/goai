package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// RetNetBlock is the full Multi-Scale Retention (MSR) block of RetNet (§R114; Sun et al. 2023,
// arXiv:2307.08621, Eq.8) — the T79-analog that wraps the retention operator (§T171) into a usable
// layer, the way MambaBlock wraps the selective scan. For X ∈ [L, d_model] with H heads (per-head
// dim k = d_model/H):
//
//	head h:  Qₕ=RoPE(X·Wqₕ), Kₕ=RoPE(X·Wkₕ) (∈[L,k]), Vₕ=X·Wvₕ (∈[L,2k]); γₕ = 1−2^(−5−h)
//	         retₕ = Retention(Qₕ, k^(−½)·Kₕ, Vₕ, γₕ)               (§T171, d_v=2·k value expansion)
//	         gnₕ  = RMSNorm(retₕ)   (per head, AFFINE-FREE — the paper's "GroupNorm_h")
//	         outₕ = (swish(X·Wgₕ) ⊙ gnₕ)·Woₕ                        (Woₕ: [2k]→[d_model])
//	MSR(X) = Σₕ outₕ
//
// The per-head decomposition (separate per-head projections, summed output) is mathematically
// identical to RetNet's single big Q/K/V/G/O projections + head split + concat, but keeps every
// tensor a tracked leaf so gradients flow without a differentiable slice/concat op. All projections
// are bias-free (RetNet). The affine-free RMSNorm is scale-invariant, so it absorbs the k^(−½) QK
// scaling (RetNet's GroupNorm plays the same role) — the scaling is kept for faithfulness. The value
// head dim is 2·k (RetNet's value expansion, value_dim = 2·d_model). Every op is dispatched, so the
// block trains on the GPU (projections + retention via the GPU kernels; the ref-only RoPE rotation
// falls back to the reference §I4). The RoPE rotation on Q,K is the paper's Θ_n=e^{inθ} (§R114
// Eq.4-5) — rotation ONLY, since the retention γ^(n−m) mask already carries the magnitude decay;
// applying the full xPos ζ-scaling would double-count it (research-lite CONFIRMED). Per-head key dim
// (d_model/heads) must be even for the rotation.
type RetNetBlock struct {
	DModel, Heads  int // model/embedding width and number of retention heads
	KeyDim, ValDim int // per-head key dim (d_model/H) and value head dim (2·KeyDim, RetNet expansion)

	Wq, Wk, Wv, Wg []*Linear // per-head projections; Wq/Wk: DModel→KeyDim, Wv/Wg: DModel→ValDim
	Wo             []*Linear // per-head output: ValDim→DModel

	scale  float64        // KeyDim^(−½) QK scaling
	gammas []float64      // per-head decay γₕ
	normG  *tensor.Tensor // affine-free RMSNorm gain (fixed ones[ValDim], NOT a parameter)
	eps    float64        // RMSNorm epsilon
}

// NewRetNetBlock builds an MSR block over d_model with heads heads (d_model must be divisible by
// heads). Projections are bias-free with Xavier-uniform weights; per-head decays are γₕ=1−2^(−5−h).
func NewRetNetBlock(dtype tensor.Dtype, dModel, heads int, seed uint64) (*RetNetBlock, error) {
	if heads <= 0 || dModel%heads != 0 {
		return nil, fmt.Errorf("nn: RetNetBlock d_model %d not divisible by heads %d", dModel, heads)
	}
	k := dModel / heads
	if k%2 != 0 {
		return nil, fmt.Errorf("nn: RetNetBlock per-head key dim %d (d_model/heads) must be even for the rotary position encoding (§R114)", k)
	}
	v := 2 * k // RetNet value expansion: value head dim = 2·k, value_dim = 2·d_model (§R114 Eq.8)
	b := &RetNetBlock{
		DModel: dModel, Heads: heads, KeyDim: k, ValDim: v,
		scale: math.Pow(float64(k), -0.5), gammas: RetentionDecays(heads), eps: 1e-5,
	}
	noBias := func(l *Linear) *Linear { l.B = nil; return l } // RetNet projections are bias-free
	var s uint64 = seed
	for range heads {
		b.Wq = append(b.Wq, noBias(NewLinear(dtype, dModel, k, s+1)))
		b.Wk = append(b.Wk, noBias(NewLinear(dtype, dModel, k, s+2)))
		b.Wv = append(b.Wv, noBias(NewLinear(dtype, dModel, v, s+3)))
		b.Wg = append(b.Wg, noBias(NewLinear(dtype, dModel, v, s+4)))
		b.Wo = append(b.Wo, noBias(NewLinear(dtype, v, dModel, s+5)))
		s += 8
	}
	b.normG = tensor.New(dtype, tensor.Shape{v}) // fixed ones → affine-free RMSNorm
	for i := range v {
		b.normG.SetF64(1, i)
	}
	return b, nil
}

// Forward runs the MSR block on x[L, d_model] → [L, d_model].
func (b *RetNetBlock) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 2 || x.Shape()[1] != b.DModel {
		return nil, fmt.Errorf("nn: RetNetBlock expects x [L,%d], got %v", b.DModel, x.Shape())
	}
	var out *tensor.Tensor
	for h := range b.Heads {
		q, err := b.Wq[h].Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		// RetNet rotation Θ_n=e^{inθ} (§R114 Eq.4-5): the SAME RoPE rotation on Q and K, so the
		// score picks up the relative rotation e^{i(n−m)θ}. This is rotation-ONLY (RoPE) — the
		// magnitude decay is carried entirely by the retention γ^(n−m) mask, so applying the full
		// xPos ζ-scaling here would double-count the decay (research-lite CONFIRMED vs paper §2 +
		// torchscale theta_shift). ref-only op; the GPU path falls back for it (§I4).
		//perfscan:ignore PS3024 per-head RoPE dispatch; matmul-dominated false positive
		q, err = b.exec(ctx, backend.OpRoPE, backend.RoPEAttrs{}, q)
		if err != nil {
			return nil, err
		}
		k, err := b.Wk[h].Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 per-head RoPE dispatch; matmul-dominated false positive
		k, err = b.exec(ctx, backend.OpRoPE, backend.RoPEAttrs{}, k)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024,PS6016 per-head AXPY dispatch; matmul-dominated false positive | loop-invariant attr struct literal; resource-only pe
		ks, err := b.exec(ctx, backend.OpAXPY, backend.AXPYAttrs{Alpha: b.scale}, k, tensor.New(x.Dtype(), k.Shape()))
		if err != nil {
			return nil, err
		}
		v, err := b.Wv[h].Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 per-head retention dispatch; kernel-dominated false positive
		ret, err := b.exec(ctx, backend.OpRetention, backend.RetentionAttrs{Gamma: b.gammas[h]}, q, ks, v)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024,PS6016 per-head RMSNorm dispatch; matmul-dominated false positive | loop-invariant norm-attr literal; resource-only p
		gn, err := b.exec(ctx, backend.OpRMSNorm, backend.NormAttrs{Eps: b.eps}, ret, b.normG)
		if err != nil {
			return nil, err
		}
		g, err := b.Wg[h].Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 per-head SiLU dispatch; matmul-dominated false positive
		gact, err := b.exec(ctx, backend.OpSiLU, nil, g)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS3024 per-head Mul dispatch; matmul-dominated false positive
		gated, err := b.exec(ctx, backend.OpMul, nil, gact, gn)
		if err != nil {
			return nil, err
		}
		oh, err := b.Wo[h].Forward(ctx, gated)
		if err != nil {
			return nil, err
		}
		if out == nil {
			out = oh
			//perfscan:ignore PS3024 per-head Add dispatch; matmul-dominated false positive
		} else if out, err = b.exec(ctx, backend.OpAdd, nil, out, oh); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (b *RetNetBlock) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Params returns every trainable weight (all per-head projections; the affine-free RMSNorm gain is
// fixed and excluded).
func (b *RetNetBlock) Params() []*tensor.Tensor {
	var p []*tensor.Tensor
	for h := range b.Heads {
		p = append(p, b.Wq[h].W, b.Wk[h].W, b.Wv[h].W, b.Wg[h].W, b.Wo[h].W)
	}
	return p
}
