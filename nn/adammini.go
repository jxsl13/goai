package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// AdamMini is the Adam-mini optimizer (Zhang et al. 2024, "Adam-mini: Use Fewer
// Learning Rates To Gain More", arXiv:2406.16793): Adam with the per-parameter
// second moment v replaced by ONE scalar per parameter BLOCK, cutting the v state
// from O(#params) to O(#blocks) — about half the optimizer memory — at comparable
// training quality. The first moment m stays per-parameter, exactly like Adam.
//
// In plain terms: Adam remembers two numbers for every weight (a direction and a
// step-size scale); Adam-mini notices that weights inside one block (e.g. the row
// feeding one output neuron) can share a single step-size scale, so it stores one
// scale per block instead of one per weight — the same idea as giving each block
// one learning rate. Memory shrinks by nearly half and, per the paper, training
// loss tracks Adam closely.
//
// Per step t, with the parameters of each block B and its shared scalar v_B:
//
//	m ← β₁m + (1−β₁)g                       (per parameter, as Adam)
//	v_B ← β₂v_B + (1−β₂)·mean(g² over B)    (ONE scalar per block)
//	m̂ = m/(1−β₁ᵗ);  v̂_B = v_B/(1−β₂ᵗ);  p ← p − lr·m̂/(√v̂_B + ε)
//
// with the same bias correction and eps placement as [Adam], and optional
// decoupled (AdamW-style) weight decay. Optimizer state is float64 (§V10).
//
// BLOCK PARTITIONING. The paper partitions by parameter ROLE (Hessian block
// structure): Linear weights per output neuron, embeddings per token row,
// attention Q/K/V per head. A generic optimizer over []*tensor.Tensor cannot see
// roles, so AdamMini uses a documented shape HEURISTIC with an override:
//
//   - ≥2-D parameter [d₀, d₁, …]: one block per leading index (d₀ blocks, each
//     covering the flattened trailing dims). For a 2-D weight [out, in] this is
//     one block per ROW = per output neuron — the paper's main case, and the
//     sensible generic default because the leading dim is the output/neuron/token
//     axis in this library's Linear ([out,in]) and Embedding ([vocab,dim]) layouts.
//   - 1-D or scalar parameter (bias, norm gain): ONE block for the whole tensor —
//     the paper keeps these small vectors Adam-like; a single shared scalar is the
//     minimal-state choice and they are a negligible fraction of the model anyway.
//
// A caller who KNOWS the structure overrides the heuristic with
// [WithAdamMiniBlockSizes] — e.g. block size head_dim·model_dim on a fused Q/K/V
// weight partitions it per attention head as the paper prescribes.
//
// Further reading: arXiv:2406.16793 (Adam-mini); arXiv:1412.6980 (Adam, the
// baseline it halves — see [Adam] in this package).
type AdamMini struct {
	Params []*tensor.Tensor // parameters this optimizer updates

	// LR is the learning rate (step size). In plain terms: how big a step to take
	// downhill each update — too small and training crawls, too large and it diverges.
	// Boundary behavior: LR→0 stalls learning; past the (problem-specific) stability
	// threshold the loss oscillates or explodes. No universal default — Adam-mini uses
	// the SAME lr you would give Adam (arXiv:2406.16793 §4 tunes it on the Adam grid;
	// 1e-3 is the classic Adam starting point, 1e-4–3e-4 typical for transformers).
	LR float64
	// Beta1 is the first-moment (momentum) EMA decay, exactly Adam's β₁. In plain
	// terms: how much the step direction is smoothed over recent gradients — higher =
	// more inertia. Boundary behavior: 0 = no momentum (react only to the current
	// gradient); →1 = sluggish. Default 0.9 (arXiv:2406.16793 follows Adam's canonical
	// value, Kingma & Ba 2015 §6).
	Beta1 float64
	// Beta2 is the second-moment EMA decay for the per-BLOCK scale v_B. In plain
	// terms: how long a memory each block's shared step-size scale has. Boundary
	// behavior: too low = noisy scaling; →1 = slow to adapt. Default 0.999
	// (arXiv:2406.16793 follows Adam's canonical value, Kingma & Ba 2015 §6).
	Beta2 float64
	// Eps is the denominator epsilon added outside the square root, exactly as in
	// Adam. In plain terms: a tiny floor that stops division by (near-)zero when a
	// block's gradient variance is small. Boundary behavior: too small risks huge
	// steps on quiet blocks; larger eps caps the effective step. Default 1e-8
	// (arXiv:2406.16793 follows Adam's convention, Kingma & Ba 2015 §6).
	Eps float64
	// WeightDecay > 0 selects decoupled (AdamW-style) decay: p ← p·(1−lr·wd) − step.
	// In plain terms: gently shrink every weight toward zero each update,
	// independently of the adaptive step. Boundary behavior: large wd underfits.
	// SPECIAL VALUE: 0 = disabled (plain Adam-mini). Typical 0.01–0.1 for
	// transformer training, as with AdamW.
	WeightDecay float64

	m [][]float64 // per-param first moment, full parameter size (as Adam)
	v [][]float64 // per-param second moment, ONE float64 per block (the memory win)

	blockSize []int // per-param block size in elements; the last block may be shorter
	blockSpec []int // WithAdamMiniBlockSizes override (nil = shape heuristic)
	err       error // deferred construction error, surfaced by Step
	t         int
}

// AdamMiniOption configures an AdamMini optimizer (functional-options idiom, §C12).
type AdamMiniOption func(*AdamMini)

// WithAdamMiniBetas sets the two EMA decays: β₁ for the per-parameter momentum, β₂ for
// the per-block second-moment scalar.
//
// In plain terms: the same two smoothing knobs as Adam — β₁ smooths the step direction,
// β₂ sets how long each block's shared step-size scale remembers. Boundary behavior:
// β₁=0 removes momentum; either →1 becomes sluggish; β₂ too low makes the shared scale
// noisy. Defaults 0.9, 0.999 (arXiv:2406.16793 keeps Adam's canonical values,
// Kingma & Ba 2015 §6).
func WithAdamMiniBetas(beta1, beta2 float64) AdamMiniOption {
	return func(a *AdamMini) { a.Beta1, a.Beta2 = beta1, beta2 }
}

// WithAdamMiniEps sets the denominator epsilon ε (see the Eps field).
//
// In plain terms: the near-zero-variance guard, added outside the square root exactly as
// in Adam. Boundary behavior: too small risks huge steps on quiet blocks; larger caps
// the step. Default 1e-8 (arXiv:2406.16793, matching Adam's convention).
func WithAdamMiniEps(eps float64) AdamMiniOption { return func(a *AdamMini) { a.Eps = eps } }

// WithAdamMiniWeightDecay sets the decoupled (AdamW-style) weight decay wd.
//
// In plain terms: shrink weights toward zero each step, kept out of the adaptive
// scaling like AdamW. Boundary behavior: large wd underfits. SPECIAL VALUE: 0 =
// disabled. Default 0 (set per task exactly as you would for AdamW; 0.01–0.1 typical).
func WithAdamMiniWeightDecay(wd float64) AdamMiniOption {
	return func(a *AdamMini) { a.WeightDecay = wd }
}

// WithAdamMiniBlockSizes overrides the shape heuristic with an explicit block size per
// parameter: sizes[i] elements of Params[i] (in flat row-major order) share one
// second-moment scalar; the last block is shorter when the size does not divide Numel.
//
// In plain terms: use this when you KNOW the parameter's structure better than its
// shape reveals — the paper's role-based partition. Examples: sizes[i] = head_dim ×
// model_dim partitions a fused [n_heads·head_dim, model_dim] Q/K/V weight per attention
// head (arXiv:2406.16793 §3.2); sizes[i] = 1 gives every element its own scalar, which
// is exactly Adam; sizes[i] = Numel is one shared scale for the whole tensor (maximum
// compression). Boundary behavior: each size must be in [1, Params[i].Numel()] and
// len(sizes) must equal len(Params) — violations surface as an error from Step.
// Default: unset = the per-row heuristic documented on [AdamMini].
func WithAdamMiniBlockSizes(sizes []int) AdamMiniOption {
	return func(a *AdamMini) { a.blockSpec = sizes }
}

// NewAdamMini builds an Adam-mini optimizer (Zhang et al. 2024, arXiv:2406.16793) over
// params with learning rate lr and the paper's defaults β₁=0.9, β₂=0.999, ε=1e-8 (Adam's
// canonical values, which the paper keeps) and no weight decay. Blocks default to the
// shape heuristic documented on [AdamMini] (per leading row for ≥2-D, whole-tensor for
// 1-D); override with [WithAdamMiniBlockSizes]. Use the lr you would give Adam.
func NewAdamMini(params []*tensor.Tensor, lr float64, opts ...AdamMiniOption) *AdamMini {
	a := &AdamMini{Params: params, LR: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8}
	for _, o := range opts {
		o(a)
	}
	if a.blockSpec != nil && len(a.blockSpec) != len(params) {
		a.err = fmt.Errorf("nn: AdamMini block sizes: got %d sizes for %d params",
			len(a.blockSpec), len(params))
		return a
	}
	a.m = make([][]float64, len(params))
	a.v = make([][]float64, len(params))
	a.blockSize = make([]int, len(params))
	for i, p := range params {
		n := p.Numel()
		bs := 0
		if a.blockSpec != nil {
			bs = a.blockSpec[i]
			if bs < 1 || bs > n {
				a.err = fmt.Errorf("nn: AdamMini block size %d for param %d out of range [1, %d]",
					bs, i, n)
				return a
			}
		} else if len(p.Shape()) >= 2 {
			bs = n / p.Shape()[0] // one block per leading index (per row for 2-D)
		} else {
			bs = n // 1-D/scalar: one block for the whole tensor
		}
		if bs < 1 {
			bs = 1 // guards degenerate zero-element shapes
		}
		a.blockSize[i] = bs
		a.m[i] = make([]float64, n)
		a.v[i] = make([]float64, (n+bs-1)/bs) // ceil(n/bs) blocks — the memory win
	}
	return a
}

// SecondMomentSize returns the total number of second-moment scalars this optimizer
// stores (the sum of block counts over all parameters). In plain terms: this is THE
// number Adam-mini shrinks — Adam would store one scalar per parameter element
// (Σ Numel); Adam-mini stores one per block, typically ~Numel/cols for weight matrices,
// so comparing SecondMomentSize against Σ Numel shows the ~50% optimizer-memory saving
// the method exists for.
func (a *AdamMini) SecondMomentSize() int {
	total := 0
	for _, v := range a.v {
		total += len(v)
	}
	return total
}

// Step applies one Adam-mini update. Parameters with nil gradient are skipped. The
// timestep advances once per Step call, as in Adam.
func (a *AdamMini) Step(grad GradFn) error {
	if a.err != nil {
		return a.err
	}
	a.t++
	c1 := 1 - math.Pow(a.Beta1, float64(a.t))
	c2 := 1 - math.Pow(a.Beta2, float64(a.t))
	decay := 1 - a.LR*a.WeightDecay // 1 when wd==0 → plain Adam-mini
	for pi, p := range a.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: AdamMini grad shape %v != param %v", g.Shape(), p.Shape())
		}
		m, v, bs := a.m[pi], a.v[pi], a.blockSize[pi]
		n := p.Numel()
		// Typed fast paths (contiguous f64/f32 pairs): per block, one pass to fold
		// mean(g²) into the shared scalar, one pass for the per-parameter m and the
		// update — all arithmetic in float64 exactly as the generic path computes it.
		if pf := flatF64(p); pf != nil {
			if gf := flatF64(g); gf != nil {
				for b := range v {
					lo := b * bs
					hi := min(lo+bs, n)
					sum := 0.0
					for i := lo; i < hi; i++ {
						gv := gf[i]
						sum += gv * gv
					}
					v[b] = a.Beta2*v[b] + (1-a.Beta2)*(sum/float64(hi-lo))
					den := math.Sqrt(v[b]/c2) + a.Eps
					for i := lo; i < hi; i++ {
						m[i] = a.Beta1*m[i] + (1-a.Beta1)*gf[i]
						pf[i] = pf[i]*decay - a.LR*(m[i]/c1)/den
					}
				}
				continue
			}
		} else if pf := flatF32(p); pf != nil {
			if gf := flatF32(g); gf != nil {
				for b := range v {
					lo := b * bs
					hi := min(lo+bs, n)
					sum := 0.0
					for i := lo; i < hi; i++ {
						gv := float64(gf[i])
						sum += gv * gv
					}
					v[b] = a.Beta2*v[b] + (1-a.Beta2)*(sum/float64(hi-lo))
					den := math.Sqrt(v[b]/c2) + a.Eps
					for i := lo; i < hi; i++ {
						gv := float64(gf[i])
						m[i] = a.Beta1*m[i] + (1-a.Beta1)*gv
						pf[i] = float32(float64(pf[i])*decay - a.LR*(m[i]/c1)/den)
					}
				}
				continue
			}
		}
		// Generic fallback: any dtype/layout via the widening accessors, in the exact
		// arithmetic (and order) of the fast paths.
		shape := p.Shape()
		for b := range v {
			lo := b * bs
			hi := min(lo+bs, n)
			sum := 0.0
			for i := lo; i < hi; i++ {
				gv := g.AtF64(tensor.Unravel(i, shape)...)
				sum += gv * gv
			}
			v[b] = a.Beta2*v[b] + (1-a.Beta2)*(sum/float64(hi-lo))
			den := math.Sqrt(v[b]/c2) + a.Eps
			for i := lo; i < hi; i++ {
				idx := tensor.Unravel(i, shape)
				gv := g.AtF64(idx...)
				m[i] = a.Beta1*m[i] + (1-a.Beta1)*gv
				p.SetF64(p.AtF64(idx...)*decay-a.LR*(m[i]/c1)/den, idx...)
			}
		}
	}
	return nil
}
