package nn

import "github.com/jxsl13/goai/tensor"

// Grokfast wraps a base optimizer with the Grokfast-EMA gradient filter (Lee, Ahn,
// Kim & Kim 2024, "Grokfast: Accelerated Grokking by Amplifying Slow Gradients",
// arXiv:2405.20233). Grokking — delayed generalization long after fitting the train
// set — is driven by a slow-varying (low-frequency) component of the gradient.
// Grokfast isolates that component with a per-parameter exponential moving average and
// AMPLIFIES it, accelerating generalization by up to ~50× at negligible cost:
//
//	μ_t = α·μ_{t−1} + (1−α)·g_t          // low-pass (EMA) of the raw gradient
//	ĝ_t = g_t + λ·μ_t                    // amplify the slow component, pass ĝ to the base
//
// It is a drop-in gradient transform applied BEFORE the base optimizer's step — the
// base (SGD, Adam, …) is used unchanged over the same parameters. The EMA μ is
// initialized to the first gradient (matching the reference gradfilter_ema). λ=0
// recovers the base optimizer exactly. State is float64 (§V10). Defaults λ=2.0,
// α=0.98 (the paper's Grokfast-EMA).
type Grokfast struct {
	Base   Optimizer        // the wrapped base optimizer, stepped over the same params
	Params []*tensor.Tensor // the parameters (shared with Base)
	Lambda float64          // amplification λ of the slow (EMA) gradient component
	Alpha  float64          // EMA momentum α ∈ [0,1); larger = slower/lower cutoff

	ema    [][]float64 // per-parameter gradient EMA μ
	inited bool        // false until μ has been seeded with the first gradient

	// Reused across steps: the filtered-gradient tensors handed to the base optimizer
	// (one per param, fully overwritten each step) and the p→ĝ map. The base reads each
	// gradient transiently within its Step (the GradFn contract), so reusing them avoids
	// a fresh tensor + map allocation on every call — bit-identical output.
	out      []*tensor.Tensor
	filtered map[*tensor.Tensor]*tensor.Tensor
}

// GrokfastOption configures a Grokfast optimizer (functional-options idiom, §C12).
type GrokfastOption func(*Grokfast)

// WithGrokfastLambda sets λ, how strongly the slow (low-frequency) gradient component is
// amplified before the base optimizer sees it.
//
// In plain terms: Grokfast speeds up "grokking" (the delayed jump to generalization) by
// boosting the slowly-changing part of the gradient; λ is the size of that boost. This is the
// EMA variant (a running average, cheaper than the moving-window GrokfastMA). Boundary behavior
// — λ=0 disables the filter (base optimizer runs unmodified); large λ over-weights the slow
// component and can destabilize training. SPECIAL VALUE: 0 = off.
//
// Default 2.0 (research-grounded: the Grokfast-EMA reference λ, §R153 — smaller than the
// moving-average variant's 5.0 because the EMA weights the slow component differently).
func WithGrokfastLambda(l float64) GrokfastOption { return func(g *Grokfast) { g.Lambda = l } }

// WithGrokfastAlpha sets α, the EMA decay that defines the "slow" gradient component.
//
// In plain terms: how much history the running average keeps — closer to 1 means a longer
// memory and a slower-moving trend. Boundary behavior — α→0 makes the "slow" component track
// the raw gradient (no separation, filter does nothing useful); α→1 remembers almost forever
// and reacts very slowly. Default 0.98 (research-grounded: the Grokfast-EMA reference α, §R153).
func WithGrokfastAlpha(a float64) GrokfastOption { return func(g *Grokfast) { g.Alpha = a } }

// NewGrokfast wraps base (which optimizes params) with the Grokfast-EMA filter. params
// must be the same slice the base optimizer updates.
func NewGrokfast(base Optimizer, params []*tensor.Tensor, opts ...GrokfastOption) *Grokfast {
	g := &Grokfast{Base: base, Params: params, Lambda: 2.0, Alpha: 0.98}
	for _, o := range opts {
		o(g)
	}
	g.ema = make([][]float64, len(params))
	g.out = make([]*tensor.Tensor, len(params))
	g.filtered = make(map[*tensor.Tensor]*tensor.Tensor, len(params))
	for i, p := range params {
		g.ema[i] = make([]float64, p.Numel())
		g.out[i] = tensor.New(p.Dtype(), p.Shape())
	}
	return g
}

// Step updates the gradient EMA and passes the amplified gradient ĝ = g + λ·μ to the
// base optimizer. Gradients are queried once per parameter (the EMA advances exactly
// one step regardless of how the base optimizer reads them).
func (g *Grokfast) Step(grad GradFn) error {
	filtered := g.filtered
	clear(filtered) // reused map: drop last step's entries
	for i, p := range g.Params {
		gr := grad(p)
		if gr == nil {
			filtered[p] = nil
			continue
		}
		ghat := g.out[i] // reused per-param output tensor, overwritten below
		ema := g.ema[i]
		// Contiguous fast paths (ghat is fresh → contiguous; gr usually is): flat loops,
		// arithmetic in float64 exactly as the generic path (§base-perf: no per-element
		// Unravel/AtF64/SetF64 dispatch).
		switch gf64, hf64 := flatF64(gr), flatF64(ghat); {
		case gf64 != nil && hf64 != nil:
			for k, gv := range gf64 {
				if !g.inited {
					ema[k] = gv // seed μ with the first gradient
				}
				ema[k] = g.Alpha*ema[k] + (1-g.Alpha)*gv // μ = α·μ + (1−α)·g
				hf64[k] = gv + g.Lambda*ema[k]           // ĝ = g + λ·μ
			}
		default:
			if gf32, hf32 := flatF32(gr), flatF32(ghat); gf32 != nil && hf32 != nil {
				for k := range gf32 {
					gv := float64(gf32[k])
					if !g.inited {
						ema[k] = gv
					}
					ema[k] = g.Alpha*ema[k] + (1-g.Alpha)*gv
					hf32[k] = float32(gv + g.Lambda*ema[k])
				}
			} else {
				for k := range p.Numel() {
					idx := tensor.Unravel(k, p.Shape())
					gv := gr.AtF64(idx...)
					if !g.inited {
						ema[k] = gv
					}
					ema[k] = g.Alpha*ema[k] + (1-g.Alpha)*gv
					ghat.SetF64(gv+g.Lambda*ema[k], idx...)
				}
			}
		}
		filtered[p] = ghat
	}
	g.inited = true
	return g.Base.Step(func(p *tensor.Tensor) *tensor.Tensor { return filtered[p] })
}
