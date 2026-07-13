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
}

// GrokfastOption configures a Grokfast optimizer (functional-options idiom, §C12).
type GrokfastOption func(*Grokfast)

// WithGrokfastLambda sets the slow-gradient amplification λ (default 2.0; 0 = off).
func WithGrokfastLambda(l float64) GrokfastOption { return func(g *Grokfast) { g.Lambda = l } }

// WithGrokfastAlpha sets the EMA momentum α (default 0.98).
func WithGrokfastAlpha(a float64) GrokfastOption { return func(g *Grokfast) { g.Alpha = a } }

// NewGrokfast wraps base (which optimizes params) with the Grokfast-EMA filter. params
// must be the same slice the base optimizer updates.
func NewGrokfast(base Optimizer, params []*tensor.Tensor, opts ...GrokfastOption) *Grokfast {
	g := &Grokfast{Base: base, Params: params, Lambda: 2.0, Alpha: 0.98}
	for _, o := range opts {
		o(g)
	}
	g.ema = make([][]float64, len(params))
	for i, p := range params {
		g.ema[i] = make([]float64, p.Numel())
	}
	return g
}

// Step updates the gradient EMA and passes the amplified gradient ĝ = g + λ·μ to the
// base optimizer. Gradients are queried once per parameter (the EMA advances exactly
// one step regardless of how the base optimizer reads them).
func (g *Grokfast) Step(grad GradFn) error {
	filtered := make(map[*tensor.Tensor]*tensor.Tensor, len(g.Params))
	for i, p := range g.Params {
		gr := grad(p)
		if gr == nil {
			filtered[p] = nil
			continue
		}
		ghat := tensor.New(p.Dtype(), p.Shape())
		for k := range p.Numel() {
			idx := tensor.Unravel(k, p.Shape())
			gv := gr.AtF64(idx...)
			if !g.inited {
				g.ema[i][k] = gv // seed μ with the first gradient
			}
			g.ema[i][k] = g.Alpha*g.ema[i][k] + (1-g.Alpha)*gv // μ = α·μ + (1−α)·g
			ghat.SetF64(gv+g.Lambda*g.ema[i][k], idx...)       // ĝ = g + λ·μ
		}
		filtered[p] = ghat
	}
	g.inited = true
	return g.Base.Step(func(p *tensor.Tensor) *tensor.Tensor { return filtered[p] })
}
