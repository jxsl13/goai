package nn

import "github.com/jxsl13/goai/tensor"

// GrokfastMA wraps a base optimizer with the Grokfast-MA gradient filter (Lee, Ahn, Kim & Kim
// 2024, "Grokfast: Accelerated Grokking by Amplifying Slow Gradients", arXiv:2405.20233; the
// windowed moving-average variant, ironjr/grokfast gradfilter_ma). Like Grokfast-EMA it isolates
// and amplifies the slow-varying (low-frequency) gradient component that drives grokking, but the
// low-pass filter is a plain MOVING AVERAGE over the last W gradients instead of an EMA:
//
//	μ_t = (1/|Wₜ|)·Σ_{g∈Wₜ} g        // mean of the last W raw gradients (Wₜ = window at step t)
//	ĝ_t = g_t + λ·μ_t                 // amplify the slow component, pass ĝ to the base
//
// It is a drop-in gradient transform applied BEFORE the base optimizer's step. Following the
// reference, when Warmup is true the amplification is applied ONLY once the window is full (the
// first W−1 steps pass the raw gradient through, ĝ = g); with Warmup false it averages over
// however many gradients have been seen so far. FilterSum switches μ from the mean to the plain
// SUM over the window. λ=0 recovers the base optimizer exactly. State is float64 (§V10). Defaults
// λ=5.0, W=100, Warmup=true, mean (the paper's Grokfast-MA).
type GrokfastMA struct {
	Base      Optimizer        // the wrapped base optimizer, stepped over the same params
	Params    []*tensor.Tensor // the parameters (shared with Base)
	Lambda    float64          // amplification λ of the slow (moving-average) gradient component
	Window    int              // moving-average window size W (number of recent gradients averaged)
	Warmup    bool             // if true, amplify only once the window is full (first W−1 steps = raw)
	FilterSum bool             // false = window mean (default); true = window sum

	ring   [][][]float64 // per-parameter ring buffer of the last W flattened gradients
	sum    [][]float64   // per-parameter running sum of the gradients currently in the ring
	filled []int         // per-parameter count of occupied ring slots (min(steps, W))
	pos    []int         // per-parameter ring write position
}

// GrokfastMAOption configures a GrokfastMA optimizer (functional-options idiom, §C12).
type GrokfastMAOption func(*GrokfastMA)

// WithGrokfastMALambda sets the slow-gradient amplification λ (default 5.0; 0 = off).
func WithGrokfastMALambda(l float64) GrokfastMAOption { return func(g *GrokfastMA) { g.Lambda = l } }

// WithGrokfastMAWindow sets the moving-average window size W (default 100; clamped to ≥ 1).
func WithGrokfastMAWindow(w int) GrokfastMAOption { return func(g *GrokfastMA) { g.Window = w } }

// WithGrokfastMAWarmup sets whether amplification waits for the window to fill (default true).
func WithGrokfastMAWarmup(b bool) GrokfastMAOption { return func(g *GrokfastMA) { g.Warmup = b } }

// WithGrokfastMASum uses the window SUM instead of the mean as the slow component (default mean).
func WithGrokfastMASum(b bool) GrokfastMAOption { return func(g *GrokfastMA) { g.FilterSum = b } }

// NewGrokfastMA wraps base (which optimizes params) with the Grokfast-MA filter. params must be
// the same slice the base optimizer updates.
func NewGrokfastMA(base Optimizer, params []*tensor.Tensor, opts ...GrokfastMAOption) *GrokfastMA {
	g := &GrokfastMA{Base: base, Params: params, Lambda: 5.0, Window: 100, Warmup: true}
	for _, o := range opts {
		o(g)
	}
	if g.Window < 1 {
		g.Window = 1
	}
	g.ring = make([][][]float64, len(params))
	g.sum = make([][]float64, len(params))
	g.filled = make([]int, len(params))
	g.pos = make([]int, len(params))
	for i, p := range params {
		g.ring[i] = make([][]float64, g.Window)
		g.sum[i] = make([]float64, p.Numel())
	}
	return g
}

// Step appends each parameter's current gradient to its window and passes the amplified gradient
// ĝ = g + λ·μ (μ = window mean or sum) to the base optimizer. Gradients are queried once per
// parameter (the window advances exactly one step regardless of how the base reads them).
func (g *GrokfastMA) Step(grad GradFn) error {
	filtered := make(map[*tensor.Tensor]*tensor.Tensor, len(g.Params))
	for i, p := range g.Params {
		gr := grad(p)
		if gr == nil {
			filtered[p] = nil
			continue
		}
		n := p.Numel()
		flat := make([]float64, n)
		for k := range n {
			flat[k] = gr.AtF64(tensor.Unravel(k, p.Shape())...)
		}
		// Push flat into the ring, maintaining the running sum (evict the oldest when full).
		if g.filled[i] == g.Window {
			old := g.ring[i][g.pos[i]]
			for k := range n {
				g.sum[i][k] -= old[k]
			}
		} else {
			g.filled[i]++
		}
		g.ring[i][g.pos[i]] = flat
		for k := range n {
			g.sum[i][k] += flat[k]
		}
		g.pos[i] = (g.pos[i] + 1) % g.Window

		ghat := tensor.New(p.Dtype(), p.Shape())
		amplify := !g.Warmup || g.filled[i] == g.Window
		div := 1.0
		if !g.FilterSum {
			div = float64(g.filled[i]) // mean over the gradients currently in the window
		}
		for k := range n {
			idx := tensor.Unravel(k, p.Shape())
			gv := gr.AtF64(idx...)
			if amplify {
				gv += g.Lambda * (g.sum[i][k] / div) // ĝ = g + λ·μ
			}
			ghat.SetF64(gv, idx...)
		}
		filtered[p] = ghat
	}
	return g.Base.Step(func(p *tensor.Tensor) *tensor.Tensor { return filtered[p] })
}
