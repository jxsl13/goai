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

	// Reused across steps: the amplified-gradient tensors handed to the base optimizer
	// (one per param, fully overwritten each step) and the p→ĝ map. The ring slots are
	// pre-allocated and overwritten in place too, so a step allocates nothing — the base
	// reads each gradient transiently within its Step (the GradFn contract).
	out      []*tensor.Tensor
	filtered map[*tensor.Tensor]*tensor.Tensor
}

// GrokfastMAOption configures a GrokfastMA optimizer (functional-options idiom, §C12).
type GrokfastMAOption func(*GrokfastMA)

// WithGrokfastMALambda sets λ, how strongly the slow (low-frequency) gradient component is
// amplified before the base optimizer sees it.
//
// In plain terms: Grokfast accelerates "grokking" (the delayed jump to generalization) by
// boosting the slowly-changing part of the gradient. λ is the size of that boost. Boundary
// behavior — λ=0 disables the filter (the base optimizer runs unmodified); large λ over-weights
// the slow component and can destabilize training. SPECIAL VALUE: 0 = off.
//
// Default 5.0 (research-grounded: the Grokfast paper's reference λ, §R153).
func WithGrokfastMALambda(l float64) GrokfastMAOption { return func(g *GrokfastMA) { g.Lambda = l } }

// WithGrokfastMAWindow sets W, the moving-average window (in steps) used to extract the slow
// gradient component.
//
// In plain terms: how many recent steps are averaged to find the "slow trend" of the gradient.
// Boundary behavior — small W tracks a shorter, noisier trend (less smoothing); large W is
// smoother but uses more memory (a ring buffer per parameter) and reacts slower. Clamped to ≥1.
//
// Default 100 (research-grounded: the Grokfast-MA reference window, §R153).
func WithGrokfastMAWindow(w int) GrokfastMAOption { return func(g *GrokfastMA) { g.Window = w } }

// WithGrokfastMAWarmup sets whether amplification waits until the moving-average window is full
// before it kicks in.
//
// In plain terms: if true, Grokfast holds off boosting until it has W steps of history, so the
// early estimate isn't based on too few samples. Boundary behavior — a boolean; false starts
// amplifying immediately with a partial window. Default true (research-grounded, §R153).
func WithGrokfastMAWarmup(b bool) GrokfastMAOption { return func(g *GrokfastMA) { g.Warmup = b } }

// WithGrokfastMASum uses the window SUM instead of the MEAN as the slow component.
//
// In plain terms: two ways to combine the window — averaging (mean) or summing. Summing scales
// the slow component by W, effectively folding the window size into λ. Boundary behavior — a
// boolean; leave it on mean unless matching a reference that used the sum form. Default false
// (mean), research-grounded: the Grokfast-MA reference default (§R153).
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
	g.out = make([]*tensor.Tensor, len(params))
	g.filtered = make(map[*tensor.Tensor]*tensor.Tensor, len(params))
	for i, p := range params {
		n := p.Numel()
		g.ring[i] = make([][]float64, g.Window)
		for j := range g.ring[i] {
			g.ring[i][j] = make([]float64, n) // pre-allocate slots; overwritten in place each step
		}
		g.sum[i] = make([]float64, n)
		g.out[i] = tensor.New(p.Dtype(), p.Shape())
	}
	return g
}

// Step appends each parameter's current gradient to its window and passes the amplified gradient
// ĝ = g + λ·μ (μ = window mean or sum) to the base optimizer. Gradients are queried once per
// parameter (the window advances exactly one step regardless of how the base reads them).
func (g *GrokfastMA) Step(grad GradFn) error {
	filtered := g.filtered
	clear(filtered) // reused map: drop last step's entries
	//perfscan:ignore PS3044 outer param loop, one grad query per param
	for i, p := range g.Params {
		gr := grad(p)
		if gr == nil {
			filtered[p] = nil
			continue
		}
		n := p.Numel()
		// flat is the pre-allocated ring slot at pos: it currently holds the gradient from
		// W steps ago (the one to evict when full). Evict it FROM the running sum before
		// overwriting the slot in place with the current gradient — same sum−old+new as the
		// old allocate-a-fresh-slot version, bit-identical, but with no per-step allocation.
		flat := g.ring[i][g.pos[i]]
		evict := g.filled[i] == g.Window
		if !evict {
			g.filled[i]++
		}
		//perfscan:ignore PS6024 scalar ring-pos bookkeeping once per param
		g.pos[i] = (g.pos[i] + 1) % g.Window

		ghat := g.out[i] // reused per-param output tensor, overwritten below
		amplify := !g.Warmup || g.filled[i] == g.Window
		div := 1.0
		if !g.FilterSum {
			div = float64(g.filled[i]) // mean over the gradients currently in the window
		}
		sum := g.sum[i]
		// Fused single pass for the all-contiguous-F64 case (the common path): evict the
		// W-steps-ago slot, overwrite it with the current gradient, update the running sum
		// and emit ĝ = g + λ·μ in ONE sweep over n instead of four (evict, copy-in, sum-add,
		// write-out). s = (sum[k] − flat[k]) + gf[k] keeps the exact eviction association of
		// the multi-pass form, so the running sum and every output are bit-identical.
		if gf, hf := flatF64(gr), flatF64(ghat); gf != nil && hf != nil {
			if evict {
				//perfscan:ignore PS4004,PS5001 flat[k]=gf[k] fused into single-pass, not standalone memmove | div in memory-streaming fused optimizer sweep,
				for k := 0; k < n; k++ {
					s := sum[k] - flat[k] + gf[k]
					sum[k] = s
					flat[k] = gf[k]
					gv := gf[k]
					if amplify {
						gv += g.Lambda * (s / div)
					}
					hf[k] = gv
				}
			} else {
				//perfscan:ignore PS4004,PS5001 fused single-pass write, not standalone memmove | div hidden under 6-array streaming fused sweep
				for k := 0; k < n; k++ {
					s := sum[k] + gf[k]
					sum[k] = s
					flat[k] = gf[k]
					gv := gf[k]
					if amplify {
						gv += g.Lambda * (s / div)
					}
					hf[k] = gv
				}
			}
			filtered[p] = ghat
			continue
		}
		// General fallback: evict, read the gradient into the slot, add it to the running
		// sum, then write ĝ — handles F32 / non-contiguous gradients and outputs.
		if evict {
			for k := range n {
				sum[k] -= flat[k]
			}
		}
		if gf := flatF64(gr); gf != nil {
			copy(flat, gf)
		} else if gf := flatF32(gr); gf != nil {
			for k := range gf {
				flat[k] = float64(gf[k])
			}
		} else {
			for k := range n {
				flat[k] = gr.AtF64(tensor.Unravel(k, p.Shape())...)
			}
		}
		for k := range n {
			sum[k] += flat[k]
		}
		if hf := flatF32(ghat); hf != nil {
			//perfscan:ignore PS5001 declined-dtype F32/general fallback path
			for k := range n {
				gv := flat[k]
				if amplify {
					gv += g.Lambda * (sum[k] / div)
				}
				hf[k] = float32(gv)
			}
		} else {
			//perfscan:ignore PS5001 declined-dtype general fallback path
			for k := range n {
				gv := flat[k]
				if amplify {
					gv += g.Lambda * (sum[k] / div)
				}
				ghat.SetF64(gv, tensor.Unravel(k, p.Shape())...)
			}
		}
		filtered[p] = ghat
	}
	return g.Base.Step(func(p *tensor.Tensor) *tensor.Tensor { return filtered[p] })
}
