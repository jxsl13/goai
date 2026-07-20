package nn

import "github.com/jxsl13/goai/tensor"

// Lookahead wraps a base ("fast") optimizer with the Lookahead algorithm (Zhang,
// Lucas, Ba & Hinton 2019, "Lookahead Optimizer: k steps forward, 1 step back",
// arXiv:1907.08610, §R98). It keeps a second set of SLOW weights φ and, every k inner
// steps of the base optimizer (which updates the fast weights θ = the parameters),
// nudges the slow weights toward the fast ones and resets the fast weights to them:
//
//	every k steps:  φ ← (1−α)·φ + α·θ ;  θ ← φ
//
// This "k steps forward, 1 step back" reduces the base optimizer's variance and makes
// training more robust to the learning rate — at negligible extra cost — while the base
// optimizer is used unchanged (SGD, Adam, …). The slow weights (which the parameters
// hold after each k-th step) are the ones to use for inference; they generalize better.
// State is kept in float64 (§V10). Defaults k=5, α=0.5 (the paper's).
type Lookahead struct {
	Base   Optimizer        // the inner "fast" optimizer, updated over the same params
	Params []*tensor.Tensor // the parameters (fast weights θ); shared with Base
	K      int              // synchronization period (inner steps between slow updates)
	Alpha  float64          // slow-weight interpolation factor α

	slow [][]float64 // slow weights φ
	t    int         // inner-step counter
}

// LookaheadOption configures a Lookahead optimizer (functional-options idiom, §C12).
type LookaheadOption func(*Lookahead)

// WithLookaheadK sets the synchronization period k — how many fast (base-optimizer) steps run
// between each slow-weight update.
//
// In plain terms: Lookahead keeps a second, slow set of weights that it snaps toward the fast
// weights every k steps; this smooths out the fast optimizer's noise. k is how often that snap
// happens. Boundary behavior — k=1 syncs every step (little smoothing, near the base
// optimizer); large k lets the fast weights wander far before each correction, which can
// destabilize. SPECIAL VALUE: k ≤ 0 is ignored (keeps the current value).
//
// Default 5 (research-grounded: the Lookahead paper's reference k, Zhang et al. 2019, §R98).
func WithLookaheadK(k int) LookaheadOption {
	return func(l *Lookahead) {
		if k > 0 {
			l.K = k
		}
	}
}

// WithLookaheadAlpha sets α, how far the slow weights move toward the fast weights at each
// synchronization.
//
// In plain terms: at each sync, the slow weights interpolate a fraction α of the way to the
// fast weights. Boundary behavior — α→0 barely moves the slow weights (very conservative,
// slow to follow progress); α=1 snaps them fully onto the fast weights (no smoothing, Lookahead
// has no effect). 0.5 sits in the middle.
//
// Default 0.5 (research-grounded: the Lookahead paper's reference α, Zhang et al. 2019, §R98).
func WithLookaheadAlpha(a float64) LookaheadOption {
	return func(l *Lookahead) { l.Alpha = a }
}

// NewLookahead wraps base (which optimizes params) with Lookahead. params must be the
// same slice the base optimizer updates. The slow weights start at the current values.
func NewLookahead(base Optimizer, params []*tensor.Tensor, opts ...LookaheadOption) *Lookahead {
	l := &Lookahead{Base: base, Params: params, K: 5, Alpha: 0.5}
	for _, o := range opts {
		o(l)
	}
	l.slow = make([][]float64, len(params))
	for i, p := range params {
		l.slow[i] = make([]float64, p.Numel())
		for k := range p.Numel() {
			l.slow[i][k] = p.AtF64(tensor.Unravel(k, p.Shape())...)
		}
	}
	return l
}

// Step runs one inner step of the base optimizer, then every k steps interpolates the
// slow weights toward the fast ones (φ ← (1−α)φ + α·θ) and resets the fast weights to
// the slow ones (θ ← φ).
func (l *Lookahead) Step(grad GradFn) error {
	if err := l.Base.Step(grad); err != nil {
		return err
	}
	l.t++
	if l.t%l.K != 0 {
		return nil
	}
	for pi, p := range l.Params {
		slow := l.slow[pi]
		// Contiguous fast paths (§base-perf: no per-element Unravel/AtF64/SetF64).
		if pf := flatF64(p); pf != nil {
			for i := range pf {
				slow[i] = (1-l.Alpha)*slow[i] + l.Alpha*pf[i] // φ ← (1−α)φ + α·θ
				pf[i] = slow[i]                               // θ ← φ (reset)
			}
		} else if pf := flatF32(p); pf != nil {
			for i := range pf {
				slow[i] = (1-l.Alpha)*slow[i] + l.Alpha*float64(pf[i])
				pf[i] = float32(slow[i])
			}
		} else {
			for i := range p.Numel() {
				idx := tensor.Unravel(i, p.Shape())
				theta := p.AtF64(idx...)
				slow[i] = (1-l.Alpha)*slow[i] + l.Alpha*theta
				p.SetF64(slow[i], idx...)
			}
		}
	}
	return nil
}
