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

// WithLookaheadK sets the synchronization period k (default 5).
func WithLookaheadK(k int) LookaheadOption {
	return func(l *Lookahead) {
		if k > 0 {
			l.K = k
		}
	}
}

// WithLookaheadAlpha sets the slow-weight interpolation factor α (default 0.5).
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
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			theta := p.AtF64(idx...)
			l.slow[pi][i] = (1-l.Alpha)*l.slow[pi][i] + l.Alpha*theta // φ ← (1−α)φ + α·θ
			p.SetF64(l.slow[pi][i], idx...)                           // θ ← φ (reset)
		}
	}
	return nil
}
