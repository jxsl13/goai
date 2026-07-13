package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// Optimizers (§T17). Step reads gradients through a lookup function — pass
// tape.Grad directly (the method value fits) — and updates parameters in place.
// Optimizer state (momentum/moments) is kept in float64 regardless of parameter
// dtype: the master-state analog of §V10, so f32 training does not lose update
// precision to state rounding.

// GradFn maps a parameter tensor to its gradient (nil = no gradient this step).
type GradFn func(*tensor.Tensor) *tensor.Tensor

// Optimizer updates its parameters from gradients.
type Optimizer interface {
	Step(grad GradFn) error
}

// --- SGD ---

// SGD is stochastic gradient descent, optionally with classical momentum in the
// torch formulation: v ← μ·v + g; p ← p − lr·v.
type SGD struct {
	Params   []*tensor.Tensor // parameters this optimizer updates
	LR       float64          // learning rate (step size)
	Momentum float64          // 0 = plain SGD
	vel      [][]float64
}

// NewSGD builds an SGD optimizer over params.
func NewSGD(params []*tensor.Tensor, lr, momentum float64) *SGD {
	s := &SGD{Params: params, LR: lr, Momentum: momentum}
	if momentum != 0 {
		s.vel = make([][]float64, len(params))
		for i, p := range params {
			s.vel[i] = make([]float64, p.Numel())
		}
	}
	return s
}

// Step applies one update. Parameters with nil gradient are skipped.
func (s *SGD) Step(grad GradFn) error {
	for pi, p := range s.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: SGD grad shape %v != param %v", g.Shape(), p.Shape())
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			gv := g.AtF64(idx...)
			if s.Momentum != 0 {
				s.vel[pi][i] = s.Momentum*s.vel[pi][i] + gv
				gv = s.vel[pi][i]
			}
			p.SetF64(p.AtF64(idx...)-s.LR*gv, idx...)
		}
	}
	return nil
}

// --- Adam ---

// Adam implements Kingma & Ba (2015), "Adam: A Method for Stochastic
// Optimization", with bias correction and eps added outside the square root
// (the paper's and torch's formulation):
//
//	m ← β₁m + (1−β₁)g;  v ← β₂v + (1−β₂)g²
//	m̂ = m/(1−β₁ᵗ);  v̂ = v/(1−β₂ᵗ);  p ← p − lr·m̂/(√v̂ + ε)
type Adam struct {
	Params []*tensor.Tensor // parameters this optimizer updates
	LR     float64          // learning rate (step size)
	Beta1  float64          // 1st-moment EMA decay (~0.9)
	Beta2  float64          // 2nd-moment EMA decay (~0.999)
	Eps    float64          // denominator epsilon for numerical stability
	// WeightDecay > 0 selects AdamW's decoupled decay (Loshchilov & Hutter 2019):
	// p ← p·(1−lr·wd) − lr·m̂/(√v̂+ε). 0 = plain Adam.
	WeightDecay float64

	m, v [][]float64
	t    int
}

// NewAdam builds an Adam optimizer with the canonical defaults β₁=0.9,
// β₂=0.999, ε=1e-8, no weight decay.
func NewAdam(params []*tensor.Tensor, lr float64) *Adam {
	a := &Adam{Params: params, LR: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8}
	a.m = make([][]float64, len(params))
	a.v = make([][]float64, len(params))
	for i, p := range params {
		a.m[i] = make([]float64, p.Numel())
		a.v[i] = make([]float64, p.Numel())
	}
	return a
}

// NewAdamW builds AdamW (Loshchilov & Hutter 2019) — Adam with decoupled weight
// decay wd, the standard LLM optimizer.
func NewAdamW(params []*tensor.Tensor, lr, wd float64) *Adam {
	a := NewAdam(params, lr)
	a.WeightDecay = wd
	return a
}

// Step applies one Adam/AdamW update. The timestep advances once per Step call.
func (a *Adam) Step(grad GradFn) error {
	a.t++
	c1 := 1 - math.Pow(a.Beta1, float64(a.t))
	c2 := 1 - math.Pow(a.Beta2, float64(a.t))
	decay := 1 - a.LR*a.WeightDecay // 1 when wd==0 → plain Adam
	for pi, p := range a.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: Adam grad shape %v != param %v", g.Shape(), p.Shape())
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			gv := g.AtF64(idx...)
			a.m[pi][i] = a.Beta1*a.m[pi][i] + (1-a.Beta1)*gv
			a.v[pi][i] = a.Beta2*a.v[pi][i] + (1-a.Beta2)*gv*gv
			mh := a.m[pi][i] / c1
			vh := a.v[pi][i] / c2
			p.SetF64(p.AtF64(idx...)*decay-a.LR*mh/(math.Sqrt(vh)+a.Eps), idx...)
		}
	}
	return nil
}

// ClipGradNorm rescales all gradients so their combined global L2 norm does not
// exceed maxNorm (the standard transformer-training stabilizer). It returns a
// GradFn wrapping grad — evaluate the global norm once, then scale each grad by
// min(1, maxNorm/norm). The reported pre-clip norm is returned too.
func ClipGradNorm(params []*tensor.Tensor, grad GradFn, maxNorm float64) (GradFn, float64) {
	var sumsq float64
	for _, p := range params {
		g := grad(p)
		if g == nil {
			continue
		}
		for i := range g.Numel() {
			v := g.AtF64(tensor.Unravel(i, g.Shape())...)
			sumsq += v * v
		}
	}
	norm := math.Sqrt(sumsq)
	scale := 1.0
	if norm > maxNorm && norm > 0 {
		scale = maxNorm / norm
	}
	if scale == 1 {
		return grad, norm
	}
	return func(p *tensor.Tensor) *tensor.Tensor {
		g := grad(p)
		if g == nil {
			return nil
		}
		out := tensor.New(g.Dtype(), g.Shape())
		for i := range g.Numel() {
			idx := tensor.Unravel(i, g.Shape())
			out.SetF64(g.AtF64(idx...)*scale, idx...)
		}
		return out
	}, norm
}

// WarmupCosine is the standard LLM learning-rate schedule: linear warmup over
// the first `warmup` steps (from 0 to baseLR), then cosine decay to minLR over
// the remaining steps to `total`. step is 0-indexed.
func WarmupCosine(step, warmup, total int, baseLR, minLR float64) float64 {
	if step < warmup {
		if warmup == 0 {
			return baseLR
		}
		return baseLR * float64(step+1) / float64(warmup)
	}
	if step >= total {
		return minLR
	}
	progress := float64(step-warmup) / float64(total-warmup)
	return minLR + 0.5*(baseLR-minLR)*(1+math.Cos(math.Pi*progress))
}

// Lion (EvoLved Sign Momentum, Chen et al. 2023) is a memory-efficient optimizer:
// it keeps only a momentum buffer (no second moment like Adam — half the state)
// and steps by the SIGN of an interpolated momentum, so every coordinate moves by
// the same magnitude lr. Update (Alg. 2, §R65): c = β1·m + (1−β1)·g; then
// θ −= lr·(sign(c) + λ·θ); then m = β2·m + (1−β2)·g (momentum updated AFTER the
// step, with β2 ≠ β1; the weight decay λ is decoupled, AdamW-style).
type Lion struct {
	Params      []*tensor.Tensor // parameters this optimizer updates
	LR          float64          // learning rate (step size)
	Beta1       float64          // update-interpolation EMA decay (~0.9)
	Beta2       float64          // momentum EMA decay (~0.99)
	WeightDecay float64          // decoupled weight-decay coefficient (λ)

	m [][]float64
}

// LionOption configures a Lion optimizer (functional-options idiom, §C12).
type LionOption func(*Lion)

// WithLionBetas sets the update/momentum interpolation coefficients (β1, β2;
// defaults 0.9, 0.99).
func WithLionBetas(beta1, beta2 float64) LionOption {
	return func(l *Lion) { l.Beta1, l.Beta2 = beta1, beta2 }
}

// WithLionWeightDecay sets the decoupled weight decay λ (default 0).
func WithLionWeightDecay(wd float64) LionOption {
	return func(l *Lion) { l.WeightDecay = wd }
}

// NewLion builds a Lion optimizer over params with learning rate lr and the
// canonical defaults β1=0.9, β2=0.99, no weight decay.
func NewLion(params []*tensor.Tensor, lr float64, opts ...LionOption) *Lion {
	l := &Lion{Params: params, LR: lr, Beta1: 0.9, Beta2: 0.99}
	for _, o := range opts {
		o(l)
	}
	l.m = make([][]float64, len(params))
	for i, p := range params {
		l.m[i] = make([]float64, p.Numel())
	}
	return l
}

// Step applies one Lion update.
func (l *Lion) Step(grad GradFn) error {
	for pi, p := range l.Params {
		g := grad(p)
		if g == nil {
			continue
		}
		if !g.Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: Lion grad shape %v != param %v", g.Shape(), p.Shape())
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			gv := g.AtF64(idx...)
			c := l.Beta1*l.m[pi][i] + (1-l.Beta1)*gv // interpolate (β1)
			pv := p.AtF64(idx...)
			p.SetF64(pv-l.LR*(signf(c)+l.WeightDecay*pv), idx...) // sign step + decoupled wd
			l.m[pi][i] = l.Beta2*l.m[pi][i] + (1-l.Beta2)*gv      // momentum after (β2)
		}
	}
	return nil
}

func signf(x float64) float64 {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	default:
		return 0
	}
}
