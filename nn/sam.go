package nn

import (
	"math"

	"github.com/jxsl13/goai/tensor"
)

// SAM is Sharpness-Aware Minimization (Foret, Kleiner, Mobahi & Neyshabur 2021, §R110) — a
// training METHODOLOGY (not an optimizer itself) that improves generalization by seeking parameters
// lying in flat loss neighborhoods rather than sharp minima. Each update is two gradient
// evaluations wrapped around a base optimizer:
//
//  1. compute g = ∇L(w) at the current weights;
//  2. ASCEND to the worst-case neighbor within an L2 ball of radius ρ — ε̂ = ρ·g/(‖g‖₂+ε), using
//     the GLOBAL gradient norm over all parameters concatenated — so w_adv = w + ε̂ climbs along g;
//  3. compute g_adv = ∇L(w_adv) at the perturbed point;
//  4. RESTORE w, then let the base optimizer take its step using g_adv (NOT g).
//
// The first gradient g is used only to build the perturbation and is then discarded; the actual
// descent direction is g_adv, the sharpness-aware gradient. Because two forward/backward passes are
// needed per step, SAM roughly doubles training compute. It composes with any base Optimizer
// (SGD, AdamW, …) and any backend, since it only perturbs the parameter tensors and re-invokes the
// caller's gradient closure. Vanilla p=2 SAM; the adaptive |w|-scaled variant (ASAM) is a follow-up.
type SAM struct {
	Params []*tensor.Tensor // parameters (shared with Base)
	Base   Optimizer        // inner optimizer applying the sharpness-aware descent
	Rho    float64          // neighborhood radius ρ (default 0.05)
	Eps    float64          // norm epsilon for the perturbation denominator (default 1e-12)

	pert [][]float64 // per-parameter perturbation ε̂, saved on ascent to restore on descent
}

// SAMOption configures a SAM wrapper (functional-options idiom, §C12).
type SAMOption func(*SAM)

// WithSAMRho sets the neighborhood radius ρ (default 0.05).
func WithSAMRho(rho float64) SAMOption { return func(s *SAM) { s.Rho = rho } }

// WithSAMEps sets the perturbation denominator epsilon (default 1e-12).
func WithSAMEps(eps float64) SAMOption { return func(s *SAM) { s.Eps = eps } }

// NewSAM wraps base with Sharpness-Aware Minimization over params (which must be the same tensors
// base optimizes). Defaults ρ=0.05, ε=1e-12.
func NewSAM(params []*tensor.Tensor, base Optimizer, opts ...SAMOption) *SAM {
	s := &SAM{Params: params, Base: base, Rho: 0.05, Eps: 1e-12}
	for _, o := range opts {
		o(s)
	}
	s.pert = make([][]float64, len(params))
	for i, p := range params {
		s.pert[i] = make([]float64, p.Numel())
	}
	return s
}

// Step performs one SAM update. grads must (re)compute the gradients at the CURRENT parameter
// values and return a lookup — it is called twice: once at w and once at the perturbed w_adv, so
// it must reflect the live parameter tensors (e.g. zero-grad, forward, backward, return tape.Grad).
func (s *SAM) Step(grads func() (GradFn, error)) error {
	g1, err := grads()
	if err != nil {
		return err
	}
	// snapshot the first gradient once (grads() may be expensive / non-idempotent to re-query)
	gs := make([]*tensor.Tensor, len(s.Params))
	var sumsq float64 // global L2 norm of the gradient over all parameters concatenated
	for pi, p := range s.Params {
		g := g1(p)
		gs[pi] = g
		if g == nil {
			continue
		}
		for i := range p.Numel() {
			v := g.AtF64(tensor.Unravel(i, p.Shape())...)
			sumsq += v * v
		}
	}
	scale := s.Rho / (math.Sqrt(sumsq) + s.Eps)
	// ascend to w_adv = w + ρ·g/‖g‖, saving the perturbation to undo it after the second grad
	for pi, p := range s.Params {
		g := gs[pi]
		if g == nil {
			for i := range s.pert[pi] {
				s.pert[pi][i] = 0 // no perturbation for a nil-grad param this step
			}
			continue
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			e := scale * g.AtF64(idx...)
			s.pert[pi][i] = e
			p.SetF64(p.AtF64(idx...)+e, idx...)
		}
	}
	// gradient at the perturbed point
	g2, err := grads()
	if err != nil {
		return err
	}
	// restore w (subtract the saved perturbation; 0 for skipped params → no-op), then let the base
	// optimizer take its step using the sharpness-aware gradient g2
	for pi, p := range s.Params {
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			p.SetF64(p.AtF64(idx...)-s.pert[pi][i], idx...)
		}
	}
	return s.Base.Step(g2)
}
