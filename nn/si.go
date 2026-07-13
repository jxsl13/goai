package nn

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// SIDefaultDamping is the default ξ damping constant in the SI importance denominator
// (Zenke, Poole & Ganguli 2017): it bounds Ω when a parameter's total displacement over a
// task is tiny, so a near-stationary weight cannot acquire an unboundedly large importance.
// The paper tunes ξ per benchmark (0.1 on permuted-MNIST, 1e-3 on split-MNIST); 0.1 is the
// permuted-MNIST value — pass an explicit xi to Consolidate to override it.
const SIDefaultDamping = 0.1

// SI accumulates the Synaptic Intelligence per-parameter importance (Zenke, Poole & Ganguli
// 2017, "Continual Learning Through Synaptic Intelligence", ICML, arXiv:1703.04200). Like
// EWC it protects parameters important to previous tasks with a quadratic anchor, but it
// estimates importance a fundamentally different way: instead of EWC's Fisher measured once at
// a task optimum (nn.EWCFisher), SI integrates each parameter's contribution to the loss drop
// ONLINE ALONG the whole training trajectory. For task μ it maintains the running path integral
//
//	ω_k = Σ_t −g_k(t)·Δθ_k(t)          // g_k = task-loss gradient, Δθ_k = the step's update
//
// (positive when moving parameter k reduced the loss), and at the task's end CONSOLIDATES it
// into the importance
//
//	Ω_k += ω_k / ((Δθ_k^task)² + ξ)     // Δθ_k^task = total displacement over the task
//
// with ξ a small damping constant (SIDefaultDamping). The resulting Ω and the end-of-task
// reference weights feed the same quadratic penalty as EWC: pass SI.Importance() and
// SI.RefParams() as the importance and refParams of nn.EWCPenalty (with lambda = 2c for the
// paper's strength c). Ω accumulates across successive tasks; the anchor advances to the latest
// weights each consolidation. Pure float64 master state (§V10); the estimator is not itself
// differentiable — the differentiability lives in EWCPenalty.
type SI struct {
	Params    []*tensor.Tensor // the live training parameters (read on Accumulate/Consolidate)
	prev      [][]float64      // θ at the previous Accumulate — for the per-step Δθ
	taskStart [][]float64      // θ at the start of the current task — for the total displacement
	omega     [][]float64      // running path integral ω for the current task
	bigOmega  [][]float64      // consolidated importance Ω (accumulated across tasks)
	ref       [][]float64      // anchor θ* (parameters at the last consolidation)
}

// NewSI builds an importance accumulator over params, snapshotting them as both the current
// task's start and the initial anchor, with zero importance.
func NewSI(params []*tensor.Tensor) *SI {
	s := &SI{
		Params:    params,
		prev:      snapshot(params),
		taskStart: snapshot(params),
		omega:     zeroLike(params),
		bigOmega:  zeroLike(params),
		ref:       snapshot(params),
	}
	return s
}

// Accumulate folds one optimizer step into the running path integral: it reads the CURRENT
// parameters (θ after the step) against the previous snapshot (θ before it) and adds
// −grad·Δθ per parameter. Call it once per step, AFTER the optimizer updates the parameters,
// passing the TASK-LOSS gradients used for that step (exclude the SI/EWC penalty's own
// gradient). grads must match the parameter layout.
func (s *SI) Accumulate(grads []*tensor.Tensor) error {
	if len(grads) != len(s.Params) {
		return fmt.Errorf("nn: SI.Accumulate needs %d grads, got %d", len(s.Params), len(grads))
	}
	for pi, p := range s.Params {
		if !grads[pi].Shape().Equal(p.Shape()) {
			return fmt.Errorf("nn: SI.Accumulate grad %d shape %v != param %v", pi, grads[pi].Shape(), p.Shape())
		}
		for i := range p.Numel() {
			c := tensor.Unravel(i, p.Shape())
			cur := p.AtF64(c...)
			dTheta := cur - s.prev[pi][i]
			s.omega[pi][i] += -grads[pi].AtF64(c...) * dTheta
			s.prev[pi][i] = cur
		}
	}
	return nil
}

// Consolidate ends the current task: it folds the running path integral into the importance
// Ω_k += ω_k/((Δθ_k^task)²+ξ), advances the anchor to the current parameters, and resets the
// running state for the next task. xi ≤ 0 uses SIDefaultDamping.
func (s *SI) Consolidate(xi float64) {
	if xi <= 0 {
		xi = SIDefaultDamping
	}
	for pi, p := range s.Params {
		for i := range p.Numel() {
			cur := p.AtF64(tensor.Unravel(i, p.Shape())...)
			dTask := cur - s.taskStart[pi][i]
			s.bigOmega[pi][i] += s.omega[pi][i] / (dTask*dTask + xi)
			s.omega[pi][i] = 0
			s.taskStart[pi][i] = cur
			s.prev[pi][i] = cur
			s.ref[pi][i] = cur
		}
	}
}

// Importance returns the consolidated per-parameter importance Ω as fresh tensors matching the
// parameter layout — pass it as the fisher/importance argument of nn.EWCPenalty.
func (s *SI) Importance() []*tensor.Tensor { return materialize(s.Params, s.bigOmega) }

// RefParams returns the anchor parameters θ* (the weights at the last Consolidate) as fresh
// tensors — pass them as the refParams argument of nn.EWCPenalty.
func (s *SI) RefParams() []*tensor.Tensor { return materialize(s.Params, s.ref) }

// snapshot copies params into a float64 master ([][]float64 by flat index).
func snapshot(params []*tensor.Tensor) [][]float64 {
	out := make([][]float64, len(params))
	for pi, p := range params {
		out[pi] = make([]float64, p.Numel())
		for i := range p.Numel() {
			out[pi][i] = p.AtF64(tensor.Unravel(i, p.Shape())...)
		}
	}
	return out
}

// zeroLike allocates a float64 master of zeros matching params.
func zeroLike(params []*tensor.Tensor) [][]float64 {
	out := make([][]float64, len(params))
	for pi, p := range params {
		out[pi] = make([]float64, p.Numel())
	}
	return out
}
