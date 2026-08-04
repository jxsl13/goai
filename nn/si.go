package nn

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

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
		siAccumulate(p, grads[pi], s.omega[pi], s.prev[pi])
	}
	return nil
}

// siAccumulate folds one step's contribution into the running path integral for a
// single parameter: omega[i] += −grad[i]·(p[i] − prev[i]) and prev[i] ← p[i]. It
// runs over EVERY parameter each optimizer step, so for contiguous, offset-0
// F64/F32 p and grad it reads the typed backing slices directly instead of the
// per-element AtF64(Unravel) dispatch on two tensors. Bit-identical to the general
// path (see the slowSIAccumulate oracle): the arithmetic and the read-prev-then-
// write-prev order are unchanged; F16/BF16 (or mixed dtypes) fall through.
func siAccumulate(p, grad *tensor.Tensor, omega, prev []float64) {
	n := p.Numel()
	if p.IsContiguous() && p.Offset() == 0 && grad.IsContiguous() && grad.Offset() == 0 {
		if p.Dtype() == tensor.F64 && grad.Dtype() == tensor.F64 {
			pd, gd := p.Storage().F64(), grad.Storage().F64()
			for i := range n {
				cur := pd[i]
				//perfscan:ignore PS3025 SI path-integral update, bandwidth-bound streaming fastpath
				omega[i] += -gd[i] * (cur - prev[i])
				prev[i] = cur
			}
			return
		}
		if p.Dtype() == tensor.F32 && grad.Dtype() == tensor.F32 {
			pd, gd := p.Storage().F32(), grad.Storage().F32()
			for i := range n {
				cur := float64(pd[i])
				//perfscan:ignore PS3025 F32 fastpath update, memory-bound streaming
				omega[i] += -float64(gd[i]) * (cur - prev[i])
				prev[i] = cur
			}
			return
		}
	}
	for i := range n {
		c := tensor.Unravel(i, p.Shape())
		cur := p.AtF64(c...)
		//perfscan:ignore PS3025 generic declined-dtype fallback, correct to keep
		omega[i] += -grad.AtF64(c...) * (cur - prev[i])
		prev[i] = cur
	}
}

// Consolidate ends the current task: it folds the running path integral into the importance
// Ω_k += ω_k/((Δθ_k^task)²+ξ), advances the anchor to the current parameters, and resets the
// running state for the next task. xi ≤ 0 uses SIDefaultDamping.
func (s *SI) Consolidate(xi float64) {
	if xi <= 0 {
		xi = SIDefaultDamping
	}
	// Each parameter's consolidation touches only its OWN master slices
	// (taskStart/bigOmega/omega/prev/ref at index pi are disjoint per parameter),
	// so the parameters are independent and consolidate concurrently — the same
	// atomic work-steal fan-out EWCFisher/MASImportance use. Per-parameter
	// arithmetic and write order are unchanged, so the result is bit-identical to
	// the serial walk.
	do := func(pi int) {
		p := s.Params[pi]
		// The importance math already runs on the flat []float64 masters; the only
		// per-element tensor dispatch is reading cur from p. For contiguous F64/F32 p
		// read the typed backing slice directly (cur is float64 either way, so the
		// arithmetic and write order are unchanged — bit-identical to the generic
		// walk). Same fast-path idiom as siAccumulate above.
		ts, bo, om, pr, rf := s.taskStart[pi], s.bigOmega[pi], s.omega[pi], s.prev[pi], s.ref[pi]
		if pf := flatF64(p); pf != nil {
			for i := range pf {
				cur := pf[i]
				dTask := cur - ts[i]
				bo[i] += om[i] / (dTask*dTask + xi)
				om[i] = 0
				ts[i], pr[i], rf[i] = cur, cur, cur
			}
			return
		}
		if pf := flatF32(p); pf != nil {
			for i := range pf {
				cur := float64(pf[i])
				dTask := cur - ts[i]
				bo[i] += om[i] / (dTask*dTask + xi)
				om[i] = 0
				ts[i], pr[i], rf[i] = cur, cur, cur
			}
			return
		}
		for i := range p.Numel() {
			cur := p.AtF64(tensor.Unravel(i, p.Shape())...)
			dTask := cur - ts[i]
			bo[i] += om[i] / (dTask*dTask + xi)
			om[i] = 0
			ts[i], pr[i], rf[i] = cur, cur, cur
		}
	}
	nP := len(s.Params)
	workers := min(runtime.GOMAXPROCS(0), nP)
	if workers > 1 {
		var next atomic.Int64
		var wg sync.WaitGroup
		for range workers {
			wg.Go(func() {
				for {
					i := int(next.Add(1)) - 1
					if i >= nP {
						return
					}
					do(i)
				}
			})
		}
		wg.Wait()
	} else {
		for i := range nP {
			do(i)
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
		n := p.Numel()
		//perfscan:ignore PS2008 snapshot per-param alloc, task-boundary resource-only
		o := make([]float64, n)
		out[pi] = o
		// Contiguous F64/F32 read the typed backing slice directly (F64 is a plain
		// copy) instead of the per-element AtF64(Unravel) dispatch; bit-identical.
		if pf := flatF64(p); pf != nil {
			copy(o, pf)
			continue
		}
		if pf := flatF32(p); pf != nil {
			for i := range o {
				o[i] = float64(pf[i])
			}
			continue
		}
		for i := range n {
			o[i] = p.AtF64(tensor.Unravel(i, p.Shape())...)
		}
	}
	return out
}

// zeroLike allocates a float64 master of zeros matching params.
func zeroLike(params []*tensor.Tensor) [][]float64 {
	out := make([][]float64, len(params))
	for pi, p := range params {
		//perfscan:ignore PS2008,PS3064 zeroLike init alloc, one-time resource-only | zeroLike make-in-loop, one-time init, no wallclock
		out[pi] = make([]float64, p.Numel())
	}
	return out
}
