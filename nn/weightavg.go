package nn

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// Weight averaging (§R112) — two training-generalization methods that average the parameters in
// weight space to land in a flatter, wider optimum than the raw last-iterate weights:
//
//   - SWA (Stochastic Weight Averaging, Izmailov et al. 2018): an EQUAL-weight running average of
//     parameter snapshots captured periodically during training (e.g. at the end of each cyclical /
//     high-constant-LR cycle).
//   - EMA (exponential moving average of weights / Polyak averaging): an EXPONENTIALLY-weighted
//     average updated every step, decay≈0.999–0.9999 — the ubiquitous variant used for evaluation
//     in diffusion models and LLM training.
//
// Both keep their own float64 master copy of the averaged weights (§V10), read the live parameters on
// Update, and expose the result via Weights() for evaluation. Neither touches the training parameters.
// SWA's post-hoc BatchNorm-statistics recomputation is irrelevant here — LayerNorm/RMSNorm (used by
// transformers) keep no running statistics.

// SWA maintains an equal-weight running average of the parameters (Izmailov et al. 2018). Call Update
// once per snapshot (typically at a cycle end); Weights() returns the mean of all captured snapshots.
type SWA struct {
	Params []*tensor.Tensor // the training parameters (read on Update)
	avg    [][]float64      // equal-weight running average (float64 master, §V10)
	n      int              // snapshots averaged so far
}

// NewSWA builds an SWA accumulator over params. It holds no snapshot until the first Update.
func NewSWA(params []*tensor.Tensor) *SWA {
	s := &SWA{Params: params, avg: make([][]float64, len(params))}
	for i, p := range params {
		s.avg[i] = make([]float64, p.Numel())
	}
	return s
}

// Update captures the current parameters as a new snapshot and folds them into the equal-weight
// running average: avg ← avg + (w − avg)/(n+1), so avg is the arithmetic mean of all snapshots.
func (s *SWA) Update() error {
	denom := float64(s.n + 1)
	//perfscan:ignore PS5001 SWA divide, infrequent memory-bound optimizer bookkeeping
	for pi, p := range s.Params {
		if p.Numel() != len(s.avg[pi]) {
			return fmt.Errorf("nn: SWA param %d size changed: %d != %d", pi, p.Numel(), len(s.avg[pi]))
		}
		avg := s.avg[pi]
		// Contiguous fast path: read the raw parameter slice in a tight loop
		// instead of readGen's per-element closure (the same arithmetic, so
		// bit-identical to the general path). denom is loop-invariant (s.n is
		// bumped only after all params), matching the closure's / float64(s.n+1).
		if p.IsContiguous() && p.Offset() == 0 {
			switch p.Dtype() {
			case tensor.F64:
				d := p.Storage().F64()
				//perfscan:ignore PS5001 SWA fast-path streaming divide, memory-bound
				for i := range avg {
					avg[i] += (d[i] - avg[i]) / denom
				}
				continue
			case tensor.F32:
				d := p.Storage().F32()
				//perfscan:ignore PS5001 SWA F32 fast-path divide, memory-bound rare update
				for i := range avg {
					avg[i] += (float64(d[i]) - avg[i]) / denom
				}
				continue
			}
		}
		idx := 0
		//perfscan:ignore PS1002 readGen non-contiguous fallback; contiguous fast path exists above
		readGen(p, func(w float64) {
			avg[idx] += (w - avg[idx]) / denom
			idx++
		})
	}
	s.n++
	return nil
}

// N returns the number of snapshots averaged so far.
func (s *SWA) N() int { return s.n }

// Weights returns the averaged parameters as fresh tensors (same shapes/dtypes as Params) for
// evaluation. Before the first Update they are all zero.
func (s *SWA) Weights() []*tensor.Tensor { return materialize(s.Params, s.avg) }

// EMA maintains an exponential moving average of the parameters: avg ← decay·avg + (1−decay)·w,
// initialized to the current weights. Call Update every step (or every k steps).
type EMA struct {
	Params []*tensor.Tensor // the training parameters (read on Update)
	Decay  float64          // EMA decay (~0.999–0.9999)
	avg    [][]float64      // exponentially-weighted average (float64 master, §V10)
}

// NewEMA builds a weight EMA over params with the given decay, initialized to the current parameter
// values (so the average tracks from the starting weights).
func NewEMA(params []*tensor.Tensor, decay float64) *EMA {
	e := &EMA{Params: params, Decay: decay, avg: make([][]float64, len(params))}
	for i, p := range params {
		avg := make([]float64, p.Numel())
		idx := 0
		//perfscan:ignore PS1002 NewEMA init closure, one-time construction
		readGen(p, func(w float64) {
			avg[idx] = w
			idx++
		})
		e.avg[i] = avg
	}
	return e
}

// Update folds the current parameters into the exponential moving average:
// avg ← decay·avg + (1−decay)·w.
func (e *EMA) Update() error {
	for pi, p := range e.Params {
		if p.Numel() != len(e.avg[pi]) {
			return fmt.Errorf("nn: EMA param %d size changed: %d != %d", pi, p.Numel(), len(e.avg[pi]))
		}
		avg := e.avg[pi]
		// Contiguous fast path: raw slice loop, no per-element closure — the same
		// decay update, so bit-identical to the general readGen path.
		if p.IsContiguous() && p.Offset() == 0 {
			switch p.Dtype() {
			case tensor.F64:
				d := p.Storage().F64()
				for i := range avg {
					avg[i] = float64(e.Decay*avg[i]) + float64((1-e.Decay)*d[i])
				}
				continue
			case tensor.F32:
				d := p.Storage().F32()
				for i := range avg {
					avg[i] = float64(e.Decay*avg[i]) + float64((1-e.Decay)*float64(d[i]))
				}
				continue
			}
		}
		idx := 0
		//perfscan:ignore PS1002 EMA readGen fallback branch; fast path above
		readGen(p, func(w float64) {
			avg[idx] = float64(e.Decay*avg[idx]) + float64((1-e.Decay)*w)
			idx++
		})
	}
	return nil
}

// Weights returns the EMA-averaged parameters as fresh tensors (same shapes/dtypes as Params) for
// evaluation.
func (e *EMA) Weights() []*tensor.Tensor { return materialize(e.Params, e.avg) }

// materialize writes float64 master values into fresh tensors shaped like the reference params.
func materialize(ref []*tensor.Tensor, avg [][]float64) []*tensor.Tensor {
	out := make([]*tensor.Tensor, len(ref))
	for pi, p := range ref {
		t := tensor.New(p.Dtype(), p.Shape())
		a := avg[pi]
		idx := 0
		//perfscan:ignore PS1002 fillGen in materialize, infrequent eval-time only
		fillGen(t, func() float64 {
			v := a[idx]
			idx++
			return v
		})
		out[pi] = t
	}
	return out
}
