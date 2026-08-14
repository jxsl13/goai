package nn

import "github.com/jxsl13/goai/tensor"

// GradAccumulator sums the gradients of several microbatches so a large effective
// batch can be trained on limited memory (§R56). For a mean-reduction loss the
// averaged accumulation is mathematically identical to one optimizer step on the
// full concatenated batch — gradient accumulation changes only memory/compute
// scheduling, not the math (by linearity of differentiation).
//
// Usage: per microbatch compute gradients on a tape, call Add(tape.Grad); after
// the intended number of microbatches call opt.Step(acc.GradFn()) then Reset().
type GradAccumulator struct {
	params []*tensor.Tensor
	sums   map[*tensor.Tensor][]float64 // flat accumulated gradient per parameter
	steps  int
}

// NewGradAccumulator builds an accumulator over the given parameters.
func NewGradAccumulator(params []*tensor.Tensor) *GradAccumulator {
	a := &GradAccumulator{params: params, sums: make(map[*tensor.Tensor][]float64, len(params))}
	for _, p := range params {
		a.sums[p] = make([]float64, p.Numel())
	}
	return a
}

// Add accumulates one microbatch's gradients (a nil gradient contributes zero).
func (a *GradAccumulator) Add(grad GradFn) {
	for _, p := range a.params {
		g := grad(p)
		if g == nil {
			continue
		}
		s := a.sums[p]
		// Contiguous fast path: accumulate the raw gradient slice in a tight loop
		// instead of readGen's per-element closure — the identical running sum
		// (same order), so bit-identical to the general path. Add runs once per
		// microbatch, so the ~1M-closure-calls-per-param overhead is on the hot
		// training loop; non-contiguous or F16/BF16 grads keep the fallback.
		if g.IsContiguous() && g.Offset() == 0 {
			switch g.Dtype() {
			case tensor.F64:
				d := g.Storage().F64()
				for i := range s {
					s[i] += d[i]
				}
				continue
			case tensor.F32:
				d := g.Storage().F32()
				for i := range s {
					s[i] += float64(d[i])
				}
				continue
			}
		}
		idx := 0
		readGen(g, func(v float64) {
			s[idx] += v
			idx++
		})
	}
	a.steps++
}

// Steps reports how many microbatches have been accumulated since the last Reset.
func (a *GradAccumulator) Steps() int { return a.steps }

// GradFn returns a GradFn giving the AVERAGE gradient over the accumulated
// microbatches (sum ÷ Steps) — for a per-microbatch mean-reduction loss this
// equals the full-batch mean gradient. Returns nil grads before any Add.
func (a *GradAccumulator) GradFn() GradFn {
	k := float64(a.steps)
	return func(p *tensor.Tensor) *tensor.Tensor {
		s, ok := a.sums[p]
		if !ok || a.steps == 0 {
			return nil
		}
		out := tensor.New(p.Dtype(), p.Shape())
		// out is freshly allocated (contiguous, offset 0), so write its raw slice
		// directly rather than through fillGen's per-element closure — the same
		// sum÷Steps average with the identical float32 rounding, so bit-identical.
		//
		// Divide per element; do NOT hoist ik := 1/k and multiply. 1/k is exact only
		// when steps is a power of two, so x*(1/k) rounds twice and drifts up to 1 ULP
		// from x/k (steps=3 reproduces it). The loop is memory-bound, so the divide is
		// free here — see TestGradFnBitIdenticalToSlowPath, which caught exactly this.
		switch out.Dtype() {
		case tensor.F64:
			d := out.Storage().F64()
			for i := range s {
				d[i] = s[i] / k
			}
			return out
		case tensor.F32:
			d := out.Storage().F32()
			for i := range s {
				d[i] = float32(s[i] / k)
			}
			return out
		}
		idx := 0
		fillGen(out, func() float64 {
			v := s[idx] / k
			idx++
			return v
		})
		return out
	}
}

// Reset zeroes the accumulated gradients and the step count.
func (a *GradAccumulator) Reset() {
	for _, s := range a.sums {
		for i := range s {
			s[i] = 0
		}
	}
	a.steps = 0
}
