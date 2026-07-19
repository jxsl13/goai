package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// samStepSlow is a VERBATIM copy of SAM.Step's pre-devirtualization per-element loops
// (AtF64(Unravel)/SetF64), kept as the bit-identity oracle (§V13/§V22): the typed
// contiguous-f64/f32 fast paths added to SAM.Step must reproduce it exactly, seeded by
// the same weights and gradients. pert is the caller-owned perturbation buffer (one
// []float64 per param, len == Numel), reused across steps exactly as the SAM struct field
// was; a fresh-vs-reused buffer cannot change values (it is fully written before read
// within a step).
func samStepSlow(params []*tensor.Tensor, base nn.Optimizer, rho, eps float64,
	pert [][]float64, grads func() (nn.GradFn, error)) error {
	g1, err := grads()
	if err != nil {
		return err
	}
	gs := make([]*tensor.Tensor, len(params))
	var sumsq float64
	for pi, p := range params {
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
	scale := rho / (math.Sqrt(sumsq) + eps)
	for pi, p := range params {
		g := gs[pi]
		if g == nil {
			for i := range pert[pi] {
				pert[pi][i] = 0
			}
			continue
		}
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			e := scale * g.AtF64(idx...)
			pert[pi][i] = e
			p.SetF64(p.AtF64(idx...)+e, idx...)
		}
	}
	g2, err := grads()
	if err != nil {
		return err
	}
	for pi, p := range params {
		for i := range p.Numel() {
			idx := tensor.Unravel(i, p.Shape())
			p.SetF64(p.AtF64(idx...)-pert[pi][i], idx...)
		}
	}
	return base.Step(g2)
}

// samNoopBase is a do-nothing base optimizer, used to isolate SAM's own three loops
// (norm / ascend / restore) in the A/B benchmarks — no base-optimizer work is measured.
type samNoopBase struct{}

func (samNoopBase) Step(nn.GradFn) error { return nil }

// samQuadGrad returns a weight-dependent SAM gradient closure for f(w)=½‖w−c‖² (grad = w−c),
// snapshotting at the CURRENT weights like the tape.Grad contract. Each gradient tensor is
// built with the SAME dtype as its parameter and the requested layout, so contiguous=true
// actually exercises SAM's typed fast paths and contiguous=false forces the generic path.
func samQuadGrad(t *testing.T, params []*tensor.Tensor, c float64, contiguous bool) func() (nn.GradFn, error) {
	return func() (nn.GradFn, error) {
		snap := make(map[*tensor.Tensor]*tensor.Tensor, len(params))
		for _, p := range params {
			dt, shape := p.Dtype(), p.Shape()
			var g *tensor.Tensor
			if contiguous {
				g = tensor.New(dt, shape)
			} else {
				g = mustTransposedView(t, dt, shape)
			}
			for i := range shape.Numel() {
				idx := tensor.Unravel(i, shape)
				g.SetF64(p.AtF64(idx...)-c, idx...)
			}
			snap[p] = g
		}
		return func(p *tensor.Tensor) *tensor.Tensor { return snap[p] }, nil
	}
}

// TestSAMFastPathBitIdentical guards SAM.Step's typed flat-loop fast paths (§V13/§V22): the
// devirtualized Step must produce a trajectory BIT-IDENTICAL to the verbatim per-element
// oracle [samStepSlow]. It runs both over parameters with identical logical values across a
// real weight-dependent gradient, for F64 and F32, on contiguous tensors (fast path fires)
// and on transposed views (non-contiguous, so flatF64/flatF32 return nil and the generic
// path runs), and requires exact math.Float64bits equality after every step. The base
// optimizer (SGD with momentum) is deterministic and identical between the two runs, so any
// divergence is attributable to SAM's three loops.
func TestSAMFastPathBitIdentical(t *testing.T) {
	const lr, mom, rho, eps, c = 1e-2, 0.9, 0.05, 1e-12, 0.25
	const steps = 6
	shapes := []tensor.Shape{{3, 5}, {2, 3, 4}}
	pval := func(i int) float64 { return math.Sin(float64(i)*0.7) + 0.1 }

	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, contiguous := range []bool{true, false} {
			name := fmt.Sprintf("%v/contig=%v", dt, contiguous)
			t.Run(name, func(t *testing.T) {
				newPs := make([]*tensor.Tensor, len(shapes))
				slowPs := make([]*tensor.Tensor, len(shapes))
				pert := make([][]float64, len(shapes))
				for pi, s := range shapes {
					if contiguous {
						newPs[pi] = tensor.New(dt, s)
						slowPs[pi] = tensor.New(dt, s)
					} else {
						newPs[pi] = mustTransposedView(t, dt, s)
						slowPs[pi] = mustTransposedView(t, dt, s)
						if newPs[pi].IsContiguous() {
							t.Fatalf("view for %v is contiguous; generic path would not run", s)
						}
					}
					for i := range s.Numel() {
						idx := tensor.Unravel(i, s)
						newPs[pi].SetF64(pval(i), idx...)
						slowPs[pi].SetF64(pval(i), idx...)
					}
					pert[pi] = make([]float64, s.Numel())
				}

				samNew := nn.NewSAM(newPs, nn.NewSGD(newPs, lr, mom),
					nn.WithSAMRho(rho), nn.WithSAMEps(eps))
				baseSlow := nn.NewSGD(slowPs, lr, mom)

				newGrads := samQuadGrad(t, newPs, c, contiguous)
				slowGrads := samQuadGrad(t, slowPs, c, contiguous)

				for s := range steps {
					if err := samNew.Step(newGrads); err != nil {
						t.Fatalf("step %d new: %v", s, err)
					}
					if err := samStepSlow(slowPs, baseSlow, rho, eps, pert, slowGrads); err != nil {
						t.Fatalf("step %d slow: %v", s, err)
					}
					for pi, shape := range shapes {
						for i := range shape.Numel() {
							idx := tensor.Unravel(i, shape)
							a := newPs[pi].AtF64(idx...)
							b := slowPs[pi].AtF64(idx...)
							if math.Float64bits(a) != math.Float64bits(b) {
								t.Fatalf("step %d param %d idx %v: fast %.17g != oracle %.17g (not bit-identical)",
									s, pi, idx, a, b)
							}
						}
					}
				}
			})
		}
	}
}

// benchmarkSAMStep isolates SAM's three per-step loops: a fixed (weight-independent) gradient
// makes grads() O(1), and samNoopBase removes the base optimizer, so the measurement is the
// norm/ascend/restore loops over a 512×512 parameter. fast=true runs the devirtualized
// SAM.Step; fast=false runs the verbatim per-element oracle [samStepSlow].
func benchmarkSAMStep(b *testing.B, dt tensor.Dtype, fast bool) {
	shape := tensor.Shape{512, 512}
	p := tensor.New(dt, shape)
	g := tensor.New(dt, shape)
	for i := range shape.Numel() {
		idx := tensor.Unravel(i, shape)
		p.SetF64(0.01*float64(i%97-48), idx...)
		g.SetF64(0.001*float64(i%53-26), idx...)
	}
	ps := []*tensor.Tensor{p}
	grads := func() (nn.GradFn, error) {
		return func(*tensor.Tensor) *tensor.Tensor { return g }, nil
	}
	b.ResetTimer()
	if fast {
		sam := nn.NewSAM(ps, samNoopBase{}, nn.WithSAMRho(0.05))
		for i := 0; i < b.N; i++ {
			if err := sam.Step(grads); err != nil {
				b.Fatal(err)
			}
		}
	} else {
		pert := [][]float64{make([]float64, p.Numel())}
		for i := 0; i < b.N; i++ {
			if err := samStepSlow(ps, samNoopBase{}, 0.05, 1e-12, pert, grads); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkSAMStepFastF64(b *testing.B) { benchmarkSAMStep(b, tensor.F64, true) }
func BenchmarkSAMStepSlowF64(b *testing.B) { benchmarkSAMStep(b, tensor.F64, false) }
func BenchmarkSAMStepFastF32(b *testing.B) { benchmarkSAMStep(b, tensor.F32, true) }
func BenchmarkSAMStepSlowF32(b *testing.B) { benchmarkSAMStep(b, tensor.F32, false) }
