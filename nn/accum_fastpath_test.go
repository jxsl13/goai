package nn

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// T870 follow-up: GradAccumulator.Add reads a full gradient tensor (parameter-
// shaped, potentially huge) via AtF64(Unravel(i)...) on EVERY microbatch —
// squarely LARGE/hot. It converts via readGen, nn/init.go's read-side mirror of
// fillGen (see weightavg_fastpath_test.go for readGen's own gate). GradFn()'s
// returned closure writes the averaged sum into a fresh tensor via
// SetF64(Unravel(i)...) — a pure fillGen shape — and reuses fillGen directly.
// Reset() zeroes a plain []float64, not a tensor, so is untouched.

func accumParams(dtype tensor.Dtype, shape tensor.Shape, n int) []*tensor.Tensor {
	out := make([]*tensor.Tensor, n)
	for i := range out {
		out[i] = tensor.New(dtype, shape)
	}
	return out
}

func accumMutate(t *tensor.Tensor, seed uint64) {
	rng := rand.New(rand.NewPCG(seed, 0xabcdef0123456789))
	shape := t.Shape()
	for i := range t.Numel() {
		t.SetF64(rng.NormFloat64(), tensor.Unravel(i, shape)...)
	}
}

// --- GradAccumulator.Add ---

// slowGradAccumulatorAdd is the verbatim pre-conversion per-element path.
func slowGradAccumulatorAdd(a *GradAccumulator, grad GradFn) {
	for _, p := range a.params {
		g := grad(p)
		if g == nil {
			continue
		}
		s := a.sums[p]
		for i := range s {
			s[i] += g.AtF64(tensor.Unravel(i, p.Shape())...)
		}
	}
	a.steps++
}

func TestGradAccumulatorAddBitIdenticalToSlowPath(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range []tensor.Shape{{7}, {4, 5}, {33, 17}} {
			params := accumParams(dt, shape, 3)
			fast := NewGradAccumulator(params)
			slow := NewGradAccumulator(params)
			for round := 0; round < 3; round++ {
				grads := make(map[*tensor.Tensor]*tensor.Tensor, len(params))
				for i, p := range params {
					g := tensor.New(dt, shape)
					accumMutate(g, uint64(round*10+i))
					grads[p] = g
				}
				gradFn := func(p *tensor.Tensor) *tensor.Tensor { return grads[p] }
				fast.Add(gradFn)
				slowGradAccumulatorAdd(slow, gradFn)
			}
			for _, p := range params {
				fs, ss := fast.sums[p], slow.sums[p]
				for i := range fs {
					if math.Float64bits(fs[i]) != math.Float64bits(ss[i]) {
						t.Fatalf("dt=%v shape=%v elem %d: fast %v != slow %v (not bit-identical)", dt, shape, i, fs[i], ss[i])
					}
				}
			}
		}
	}
}

// A non-contiguous gradient tensor (e.g. a transposed view returned by a custom
// GradFn) must still accumulate correctly, matching the slow oracle exactly.
func TestGradAccumulatorAddNonContiguousGrad(t *testing.T) {
	p := tensor.New(tensor.F64, tensor.Shape{3, 4})
	fast := NewGradAccumulator([]*tensor.Tensor{p})
	slow := NewGradAccumulator([]*tensor.Tensor{p})

	base := tensor.Zeros(tensor.F64, tensor.Shape{4, 3})
	for i := range 4 {
		for j := range 3 {
			base.SetF64(float64(i*3+j)+0.25, i, j)
		}
	}
	gView, err := base.Transpose(0, 1) // [3,4], non-contiguous, matches p's shape
	if err != nil {
		t.Fatal(err)
	}
	gradFn := func(q *tensor.Tensor) *tensor.Tensor {
		if q == p {
			return gView
		}
		return nil
	}
	fast.Add(gradFn)
	slowGradAccumulatorAdd(slow, gradFn)
	for i := range fast.sums[p] {
		a, b := fast.sums[p][i], slow.sums[p][i]
		if math.Float64bits(a) != math.Float64bits(b) {
			t.Fatalf("elem %d: fast %v != slow %v (not bit-identical)", i, a, b)
		}
	}
}

func BenchmarkGradAccumulatorAddFast(b *testing.B) {
	params := accumParams(tensor.F32, tensor.Shape{512, 512}, 4)
	a := NewGradAccumulator(params)
	grads := make(map[*tensor.Tensor]*tensor.Tensor, len(params))
	for _, p := range params {
		g := tensor.New(tensor.F32, tensor.Shape{512, 512})
		accumMutate(g, 1)
		grads[p] = g
	}
	gradFn := func(p *tensor.Tensor) *tensor.Tensor { return grads[p] }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Add(gradFn)
	}
}

func BenchmarkGradAccumulatorAddSlow(b *testing.B) {
	params := accumParams(tensor.F32, tensor.Shape{512, 512}, 4)
	a := NewGradAccumulator(params)
	grads := make(map[*tensor.Tensor]*tensor.Tensor, len(params))
	for _, p := range params {
		g := tensor.New(tensor.F32, tensor.Shape{512, 512})
		accumMutate(g, 1)
		grads[p] = g
	}
	gradFn := func(p *tensor.Tensor) *tensor.Tensor { return grads[p] }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		slowGradAccumulatorAdd(a, gradFn)
	}
}

// --- GradAccumulator.GradFn() closure ---

// slowGradFnClosure is the verbatim pre-conversion per-element path for the
// closure GradFn() returns.
func slowGradFnClosure(a *GradAccumulator) GradFn {
	k := float64(a.steps)
	return func(p *tensor.Tensor) *tensor.Tensor {
		s, ok := a.sums[p]
		if !ok || a.steps == 0 {
			return nil
		}
		out := tensor.New(p.Dtype(), p.Shape())
		for i := range s {
			out.SetF64(s[i]/k, tensor.Unravel(i, p.Shape())...)
		}
		return out
	}
}

func TestGradFnBitIdenticalToSlowPath(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range []tensor.Shape{{1}, {7}, {4, 5}, {33, 17}} {
			params := accumParams(dt, shape, 2)
			fast := NewGradAccumulator(params)
			slow := NewGradAccumulator(params)
			for round := 0; round < 3; round++ {
				grads := make(map[*tensor.Tensor]*tensor.Tensor, len(params))
				for i, p := range params {
					g := tensor.New(dt, shape)
					accumMutate(g, uint64(round*10+i)+7)
					grads[p] = g
				}
				gradFn := func(p *tensor.Tensor) *tensor.Tensor { return grads[p] }
				fast.Add(gradFn)
				slowGradAccumulatorAdd(slow, gradFn)
			}
			fastOut := fast.GradFn()
			slowOut := slowGradFnClosure(slow)
			for _, p := range params {
				a, b := fastOut(p), slowOut(p)
				n := a.Numel()
				for i := range n {
					idx := tensor.Unravel(i, shape)
					av, bv := a.AtF64(idx...), b.AtF64(idx...)
					if math.Float64bits(av) != math.Float64bits(bv) {
						t.Fatalf("dt=%v shape=%v elem %d: fast %v != slow %v (not bit-identical)", dt, shape, i, av, bv)
					}
				}
			}
		}
	}
}

func BenchmarkGradFnFast(b *testing.B) {
	params := accumParams(tensor.F32, tensor.Shape{512, 512}, 4)
	a := NewGradAccumulator(params)
	grads := make(map[*tensor.Tensor]*tensor.Tensor, len(params))
	for _, p := range params {
		g := tensor.New(tensor.F32, tensor.Shape{512, 512})
		accumMutate(g, 1)
		grads[p] = g
	}
	a.Add(func(p *tensor.Tensor) *tensor.Tensor { return grads[p] })
	fn := a.GradFn()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, p := range params {
			_ = fn(p)
		}
	}
}

func BenchmarkGradFnSlow(b *testing.B) {
	params := accumParams(tensor.F32, tensor.Shape{512, 512}, 4)
	a := NewGradAccumulator(params)
	grads := make(map[*tensor.Tensor]*tensor.Tensor, len(params))
	for _, p := range params {
		g := tensor.New(tensor.F32, tensor.Shape{512, 512})
		accumMutate(g, 1)
		grads[p] = g
	}
	a.Add(func(p *tensor.Tensor) *tensor.Tensor { return grads[p] })
	fn := slowGradFnClosure(a)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, p := range params {
			_ = fn(p)
		}
	}
}
