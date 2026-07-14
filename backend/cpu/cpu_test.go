package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func cpuBackend(t *testing.T) backend.Backend {
	t.Helper()
	b, ok := backend.Get(backend.CPU)
	if !ok {
		t.Fatal("cpu backend not registered")
	}
	return b
}

func run(t *testing.T, b backend.Backend, op backend.Op, ins ...*tensor.Tensor) *tensor.Tensor {
	t.Helper()
	out, err := backend.Execute(backend.NewContext().WithBackend(b), op, ins, nil)
	if err != nil {
		t.Fatalf("%s %v: %v", b.Name(), op, err)
	}
	return out[0]
}

// cpu is registered as the preferred Default (§T11).
func TestCPUIsDefault(t *testing.T) {
	if backend.Default().Name() != backend.CPU {
		t.Errorf("Default = %q, want cpu", backend.Default().Name())
	}
}

// §V3/§V11 CROSS: the optimized cpu backend is bit-identical to the reference for
// elementwise ops (same f64 math) — tolerance 0.
func TestCPUCrossReferenceExact(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)

	binOps := []backend.Op{backend.OpAdd, backend.OpSub, backend.OpMul, backend.OpDiv}
	unOps := []backend.Op{backend.OpNeg, backend.OpExp, backend.OpLog, backend.OpTanh, backend.OpReLU, backend.OpGELU, backend.OpSigmoid}

	for _, dtype := range []tensor.Dtype{tensor.F64, tensor.F32} {
		mk := func(seed uint64) *tensor.Tensor {
			if dtype == tensor.F64 {
				return bench.RandF64(tensor.Shape{257}, seed)
			}
			return bench.RandF32(tensor.Shape{257}, seed)
		}
		a, b := mk(1), mk(2)
		// make b strictly positive for log domain
		bpos := run(t, cpu, backend.OpExp, b) // exp>0

		for _, op := range binOps {
			gc := run(t, cpu, op, a, b)
			gr := run(t, ref, op, a, b)
			assertEqualExact(t, gc, gr, op.String()+"/"+dtype.String())
		}
		for _, op := range unOps {
			in := a
			if op == backend.OpLog {
				in = bpos
			}
			gc := run(t, cpu, op, in)
			gr := run(t, ref, op, in)
			assertEqualExact(t, gc, gr, op.String()+"/"+dtype.String())
		}
	}
}

func assertEqualExact(t *testing.T, got, want *tensor.Tensor, label string) {
	t.Helper()
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s: shape %v vs %v", label, got.Shape(), want.Shape())
	}
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		g, w := got.AtF64(idx...), want.AtF64(idx...)
		if g != w && !(math.IsNaN(g) && math.IsNaN(w)) {
			t.Fatalf("%s [%d]: cpu %v != ref %v", label, i, g, w)
		}
	}
}

// Parallel path (n > parThreshold) must match the serial reference.
func TestCPUParallelCorrect(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)
	const n = 1 << 18 // above parThreshold → exercises goroutines
	a := bench.RandF64(tensor.Shape{n}, 7)
	b := bench.RandF64(tensor.Shape{n}, 8)
	gc := run(t, cpu, backend.OpMul, a, b)
	gr := run(t, ref, backend.OpMul, a, b)
	assertEqualExact(t, gc, gr, "parallel-mul")

	ge := run(t, cpu, backend.OpTanh, a)
	gre := run(t, ref, backend.OpTanh, a)
	assertEqualExact(t, ge, gre, "parallel-tanh")
}

// Non-contiguous inputs are handled (materialized) and stay correct.
func TestCPUNonContiguous(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)
	m := tensor.FromFloat64(tensor.Shape{3, 4}, bench.RandF64(tensor.Shape{12}, 5).Storage().F64())
	tr, _ := m.Transpose(0, 1)
	gc := run(t, cpu, backend.OpNeg, tr)
	gr := run(t, ref, backend.OpNeg, tr)
	assertEqualExact(t, gc, gr, "noncontig-neg")
}

// TestCPUActivationEdge locks in the NaN/±Inf behavior of the devirtualized unary
// activations (relu/silu/gelu, plus exp/neg) against the reference. The parity test
// used only finite inputs, so this guards the edge cases the closure→native rewrite
// could have changed (e.g. relu(NaN): NaN>0 is false → 0; silu/gelu propagate NaN).
func TestCPUActivationEdge(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)
	inf, ninf, nan := math.Inf(1), math.Inf(-1), math.NaN()
	edge := []float32{float32(nan), float32(inf), float32(ninf), 2, -2}
	for _, op := range []backend.Op{backend.OpReLU, backend.OpSiLU, backend.OpGELU, backend.OpExp, backend.OpNeg} {
		x := tensor.New(tensor.F32, tensor.Shape{len(edge)})
		copy(x.Storage().F32(), edge)
		gc, err := backend.Execute(backend.NewContext().WithBackend(cpu), op, []*tensor.Tensor{x}, nil)
		if err != nil {
			t.Fatalf("%v cpu: %v", op, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), op, []*tensor.Tensor{x}, nil)
		if err != nil {
			t.Fatalf("%v ref: %v", op, err)
		}
		c, r := gc[0].Storage().F32(), gr[0].Storage().F32()
		for i := range c {
			cf, rf := float64(c[i]), float64(r[i])
			if math.IsNaN(cf) != math.IsNaN(rf) || (!math.IsNaN(cf) && cf != rf) {
				t.Errorf("%v[%d] in=%v: cpu=%v ref=%v", op, i, edge[i], c[i], r[i])
			}
		}
	}
}
