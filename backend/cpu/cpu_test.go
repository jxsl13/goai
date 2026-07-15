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
	unOps := []backend.Op{backend.OpNeg, backend.OpExp, backend.OpLog, backend.OpTanh, backend.OpReLU, backend.OpGELU, backend.OpSigmoid, backend.OpSiLU}

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
			if op == backend.OpSiLU && dtype == tensor.F64 {
				// Pre-existing ulp split: cpu evaluates x/(1+e^−x), ref
				// x·σ(x) — identical after F32 rounding (asserted below) but
				// not in F64, so only the F32 lane is cross-checked here.
				continue
			}
			gc := run(t, cpu, op, in)
			gr := run(t, ref, op, in)
			vexpVectorized := op == backend.OpGELU || op == backend.OpSigmoid || op == backend.OpSiLU
			if vexpVectorized && dtype == tensor.F32 && geluF32Tolerant {
				// arm64 perf build: F32 GELU/sigmoid/SiLU are the f32-native
				// NEON pipelines (vexp.go) — the TestGeluF32Accuracy /
				// TestSigmoidF32Accuracy / TestSiluF32Accuracy budget, not
				// bit-exact.
				assertCloseGelu(t, gc, gr, op.String()+"/"+dtype.String())
				continue
			}
			assertEqualExact(t, gc, gr, op.String()+"/"+dtype.String())
		}
	}
}

// assertCloseGelu: the F32 vexp-pipeline budget on the arm64 perf build
// (geluF32Tolerant) — |got−ref| ≤ 1e-6 + 2e-4·|ref|, the shared
// TestGeluF32Accuracy / TestSiluF32Accuracy / TestSigmoidF32Accuracy
// envelope, an order inside the ADR-0021 f32 tolerance. NaN must agree
// exactly.
func assertCloseGelu(t *testing.T, got, want *tensor.Tensor, label string) {
	t.Helper()
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s: shape %v vs %v", label, got.Shape(), want.Shape())
	}
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		g, w := got.AtF64(idx...), want.AtF64(idx...)
		if math.IsNaN(g) != math.IsNaN(w) {
			t.Fatalf("%s [%d]: cpu %v vs ref %v (NaN mismatch)", label, i, g, w)
		}
		if !math.IsNaN(g) && math.Abs(g-w) > 1e-6+2e-4*math.Abs(w) {
			t.Fatalf("%s [%d]: cpu %v vs ref %v", label, i, g, w)
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

// §V3/§V11 CROSS for the GELU VJP (§T664): on the default build (and amd64,
// and F64 everywhere) the cpu backend registers no gelu_backward kernel, so
// dispatch falls back to ref — asserted BIT-EXACT here. On the arm64 perf
// build the F32 kernel is the f32-native NEON vgeluGrad pipeline (vexp.go) —
// asserted within the TestGeluGradF32Accuracy envelope (geluF32Tolerant gates
// on exactly the vexpNeon build combination). The FFN training shape
// [256,2048] exceeds parThreshold, so the parallel chunking (and its NEON
// lane/scalar-tail splits) is exercised too.
func TestCPUGeluBackwardCrossReference(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)
	shape := tensor.Shape{256, 2048}
	for _, dtype := range []tensor.Dtype{tensor.F64, tensor.F32} {
		mk := func(seed uint64, scale float64) *tensor.Tensor {
			var tt *tensor.Tensor
			if dtype == tensor.F64 {
				tt = bench.RandF64(shape, seed)
				d := tt.Storage().F64()
				for i := range d {
					d[i] = scale * (d[i] - 0.5)
				}
			} else {
				tt = bench.RandF32(shape, seed)
				d := tt.Storage().F32()
				for i := range d {
					d[i] = float32(scale) * (d[i] - 0.5)
				}
			}
			return tt
		}
		x := mk(21, 8) // pre-activations spanning GELU's active region
		g := mk(22, 4) // upstream gradients, both signs
		gc, err := backend.Execute(backend.NewContext().WithBackend(cpu), backend.OpGELUBackward, []*tensor.Tensor{x, g}, nil)
		if err != nil {
			t.Fatalf("cpu gelu_backward/%v: %v", dtype, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpGELUBackward, []*tensor.Tensor{x, g}, nil)
		if err != nil {
			t.Fatalf("ref gelu_backward/%v: %v", dtype, err)
		}
		if dtype == tensor.F32 && geluF32Tolerant {
			c, r := gc[0].Storage().F32(), gr[0].Storage().F32()
			for i := range c {
				cf, rf := float64(c[i]), float64(r[i])
				if math.Abs(cf-rf) > 1e-6+2e-4*math.Abs(rf) {
					t.Fatalf("gelu_backward/F32 [%d]: cpu %v vs ref %v", i, c[i], r[i])
				}
			}
			continue
		}
		assertEqualExact(t, gc[0], gr[0], "gelu_backward/"+dtype.String())
	}
}

// §V3/§V11 CROSS for the SiLU VJP (§T665): on the default build (and amd64,
// and F64 everywhere) the cpu backend registers no silu_backward kernel, so
// dispatch falls back to ref — asserted BIT-EXACT here. On the arm64 perf
// build the F32 kernel is the f32-native NEON vsiluGrad pipeline (vexp.go) —
// asserted within the TestSiluGradF32Accuracy envelope (geluF32Tolerant gates
// on exactly the vexpNeon build combination). The SwiGLU FFN training shape
// [256,1365] (hidden ≈ 8·512/3 for Dim512) exceeds parThreshold, so the
// parallel chunking (and its NEON lane/scalar-tail splits — 1365 is not a
// multiple of 4) is exercised too.
func TestCPUSiluBackwardCrossReference(t *testing.T) {
	cpu := cpuBackend(t)
	ref, _ := backend.Get(backend.Ref)
	shape := tensor.Shape{256, 1365}
	for _, dtype := range []tensor.Dtype{tensor.F64, tensor.F32} {
		mk := func(seed uint64, scale float64) *tensor.Tensor {
			var tt *tensor.Tensor
			if dtype == tensor.F64 {
				tt = bench.RandF64(shape, seed)
				d := tt.Storage().F64()
				for i := range d {
					d[i] = scale * (d[i] - 0.5)
				}
			} else {
				tt = bench.RandF32(shape, seed)
				d := tt.Storage().F32()
				for i := range d {
					d[i] = float32(scale) * (d[i] - 0.5)
				}
			}
			return tt
		}
		x := mk(31, 8) // pre-activations spanning SiLU's active region
		g := mk(32, 4) // upstream gradients, both signs
		gc, err := backend.Execute(backend.NewContext().WithBackend(cpu), backend.OpSiLUBackward, []*tensor.Tensor{x, g}, nil)
		if err != nil {
			t.Fatalf("cpu silu_backward/%v: %v", dtype, err)
		}
		gr, err := backend.Execute(backend.NewContext().WithBackend(ref), backend.OpSiLUBackward, []*tensor.Tensor{x, g}, nil)
		if err != nil {
			t.Fatalf("ref silu_backward/%v: %v", dtype, err)
		}
		if dtype == tensor.F32 && geluF32Tolerant {
			c, r := gc[0].Storage().F32(), gr[0].Storage().F32()
			for i := range c {
				cf, rf := float64(c[i]), float64(r[i])
				if math.Abs(cf-rf) > 1e-6+2e-4*math.Abs(rf) {
					t.Fatalf("silu_backward/F32 [%d]: cpu %v vs ref %v", i, c[i], r[i])
				}
			}
			continue
		}
		assertEqualExact(t, gc[0], gr[0], "silu_backward/"+dtype.String())
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
			if math.IsNaN(cf) != math.IsNaN(rf) {
				t.Errorf("%v[%d] in=%v: cpu=%v ref=%v (NaN mismatch)", op, i, edge[i], c[i], r[i])
				continue
			}
			if math.IsNaN(cf) {
				continue
			}
			if (op == backend.OpGELU || op == backend.OpSiLU) && geluF32Tolerant {
				// arm64 perf build: finite F32 GELU/SiLU values carry the
				// NEON pipeline's budget; ±Inf still match exactly
				// (|Inf−Inf|=NaN fails the bound only if they differ, which
				// the IsInf check below rules out).
				if math.IsInf(cf, 0) || math.IsInf(rf, 0) {
					if cf != rf {
						t.Errorf("%v[%d] in=%v: cpu=%v ref=%v", op, i, edge[i], c[i], r[i])
					}
				} else if math.Abs(cf-rf) > 1e-6+2e-4*math.Abs(rf) {
					t.Errorf("%v[%d] in=%v: cpu=%v ref=%v", op, i, edge[i], c[i], r[i])
				}
				continue
			}
			if cf != rf {
				t.Errorf("%v[%d] in=%v: cpu=%v ref=%v", op, i, edge[i], c[i], r[i])
			}
		}
	}
}
