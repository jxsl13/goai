package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

func executeFocal(t *testing.T, be backend.Backend, op backend.Op, attrs backend.SigmoidFocalAttrs, in ...*tensor.Tensor) *tensor.Tensor {
	t.Helper()
	out, err := backend.Execute(backend.NewContext().WithBackend(be), op, in, attrs)
	if err != nil {
		t.Fatal(err)
	}
	return out[0]
}

func requireFocalBits(t *testing.T, got, want *tensor.Tensor) {
	t.Helper()
	switch got.Dtype() {
	case tensor.F32:
		for i, v := range got.Storage().F32() {
			w := want.Storage().F32()[i]
			if math.IsNaN(float64(v)) || math.IsNaN(float64(w)) {
				if !math.IsNaN(float64(v)) || !math.IsNaN(float64(w)) {
					t.Fatalf("element %d: got %g want %g", i, v, w)
				}
				continue
			}
			if geluF32Tolerant {
				if math.IsInf(float64(v), 0) || math.IsInf(float64(w), 0) {
					if v != w {
						t.Fatalf("element %d: got %g want %g", i, v, w)
					}
					continue
				}
				if d := math.Abs(float64(v) - float64(w)); d > 1e-6+2e-3*math.Abs(float64(w)) {
					t.Fatalf("element %d: got %g want %g (|err|=%g)", i, v, w, d)
				}
				continue
			}
			if gb, wb := math.Float32bits(v), math.Float32bits(w); gb != wb {
				t.Fatalf("element %d: got %08x want %08x", i, gb, wb)
			}
		}
	case tensor.F64:
		for i, v := range got.Storage().F64() {
			w := want.Storage().F64()[i]
			if math.IsNaN(v) || math.IsNaN(w) {
				if !math.IsNaN(v) || !math.IsNaN(w) {
					t.Fatalf("element %d: got %g want %g", i, v, w)
				}
				continue
			}
			if gb, wb := math.Float64bits(v), math.Float64bits(w); gb != wb {
				t.Fatalf("element %d: got %016x want %016x", i, gb, wb)
			}
		}
	}
}

func TestSigmoidFocalCPUReferenceParity(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	refBE := backend.Reference()
	for _, n := range []int{0, 21, 300000} {
		for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
			name := dt.String()
			var logits, targets, grad *tensor.Tensor
			if dt == tensor.F32 {
				logits = bench.RandF32(tensor.Shape{n}, 211)
				targets = tensor.New(tensor.F32, tensor.Shape{n})
				grad = bench.RandF32(tensor.Shape{n}, 223)
				for i := range targets.Storage().F32() {
					targets.Storage().F32()[i] = float32(i & 1)
				}
				if n == 21 {
					copy(logits.Storage().F32(), []float32{
						0, math.Float32frombits(1 << 31), math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
						1, -1, 20, -20, 88, -88, 100, -100, math.MaxFloat32, -math.MaxFloat32,
						0.25, -0.25, 7.5, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
						math.Float32frombits(0xFFC00001),
					})
				}
			} else {
				logits = bench.RandF64(tensor.Shape{n}, 211)
				targets = tensor.New(tensor.F64, tensor.Shape{n})
				grad = bench.RandF64(tensor.Shape{n}, 223)
				for i := range targets.Storage().F64() {
					targets.Storage().F64()[i] = float64(i & 1)
				}
				if n == 21 {
					copy(logits.Storage().F64(), []float64{
						0, math.Copysign(0, -1), math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
						1, -1, 20, -20, 700, -700, 1000, -1000, math.MaxFloat64, -math.MaxFloat64,
						0.25, -0.25, 7.5, math.Inf(1), math.Inf(-1), math.NaN(),
						math.Float64frombits(0xFFF8000000000001),
					})
				}
			}
			for _, attrs := range []backend.SigmoidFocalAttrs{{Gamma: 0, Alpha: -1}, {Gamma: 2, Alpha: 0.25}} {
				t.Run(name, func(t *testing.T) {
					requireFocalBits(t,
						executeFocal(t, cpuBE, backend.OpSigmoidFocalCore, attrs, logits, targets),
						executeFocal(t, refBE, backend.OpSigmoidFocalCore, attrs, logits, targets))
					requireFocalBits(t,
						executeFocal(t, cpuBE, backend.OpSigmoidFocalCoreBackward, attrs, logits, targets, grad),
						executeFocal(t, refBE, backend.OpSigmoidFocalCoreBackward, attrs, logits, targets, grad))
				})
			}
		}
	}
}

func TestSigmoidFocalCoreDoesNotMutateInputs(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	logits := tensor.FromFloat32(tensor.Shape{5}, []float32{-3, -0, 0.5, 2, 9})
	targets := tensor.FromFloat32(tensor.Shape{5}, []float32{0, 1, 0, 1, 1})
	grad := tensor.FromFloat32(tensor.Shape{5}, []float32{1, -2, 0.25, 4, -0.5})
	wantLogits := append([]float32(nil), logits.Storage().F32()...)
	wantTargets := append([]float32(nil), targets.Storage().F32()...)
	wantGrad := append([]float32(nil), grad.Storage().F32()...)
	attrs := backend.SigmoidFocalAttrs{Gamma: 2, Alpha: 0.25}
	executeFocal(t, cpuBE, backend.OpSigmoidFocalCore, attrs, logits, targets)
	executeFocal(t, cpuBE, backend.OpSigmoidFocalCoreBackward, attrs, logits, targets, grad)
	for i, want := range wantLogits {
		if math.Float32bits(logits.Storage().F32()[i]) != math.Float32bits(want) {
			t.Fatalf("logits[%d] mutated", i)
		}
	}
	for i, want := range wantTargets {
		if math.Float32bits(targets.Storage().F32()[i]) != math.Float32bits(want) {
			t.Fatalf("targets[%d] mutated", i)
		}
	}
	for i, want := range wantGrad {
		if math.Float32bits(grad.Storage().F32()[i]) != math.Float32bits(want) {
			t.Fatalf("grad[%d] mutated", i)
		}
	}
}

func TestSigmoidFocalCoreRejectsInvalidInputs(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(cpuBE)
	x := tensor.New(tensor.F32, tensor.Shape{2})
	y := tensor.New(tensor.F32, tensor.Shape{3})
	if _, err := backend.Execute(ctx, backend.OpSigmoidFocalCore, []*tensor.Tensor{x, y}, backend.SigmoidFocalAttrs{}); err == nil {
		t.Fatal("shape mismatch: want error")
	}
	if _, err := backend.Execute(ctx, backend.OpSigmoidFocalCoreBackward, []*tensor.Tensor{x, x}, backend.SigmoidFocalAttrs{}); err == nil {
		t.Fatal("arity mismatch: want error")
	}
}
