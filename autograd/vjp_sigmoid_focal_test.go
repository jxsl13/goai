package autograd_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func requireSigmoidFocalParity(t *testing.T, got, want *tensor.Tensor) {
	t.Helper()
	if got.Dtype() != tensor.F32 || !sigmoidFocalF32Tolerant {
		requireGradBits(t, got, want)
		return
	}
	for i, v := range got.Storage().F32() {
		w := want.Storage().F32()[i]
		if d := math.Abs(float64(v) - float64(w)); d > 1e-6+2e-3*math.Abs(float64(w)) {
			t.Fatalf("element %d: got %g want %g (|err|=%g)", i, v, w, d)
		}
	}
}

type focalCompositeBackend struct{ backend.Backend }

func (b focalCompositeBackend) Name() backend.Name { return "cpu-focal-composite-test" }

func (b focalCompositeBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpSigmoidFocalCore || op == backend.OpSigmoidFocalCoreBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

func focalInputs(dt tensor.Dtype, n int) (*tensor.Tensor, *tensor.Tensor) {
	if dt == tensor.F32 {
		x := bench.RandF32(tensor.Shape{n}, 271)
		y := tensor.New(tensor.F32, tensor.Shape{n})
		for i := range y.Storage().F32() {
			y.Storage().F32()[i] = float32(i & 1)
		}
		return x, y
	}
	x := bench.RandF64(tensor.Shape{n}, 271)
	y := tensor.New(tensor.F64, tensor.Shape{n})
	for i := range y.Storage().F64() {
		y.Storage().F64()[i] = float64(i & 1)
	}
	return x, y
}

func TestSigmoidFocalCoreExactCompositeVJPParity(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	controlBE := focalCompositeBackend{cpuBE}
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, attrs := range []backend.SigmoidFocalAttrs{{Gamma: 0, Alpha: -1}, {Gamma: 2, Alpha: 0.25}} {
			t.Run(dt.String(), func(t *testing.T) {
				logits, targets := focalInputs(dt, 65537)

				fusedTape := autograd.NewTapeOn(cpuBE)
				fusedCore, err := backend.Execute(fusedTape.Context(), backend.OpSigmoidFocalCore, []*tensor.Tensor{logits, targets}, attrs)
				if err != nil {
					t.Fatal(err)
				}
				fusedLoss, err := backend.Execute(fusedTape.Context(), backend.OpMean, fusedCore, nil)
				if err != nil {
					t.Fatal(err)
				}

				controlTape := autograd.NewTapeOn(controlBE)
				controlLoss, err := nn.SigmoidFocalLoss(controlTape.Context(), logits, targets, attrs.Gamma, attrs.Alpha)
				if err != nil {
					t.Fatal(err)
				}
				requireSigmoidFocalParity(t, fusedLoss[0], controlLoss)
				if err := fusedTape.Backward(fusedLoss[0]); err != nil {
					t.Fatal(err)
				}
				if err := controlTape.Backward(controlLoss); err != nil {
					t.Fatal(err)
				}
				requireSigmoidFocalParity(t, fusedTape.Grad(logits), controlTape.Grad(logits))
				if fusedTape.Grad(targets) != nil || controlTape.Grad(targets) != nil {
					t.Fatal("targets must remain detached")
				}
			})
		}
	}
}

func TestSigmoidFocalCoreGradCheck(t *testing.T) {
	targets := tensor.FromFloat64(tensor.Shape{6}, []float64{0, 1, 0, 1, 1, 0})
	forward := func(ctx *backend.Context, xs []*tensor.Tensor) (*tensor.Tensor, error) {
		out, err := backend.Execute(ctx, backend.OpSigmoidFocalCore, []*tensor.Tensor{xs[0], targets}, backend.SigmoidFocalAttrs{Gamma: 2, Alpha: 0.25})
		if err != nil {
			return nil, err
		}
		return out[0], nil
	}
	logits := tensor.FromFloat64(tensor.Shape{6}, []float64{-2.4, -0.7, 0.2, 0.8, 1.7, 3.1})
	if err := autograd.GradCheck(forward, []*tensor.Tensor{logits}); err != nil {
		t.Fatal(err)
	}
}
