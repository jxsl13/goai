package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type noFocalBackend struct{ backend.Backend }

func (b noFocalBackend) Name() backend.Name { return "cpu-no-focal-test" }

func (b noFocalBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpSigmoidFocalCore || op == backend.OpSigmoidFocalCoreBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

func TestSigmoidFocalCapabilityRoute(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	logits := tensor.FromFloat32(tensor.Shape{4}, []float32{-1, 2, 0.5, -0.25})
	targets := tensor.FromFloat32(tensor.Shape{4}, []float32{0, 1, 1, 0})

	fused := autograd.NewTapeOn(cpuBE)
	if _, err := nn.SigmoidFocalLoss(fused.Context(), logits, targets, 2, 0.25); err != nil {
		t.Fatal(err)
	}
	if got := fused.Len(); got != 2 {
		t.Fatalf("fused tape length=%d, want core+mean=2", got)
	}

	control := autograd.NewTapeOn(noFocalBackend{cpuBE})
	if _, err := nn.SigmoidFocalLoss(control.Context(), logits, targets, 2, 0.25); err != nil {
		t.Fatal(err)
	}
	if got := control.Len(); got != 9 {
		t.Fatalf("fallback tape length=%d, want established gamma!=0 composite=9", got)
	}
}

func TestSigmoidFocalFusionGuards(t *testing.T) {
	cpuBE, _ := backend.Get(backend.CPU)
	stridedLogits, err := tensor.FromFloat32(tensor.Shape{2, 2}, []float32{-1, 2, 0.5, -0.25}).Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	stridedTargets, err := tensor.FromFloat32(tensor.Shape{2, 2}, []float32{0, 1, 1, 0}).Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name            string
		logits, targets *tensor.Tensor
	}{
		{
			name:    "mixed-dtype",
			logits:  tensor.FromFloat32(tensor.Shape{4}, []float32{-1, 2, 0.5, -0.25}),
			targets: tensor.FromFloat64(tensor.Shape{4}, []float64{0, 1, 1, 0}),
		},
		{
			name:    "strided",
			logits:  stridedLogits,
			targets: stridedTargets,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tape := autograd.NewTapeOn(cpuBE)
			if _, err := nn.SigmoidFocalLoss(tape.Context(), tc.logits, tc.targets, 2, 0.25); err != nil {
				t.Fatal(err)
			}
			if got := tape.Len(); got != 9 {
				t.Fatalf("guarded fallback tape length=%d, want 9", got)
			}
		})
	}
}
