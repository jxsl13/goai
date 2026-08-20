//go:build darwin && cgo && arm64 && goexperiment.simd

package metal

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// This mutation-resistant selector test pins both measured route arms for all
// four activations: a small contiguous tensor must match the optimized CPU
// kernel bytewise, while strided and max+1 tensors must match direct Metal.
func TestMetalSIMDActivationMeasuredThresholdRoutesBothArms(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	if !hostSIMDActivationRouteEnabled {
		t.Fatal("arm64 SIMD activation route unexpectedly disabled")
	}
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("Metal available but not registered")
	}
	ctx := backend.NewContext().WithBackend(mb)
	type routeCase struct {
		name       string
		op         backend.Op
		arity      int
		production backend.Kernel
		direct     backend.Kernel
	}
	for _, activation := range []routeCase{
		{name: "gelu", op: backend.OpGELU, arity: 1, production: unaryF32(backend.OpGELU, unaryGELU), direct: unaryMetalF32(backend.OpGELU, unaryGELU)},
		{name: "gelu_backward", op: backend.OpGELUBackward, arity: 2, production: geluBackwardF32, direct: geluBackwardMetalF32},
		{name: "silu", op: backend.OpSiLU, arity: 1, production: unaryF32(backend.OpSiLU, unarySiLU), direct: unaryMetalF32(backend.OpSiLU, unarySiLU)},
		{name: "silu_backward", op: backend.OpSiLUBackward, arity: 2, production: siluBackwardF32, direct: siluBackwardMetalF32},
	} {
		activation := activation
		cpu, ok := cpuPrefers(activation.op, tensor.F32)
		if !ok {
			t.Fatalf("optimized CPU %s unavailable", activation.name)
		}
		for _, arm := range []struct {
			name      string
			shape     tensor.Shape
			transpose bool
			want      func([]*tensor.Tensor) ([]*tensor.Tensor, error)
		}{
			{name: "within-to-cpu", shape: tensor.Shape{256, 512}, want: func(in []*tensor.Tensor) ([]*tensor.Tensor, error) {
				return backend.Execute(ctx.WithBackend(cpu).WithRecorder(nil), activation.op, in, nil)
			}},
			{name: "strided-to-metal", shape: tensor.Shape{512, 256}, transpose: true, want: func(in []*tensor.Tensor) ([]*tensor.Tensor, error) {
				return activation.direct(ctx, in, nil)
			}},
			{name: "above-to-metal", shape: tensor.Shape{maxHostSIMDActivationElements + 1}, want: func(in []*tensor.Tensor) ([]*tensor.Tensor, error) {
				return activation.direct(ctx, in, nil)
			}},
		} {
			arm := arm
			t.Run(activation.name+"/"+arm.name, func(t *testing.T) {
				makeInput := func(seed uint64) *tensor.Tensor {
					x := bench.RandF32(arm.shape, seed)
					if !arm.transpose {
						return x
					}
					x, err := x.Transpose(0, 1)
					if err != nil {
						t.Fatal(err)
					}
					return x
				}
				inputs := []*tensor.Tensor{makeInput(3)}
				if activation.arity == 2 {
					inputs = append(inputs, makeInput(5))
				}
				got, err := activation.production(ctx, inputs, nil)
				if err != nil {
					t.Fatal(err)
				}
				want, err := arm.want(inputs)
				if err != nil {
					t.Fatal(err)
				}
				gotValues, wantValues := got[0].Storage().F32(), want[0].Storage().F32()
				for i := range gotValues {
					if math.Float32bits(gotValues[i]) != math.Float32bits(wantValues[i]) {
						t.Fatalf("route [%d]: production %v vs selected arm %v", i, gotValues[i], wantValues[i])
					}
				}
			})
		}
	}
}
