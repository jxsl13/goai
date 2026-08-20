//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

type measuredUnaryRouteCase struct {
	name     string
	op       backend.Op
	selector int
	positive bool
}

func measuredUnaryRouteCases() []measuredUnaryRouteCase {
	return []measuredUnaryRouteCase{
		{name: "neg", op: backend.OpNeg, selector: unaryNeg},
		{name: "exp", op: backend.OpExp, selector: unaryExp},
		{name: "log", op: backend.OpLog, selector: unaryLog, positive: true},
		{name: "tanh", op: backend.OpTanh, selector: unaryTanh},
		{name: "relu", op: backend.OpReLU, selector: unaryReLU},
		{name: "sigmoid", op: backend.OpSigmoid, selector: unarySigmoid},
		{name: "sqrt", op: backend.OpSqrt, selector: unarySqrt, positive: true},
		{name: "abs", op: backend.OpAbs, selector: unaryAbs},
	}
}

func TestMetalMeasuredUnaryRouteThresholds(t *testing.T) {
	cases := measuredUnaryRouteCases()
	if got := len(cases); got != 8 {
		t.Fatalf("measured unary route case count = %d, want 8", got)
	}
	for _, unary := range cases {
		if got, want := measuredHostUnaryMaxElements(unary.op), expectedMeasuredHostUnaryMaxElements(unary.op); got != want {
			t.Errorf("%s threshold = %d, want %d", unary.name, got, want)
		}
	}
}

// Each enabled operation is pinned to the optimized CPU arm at its measured
// ceiling and to direct Metal immediately above it. A strided tensor separately
// proves that an unmeasured layout remains on Metal even inside the size bound.
func TestMetalMeasuredUnaryRoutesBothArms(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("Metal available but not registered")
	}
	ctx := backend.NewContext().WithBackend(mb)
	for _, unary := range measuredUnaryRouteCases() {
		unary := unary
		maxElements := measuredHostUnaryMaxElements(unary.op)
		if maxElements == 0 {
			continue
		}
		cpu, ok := cpuPrefers(unary.op, tensor.F32)
		if !ok {
			t.Fatalf("optimized CPU %s unavailable", unary.name)
		}
		production := unaryF32(unary.op, unary.selector)
		direct := unaryMetalF32(unary.op, unary.selector)
		for _, arm := range []struct {
			name      string
			shape     tensor.Shape
			transpose bool
			want      func([]*tensor.Tensor) ([]*tensor.Tensor, error)
		}{
			{name: "ceiling-to-cpu", shape: tensor.Shape{maxElements}, want: func(in []*tensor.Tensor) ([]*tensor.Tensor, error) {
				return backend.Execute(ctx.WithBackend(cpu).WithRecorder(nil), unary.op, in, nil)
			}},
			{name: "above-to-metal", shape: tensor.Shape{maxElements + 1}, want: func(in []*tensor.Tensor) ([]*tensor.Tensor, error) {
				return direct(ctx, in, nil)
			}},
			{name: "strided-to-metal", shape: tensor.Shape{64, 32}, transpose: true, want: func(in []*tensor.Tensor) ([]*tensor.Tensor, error) {
				return direct(ctx, in, nil)
			}},
		} {
			arm := arm
			t.Run(unary.name+"/"+arm.name, func(t *testing.T) {
				x := bench.RandF32(arm.shape, 23)
				if unary.positive {
					values := x.Storage().F32()
					for i, value := range values {
						values[i] = float32(math.Abs(float64(value))) + 0.125
					}
				}
				if arm.transpose {
					var err error
					x, err = x.Transpose(0, 1)
					if err != nil {
						t.Fatal(err)
					}
				}
				inputs := []*tensor.Tensor{x}
				got, err := production(ctx, inputs, nil)
				if err != nil {
					t.Fatal(err)
				}
				want, err := arm.want(inputs)
				if err != nil {
					t.Fatal(err)
				}
				gotValues, wantValues := got[0].Storage().F32(), want[0].Storage().F32()
				if len(gotValues) == 0 || len(gotValues) != len(wantValues) {
					t.Fatalf("route output lengths: production %d, selected arm %d", len(gotValues), len(wantValues))
				}
				for i := range gotValues {
					if math.Float32bits(gotValues[i]) != math.Float32bits(wantValues[i]) {
						t.Fatalf("route [%d]: production %v vs selected arm %v", i, gotValues[i], wantValues[i])
					}
				}
			})
		}
	}
}
