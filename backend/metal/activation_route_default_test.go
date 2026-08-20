//go:build darwin && cgo && !(arm64 && goexperiment.simd)

package metal

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Default builds must remain byte-identical to the incumbent direct Metal
// route, even where the CPU backend has scalar forward kernels registered.
func TestMetalDefaultActivationRouteRemainsDirect(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	if hostSIMDActivationRouteEnabled {
		t.Fatal("default activation route unexpectedly enabled")
	}
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("Metal available but not registered")
	}
	ctx := backend.NewContext().WithBackend(mb)
	shape := tensor.Shape{2048}
	for _, activation := range []struct {
		name       string
		arity      int
		production backend.Kernel
		direct     backend.Kernel
	}{
		{name: "gelu", arity: 1, production: unaryF32(backend.OpGELU, unaryGELU), direct: unaryMetalF32(backend.OpGELU, unaryGELU)},
		{name: "gelu_backward", arity: 2, production: geluBackwardF32, direct: geluBackwardMetalF32},
		{name: "silu", arity: 1, production: unaryF32(backend.OpSiLU, unarySiLU), direct: unaryMetalF32(backend.OpSiLU, unarySiLU)},
		{name: "silu_backward", arity: 2, production: siluBackwardF32, direct: siluBackwardMetalF32},
	} {
		activation := activation
		t.Run(activation.name, func(t *testing.T) {
			inputs := []*tensor.Tensor{bench.RandF32(shape, 3)}
			if activation.arity == 2 {
				inputs = append(inputs, bench.RandF32(shape, 5))
			}
			got, err := activation.production(ctx, inputs, nil)
			if err != nil {
				t.Fatal(err)
			}
			want, err := activation.direct(ctx, inputs, nil)
			if err != nil {
				t.Fatal(err)
			}
			gotValues, wantValues := got[0].Storage().F32(), want[0].Storage().F32()
			for i := range gotValues {
				if math.Float32bits(gotValues[i]) != math.Float32bits(wantValues[i]) {
					t.Fatalf("route [%d]: production %v vs direct Metal %v", i, gotValues[i], wantValues[i])
				}
			}
		})
	}
}
