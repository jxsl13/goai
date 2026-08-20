//go:build darwin && cgo

package metal

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// The CPU winner zone is deliberately bounded by maxHostBiasGradElements. This
// pins both selector arms: a measured shape must match the optimized CPU route,
// while a shape one row beyond the ceiling must match direct Metal bytewise.
func TestMetalAddBiasBackwardMeasuredThresholdRoutesBothArms(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	mb, ok := backend.Get(backend.Metal)
	if !ok {
		t.Fatal("Metal available but not registered")
	}
	ctx := backend.NewContext().WithBackend(mb)
	cpu, ok := cpuPrefers(backend.OpAddBiasBackward, tensor.F32)
	if !ok {
		t.Fatal("optimized CPU add-bias backward unavailable")
	}
	const (
		cols      = 512
		aboveRows = 4097
	)
	if aboveRows*cols <= maxHostBiasGradElements {
		t.Fatal("above-threshold test shape no longer crosses the measured bound")
	}
	for _, tc := range []struct {
		name string
		rows int
		want func([]*tensor.Tensor) ([]*tensor.Tensor, error)
	}{
		{name: "below-to-cpu", rows: 256, want: func(in []*tensor.Tensor) ([]*tensor.Tensor, error) {
			return backend.Execute(ctx.WithBackend(cpu).WithRecorder(nil), backend.OpAddBiasBackward, in, nil)
		}},
		{name: "above-to-metal", rows: aboveRows, want: func(in []*tensor.Tensor) ([]*tensor.Tensor, error) {
			return addBiasBackwardMetalF32(ctx, in, nil)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := bench.RandF32(tensor.Shape{tc.rows, cols}, 3)
			got, err := addBiasBackwardF32(ctx, []*tensor.Tensor{g}, nil)
			if err != nil {
				t.Fatal(err)
			}
			want, err := tc.want([]*tensor.Tensor{g})
			if err != nil {
				t.Fatal(err)
			}
			gs, ws := got[0].Storage().F32(), want[0].Storage().F32()
			for i := range gs {
				if math.Float32bits(gs[i]) != math.Float32bits(ws[i]) {
					t.Fatalf("route [%d]: production %v vs selected arm %v", i, gs[i], ws[i])
				}
			}
		})
	}
}
