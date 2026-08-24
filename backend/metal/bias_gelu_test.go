//go:build darwin && cgo

package metal_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
)

func biasGELUBuffer(t *testing.T, data []float32) *metal.DeviceBuffer {
	t.Helper()
	b, err := metal.NewDeviceBufferF32(data)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(b.Release)
	return b
}

func TestRecorderBiasGELUMatchesSplitPrefixAndPreservesTail(t *testing.T) {
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	const rows, width, capacity = 2, 7, 31
	input := make([]float32, capacity)
	for i := range input {
		input[i] = float32((i*11)%29-14) * 0.125
	}
	bias := make([]float32, width)
	for i := range bias {
		bias[i] = float32(i-3) * 0.0625
	}

	candidate := biasGELUBuffer(t, input)
	control := biasGELUBuffer(t, input)
	b := biasGELUBuffer(t, bias)

	fused, err := metal.NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := fused.BiasGELU(candidate, b, candidate, rows, width); err != nil {
		fused.Free()
		t.Fatal(err)
	}
	if err := fused.Finish(); err != nil {
		fused.Free()
		t.Fatal(err)
	}
	fused.Free()

	split, err := metal.NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := split.AddBias(control, b, control, rows, width); err != nil {
		split.Free()
		t.Fatal(err)
	}
	if err := split.Unary(control, control, 9); err != nil {
		split.Free()
		t.Fatal(err)
	}
	if err := split.Finish(); err != nil {
		split.Free()
		t.Fatal(err)
	}
	split.Free()

	got := make([]float32, capacity)
	want := make([]float32, capacity)
	if err := candidate.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	if err := control.DownloadF32(want); err != nil {
		t.Fatal(err)
	}
	active := rows * width
	for i := range active {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("active[%d] fused=%v split=%v", i, got[i], want[i])
		}
	}
	if !reflect.DeepEqual(got[active:], input[active:]) {
		t.Fatalf("inactive tail mutated: got %v want %v", got[active:], input[active:])
	}
}

func TestRecorderBiasGELUProfileIsOneDispatch(t *testing.T) {
	if !metal.Available() || !metal.RecorderProfilingAvailable() {
		t.Skip("Metal timestamp profiling unavailable")
	}
	x := biasGELUBuffer(t, make([]float32, 32))
	b := biasGELUBuffer(t, make([]float32, 8))

	profile := func(fused bool) []string {
		r, err := metal.NewProfilingRecorder(2)
		if err != nil {
			t.Fatal(err)
		}
		defer r.Free()
		if fused {
			err = r.BiasGELU(x, b, x, 1, 8)
		} else {
			err = r.AddBias(x, b, x, 1, 8)
			if err == nil {
				err = r.Unary(x, x, 9)
			}
		}
		if err != nil {
			t.Fatal(err)
		}
		if err := r.Finish(); err != nil {
			t.Fatal(err)
		}
		p, err := r.Profile()
		if err != nil {
			t.Fatal(err)
		}
		labels := make([]string, len(p.Events))
		for i := range p.Events {
			labels[i] = p.Events[i].Label
		}
		return labels
	}

	if got := profile(true); !reflect.DeepEqual(got, []string{"bias_gelu"}) {
		t.Fatalf("fused labels = %v", got)
	}
	if got := profile(false); !reflect.DeepEqual(got, []string{"addbias", "unary.gelu"}) {
		t.Fatalf("split labels = %v", got)
	}
}
