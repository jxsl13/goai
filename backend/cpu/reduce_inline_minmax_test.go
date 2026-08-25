package cpu

import (
	"math"
	"testing"
)

func TestExactMinMaxMatchesMathBits(t *testing.T) {
	values := []float64{
		0,
		math.Copysign(0, -1),
		1,
		-1,
		math.Inf(1),
		math.Inf(-1),
		math.Float64frombits(0x7ff8000000000001),
		math.Float64frombits(0x7ff8000000000042),
		math.Float64frombits(0xfff8000000000042),
	}
	for _, x := range values {
		for _, y := range values {
			if got, want := exactMax(x, y), math.Max(x, y); math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("max(%016x,%016x): got %016x want %016x", math.Float64bits(x), math.Float64bits(y), math.Float64bits(got), math.Float64bits(want))
			}
			if got, want := exactMin(x, y), math.Min(x, y); math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("min(%016x,%016x): got %016x want %016x", math.Float64bits(x), math.Float64bits(y), math.Float64bits(got), math.Float64bits(want))
			}
		}
	}
}
