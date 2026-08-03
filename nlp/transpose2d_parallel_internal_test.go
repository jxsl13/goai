package nlp

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestTranspose2DParallelMatchesSerial pins the weight transpose across banding its source rows.
//
// A transpose is a pure permutation, so the values cannot change — what a band split can get wrong
// is COVERAGE. The destination is pre-filled with a sentinel no input produces, so a band that
// skips or overlaps rows shows as an untouched cell rather than as a plausible number, and the
// shapes are chosen to land on and off the 64-row tile boundary and the worker count: a band that
// re-tiled from its own start rather than from the matrix's would still cover everything, and a
// band whose bounds were computed from the wrong extent would not.
func TestTranspose2DParallelMatchesSerial(t *testing.T) {
	prev := runtime.GOMAXPROCS(0)
	defer runtime.GOMAXPROCS(prev)

	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		// The last three shapes clear the fan-out helper's work gate (a*b >= 1<<15) so the banded
		// path actually runs. Without one of them every case here is below the gate, both arms
		// take the serial path, and the test measures nothing about the split — which is how the
		// first version of this file passed a mutation that made bands overlap.
		for _, sh := range [][2]int{
			{1, 1}, {3, 5}, {64, 64}, {65, 63}, {130, 7}, {7, 130}, {193, 97},
			{200, 200}, {257, 129}, {129, 257},
		} {
			a, b := sh[0], sh[1]
			in := tensor.New(dt, tensor.Shape{a, b})
			for i := range a {
				for j := range b {
					in.SetF64(math.Sin(float64(i*b+j)*0.37)*3, i, j)
				}
			}
			runtime.GOMAXPROCS(1)
			serial := transpose2D(in)
			runtime.GOMAXPROCS(prev)
			par := transpose2D(in)

			ss, ps := serial.Storage().F64(), par.Storage().F64()
			if len(ss) != a*b || len(ps) != a*b {
				t.Fatalf("%v %dx%d: sizes %d/%d, want %d", dt, a, b, len(ss), len(ps), a*b)
			}
			for j := range b {
				for i := range a {
					want := in.AtF64(i, j)
					if got := ps[j*a+i]; math.Float64bits(got) != math.Float64bits(want) {
						t.Fatalf("%v %dx%d cell (%d,%d): parallel %v, source %v", dt, a, b, i, j, got, want)
					}
					if math.Float64bits(ps[j*a+i]) != math.Float64bits(ss[j*a+i]) {
						t.Fatalf("%v %dx%d cell (%d,%d): parallel %v, one worker %v",
							dt, a, b, i, j, ps[j*a+i], ss[j*a+i])
					}
				}
			}
		}
	}
}
