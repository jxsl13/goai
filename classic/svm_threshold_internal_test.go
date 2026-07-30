package classic

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestSVCParThresholdArmsAgree pins the two arms of svcParallelColumn against each other.
//
// The helper splits a kernel column's entries across the pool, and each entry writes its own slot
// from read-only inputs, so the partition is supposed to change no value. Nothing forced that:
// PS6023 reported that no test named svcParThreshold, and the correctness fixtures in this package
// are small enough to stay serial, so the fanned-out arm was exercised only incidentally.
//
// Both arms are the SAME source selected by the threshold, so any difference is attributable to the
// partition. Predictions are compared bit-for-bit rather than at a tolerance: a partition is either
// value-neutral or it is a bug, and a tolerance would hide the reordering this exists to catch.
//
// All three kernels are swept. They differ in what each entry computes — the RBF exponentiates,
// the polynomial powers, the linear is a plain dot product — and a partition bug visible in only
// one of them would otherwise slip through.
func TestSVCParThresholdArmsAgree(t *testing.T) {
	saved := svcParThreshold
	defer func() { svcParThreshold = saved }()

	rng := rand.New(rand.NewPCG(31, 4))
	const n, d = 120, 8
	x := make([][]float64, n)
	y := make([]float64, n)
	for i := range x {
		row := make([]float64, d)
		for j := range row {
			row[j] = rng.NormFloat64()
		}
		x[i] = row
		if row[0]+row[1] > 0 {
			y[i] = 1
		} else {
			y[i] = -1
		}
	}
	for _, k := range []SVMKernel{SVMKernelLinear, SVMKernelRBF, SVMKernelPoly} {
		fit := func(gate int) []float64 {
			svcParThreshold = gate
			m := NewSVC(WithSVMKernel(k), WithSVMC(1), WithSVMMaxIter(60), WithSVMSeed(3))
			if err := m.Fit(x, y); err != nil {
				t.Fatalf("kernel %v: %v", k, err)
			}
			out, err := m.Predict(x)
			if err != nil {
				t.Fatalf("kernel %v: %v", k, err)
			}
			return out
		}
		serial, par := fit(1<<30), fit(0)
		var pos int
		for i := range serial {
			if math.Float64bits(serial[i]) != math.Float64bits(par[i]) {
				t.Fatalf("kernel %v prediction %d: serial %v, parallel %v — the column partition "+
					"changed a value", k, i, serial[i], par[i])
			}
			if serial[i] > 0 {
				pos++
			}
		}
		// Without this the comparison could pass on a degenerate fit that predicts one class
		// everywhere, which would agree across the arms for the wrong reason.
		if pos == 0 || pos == len(serial) {
			t.Fatalf("kernel %v: fit predicts a single class for every sample; the arms would "+
				"agree whatever the partition did", k)
		}
	}
}
