package classic

import (
	"fmt"
	"math"
	"testing"
)

// Kernel columns are now evaluated in parallel. Each entry is an independent kernel call
// writing its own slot, so this is bit-identical by construction — but SMO's ITERATION PATH
// depends on those values, so any divergence at all would compound into a different support
// vector set rather than a small numeric difference. That makes the fitted model the right
// thing to pin.
//
// n=4000 crosses the work threshold and runs parallel; n=1000 stays serial. Both are here so
// the test covers the two paths, and so a threshold change cannot silently move a case out of
// coverage.
func TestSVCFitBitStableAcrossParallelThreshold(t *testing.T) {
	cases := []struct {
		n, d     int
		sum, xor uint64
	}{
		{1000, 20, 0x3fe4c226a2b12ba4, 0x80424d0ce4018540}, // serial path
		{4000, 20, 0x3fe91af93bd19efc, 0xbf9e92567edbab4e}, // parallel path
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("n%d", tc.n), func(t *testing.T) {
			x, y := svmBenchData(tc.n, tc.d)
			m := NewSVC(WithSVMKernel(SVMKernelRBF))
			if err := m.Fit(x, y); err != nil {
				t.Fatal(err)
			}
			var sum float64
			var xr uint64
			for _, v := range m.DualCoef() {
				sum += v
				xr ^= math.Float64bits(v)
			}
			sum += m.Intercept()
			xr ^= math.Float64bits(m.Intercept())
			if got := math.Float64bits(sum); got != tc.sum || xr != tc.xor {
				t.Errorf("n=%d fitted model digest = %016x/%016x, want %016x/%016x — a kernel column "+
					"value changed, and SMO compounds that into a different support vector set",
					tc.n, got, xr, tc.sum, tc.xor)
			}
		})
	}
}
