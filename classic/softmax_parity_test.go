package classic

import (
	"fmt"
	"math"
	"testing"
)

// The Hessian Gram accumulation was interchanged to put the feature index outside the sample
// loop and then parallelized over it. That is bit-identical only because each Gram element
// accumulates over exactly ONE axis — the sample index — so its sum stays ascending however the
// other indices are ordered. Parallelizing over samples instead would reassociate.
//
// This pins the learned weights exactly. The values were captured from the serial
// implementation before the interchange; any reassociation moves them.
func TestSoftmaxFitBitStable(t *testing.T) {
	cases := []struct {
		n, d, k, steps int
		sum, xor       uint64
	}{
		// the benchmark's shape
		{4000, 20, 3, 40, 0xc003fd8710e5b214, 0x80c96e1a4e81a75f},
		// more classes: numPairs 6, so several pair blocks
		{300, 7, 4, 30, 0x4010d397886be23b, 0x7ffc4e455e7bc83a},
		// kEff 1: a single pair block, the degenerate case
		{200, 3, 2, 25, 0xc003e1c2928c4e20, 0x8010aad76cb4e54a},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("n%d_d%d_k%d", tc.n, tc.d, tc.k), func(t *testing.T) {
			x, y := softmaxSynthetic(tc.n, tc.d, tc.k, 1)
			var m SoftmaxRegression
			if err := m.Fit(x, y, tc.k, tc.steps, 0.05); err != nil {
				t.Fatal(err)
			}
			var sum float64
			var xr uint64
			for _, v := range m.W.Contiguous().Storage().F64() {
				sum += v
				xr ^= math.Float64bits(v)
			}
			for _, v := range m.B.Contiguous().Storage().F64() {
				sum += v
				xr ^= math.Float64bits(v)
			}
			if got := math.Float64bits(sum); got != tc.sum || xr != tc.xor {
				t.Errorf("weights digest = %016x/%016x, want %016x/%016x — the Gram accumulation "+
					"reassociated; each element must still sum over the sample index ascending",
					got, xr, tc.sum, tc.xor)
			}
		})
	}
}
