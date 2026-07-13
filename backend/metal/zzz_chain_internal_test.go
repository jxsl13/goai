//go:build darwin && cgo

package metal

import (
	"math"
	"testing"
)

// §T369 (ADR-0019 Phase 2 feasibility): MPS matmul + a custom ReLU kernel chained in ONE
// command buffer over a DEVICE-RESIDENT intermediate produce the same result as the two
// ops run separately — proving the record-mode chaining Phase 2 relies on (with Metal's
// default hazard tracking ordering the MPS write before the ReLU read).
func TestChainMatMulReLU(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	const m, k, n = 5, 7, 6
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32((i%13)-6) * 0.3
	}
	for i := range b {
		b[i] = float32((i%11)-5) * 0.4
	}
	got, err := chainMatMulReLU(a, b, m, k, n)
	if err != nil {
		t.Fatal(err)
	}
	// reference: C = A·B then ReLU
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for p := 0; p < k; p++ {
				s += float64(a[i*k+p]) * float64(b[p*n+j])
			}
			if s < 0 {
				s = 0
			}
			if math.Abs(float64(got[i*n+j])-s) > 1e-4*math.Max(1, math.Abs(s)) {
				t.Fatalf("[%d,%d]: chained %v vs ref %v", i, j, got[i*n+j], s)
			}
		}
	}
}
