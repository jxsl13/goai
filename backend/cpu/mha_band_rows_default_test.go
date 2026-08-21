//go:build !arm64 || !goexperiment.simd

package cpu

import "testing"

func TestMHAFwdBandRowsDefaultUnchanged(t *testing.T) {
	if mhaFwdBandRows != 30 {
		t.Fatalf("default MHA band rows = %d, want 30", mhaFwdBandRows)
	}
}
