//go:build darwin && cgo

package metal

import "testing"

func TestQ2KWordLoadRouteScope(t *testing.T) {
	if !Available() {
		t.Skip("metal device unavailable")
	}
	previousCooperative := SetQ2KCooperative(true)
	previousWord := SetQ2KWordLoad(true)
	defer func() {
		SetQ2KWordLoad(previousWord)
		SetQ2KCooperative(previousCooperative)
	}()
	if !q2KWordLoadActive(1, 2048, 3072) {
		t.Fatal("eligible M=1 projection did not select Q2_K word loads")
	}
	for _, m := range []int{2, 8, 64, 65} {
		if q2KWordLoadActive(m, 2048, 3072) {
			t.Fatalf("M=%d selected the M=1-only Q2_K word-load pipeline", m)
		}
	}
	for _, shape := range []struct{ k, n int }{{2048, 256}, {2048, 2048}, {3072, 2047}} {
		if q2KWordLoadActive(1, shape.k, shape.n) {
			t.Fatalf("K=%d N=%d selected Q2_K word loads below the measured threshold", shape.k, shape.n)
		}
	}
	SetQ2KWordLoad(false)
	if q2KWordLoadActive(1, 2048, 3072) {
		t.Fatal("disabled word-load route remained active")
	}
	SetQ2KWordLoad(true)
	SetQ2KCooperative(false)
	if q2KWordLoadActive(1, 2048, 3072) {
		t.Fatal("disabled cooperative route still selected Q2_K word loads")
	}
}
