//go:build darwin && cgo

package metal

import "testing"

func TestQ5KWideLoadRouteScope(t *testing.T) {
	if !Available() {
		t.Skip("metal device unavailable")
	}
	previousCooperative := SetQ5KCooperative(true)
	previousWideLoad := SetQ5KWideLoad(true)
	defer func() {
		SetQ5KWideLoad(previousWideLoad)
		SetQ5KCooperative(previousCooperative)
	}()
	if !q5KWideLoadActive(1, 2048, 3072) {
		t.Fatal("eligible M=1 projection did not select the supported Q5_K vector-load pipeline")
	}
	for _, m := range []int{2, 8, 64, 65} {
		if q5KWideLoadActive(m, 2048, 3072) {
			t.Fatalf("M=%d selected the M=1-only Q5_K vector-load pipeline", m)
		}
	}
	for _, shape := range []struct{ k, n int }{{2048, 256}, {2048, 2048}, {3072, 2047}} {
		if q5KWideLoadActive(1, shape.k, shape.n) {
			t.Fatalf("K=%d N=%d selected Q5_K vector loads below the measured threshold", shape.k, shape.n)
		}
	}
	SetQ5KCooperative(false)
	if q5KWideLoadActive(1, 2048, 3072) {
		t.Fatal("disabled cooperative route still selected Q5_K vector loads")
	}
}
