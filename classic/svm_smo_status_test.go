package classic

import "testing"

func TestSVCSMOStatusMatchesIndexPredicates(t *testing.T) {
	const C = 2.0
	for _, y := range []float64{-1, 1} {
		for _, alpha := range []float64{0, 0.5, C} {
			status := svcSMOStatus(y, alpha, C)
			wantUp := (y > 0 && alpha < C) || (y < 0 && alpha > 0)
			wantLow := (y > 0 && alpha > 0) || (y < 0 && alpha < C)
			if got := status&svcSMOIUp != 0; got != wantUp {
				t.Fatalf("I_up mismatch for y=%g alpha=%g: got %t want %t", y, alpha, got, wantUp)
			}
			if got := status&svcSMOILow != 0; got != wantLow {
				t.Fatalf("I_low mismatch for y=%g alpha=%g: got %t want %t", y, alpha, got, wantLow)
			}
		}
	}
}
