package linalg

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestLstsqBitStable freezes the exact bits of Lstsq's solution.
//
// Nothing else in this package could gate a restructuring of Lstsq as bit-identical. The
// correctness tests assert residuals and recovered coefficients at a tolerance, which any
// reassociation slips straight through, and TestLstsqParallelArmIsReached compares the serial and
// parallel arms — the same source selected by GOMAXPROCS, so a change to the shared body moves both
// arms together and they still agree (TEST-ORACLE-SHARES-THE-SUBJECT-001).
//
// The geometry exercises what a golden here is for: n=40 makes the back substitution run deep
// enough that its accumulation order matters, and cols is sized so the column fan-out clears
// solveColsThreshold, putting the PARALLEL arm under the golden rather than only the serial one.
// That is asserted below rather than asserted in prose — this test is internal precisely so it can
// read the threshold it depends on.
func TestLstsqBitStable(t *testing.T) {
	const m, n = 96, 40
	const cols = 24
	const wantHash uint64 = 0xb00a042a130996b0

	if cols*n*n < solveColsThreshold {
		t.Fatalf("geometry no longer reaches the column fan-out: cols*n² = %d < %d; the golden would "+
			"cover only the serial arm", cols*n*n, solveColsThreshold)
	}
	if runtime.NumCPU() < 2 {
		t.Log("single-CPU host: solveCols runs its serial branch; the golden still applies")
	}

	a := tensor.New(tensor.F64, tensor.Shape{m, n})
	for i := range m {
		for j := range n {
			// Deterministic, well-conditioned, and not symmetric: a diagonal-dominant term plus a
			// smooth off-diagonal fill.
			v := math.Sin(float64(i*7+j*3)) + 0.25*float64((i*j)%11)
			if i == j {
				v += float64(n)
			}
			a.SetF64(v, i, j)
		}
	}
	b := tensor.New(tensor.F64, tensor.Shape{m, cols})
	for i := range m {
		for c := range cols {
			b.SetF64(math.Cos(float64(i*5+c*13))*3+float64((i+c)%7), i, c)
		}
	}

	x, err := Lstsq(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if got := x.Shape(); len(got) != 2 || got[0] != n || got[1] != cols {
		t.Fatalf("shape %v, want [%d %d]", got, n, cols)
	}
	var h uint64 = 14695981039346656037
	for i := range n {
		for c := range cols {
			v := x.AtF64(i, c)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("x[%d][%d] = %v", i, c, v)
			}
			h = (h ^ math.Float64bits(v)) * 1099511628211
		}
	}
	if wantHash == 0 {
		t.Fatalf("CAPTURE: set wantHash to %#x", h)
	}
	if h != wantHash {
		t.Fatalf("Lstsq solution hash %#x, want %#x — the Q^T b application or the back "+
			"substitution changed a bit", h, wantHash)
	}
}
