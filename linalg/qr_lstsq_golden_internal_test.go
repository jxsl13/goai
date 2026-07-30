package linalg

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// lstsqGoldenFixture builds the deterministic, well-conditioned system the golden below solves.
func lstsqGoldenFixture(m, n, cols int) (a, b *tensor.Tensor) {
	a = tensor.New(tensor.F64, tensor.Shape{m, n})
	for i := range m {
		for j := range n {
			// diagonal-dominant, not symmetric
			v := math.Sin(float64(i*7+j*3)) + 0.25*float64((i*j)%11)
			if i == j {
				v += float64(n)
			}
			a.SetF64(v, i, j)
		}
	}
	b = tensor.New(tensor.F64, tensor.Shape{m, cols})
	for i := range m {
		for c := range cols {
			b.SetF64(math.Cos(float64(i*5+c*13))*3+float64((i+c)%7), i, c)
		}
	}
	return a, b
}

// TestLstsqBitStable freezes the exact bits of Lstsq's solution, across BOTH the column fan-out and
// the four-wide column jam inside it.
//
// Nothing else in this package could gate a restructuring of Lstsq as bit-identical. The
// correctness tests assert residuals and recovered coefficients at a tolerance, which any
// reassociation slips straight through, and TestLstsqParallelArmIsReached compares the serial and
// parallel arms — the same source selected by GOMAXPROCS, so a change to the shared body moves both
// arms together and they still agree (TEST-ORACLE-SHARES-THE-SUBJECT-001).
//
// THE GEOMETRY IS THE POINT, and an earlier version of it got this wrong. Lstsq processes columns
// colJam at a time, and solveCols hands each worker only ceil(cols/nw) of them: at cols=24 on a
// 12-way host that is 2 per worker, so the jammed loop never ran and this golden covered nothing
// but the scalar remainder — it would have passed with an arbitrarily broken jam. cols=64 gives 6
// per worker, which exercises the jam AND leaves a 2-column remainder, and the guards below fail
// loudly rather than letting either drop out again.
//
// Both GOMAXPROCS arms must produce the SAME hash: serial runs one worker over all 64 columns (16
// jams), parallel runs twelve workers over 6 each. That is what makes this a gate on the jam rather
// than on one partitioning of it.
func TestLstsqBitStable(t *testing.T) {
	const m, n = 96, 40
	const cols = 64
	const wantHash uint64 = 0x6a092164cc246cd1

	if cols*n*n < solveColsThreshold {
		t.Fatalf("geometry no longer reaches the column fan-out: cols*n² = %d < %d", cols*n*n, solveColsThreshold)
	}
	a, b := lstsqGoldenFixture(m, n, cols)

	hashOf := func(procs int) uint64 {
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
		// Replicate solveCols' partitioning so the guard describes what will actually run.
		nw := min(runtime.GOMAXPROCS(0), cols)
		if nw > 1 && cols*n*n >= solveColsThreshold {
			if csz := (cols + nw - 1) / nw; csz < colJam {
				t.Fatalf("procs=%d: %d columns per worker is below colJam=%d — the jammed loop "+
					"would not run and this golden would cover only the remainder", procs, csz, colJam)
			}
		}
		x, err := Lstsq(a, b)
		if err != nil {
			t.Fatalf("procs=%d: %v", procs, err)
		}
		if got := x.Shape(); len(got) != 2 || got[0] != n || got[1] != cols {
			t.Fatalf("shape %v, want [%d %d]", got, n, cols)
		}
		var h uint64 = 14695981039346656037
		for i := range n {
			for c := range cols {
				v := x.AtF64(i, c)
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("procs=%d: x[%d][%d] = %v", procs, i, c, v)
				}
				h = (h ^ math.Float64bits(v)) * 1099511628211
			}
		}
		return h
	}

	serial := hashOf(1)
	if wantHash == 0 {
		t.Fatalf("CAPTURE: set wantHash to %#x", serial)
	}
	if serial != wantHash {
		t.Fatalf("serial-arm hash %#x, want %#x — the Q^T b application, the jam, or the back "+
			"substitution changed a bit", serial, wantHash)
	}
	if runtime.NumCPU() < 2 {
		t.Skip("single-CPU host: the parallel partitioning cannot be compared")
	}
	if par := hashOf(runtime.NumCPU()); par != serial {
		t.Fatalf("parallel-arm hash %#x != serial %#x — the result depends on how the columns were "+
			"partitioned across workers", par, serial)
	}
}
