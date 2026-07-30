package linalg_test

import (
	"math/rand"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// LU.Solve fans its per-RHS-column substitution over GOMAXPROCS workers. Each column is
// independent (writes only its own output column, reads a private forward-substitution
// scratch + the shared read-only factorization), so the parallel result must be BYTE-FOR-BYTE
// identical to the single-worker serial result. This locks that invariant by solving the same
// system at GOMAXPROCS=1 and GOMAXPROCS=N and requiring bit-exact equality.
func TestLUSolveParallelBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(20260730))
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)

	for trial := 0; trial < 60; trial++ {
		n := 2 + rng.Intn(48)          // 2..49
		cols := 1 + rng.Intn(3*n)      // include cols < n, = n, > n and the vec path (cols==1)
		ad := make([]float64, n*n)
		for i := range ad {
			ad[i] = rng.NormFloat64()
		}
		for i := 0; i < n; i++ { // diagonally dominant → non-singular
			ad[i*n+i] += float64(n)
		}
		a := tensor.FromFloat64(tensor.Shape{n, n}, ad)
		bd := make([]float64, n*cols)
		for i := range bd {
			bd[i] = rng.NormFloat64()
		}
		rhs := tensor.FromFloat64(tensor.Shape{n, cols}, bd)

		f, err := linalg.Factor(a)
		if err != nil {
			t.Fatalf("trial %d: factor: %v", trial, err)
		}

		runtime.GOMAXPROCS(1)
		serial, err := f.Solve(rhs)
		if err != nil {
			t.Fatalf("trial %d: serial solve: %v", trial, err)
		}
		runtime.GOMAXPROCS(prev)
		par, err := f.Solve(rhs)
		if err != nil {
			t.Fatalf("trial %d: parallel solve: %v", trial, err)
		}

		ss, ps := serial.Storage().F64(), par.Storage().F64()
		if len(ss) != len(ps) {
			t.Fatalf("trial %d: length mismatch %d vs %d", trial, len(ss), len(ps))
		}
		for i := range ss {
			if ss[i] != ps[i] { // bit-exact, not approximate
				t.Fatalf("trial %d n=%d cols=%d idx=%d: serial %v != parallel %v", trial, n, cols, i, ss[i], ps[i])
			}
		}
	}
}
