package linalg

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// TestSolveColsParallelArmIsReached is the internal companion to the external
// TestLUSolveParallelBitExact, and it exists for one reason that test structurally cannot cover.
//
// That test toggles GOMAXPROCS and compares bit patterns, which is the right shape — but
// solveCols also gates on cols·n·n against solveColsThreshold, and the external test picks n and
// cols at random without asserting that any trial clears it. Raise solveColsThreshold and every
// trial would quietly run serial on BOTH arms: the comparison would still pass, while proving
// nothing about the fan-out. Being in package linalg_test it cannot even name the constant, which
// is the external-test case PS6023 calls out.
//
// So this test derives its geometry FROM the threshold rather than hoping to exceed it, and
// fails loudly if the arithmetic ever stops clearing it. It is deliberately narrow — one
// geometry, not a sweep — because the shape coverage already lives in the external test; the
// only thing added here is the guarantee that the parallel arm is entered at all.
func TestSolveColsParallelArmIsReached(t *testing.T) {
	if runtime.NumCPU() < 2 {
		t.Skip("single-CPU host: solveCols cannot take its parallel branch")
	}
	const n = 64
	// cols chosen so cols·n·n strictly exceeds the gate, with a margin that survives a modest
	// retune. The guard below is what makes the choice self-checking rather than a stale constant.
	cols := solveColsThreshold/(n*n) + 2
	if cols < 2 {
		cols = 2
	}
	if cols*n*n < solveColsThreshold || cols < 2 {
		t.Fatalf("geometry no longer reaches solveCols' parallel arm: cols=%d n=%d gives %d, gate %d",
			cols, n, cols*n*n, solveColsThreshold)
	}

	rng := rand.New(rand.NewPCG(3, 14))
	ad := make([]float64, n*n)
	for i := range ad {
		ad[i] = rng.NormFloat64()
	}
	for i := range n {
		ad[i*n+i] += float64(n) // diagonally dominant, so the factorization is non-singular
	}
	a := tensor.FromFloat64(tensor.Shape{n, n}, ad)
	bd := make([]float64, n*cols)
	for i := range bd {
		bd[i] = rng.NormFloat64()
	}
	rhs := tensor.FromFloat64(tensor.Shape{n, cols}, bd)

	f, err := Factor(a)
	if err != nil {
		t.Fatal(err)
	}
	// Both arms are the SAME source, selected only by the worker count solveCols observes.
	prev := runtime.GOMAXPROCS(1)
	serial, err := f.Solve(rhs)
	runtime.GOMAXPROCS(prev)
	if err != nil {
		t.Fatal(err)
	}
	par, err := f.Solve(rhs)
	if err != nil {
		t.Fatal(err)
	}
	ss, ps := serial.Storage().F64(), par.Storage().F64()
	if len(ss) != len(ps) {
		t.Fatalf("%d values serial, %d parallel", len(ss), len(ps))
	}
	for i := range ss {
		if math.Float64bits(ss[i]) != math.Float64bits(ps[i]) {
			t.Fatalf("value %d: serial %016x != parallel %016x — the column partition moved a bit",
				i, math.Float64bits(ss[i]), math.Float64bits(ps[i]))
		}
	}
}

// TestLstsqParallelArmIsReached is the gate for the Lstsq column fan-out, which had none — and the
// absence was total: sharing one cvec across workers, the race the per-worker scratch exists to
// prevent, was caught by NO test and reported ZERO data races.
//
// The race detector was silent for a specific reason worth recording: it only reports races it
// actually observes, and no existing Lstsq test reaches solveCols' parallel arm. That arm needs
// cols·n² to clear solveColsThreshold, and the correctness tests in this package use matrices of a
// handful of rows, so every one of them runs the serial branch. A fan-out with a genuine race can
// therefore sit under a green suite AND a green race detector.
//
// The geometry below is derived from the threshold rather than guessed, and the guard fails loudly
// if it stops clearing it. Both arms are the same source selected by GOMAXPROCS.
func TestLstsqParallelArmIsReached(t *testing.T) {
	if runtime.NumCPU() < 2 {
		t.Skip("single-CPU host: solveCols cannot take its parallel branch")
	}
	const n = 48
	cols := solveColsThreshold/(n*n) + 2
	if cols < 2 {
		cols = 2
	}
	if cols*n*n < solveColsThreshold {
		t.Fatalf("geometry no longer reaches the parallel arm: cols=%d n=%d gives %d, gate %d",
			cols, n, cols*n*n, solveColsThreshold)
	}

	rng := rand.New(rand.NewPCG(19, 23))
	ad := make([]float64, n*n)
	for i := range ad {
		ad[i] = rng.NormFloat64()
	}
	for i := range n {
		ad[i*n+i] += float64(n) // well conditioned, so R has no zero on its diagonal
	}
	a := tensor.FromFloat64(tensor.Shape{n, n}, ad)
	bd := make([]float64, n*cols)
	for i := range bd {
		bd[i] = rng.NormFloat64()
	}
	b := tensor.FromFloat64(tensor.Shape{n, cols}, bd)

	prev := runtime.GOMAXPROCS(1)
	serial, err := Lstsq(a, b)
	runtime.GOMAXPROCS(prev)
	if err != nil {
		t.Fatal(err)
	}
	par, err := Lstsq(a, b)
	if err != nil {
		t.Fatal(err)
	}
	ss, ps := serial.Storage().F64(), par.Storage().F64()
	if len(ss) != len(ps) {
		t.Fatalf("%d values serial, %d parallel", len(ss), len(ps))
	}
	for i := range ss {
		if math.Float64bits(ss[i]) != math.Float64bits(ps[i]) {
			t.Fatalf("value %d: serial %016x != parallel %016x — the column partition moved a bit",
				i, math.Float64bits(ss[i]), math.Float64bits(ps[i]))
		}
	}
}
