package linalg_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// TestFactorLargeReconstructsExactly gates the rank-1 update's row split, which no other test in
// this package reaches: the fan-out starts at rows*rows >= 1<<14, so a factorization needs n above
// 128 before a single pivot takes it, and every existing size is below that. A mutation dropping
// the last row of each parallel band left the whole suite green.
//
// The oracle is INDEPENDENT of how the factorization works and uses only the public API: solve
// A·x = b and check that A·x comes back to b. A row the update skipped leaves stale values in U, so
// the solution misses by far more than any rounding.
//
// The tolerance is a relative residual rather than exact bits — a solve reassociates by
// construction — and 1e-9 is many orders tighter than a skipped update could survive.
func TestFactorLargeSolvesAccurately(t *testing.T) {
	const n = 160 // above 128: the first pivots clear rows*rows >= 1<<14 and fan out
	rng := rand.New(rand.NewPCG(31, 41))
	a := make([]float64, n*n)
	for i := range a {
		a[i] = rng.NormFloat64()
	}
	for i := range n { // keep it comfortably non-singular
		a[i*n+i] += float64(n)
	}
	b := make([]float64, n)
	for i := range b {
		b[i] = rng.NormFloat64()
	}
	at := tensor.FromFloat64(tensor.Shape{n, n}, append([]float64(nil), a...))
	bt := tensor.FromFloat64(tensor.Shape{n}, append([]float64(nil), b...))
	lu, err := linalg.Factor(at)
	if err != nil {
		t.Fatal(err)
	}
	x, err := lu.Solve(bt)
	if err != nil {
		t.Fatal(err)
	}
	var num, den float64
	for i := range n {
		var s float64 // (A·x)_i
		for j := range n {
			s += a[i*n+j] * x.AtF64(j)
		}
		d := s - b[i]
		num += d * d
		den += b[i] * b[i]
	}
	if rel := math.Sqrt(num / (den + 1e-300)); rel > 1e-9 {
		t.Fatalf("A·x != b at n=%d: relative residual %.3e", n, rel)
	}
}
