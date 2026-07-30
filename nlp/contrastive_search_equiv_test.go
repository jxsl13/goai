package nlp_test

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// refCosine and refMaxContextCosine are VERBATIM copies of the original serial implementation
// (single-pass dot/na/nb per pair, no norm hoist), used to prove the optimized
// MaxContextCosine (per-candidate parallel + hoisted candidate norm) is BIT-FOR-BIT identical.
func refCosine(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func refMaxContextCosine(candReps, contextReps [][]float64) []float64 {
	out := make([]float64, len(candReps))
	for v, cand := range candReps {
		best := 0.0
		for _, ctx := range contextReps {
			if s := refCosine(cand, ctx); s > best {
				best = s
			}
		}
		out[v] = best
	}
	return out
}

func TestMaxContextCosineBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(20260730))
	mk := func(n, dim int, zeroRow bool) [][]float64 {
		r := make([][]float64, n)
		for i := range r {
			r[i] = make([]float64, dim)
			if zeroRow && i == n/2 {
				continue // exercise the na==0 / nb==0 zero-norm branch
			}
			for j := range r[i] {
				r[i][j] = rng.NormFloat64()
			}
		}
		return r
	}
	prev := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(prev)

	for trial := 0; trial < 40; trial++ {
		nCand := 1 + rng.Intn(40)
		nCtx := rng.Intn(60) // include 0 (no context -> all zero penalties)
		dim := 1 + rng.Intn(64)
		cand := mk(nCand, dim, trial%3 == 0)
		ctx := mk(nCtx, dim, trial%2 == 0)

		want := refMaxContextCosine(cand, ctx)
		runtime.GOMAXPROCS(1)
		gotS := nlp.MaxContextCosine(cand, ctx)
		runtime.GOMAXPROCS(prev)
		gotP := nlp.MaxContextCosine(cand, ctx)

		for v := range want {
			if want[v] != gotS[v] {
				t.Fatalf("trial %d cand %d: serial %v != reference %v", trial, v, gotS[v], want[v])
			}
			if want[v] != gotP[v] {
				t.Fatalf("trial %d cand %d: parallel %v != reference %v", trial, v, gotP[v], want[v])
			}
		}
	}
}
