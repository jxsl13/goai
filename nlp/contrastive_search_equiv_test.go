package nlp_test

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// refCosine and refMaxContextCosine are VERBATIM copies of the original serial implementation
// (single-pass dot/na/nb per pair, no norm hoist, ONE accumulator per term), used to bound the
// optimized MaxContextCosine against the shape this package started from.
//
// This comparison was bit-for-bit until the inner loop's two accumulators were split into four
// partials each for instruction-level parallelism (geomean -62.32%, ADR-01KYTPF84PEC0). That
// reassociates the sums, so the results now differ from the reference in the last ulp and the
// check is a relative tolerance. The pin was a REGRESSION PIN — it existed to prove an earlier
// parallelization changed nothing — not a guarantee to any external consumer, which is why
// weakening it was a decision that could be taken at all.
//
// The SERIAL-VERSUS-PARALLEL half stays EXACT, and that is the half worth keeping strict: the
// partition is over candidates, and each candidate's own sums are computed by the same code in the
// same order regardless of which worker runs it. If those two ever diverge it is a real bug, not
// rounding, so tolerance there would hide exactly what this test is for.
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

// relTol bounds the last-ulp drift from reassociating the two inner sums into four partials each.
// dim here is at most 64, where the accumulated difference is on the order of 1e-15; 1e-12 leaves
// three orders of headroom while still failing any real arithmetic change.
const relTol = 1e-12

func TestMaxContextCosineMatchesReference(t *testing.T) {
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
			// Against the single-accumulator reference: rounding only. The partial sums are
			// reassociated, so the tolerance has to be relative — a cosine near zero must not be
			// held to an absolute bound a denormal difference would break.
			if d := math.Abs(want[v] - gotS[v]); d > relTol*math.Max(1, math.Abs(want[v])) {
				t.Fatalf("trial %d cand %d: serial %v differs from reference %v by %g, above the "+
					"reassociation tolerance", trial, v, gotS[v], want[v], d)
			}
			// Against itself across partitions: EXACT. Same operands, same order, same association
			// per candidate — only which goroutine runs it changes.
			if gotS[v] != gotP[v] {
				t.Fatalf("trial %d cand %d: parallel %v != serial %v; the candidate partition must "+
					"not change any candidate's own arithmetic", trial, v, gotP[v], gotS[v])
			}
		}
	}
}
