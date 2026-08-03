package nn

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// aqlmDigest folds a run of float64s into one value by their exact bit patterns, so a golden can
// name a whole codebook set in a single constant and any changed bit shows up.
func aqlmDigest(vals ...[]float64) uint64 {
	h := uint64(1469598103934665603)
	for _, v := range vals {
		for _, x := range v {
			b := math.Float64bits(x)
			for s := 0; s < 64; s += 8 {
				h = (h ^ (b>>s)&0xff) * 1099511628211
			}
		}
	}
	return h
}

func aqlmDigestInts(h uint64, xs []int) uint64 {
	for _, x := range xs {
		u := uint64(x)
		for s := 0; s < 64; s += 8 {
			h = (h ^ (u>>s)&0xff) * 1099511628211
		}
	}
	return h
}

// refitTwoStepReference is the refit as it was written before the augmented system was
// accumulated directly: build the n*n normal matrix and the n*g right-hand side, then hand both
// to solveLinearAQLM, which is deliberately left untouched.
//
// This is the oracle for the fused form. It is an independent construction of the same
// mathematics — separate buffers, separate copy into the solver — so agreement to the bit is
// evidence about the fusion and not a statement that the code equals itself.
func refitTwoStepReference(groups [][]float64, codes []int, codebooks [][]float64, m, k, g int, ridge float64) {
	n := m * k
	ata := make([][]float64, n)
	atg := make([][]float64, n)
	for i := range n {
		ata[i] = make([]float64, n)
		atg[i] = make([]float64, g)
	}
	cols := make([]int, m)
	for i := range groups {
		for a := range m {
			cols[a] = a*k + codes[i*m+a]
		}
		for a := range m {
			pa := cols[a]
			for t := range g {
				atg[pa][t] += groups[i][t]
			}
			for b := range m {
				ata[pa][cols[b]]++
			}
		}
	}
	for p := range n {
		ata[p][p] += ridge
	}
	x := solveLinearAQLM(ata, atg)
	for p := range n {
		copy(codebooks[p], x[p])
	}
}

// TestAQLMRefitMatchesTwoStepReference pins the fused refit against the two-step form it
// replaced, bit for bit, including the case that makes the ridge term matter: codes that leave
// some codebook entries entirely unused, so their rows are singular but for the ridge.
func TestAQLMRefitMatchesTwoStepReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 17))
	for _, tc := range []struct {
		name        string
		ng, m, k, g int
		usedEntries int // codes are drawn from the first usedEntries of each codebook
		ridge       float64
	}{
		{"all-entries-used", 500, 2, 8, 4, 8, 1e-3},
		{"most-entries-unused", 60, 3, 16, 5, 2, 1e-3},
		{"single-codebook", 300, 1, 12, 6, 12, 1e-6},
	} {
		t.Run(tc.name, func(t *testing.T) {
			groups := make([][]float64, tc.ng)
			for i := range groups {
				groups[i] = make([]float64, tc.g)
				for j := range groups[i] {
					groups[i][j] = math.Sin(float64(i*tc.g+j)*0.37) * 2
				}
			}
			codes := make([]int, tc.ng*tc.m)
			for i := range codes {
				codes[i] = rng.IntN(tc.usedEntries)
			}
			n := tc.m * tc.k
			got := make([][]float64, n)
			want := make([][]float64, n)
			for i := range n {
				got[i] = make([]float64, tc.g)
				want[i] = make([]float64, tc.g)
			}
			sc := newAQLMRefitScratch(n, tc.g, tc.m)
			refitCodebooksAQLM(sc, groups, codes, got, tc.m, tc.k, tc.g, tc.ridge)
			refitTwoStepReference(groups, codes, want, tc.m, tc.k, tc.g, tc.ridge)
			for p := range n {
				for j := range tc.g {
					if math.Float64bits(got[p][j]) != math.Float64bits(want[p][j]) {
						t.Fatalf("entry %d dim %d: fused %v, two-step %v", p, j, got[p][j], want[p][j])
					}
				}
			}
			// A second refit through the SAME scratch must not see the first one's system. The
			// buffer is reused across every refinement round, and the fresh allocations it
			// replaced arrived zeroed.
			again := make([][]float64, n)
			for i := range n {
				again[i] = make([]float64, tc.g)
			}
			refitCodebooksAQLM(sc, groups, codes, again, tc.m, tc.k, tc.g, tc.ridge)
			for p := range n {
				for j := range tc.g {
					if math.Float64bits(again[p][j]) != math.Float64bits(want[p][j]) {
						t.Fatalf("reused scratch, entry %d dim %d: %v, want %v", p, j, again[p][j], want[p][j])
					}
				}
			}
		})
	}
}

// TestEncodeAQLMOutputIsFrozen is the whole-encoder golden. It covers what the per-function
// oracles cannot: the k-means assignment loop now runs its nearest-centroid search in parallel
// and folds the results into the running sums serially afterwards, and the initial residual pass
// fans out too. Both are claimed bit-identical, and the claim is only worth anything against a
// value produced by the implementation that came before.
//
// The digests below were generated from the pre-change encoder and pass on both.
func TestEncodeAQLMOutputIsFrozen(t *testing.T) {
	const wantCodes, wantBooks uint64 = 5663742417524666579, 10212476675395595104
	w := tensor.New(tensor.F64, tensor.Shape{64, 128})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = math.Sin(float64(i)*0.13)*0.6 + math.Cos(float64(i)*0.021)*0.2
	}
	q, err := EncodeAQLM(w, WithAQLMCodebooks(3), WithAQLMBits(5), WithAQLMGroupSize(8), WithAQLMIters(4))
	if err != nil {
		t.Fatal(err)
	}
	if got := aqlmDigestInts(1469598103934665603, q.Codes); got != wantCodes {
		t.Fatalf("codes digest %d, want %d", got, wantCodes)
	}
	if got := aqlmDigest(q.Codebooks...); got != wantBooks {
		t.Fatalf("codebooks digest %d, want %d", got, wantBooks)
	}
}
