package nn

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

// peerBruteTopK is the reference: score ALL n² experts (score i·n+j = s1[i]+s2[j])
// and argsort. Ties break by ascending flat index, matching peerRetrieve.
func peerBruteTopK(s1, s2 []float64, n, topK int) (flat []int, scores []float64) {
	type c struct {
		idx int
		sc  float64
	}
	all := make([]c, 0, n*n)
	for i := range n {
		for j := range n {
			all = append(all, c{i*n + j, s1[i] + s2[j]})
		}
	}
	sort.SliceStable(all, func(a, b int) bool {
		if all[a].sc != all[b].sc {
			return all[a].sc > all[b].sc
		}
		return all[a].idx < all[b].idx
	})
	if topK > len(all) {
		topK = len(all)
	}
	flat = make([]int, topK)
	scores = make([]float64, topK)
	for i := range topK {
		flat[i] = all[i].idx
		scores[i] = all[i].sc
	}
	return flat, scores
}

// §V16.1 (the anchor): product-key retrieval returns EXACTLY the true top-k over
// all N=n² experts. With k' = n (retrieve all sub-keys) the candidate set is all
// N, so it is exact by construction; with the SUB-LINEAR k' = topK it is exact by
// the product-key lemma (§C21). We check both against a brute-force full-N argsort
// over many random queries, matching indices AND scores.
func TestPEERProductKeyExactness(t *testing.T) {
	const n, topK = 8, 5 // N = 64
	rng := rand.New(rand.NewPCG(7, 11))

	for _, kPrime := range []int{topK, n} { // sublinear exact, and all-sub-keys exact
		for trial := range 400 {
			s1 := make([]float64, n)
			s2 := make([]float64, n)
			for i := range n {
				s1[i] = rng.NormFloat64()
				s2[i] = rng.NormFloat64()
			}
			gotFlat, gotS1, gotS2 := peerRetrieve(s1, s2, n, topK, kPrime)
			wantFlat, wantSc := peerBruteTopK(s1, s2, n, topK)

			if len(gotFlat) != topK {
				t.Fatalf("k'=%d trial %d: got %d experts, want %d", kPrime, trial, len(gotFlat), topK)
			}
			for c := range topK {
				if gotFlat[c] != wantFlat[c] {
					t.Fatalf("k'=%d trial %d slot %d: flat idx %d != brute %d",
						kPrime, trial, c, gotFlat[c], wantFlat[c])
				}
				// Score reconstructed from the returned sub-indices must match brute.
				gotScore := s1[gotS1[c]] + s2[gotS2[c]]
				if math.Abs(gotScore-wantSc[c]) > 1e-12 {
					t.Fatalf("k'=%d trial %d slot %d: score %g != brute %g",
						kPrime, trial, c, gotScore, wantSc[c])
				}
				if gotS1[c]*n+gotS2[c] != gotFlat[c] {
					t.Fatalf("k'=%d trial %d slot %d: sub-indices (%d,%d) don't form flat %d",
						kPrime, trial, c, gotS1[c], gotS2[c], gotFlat[c])
				}
			}
		}
	}
}

// §V16.1 edge: report the k'-too-small case. The lemma guarantees exactness only
// for k' ≥ topK; with k' < topK the k'² Cartesian candidates can be FEWER than
// topK, so retrieval necessarily returns fewer experts (and can miss true top-k).
// The GLOBAL top-1 = (argmax s1, argmax s2) is always the single k'=1 candidate,
// so k'=1 still recovers the true top-1 — but nothing beyond it. NewPEER prevents
// this by raising k' so k'² ≥ topK.
func TestPEERSubKeyTooSmallEdge(t *testing.T) {
	const n, topK = 8, 4
	rng := rand.New(rand.NewPCG(3, 9))
	misses := 0
	for range 200 {
		s1 := make([]float64, n)
		s2 := make([]float64, n)
		for i := range n {
			s1[i] = rng.NormFloat64()
			s2[i] = rng.NormFloat64()
		}
		gotFlat, _, _ := peerRetrieve(s1, s2, n, topK, 1) // k'=1 ⇒ only 1 candidate
		wantFlat, _ := peerBruteTopK(s1, s2, n, topK)
		if len(gotFlat) != 1 {
			t.Fatalf("k'=1 must yield exactly 1 candidate, got %d", len(gotFlat))
		}
		if gotFlat[0] != wantFlat[0] {
			t.Fatalf("k'=1 must still recover the true top-1 (the global argmax pair)")
		}
		if len(gotFlat) < topK {
			misses++ // fewer than topK returned — the documented k'-too-small edge
		}
	}
	if misses != 200 {
		t.Fatalf("expected all 200 k'=1 trials to under-return vs topK=4, got %d", misses)
	}
	t.Logf("k'=1 (<topK=4): recovered the true top-1 in all 200 trials but returned only 1<4 experts — NewPEER raises k' to avoid this")
}

// Determinism of the pure retrieval: identical score vectors ⇒ identical routing.
func TestPEERRetrieveDeterministic(t *testing.T) {
	const n, topK, kPrime = 6, 3, 3
	rng := rand.New(rand.NewPCG(5, 5))
	s1 := make([]float64, n)
	s2 := make([]float64, n)
	for i := range n {
		s1[i] = rng.NormFloat64()
		s2[i] = rng.NormFloat64()
	}
	a, _, _ := peerRetrieve(s1, s2, n, topK, kPrime)
	b, _, _ := peerRetrieve(s1, s2, n, topK, kPrime)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("retrieval not deterministic: %v vs %v", a, b)
		}
	}
}
