package nlp

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestRadixSortMatchesComparisonSort is the parity gate for the LSD radix path in
// sortIdxDescByProb, which had none.
//
// Nothing covered it before. The radix arm is entered only at radixSortCutoff indices or more,
// and the one existing test that looks like an oracle — refSampleMirostat in
// mirostat_bitexact_test.go — cannot be one: it calls sortIdxDescByProb itself, so a radix defect
// moves its reference and the implementation identically and the comparison stays equal. A
// differential gate covers only what DIFFERS between its arms, and there the arms share the sort.
//
// Here the arms are two genuinely different implementations: the radix on IEEE-754 bits versus
// the comparison sort it replaced. The assertion is on the VALUE sequence rather than the index
// permutation, because that is what the callers consume and what the two are entitled to agree
// on: the radix is a stable counting sort while slices.SortFunc is not, so tied keys may be
// permuted differently by design. Index equality is asserted separately, only where keys are
// distinct and the permutation is therefore unique.
//
// n is swept ACROSS the cutoff so both arms of the size guard are exercised, and the guard below
// fails loudly if the cutoff is ever raised past the sweep — otherwise this test would quietly
// stop reaching the radix at all, which is the exact failure PS6023 exists to prevent.
func TestRadixSortMatchesComparisonSort(t *testing.T) {
	sizes := []int{radixSortCutoff - 1, radixSortCutoff, radixSortCutoff + 1, 4 * radixSortCutoff}
	if sizes[len(sizes)-1] <= radixSortCutoff {
		t.Fatalf("sweep no longer reaches the radix arm: max n %d vs cutoff %d",
			sizes[len(sizes)-1], radixSortCutoff)
	}

	rng := rand.New(rand.NewPCG(5, 9))
	dists := map[string]func(n int) []float64{
		// Distinct probabilities: the case the doc comment promises exact agreement for.
		"distinct": func(n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				v[i] = rng.Float64()
			}
			return v
		},
		// Heavy ties: a small alphabet makes equal keys the norm, which is where a stable
		// counting sort and an unstable pdqsort are allowed to disagree on indices.
		"ties": func(n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				v[i] = float64(rng.IntN(4)) * 0.25
			}
			return v
		},
		// Every key equal: the degenerate tie case.
		"uniform": func(n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				v[i] = 0.5
			}
			return v
		},
		// Zeros mixed in: truncateNucleus writes exact zeros back into the prob slice, so a
		// sort here routinely sees them. Float64bits(+0) is 0, which the radix relies on.
		"withzeros": func(n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				if i%3 == 0 {
					v[i] = 0
					continue
				}
				v[i] = rng.Float64()
			}
			return v
		},
		// Subnormals and the smallest normals: the low-exponent end of the bit pattern, where a
		// pass that mishandles leading zero bytes would show up.
		"subnormal": func(n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				switch i % 4 {
				case 0:
					v[i] = math.SmallestNonzeroFloat64 * float64(1+rng.IntN(8))
				case 1:
					v[i] = math.Float64frombits(uint64(rng.IntN(1 << 20)))
				case 2:
					v[i] = 0
				default:
					v[i] = rng.Float64()
				}
			}
			return v
		},
		// One dominant key and a long tail of tiny ones — the shape a low-temperature softmax
		// actually produces, and the shape top-p truncation depends on.
		"peaked": func(n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				v[i] = math.Exp(-float64(i) * 0.5)
			}
			return v
		},
	}

	for name, mk := range dists {
		for _, n := range sizes {
			key := mk(n)
			gotIdx := make([]int, n)
			wantIdx := make([]int, n)
			for i := range gotIdx {
				gotIdx[i], wantIdx[i] = i, i
			}
			sortIdxDescByProb(gotIdx, key) // radix above the cutoff
			sortIdxDescByKey(wantIdx, key) // comparison sort, always

			// The value sequence is what the callers consume: top-p walks it accumulating
			// probability mass, so any difference changes the truncation point.
			for i := range wantIdx {
				g, w := key[gotIdx[i]], key[wantIdx[i]]
				if math.Float64bits(g) != math.Float64bits(w) {
					t.Fatalf("%s n=%d position %d: radix value %v (%016x), comparison %v (%016x)",
						name, n, i, g, math.Float64bits(g), w, math.Float64bits(w))
				}
			}
			// Descending order, checked directly rather than only against the other arm — if
			// BOTH implementations were wrong in the same way the cross-check above would pass.
			for i := 1; i < n; i++ {
				if key[gotIdx[i-1]] < key[gotIdx[i]] {
					t.Fatalf("%s n=%d: not descending at %d: %v then %v",
						name, n, i, key[gotIdx[i-1]], key[gotIdx[i]])
				}
			}
			// Must be a permutation: a radix pass that drops or duplicates an index would
			// otherwise be invisible whenever the dropped key is a duplicate of a kept one.
			seen := make([]bool, n)
			for _, v := range gotIdx {
				if v < 0 || v >= n {
					t.Fatalf("%s n=%d: index %d out of range", name, n, v)
				}
				if seen[v] {
					t.Fatalf("%s n=%d: index %d appears twice", name, n, v)
				}
				seen[v] = true
			}
		}
	}
}

// TestRadixSortIndexOrderOnDistinctKeys pins the stronger claim the doc comment makes for
// DISTINCT keys: not merely the same values, but the same permutation. With no equal keys the
// descending order is unique, so radix and comparison sort must agree index for index — and any
// disagreement there is a defect in one of them rather than a tie-breaking difference.
func TestRadixSortIndexOrderOnDistinctKeys(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 23))
	for _, n := range []int{radixSortCutoff, radixSortCutoff + 1, 3 * radixSortCutoff} {
		// Distinct by construction: a permutation of i/n, so no two keys collide.
		key := make([]float64, n)
		for i := range key {
			key[i] = float64(i) / float64(n)
		}
		rng.Shuffle(n, func(i, j int) { key[i], key[j] = key[j], key[i] })

		gotIdx := make([]int, n)
		wantIdx := make([]int, n)
		for i := range gotIdx {
			gotIdx[i], wantIdx[i] = i, i
		}
		sortIdxDescByProb(gotIdx, key)
		sortIdxDescByKey(wantIdx, key)
		for i := range wantIdx {
			if gotIdx[i] != wantIdx[i] {
				t.Fatalf("n=%d position %d: radix index %d, comparison index %d (keys %v vs %v)",
					n, i, gotIdx[i], wantIdx[i], key[gotIdx[i]], key[wantIdx[i]])
			}
		}
	}
}

// TestRadixSortAscMatchesComparisonSort gates the OTHER radix behind radixSortCutoff.
//
// The same constant guards two distinct implementations — the descending radix in
// sortIdxDescByProb and the ascending one in sortIdxAscByScore — and covering only one of them
// would still silence PS6023, since that check keys on whether a test NAMES the threshold. So
// this exists to make the silence honest: the constant is not covered until every path it gates
// is.
//
// The distributions include +Inf because typicalTruncate actually produces it: score[i] is set
// to math.Inf(1) for every zero-probability token, and those entries then flow into this sort.
// Its bit pattern is the largest of any non-NaN positive, which is what makes the ascending
// bit-order argument hold for it, so it belongs in the fixture rather than being assumed.
func TestRadixSortAscMatchesComparisonSort(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 43))
	dists := map[string]func(n int) []float64{
		"distinct": func(n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				v[i] = rng.Float64()
			}
			return v
		},
		"ties": func(n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				v[i] = float64(rng.IntN(4)) * 0.25
			}
			return v
		},
		// The typicalTruncate shape: masked tokens carry +Inf so they sort last (least typical).
		"withinf": func(n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				if i%5 == 0 {
					v[i] = math.Inf(1)
					continue
				}
				v[i] = math.Abs(rng.NormFloat64())
			}
			return v
		},
		"withzeros": func(n int) []float64 {
			v := make([]float64, n)
			for i := range v {
				if i%3 == 0 {
					v[i] = 0
					continue
				}
				v[i] = rng.Float64()
			}
			return v
		},
	}
	for name, mk := range dists {
		for _, n := range []int{radixSortCutoff - 1, radixSortCutoff, radixSortCutoff + 1, 4 * radixSortCutoff} {
			key := mk(n)
			gotIdx := make([]int, n)
			wantIdx := make([]int, n)
			for i := range gotIdx {
				gotIdx[i], wantIdx[i] = i, i
			}
			sortIdxAscByScore(gotIdx, key) // radix above the cutoff
			sortIdxAscByKey(wantIdx, key)  // comparison sort, always
			for i := range wantIdx {
				g, w := key[gotIdx[i]], key[wantIdx[i]]
				if math.Float64bits(g) != math.Float64bits(w) {
					t.Fatalf("%s n=%d position %d: radix value %v (%016x), comparison %v (%016x)",
						name, n, i, g, math.Float64bits(g), w, math.Float64bits(w))
				}
			}
			for i := 1; i < n; i++ {
				if key[gotIdx[i-1]] > key[gotIdx[i]] {
					t.Fatalf("%s n=%d: not ascending at %d: %v then %v",
						name, n, i, key[gotIdx[i-1]], key[gotIdx[i]])
				}
			}
			seen := make([]bool, n)
			for _, v := range gotIdx {
				if v < 0 || v >= n || seen[v] {
					t.Fatalf("%s n=%d: index %d is out of range or duplicated", name, n, v)
				}
				seen[v] = true
			}
		}
	}
	// The ascending radix promises a STABLE tie-break — diffusionRefillOrder relies on equal
	// confidences keeping ascending index order to reproduce its documented tie-break — so that
	// is asserted directly rather than left to the value-sequence check, which cannot see it.
	n := 4 * radixSortCutoff
	key := make([]float64, n)
	for i := range key {
		key[i] = float64(i%3) * 0.5 // three distinct values, so every value has many ties
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sortIdxAscByScore(idx, key)
	for i := 1; i < n; i++ {
		if key[idx[i-1]] == key[idx[i]] && idx[i-1] > idx[i] {
			t.Fatalf("tie at positions %d,%d broke ascending index order: %d then %d — "+
				"diffusionRefillOrder's documented tie-break depends on this", i-1, i, idx[i-1], idx[i])
		}
	}
}
