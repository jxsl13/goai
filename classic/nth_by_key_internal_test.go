package classic

import (
	"math/rand/v2"
	"testing"
)

// TestNthByKeyContract pins what nthByKey guarantees, and it exists because NOTHING else does.
//
// ballTree.build's split is invisible to every behavioral test in this package: the kNN search is
// exact and prunes with bounds, so a badly split — even an unsplit — tree still returns the right
// neighbours, just slower. Verified rather than assumed: removing the partition call entirely, and
// separately reversing it to put LARGER keys first, both left the whole package green. So a defect
// here would surface only as a performance regression, which is exactly the kind of thing that gets
// attributed to something else months later.
//
// The contract is the one ballTree.build relies on: after the call, every element before k has a key
// <= key[idx[k]] and every element after has a key >= it. Not a full ordering — that is the point of
// using a partition instead of a sort.
func TestNthByKeyContract(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 17))
	for _, n := range []int{1, 2, 3, 7, 8, 64, 257} {
		for _, distinct := range []int{n, 1 + n/6, 2, 1} { // dense duplicates included on purpose
			key := make([]float64, n)
			for i := range key {
				key[i] = float64(rng.IntN(distinct)) * 0.5
			}
			idx := make([]int, n)
			for i := range idx {
				idx[i] = i
			}
			rng.Shuffle(n, func(a, b int) { idx[a], idx[b] = idx[b], idx[a] })
			k := n / 2
			nthByKey(idx, key, k)

			// The permutation must be preserved: same multiset of ids, none lost or duplicated.
			seen := make([]bool, n)
			for _, id := range idx {
				if id < 0 || id >= n || seen[id] {
					t.Fatalf("n=%d distinct=%d: idx is not a permutation (bad or repeated id %d)", n, distinct, id)
				}
				seen[id] = true
			}
			pivot := key[idx[k]]
			for p := 0; p < k; p++ {
				if key[idx[p]] > pivot {
					t.Fatalf("n=%d distinct=%d: key[idx[%d]]=%v exceeds the k-th key %v",
						n, distinct, p, key[idx[p]], pivot)
				}
			}
			for p := k + 1; p < n; p++ {
				if key[idx[p]] < pivot {
					t.Fatalf("n=%d distinct=%d: key[idx[%d]]=%v is below the k-th key %v",
						n, distinct, p, key[idx[p]], pivot)
				}
			}
		}
	}
}
