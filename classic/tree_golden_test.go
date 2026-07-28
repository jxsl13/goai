package classic

import "testing"

// treeGoldenHash and forestGoldenHash are FNV-1a hashes over every predicted label,
// captured before radixByFeature's small-node path was changed.
//
// They exist because the CART sweep's ordering feeds an argmax over split gains, and the
// sort at its heart is UNSTABLE. Swapping one unstable sort for another can reorder ties
// and, if the code ever depended on tie order, select a different split and grow a
// different tree. The comment on that path argues ties are irrelevant because thresholds
// sit strictly between distinct values; these hashes are what CHECK that argument rather
// than trusting it.
//
// Hashing every prediction rather than a prefix: a change that reorders ties would move a
// split deep in one branch, which a prefix of the output would not reach.
const (
	treeGoldenHash   = 0x93ace9b515551dd
	forestGoldenHash = 0x50970aa45f40ee73
)

func TestCARTBitIdenticalToGolden(t *testing.T) {
	x, lab, _ := synthFitData(2000, 10, 3)

	tr := NewDecisionTreeClassifier(WithMaxDepth(6))
	if err := tr.Fit(x, lab); err != nil {
		t.Fatal(err)
	}
	p, err := tr.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	if h := labelHash(p); h != treeGoldenHash {
		t.Fatalf("decision-tree prediction hash %#x, want %#x — the CART sweep no longer "+
			"grows the same tree", h, treeGoldenHash)
	}

	f := NewRandomForestClassifier(WithNumTrees(12), WithSeed(9), WithForestMaxDepth(6))
	if err := f.Fit(x, lab); err != nil {
		t.Fatal(err)
	}
	pf, err := f.Predict(x)
	if err != nil {
		t.Fatal(err)
	}
	if h := labelHash(pf); h != forestGoldenHash {
		t.Fatalf("random-forest prediction hash %#x, want %#x", h, forestGoldenHash)
	}
}

func labelHash(labels []int) uint64 {
	var h uint64 = 1469598103934665603
	for _, v := range labels {
		h = (h ^ uint64(v)) * 1099511628211
	}
	return h
}
