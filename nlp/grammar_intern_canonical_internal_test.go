package nlp

import (
	"math/rand/v2"
	"testing"
)

// TestInternStacksIsCanonical pins the invariant the key sort in internStacks exists for: the
// interned state id must depend on the stack SET, not on the order the stacks arrive in.
//
// Nothing covered it. That sort was converted from sort.Slice to slices.SortFunc for PS6009, and
// slices.SortFunc passes ELEMENTS where sort.Slice passed POSITIONS — transcribing the old
// keys[order[a]] expression instead of re-deriving it to keys[x] COMPILES and sorts by a
// meaningless key. The whole nlp suite passed with that transcription in place, because a wrong
// order does not produce wrong parses: the sort is a canonicalization for the intern table, so
// breaking it costs deduplication (the same stack set interns under several ids and the state
// count inflates) rather than correctness. That is a performance invariant, and it needs a test
// that checks the invariant itself instead of checking downstream parse results.
//
// The fixture uses a grammar with real ambiguity, because the property is unobservable on small
// sets: with two stacks the single comparison happens while order is still the identity
// permutation, so the correct and transcribed comparators agree. The JSON grammar below reaches
// ten stacks at one state, which is enough for the two to diverge.
func TestInternStacksIsCanonical(t *testing.T) {
	const gr = `root ::= obj
obj ::= "{" pairs "}"
pairs ::= pair | pair "," pairs | ""
pair ::= str ":" val
str ::= "\"" [a-z]+ "\""
val ::= str | num | "true" | "false" | "null"
num ::= [0-9]+`
	vocab := []string{"{", "}", "[", "]", ":", ",", "\"", "x", "y", "z", "a", "b", "1", "true", "null"}
	g, err := NewGrammarGuide(gr, vocab)
	if err != nil {
		t.Fatal(err)
	}
	// Explore so the interner holds a range of stack sets, and find the widest one.
	seen := map[int]bool{g.Start(): true}
	queue := []int{g.Start()}
	widest, widestN := -1, 0
	for len(queue) > 0 && len(seen) < 200 {
		st := queue[0]
		queue = queue[1:]
		if n := len(g.states[st].stacks); n > widestN {
			widest, widestN = st, n
		}
		for tok := range vocab {
			if nx := g.Advance(st, tok); nx >= 0 && !seen[nx] {
				seen[nx] = true
				queue = append(queue, nx)
			}
		}
	}
	if widestN < 3 {
		t.Fatalf("fixture no longer produces an ambiguous state: widest stack set is %d, need >= 3 "+
			"for a wrong comparator to be observable", widestN)
	}

	base := g.states[widest].stacks
	want := g.internStacks(append([]gStack(nil), base...))
	statesBefore := len(g.states)

	// Every permutation must land on the same id. Random shuffles rather than one reversal: a
	// reversal is a single permutation and a broken comparator can agree with the correct one on
	// it by chance, which is exactly what happened on the two-stack fixture first tried here.
	rng := rand.New(rand.NewPCG(8, 15))
	for trial := 0; trial < 64; trial++ {
		perm := append([]gStack(nil), base...)
		rng.Shuffle(len(perm), func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
		if got := g.internStacks(perm); got != want {
			t.Fatalf("trial %d: a permutation of %d stacks interned as %d, want %d — internStacks "+
				"is not canonical, so equal stack sets will occupy distinct states", trial, len(perm), got, want)
		}
	}
	// No permutation may have minted a new state: that is the dedup this sort buys, stated as an
	// assertion rather than left implicit in the id comparison above.
	if len(g.states) != statesBefore {
		t.Fatalf("interning permutations grew the state table from %d to %d", statesBefore, len(g.states))
	}
}
