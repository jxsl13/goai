package main

import (
	"strings"
	"testing"
)

// permMsg returns the PS6009 message for src, or "" if the check did not fire.
func permMsg(t *testing.T, src string) string {
	t.Helper()
	for _, f := range scanSrc(t, src) {
		if f.category == "reflect-swapper-sort" {
			return f.msg
		}
	}
	return ""
}

const permMarker = "INDEX-PERMUTATION"

// TestPS6009ClassifiesPermutationSort is the positive floor. The comparator indexes another
// slice THROUGH the sorted slice's element, so both the position and the element are ints and a
// transcribed conversion type-checks while sorting by a meaningless key. That is the one PS6009
// conversion that fails silently, so the message has to say so at the site rather than issuing a
// generic caution everywhere.
func TestPS6009ClassifiesPermutationSort(t *testing.T) {
	src := `package p

import "sort"

func f(col []int, key []float64) {
	sort.Slice(col, func(a, b int) bool { return key[col[a]] < key[col[b]] })
}`
	msg := permMsg(t, src)
	if msg == "" {
		t.Fatal("PS6009 did not fire at all")
	}
	if !strings.Contains(msg, permMarker) {
		t.Fatalf("permutation sort not classified: %s", msg)
	}
	// The message must name the two slices, since the remedy is the specific rewrite
	// key[col[x]] -> key[x] and a generic warning is what let this slip through twice.
	if !strings.Contains(msg, "key[col[x]]") || !strings.Contains(msg, "key[x]") {
		t.Fatalf("message does not spell out the rewrite: %s", msg)
	}
}

// TestPS6009ClassifiesOffsetPermutation covers the form that was actually shipped wrong: the
// index carries an offset, gsc[grp[a]-base], so a detector matching only the exact two-level
// OUTER[INNER[param]] shape would miss the real instance while passing the tidy fixture above.
func TestPS6009ClassifiesOffsetPermutation(t *testing.T) {
	src := `package p

import "sort"

func f(grp []int, gsc []float64, base int) {
	sort.SliceStable(grp, func(a, b int) bool { return gsc[grp[a]-base] < gsc[grp[b]-base] })
}`
	msg := permMsg(t, src)
	if !strings.Contains(msg, permMarker) {
		t.Fatalf("offset permutation form not classified: %s", msg)
	}
}

// Each case below is the floor for one clause of the classification. They are subtests so that
// breaking a single clause reddens exactly the one guarding it; sharing a t.Fatalf would make
// every mutation report the same failure.
//
// "Not classified" does NOT mean silent: PS6009 still fires on all of these. What must be absent
// is the permutation warning, because for these the transcribed rewrite does not compile and the
// compiler already catches it — claiming otherwise would train the reader to ignore the warning.
func TestPS6009DoesNotClassify(t *testing.T) {
	plain := func(name, src string) {
		t.Run(name, func(t *testing.T) {
			msg := permMsg(t, src)
			if msg == "" {
				t.Fatalf("%s: PS6009 should still fire on any sort.Slice", name)
			}
			if strings.Contains(msg, permMarker) {
				t.Fatalf("%s: wrongly classified as an index permutation: %s", name, msg)
			}
		})
	}

	// CLAUSE: the element must be NESTED inside another index. Here s[i] is compared directly,
	// so the index is a bare parameter, the element is the key itself, and a transcribed s[x]
	// would not even type-check as an index. Contrast TestPS6009ClassifiesSelfPermutation, where
	// the same slice IS indexed through itself and must be classified.
	plain("self-indexed", `package p

import "sort"

func f(s []int) {
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
}`)

	// CLAUSE: the element must be used as an INDEX. A direct field comparison makes the element
	// a struct, so a transcribed ms[x].r does not compile.
	plain("direct-field", `package p

import "sort"

type e struct{ r int }

func f(ms []e) {
	sort.Slice(ms, func(a, b int) bool { return ms[a].r < ms[b].r })
}`)

	// CLAUSE: the inner index must be a comparator PARAMETER. Indexing by a captured variable
	// is a constant lookup within the comparison, not a per-element key.
	plain("indexed-by-capture", `package p

import "sort"

func f(col []int, key []float64, k int) {
	sort.Slice(col, func(a, b int) bool { return key[col[k]] < float64(a+b) })
}`)

	// CLAUSE: the comparator must be a function LITERAL. A named comparator has no parameter
	// list here to bind, and its body is not available at this call site.
	plain("named-comparator", `package p

import "sort"

func f(col []int, less func(i, j int) bool) {
	sort.Slice(col, less)
}`)

	// CLAUSE: the inner indexed slice must be the one being SORTED. Indexing through a
	// different permutation array is a POSITIONAL lookup that slices.SortFunc cannot express at
	// all, so the message's "it must become key[x]" rewrite would be actively wrong there.
	plain("indexed-through-other-slice", `package p

import "sort"

func f(col []int, other []int, key []float64) {
	sort.Slice(col, func(a, b int) bool { return key[other[a]] < key[other[b]] })
}`)

	// CLAUSE: the sorted operand must be a plain identifier — otherwise there is no name the
	// message could spell a rewrite for. The nesting is present here, so this isolates the
	// identifier clause rather than being excluded by the nesting requirement.
	plain("non-ident-operand", `package p

import "sort"

func f(col []int, key []float64) {
	sort.Slice(col[1:], func(a, b int) bool { return key[col[a]] < key[col[b]] })
}`)
}

// TestPS6009ClassifiesSelfPermutation covers a slice whose elements index ITSELF. It has the
// property that matters — the element is an integer index — so a transcribed s[s[x]] compiles
// and sorts by the wrong key, exactly like the cross-slice case. An earlier version of the
// detector excluded outer == inner; this floor is what makes removing that exclusion stick.
func TestPS6009ClassifiesSelfPermutation(t *testing.T) {
	src := `package p

import "sort"

func f(s []int) {
	sort.Slice(s, func(a, b int) bool { return s[s[a]] < s[s[b]] })
}`
	if msg := permMsg(t, src); !strings.Contains(msg, permMarker) {
		t.Fatalf("self-permutation not classified: %s", msg)
	}
}
