package nlp

import (
	"math"
	"math/rand/v2"
	"testing"
)

// docMaskDirect is the pre-templated fill, transcribed verbatim: the branchy per-element
// store with the data-dependent docIDs[j] load. Reference for the parity test below.
func docMaskDirect(docIDs []int) []float64 {
	n := len(docIDs)
	md := make([]float64, n*n)
	neg := math.Inf(-1)
	for i := range n {
		di := docIDs[i]
		row := md[i*n : i*n+n]
		for j := range n {
			if di == docIDs[j] && j <= i {
				row[j] = 0
			} else {
				row[j] = neg
			}
		}
	}
	return md
}

// TestDocumentCausalMaskTemplatedParity is the §V22 gate for the templated fill. Tolerance is
// ZERO — the change performs no arithmetic, so a relative check would pass a genuine indexing
// error, and the two values involved (+0.0 and −Inf) compare exactly.
//
// The id patterns are chosen to attack the template mapping rather than the common case.
// Contiguous runs are what PackSequences produces. Interleaved and shuffled ids break the
// assumption that a document occupies one contiguous span, which the templated form must not
// rely on. Non-monotone and negative ids check that the slot map, not the id value, indexes
// the template. Single-document and all-distinct cover both ends of the document count, the
// latter also exercising the guard's fallback when it exceeds the limit.
func TestDocumentCausalMaskTemplatedParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 11))
	patterns := map[string]func(n int) []int{
		"contiguous": func(n int) []int {
			d := make([]int, n)
			for i := range d {
				d[i] = i / 8
			}
			return d
		},
		"single": func(n int) []int { return make([]int, n) },
		"interleaved": func(n int) []int {
			d := make([]int, n)
			for i := range d {
				d[i] = i % 3
			}
			return d
		},
		"shuffled": func(n int) []int {
			d := make([]int, n)
			for i := range d {
				d[i] = rng.IntN(4)
			}
			return d
		},
		"negative_nonmonotone": func(n int) []int {
			d := make([]int, n)
			for i := range d {
				d[i] = []int{-7, 3, -7, 100, 3}[i%5]
			}
			return d
		},
		"all_distinct": func(n int) []int {
			d := make([]int, n)
			for i := range d {
				d[i] = i * 13
			}
			return d
		},
	}
	for name, mk := range patterns {
		for _, n := range []int{1, 2, 5, 17, 64, 300} {
			docIDs := mk(n)
			got := DocumentCausalMask(docIDs).Storage().F64()
			want := docMaskDirect(docIDs)
			if len(got) != len(want) {
				t.Fatalf("%s n=%d: %d elements, want %d", name, n, len(got), len(want))
			}
			for k := range want {
				if math.Float64bits(got[k]) != math.Float64bits(want[k]) {
					t.Fatalf("%s n=%d: element [%d,%d] = %v (%016x), want %v (%016x)",
						name, n, k/n, k%n, got[k], math.Float64bits(got[k]), want[k], math.Float64bits(want[k]))
				}
			}
		}
	}
}

// TestDocMaskTemplatesGuard pins both sides of the document-count guard, so a future change to
// the limit cannot silently disable the fast path or let it run where it is not cheaper.
func TestDocMaskTemplatesGuard(t *testing.T) {
	n := docMaskTemplateLimit * 2
	under := make([]int, n)
	for i := range under {
		under[i] = i % docMaskTemplateLimit // exactly at the limit
	}
	if _, _, ok := docMaskTemplates(under, n, math.Inf(-1)); !ok {
		t.Fatalf("want the templated path at exactly %d documents", docMaskTemplateLimit)
	}
	over := make([]int, n)
	for i := range over {
		over[i] = i // n distinct, well over the limit
	}
	if _, _, ok := docMaskTemplates(over, n, math.Inf(-1)); ok {
		t.Fatalf("want the fallback above %d documents", docMaskTemplateLimit)
	}
}
