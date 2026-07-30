package nlp

import (
	"fmt"
	"math"
	"testing"
)

// BenchmarkRandomOrthogonal covers the modified Gram-Schmidt that builds TurboQuant's rotation.
//
// It had no benchmark, while being the single hottest own-code function across the whole nlp
// benchmark suite: 2.33s flat in a profile of all 194 benchmarks, four times the next. That is not
// because it runs often — it runs once per quantizer — but because it is O(d^3), so it dominates
// the setup of every polar/TurboQuant benchmark that constructs one.
//
// The sweep stops at 512: at d=1024 a single call is already ~1s, which says more about the cubic
// than about the code.
func BenchmarkRandomOrthogonal(b *testing.B) {
	for _, d := range []int{128, 256, 512} {
		b.Run(fmt.Sprintf("d=%d", d), func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				_ = randomOrthogonal(d, 7)
			}
		})
	}
}

// TestRandomOrthogonalBitStable freezes the exact bits of the generated matrix.
//
// The existing turboquant tests check what the matrix MEANS — that it is orthogonal, that apply and
// applyInverse round-trip — which is a tolerance-level property that survives a reassociated dot
// product or a reordered RNG draw. This pins the representation itself, so any change to the
// Gram-Schmidt loops, the allocation strategy, or the draw order moves the hash.
//
// It is the gate for treating a layout change to randomOrthogonal as bit-identical: hoisting a row
// pointer or allocating the rows from one slab must not alter a single bit, and only a golden can
// say so.
func TestRandomOrthogonalBitStable(t *testing.T) {
	const wantHash uint64 = 0x0aebe0b3a841f124
	q := randomOrthogonal(48, 7)
	if len(q) != 48 {
		t.Fatalf("%d rows, want 48", len(q))
	}
	var h uint64 = 14695981039346656037
	for _, row := range q {
		if len(row) != 48 {
			t.Fatalf("row width %d, want 48", len(row))
		}
		for _, v := range row {
			h = (h ^ math.Float64bits(v)) * 1099511628211
		}
	}
	if wantHash == 0 {
		t.Fatalf("CAPTURE: set wantHash to %#x", h)
	}
	if h != wantHash {
		t.Fatalf("randomOrthogonal hash %#x, want %#x — the matrix changed bit-for-bit", h, wantHash)
	}
}

// TestRandomOrthogonalRowsAreDisjoint guards the property a one-slab allocation could plausibly
// break and a hash could not see: the rows must not alias.
//
// A golden over the values passes just as happily whether the rows are separate allocations or
// capped views into one buffer — but it would ALSO pass if two rows accidentally shared storage and
// happened to hold equal values. Writing through one row and reading the others catches that
// directly, which matters because handing out uncapped views is the classic way to get it wrong.
func TestRandomOrthogonalRowsAreDisjoint(t *testing.T) {
	const d = 16
	q := randomOrthogonal(d, 3)
	for i := range q {
		before := make([]float64, d)
		for k := range q {
			if k != i {
				copy(before, q[k])
				break
			}
		}
		marker := float64(i) + 12345.5
		for j := range q[i] {
			q[i][j] = marker
		}
		for k := range q {
			if k == i {
				continue
			}
			for j := range q[k] {
				if q[k][j] == marker {
					t.Fatalf("writing row %d changed row %d at %d — the rows alias", i, k, j)
				}
			}
		}
		// An append to one row must not reach into the next: capped views make this a copy.
		grown := append(q[i], 999)
		if len(q) > i+1 && &grown[len(grown)-1] == &q[i+1][0] {
			t.Fatalf("append to row %d wrote into row %d — the view is not capacity-capped", i, i+1)
		}
	}
}
