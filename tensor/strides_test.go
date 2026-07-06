package tensor

import (
	"math/rand/v2"
	"testing"
)

func TestRowMajorStridesKnown(t *testing.T) {
	cases := []struct {
		s    Shape
		want Strides
	}{
		{Shape{}, Strides{}},
		{Shape{5}, Strides{1}},
		{Shape{2, 3}, Strides{3, 1}},
		{Shape{2, 3, 4}, Strides{12, 4, 1}},
		{Shape{2, 0, 3}, Strides{0, 3, 1}}, // structural, well-defined for empty
	}
	for _, c := range cases {
		got := RowMajorStrides(c.s)
		if !equalInts(got, c.want) {
			t.Errorf("RowMajorStrides(%v) = %v, want %v", c.s, got, c.want)
		}
	}
}

func TestIsContiguous(t *testing.T) {
	s := Shape{2, 3}
	if !RowMajorStrides(s).IsContiguous(s) {
		t.Error("row-major strides must be contiguous")
	}
	// transpose view of (2,3) → shape (3,2), strides (1,3): not contiguous
	if (Strides{1, 3}).IsContiguous(Shape{3, 2}) {
		t.Error("transposed strides must not be contiguous")
	}
}

func TestOffsetRankMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Offset with mismatched rank must panic")
		}
	}()
	Offset(0, Strides{3, 1}, []int{0}) // 1 index vs 2 strides
}

// Property (§V6): for a contiguous row-major tensor, walking every flat position
// 0..Numel-1, unraveling it, and re-raveling via Offset must return the same
// position. This proves Unravel and Offset are exact inverses.
func TestOffsetUnravelRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for range 500 {
		s := randShape(rng, 4, 5)
		if s.IsEmpty() {
			continue // no valid positions
		}
		st := RowMajorStrides(s)
		n := s.Numel()
		for pos := range n {
			idx := Unravel(pos, s)
			if got := Offset(0, st, idx); got != pos {
				t.Fatalf("shape %v pos %d: Unravel→%v→Offset %d", s, pos, idx, got)
			}
		}
	}
}

// Property (§V6): for a contiguous tensor, the set of offsets over all positions
// is exactly {0..Numel-1} — a bijection with no gaps or collisions.
func TestContiguousOffsetsAreBijection(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for range 300 {
		s := randShape(rng, 4, 5)
		if s.IsEmpty() {
			continue
		}
		st := RowMajorStrides(s)
		n := s.Numel()
		seen := make([]bool, n)
		for pos := range n {
			off := Offset(0, st, Unravel(pos, s))
			if off < 0 || off >= n {
				t.Fatalf("shape %v: offset %d out of range [0,%d)", s, off, n)
			}
			if seen[off] {
				t.Fatalf("shape %v: offset %d seen twice (collision)", s, off)
			}
			seen[off] = true
		}
	}
}

// randShape builds a random valid shape with rank in [0,maxRank] and each dim in
// [1,maxDim]. Dims are kept >=1 so tensors are non-empty for iteration tests;
// empty-tensor behavior is covered by table-driven edge tests.
func randShape(rng *rand.Rand, maxRank, maxDim int) Shape {
	rank := rng.IntN(maxRank + 1)
	s := make(Shape, rank)
	for i := range s {
		s[i] = 1 + rng.IntN(maxDim)
	}
	return s
}

func equalInts[T ~int](a, b []T) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
