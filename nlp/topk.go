package nlp

import (
	"math/bits"
	"slices"
)

// selectTopK partitions s in place so that s[:k] holds the k elements that come FIRST under
// cmp, in unspecified order within that prefix. It is the nth_element / quickselect answer to
// "which k are best", at O(len(s)) expected instead of a full sort's O(n log n).
//
// The caller almost always wants the prefix ordered too, and should sort s[:k] with the SAME
// comparator afterwards: select-then-sort-the-prefix costs O(n + k log k), and for the beam
// searches here k is a beam width against n = beams x vocabulary.
//
// BIT-SAFETY IS THE CALLER'S PRECONDITION, not this function's guarantee. Substituting
// selection for a sort reproduces the sort's result only when cmp is a STRICT TOTAL ORDER —
// with ties the set of "first k" is not unique, so a selection and a sort can legitimately
// keep different elements. Both beam searches in this package tie-break to (parent, token),
// which is unique per candidate, so their comparators qualify.
//
// That same property also rules out the classic Lomuto failure mode: with no equal keys there
// is no equal-key band for the partition to degenerate on, so the two-way partition below
// cannot hit its O(n^2) case on adversarial duplicate input.
func selectTopK[T any](s []T, k int, cmp func(a, b T) int) {
	if k <= 0 || k >= len(s) {
		return
	}
	// INTROSELECT, and the fallback is not a formality — it was added because the plain
	// quickselect measurably lost to the sort it replaced on real input from this package.
	//
	// Beam search feeds candidates built parent-outer over a vocabulary. When a model returns
	// similar logits for every prefix, the array becomes several near-identical copies of one
	// smooth curve, and median-of-three picks poor pivots on it again and again. Measured on
	// exactly that shape: selection 1442us against pdqsort's 1170us, while on varied input the
	// same selection took 61us — a 24x self-slowdown, not a small constant.
	//
	// Bounding the partition count at 2*log2(n) and finishing with a sort caps the worst case
	// at the full sort's cost plus the partitions already spent, so the selection can never be
	// dramatically worse than what it replaced, while keeping the linear behavior that makes
	// it 20x faster on well-conditioned input.
	budget := 2 * bits.Len(uint(len(s)))
	lo, hi := 0, len(s)-1
	for lo < hi {
		if budget <= 0 {
			slices.SortFunc(s[lo:hi+1], cmp)
			return
		}
		budget--
		p := partitionAt(s, lo, hi, cmp)
		switch {
		case p == k:
			return
		case p < k:
			lo = p + 1
		default:
			hi = p - 1
		}
	}
}

// partitionAt Lomuto-partitions s[lo:hi+1] around a median-of-three pivot and returns the
// pivot's final index. Median-of-three rather than a fixed pivot because a sorted or
// reverse-sorted input is the common case here, not the rare one: candidate lists are built
// parent-outer over a vocabulary, so runs of monotone score are ordinary.
func partitionAt[T any](s []T, lo, hi int, cmp func(a, b T) int) int {
	mid := lo + (hi-lo)/2
	// Order lo, mid, hi so the median sits at mid.
	if cmp(s[mid], s[lo]) < 0 {
		s[lo], s[mid] = s[mid], s[lo]
	}
	if cmp(s[hi], s[mid]) < 0 {
		s[mid], s[hi] = s[hi], s[mid]
		if cmp(s[mid], s[lo]) < 0 {
			s[lo], s[mid] = s[mid], s[lo]
		}
	}
	// Park the median at hi as the pivot.
	s[mid], s[hi] = s[hi], s[mid]
	pivot := s[hi]
	i := lo
	for j := lo; j < hi; j++ {
		if cmp(s[j], pivot) < 0 {
			s[i], s[j] = s[j], s[i]
			i++
		}
	}
	s[i], s[hi] = s[hi], s[i]
	return i
}
