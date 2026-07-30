package main

import "testing"

// PS3006: a full sort of a whole (n-sized) slice consumed only through a bounded top-K prefix.
func TestDetectFullSortTakeTopK(t *testing.T) {
	// prefix-reslice form: sort all n, take idx[:k].
	reslice := `package p
import "sort"
func route(probs []float64, n, k int) []int {
	idx := make([]int, n)
	for i := range idx { idx[i] = i }
	sort.SliceStable(idx, func(a, b int) bool { return probs[idx[a]] > probs[idx[b]] })
	return append([]int(nil), idx[:k]...)
}`
	if got := countCat(scanSrc(t, reslice))["full-sort-take-topk"]; got != 1 {
		t.Fatalf("reslice form: want 1 full-sort-take-topk, got %d", got)
	}
	// bounded-loop form: sort all n, read only the first topM via `for r := range topM`.
	loop := `package p
import "sort"
func top(all []int, n, topM int) (out []int) {
	sort.Slice(all, func(a, b int) bool { return all[a] < all[b] })
	out = make([]int, topM)
	for r := range topM { out[r] = all[r] }
	return out
}`
	if got := countCat(scanSrc(t, loop))["full-sort-take-topk"]; got != 1 {
		t.Fatalf("loop form: want 1 full-sort-take-topk, got %d", got)
	}
}

// Must stay silent when the sort is not actually a full-n sort or the whole slice is used.
func TestDetectFullSortTakeTopK_Silent(t *testing.T) {
	// make-size veto: the sorted slice is ALREADY topM-sized (heap select), so its sort is
	// O(K log K), not a full sort — the prefix bound equals the slice's own make size.
	alreadyK := `package p
import "sort"
func sel(all []int, topM int) []int {
	heap := make([]int, 0, topM)
	// … fill heap with the topM best …
	sort.Slice(heap, func(a, b int) bool { return heap[a] < heap[b] })
	out := make([]int, topM)
	for r := range topM { out[r] = heap[r] }
	return out
}`
	if got := countCat(scanSrc(t, alreadyK))["full-sort-take-topk"]; got != 0 {
		t.Fatalf("make-size veto: want 0, got %d", got)
	}
	// full-use veto: the sorted slice is also ranged in full.
	fullUse := `package p
import "sort"
func all(xs []int, k int) int {
	sort.Slice(xs, func(a, b int) bool { return xs[a] < xs[b] })
	_ = xs[:k]
	s := 0
	for _, v := range xs { s += v }
	return s
}`
	if got := countCat(scanSrc(t, fullUse))["full-sort-take-topk"]; got != 0 {
		t.Fatalf("full-use veto (range): want 0, got %d", got)
	}
	// return-whole veto: the caller receives the full sorted order.
	retWhole := `package p
import "sort"
func sorted(xs []int, k int) []int {
	sort.Slice(xs, func(a, b int) bool { return xs[a] < xs[b] })
	_ = xs[:k]
	return xs
}`
	if got := countCat(scanSrc(t, retWhole))["full-sort-take-topk"]; got != 0 {
		t.Fatalf("return-whole veto: want 0, got %d", got)
	}
}
