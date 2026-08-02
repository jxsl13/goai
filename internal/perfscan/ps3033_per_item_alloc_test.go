package main

import (
	"strings"
	"testing"
)

func perItemAllocFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "per-item-alloc-helper" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3033_PerItemAllocHelper is the measured shape, and it is deliberately TWO levels
// deep: the allocating helper is called from another helper, which is called from a callback
// handed to a parallel-rows function. Neither allocator sits inside anything loop-shaped, which
// is why the reachability has to be a fixed point.
func TestDetectPS3033_PerItemAllocHelper(t *testing.T) {
	src := `package p

func knnWeights(nb []neighbour, w int) []float64 {
	out := make([]float64, len(nb))
	for i := range out {
		out[i] = 1 / nb[i].dist
	}
	return out
}

func (m *M) vote(row []float64) []float64 {
	nb := m.knn(row)
	w := knnWeights(nb, m.weights)
	scores := make([]float64, len(m.classes))
	for i, n := range nb {
		scores[m.yi[n.idx]] += w[i]
	}
	return scores
}

func (m *M) Predict(x [][]float64) []int {
	out := make([]int, len(x))
	parallelRows(len(x), func(i int) {
		scores := m.vote(x[i])
		out[i] = argmax(scores)
	})
	return out
}`
	fs := perItemAllocFindings(t, src)
	if len(fs) != 2 {
		t.Fatalf("%d findings, want 2 (the allocator and the helper above it)", len(fs))
	}
	names := map[string]bool{}
	for _, f := range fs {
		names[strings.Fields(f.msg)[0]] = true
	}
	if !names["knnWeights"] || !names["vote"] {
		t.Fatalf("wrong functions reported: %v", names)
	}
	// Both traps of the conversion have to survive into the advice, and so does the metric to
	// judge it by — this change moved allocations by two orders of magnitude and time by 5%.
	if !containsAll(fs[0].msg, "allocs/op AND B/op, NOT ns/op", "CLEAR", "copied out") {
		t.Fatalf("message omits the metric or a conversion trap:\n%s", fs[0].msg)
	}
}

// TestDetectPS3033_SilentOnAppliedForm pins the applied form: the buffer moved onto a
// caller-supplied scratch, so the make targets a field and the returned local is a reslice of it.
func TestDetectPS3033_SilentOnAppliedForm(t *testing.T) {
	src := `package p

func knnWeights(nb []neighbour, w int, s *scratch) []float64 {
	if cap(s.w) < len(nb) {
		s.w = make([]float64, len(nb))
	}
	out := s.w[:len(nb)]
	for i := range out {
		out[i] = 1 / nb[i].dist
	}
	return out
}

func (m *M) Predict(x [][]float64) []int {
	out := make([]int, len(x))
	parallelRows(len(x), func(i int, s *scratch) {
		out[i] = argmax(knnWeights(m.knn(x[i]), m.weights, s))
	})
	return out
}`
	if fs := perItemAllocFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a caller-supplied scratch is the applied form:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3033_SilentWhenResultIsKept pins the escape condition. A result stored into an
// element of a container declared outside the loop outlives the iteration, so reusing one buffer
// would give every row the same aliased array — the exact bug this transform can introduce. The
// fixture keeps the allocator and the per-item call and changes only what happens to the result.
func TestDetectPS3033_SilentWhenResultIsKept(t *testing.T) {
	src := `package p

func rowProba(m *M, row []float64) []float64 {
	p := make([]float64, len(m.classes))
	for i := range p {
		p[i] = m.score(row, i)
	}
	return p
}

func (m *M) PredictProba(x [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = rowProba(m, x[i])
	}
	return out
}`
	if fs := perItemAllocFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the caller keeps the result:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3033_SilentOnOneShotCaller pins that EVERY call site must be per item. A helper
// also called once per batch cannot become scratch-based without changing that caller too, and
// its allocation is not on the per-element path at all. The fixture is the positive with one
// extra, non-loop call site.
func TestDetectPS3033_SilentOnOneShotCaller(t *testing.T) {
	src := `package p

func weights(nb []neighbour) []float64 {
	out := make([]float64, len(nb))
	for i := range out {
		out[i] = 1 / nb[i].dist
	}
	return out
}

func (m *M) Predict(x [][]float64) []int {
	out := make([]int, len(x))
	for i := range x {
		out[i] = argmax(weights(m.knn(x[i])))
	}
	return out
}

func (m *M) Summary(row []float64) float64 {
	return sum(weights(m.knn(row)))
}`
	if fs := perItemAllocFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one call site is once per batch:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3033_SilentOnExported pins the visibility condition: an exported helper's callers
// need not be in this file, and one of them may keep the result. The fixture is the positive with
// the allocator exported and nothing else changed.
func TestDetectPS3033_SilentOnExported(t *testing.T) {
	src := `package p

func Weights(nb []neighbour) []float64 {
	out := make([]float64, len(nb))
	for i := range out {
		out[i] = 1 / nb[i].dist
	}
	return out
}

func (m *M) Predict(x [][]float64) []int {
	out := make([]int, len(x))
	for i := range x {
		out[i] = argmax(Weights(m.knn(x[i])))
	}
	return out
}`
	if fs := perItemAllocFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — an exported helper has callers this file cannot see:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3033_SilentOnMemoizedBuffer pins the case that forced the escape test to look past
// assignment: a helper that allocates a row, stores it in a cache through a composite literal and
// returns it. The buffer is shared with every later reader, so reusing it would corrupt the cache
// — and the store is a CALL ARGUMENT, which no LHS test can see.
//
// This is a real site, found by running the check over the tree before shipping it.
func TestDetectPS3033_SilentOnMemoizedBuffer(t *testing.T) {
	src := `package p

func cachedTrig(k key, n int, pos float64, inv []float64) ([]float64, []float64) {
	if v, ok := trigCache.Load(k); ok {
		e := v.(entry)
		return e.cs, e.sn
	}
	cs := make([]float64, len(inv))
	sn := make([]float64, len(inv))
	for i, th := range inv {
		sn[i], cs[i] = math.Sincos(pos * th)
	}
	trigCache.Store(k, entry{cs, sn})
	return cs, sn
}

func apply(x []float64, ks []key, inv []float64) {
	for i, k := range ks {
		cs, sn := cachedTrig(k, i, float64(i), inv)
		x[i] = cs[0]*x[i] + sn[0]*x[i]
	}
}`
	if fs := perItemAllocFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the buffer is handed to a cache and shared:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3033_SilentWhenHandedToAnotherCall is the memoized case without the composite
// literal: the buffer goes straight into a call that may keep it. Whether it does cannot be known
// from this file, and the cost of being wrong is a shared buffer that changes under its readers,
// so the check declines. len, cap, copy and clear are the exceptions — none of them can retain.
func TestDetectPS3033_SilentWhenHandedToAnotherCall(t *testing.T) {
	src := `package p

func rowFor(c *cache, k key, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = float64(i)
	}
	c.Store(k, out)
	return out
}

func fill(c *cache, ks []key, n int) {
	for i, k := range ks {
		_ = rowFor(c, k, n)[i]
	}
}`
	if fs := perItemAllocFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the buffer is handed to a call that may keep it:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3033_ReportsThroughCopyAndLen keeps that exception honest: a helper whose buffer
// only ever reaches len, copy and its own indices is still reported. Without this, widening the
// escape test to every call would silence the ordinary row-copy helper, which is the single most
// common shape of this finding across the tree.
func TestDetectPS3033_ReportsThroughCopyAndLen(t *testing.T) {
	src := `package p

func rowOf(src []float64, n int) []float64 {
	out := make([]float64, n)
	copy(out, src[:len(out)])
	return out
}

func total(rows [][]float64, n int) float64 {
	var s float64
	for _, r := range rows {
		s += rowOf(r, n)[0]
	}
	return s
}`
	if fs := perItemAllocFindings(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — copy and len cannot retain the buffer", len(fs))
	}
}
