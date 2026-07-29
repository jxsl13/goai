package vision

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The headBias inference cache is keyed on an exact copy of Table, because Table is the
// trainable parameter: a cache keyed on anything weaker would serve stale biases after an
// optimizer step, silently and only in the numbers. These tests pin all three behaviors — the
// hit returns the same values, a Table mutation invalidates, and a taped context never serves
// a cached tensor.
func newTestRelBias(t *testing.T, m, heads int) *swinRelBias {
	t.Helper()
	r := newSwinRelBias(tensor.F32, m, heads, 7)
	if r == nil {
		t.Fatal("newSwinRelBias returned nil")
	}
	return r
}

func biasValues(t *testing.T, r *swinRelBias, ctx *backend.Context, hh int) []float32 {
	t.Helper()
	b, err := r.headBias(ctx, hh)
	if err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), b.Contiguous().Storage().F32()...)
}

// A repeated inference call must return the same values — and the second call must be a cache
// hit, which is what makes the mutation test below meaningful.
func TestSwinHeadBiasCacheHitMatches(t *testing.T) {
	r := newTestRelBias(t, 3, 2)
	ctx := backend.NewContext()
	first := biasValues(t, r, ctx, 1)
	if r.biasCache[1] == nil {
		t.Fatal("cache not populated after an inference call — the rest of this suite would be vacuous")
	}
	second := biasValues(t, r, ctx, 1)
	for i := range first {
		if math.Float32bits(first[i]) != math.Float32bits(second[i]) {
			t.Fatalf("element %d: %08x then %08x", i, math.Float32bits(first[i]), math.Float32bits(second[i]))
		}
	}
}

// THE ONE THAT MATTERS: mutating Table, as an optimizer step does, must invalidate. A
// generation counter or a checksum could miss this; an exact comparison cannot.
func TestSwinHeadBiasCacheInvalidatesOnTableChange(t *testing.T) {
	r := newTestRelBias(t, 3, 2)
	ctx := backend.NewContext()
	before := biasValues(t, r, ctx, 0)

	ts := r.Table.Storage().F32()
	ts[0] += 1.5 // an optimizer step
	after := biasValues(t, r, ctx, 0)

	same := true
	for i := range before {
		if math.Float32bits(before[i]) != math.Float32bits(after[i]) {
			same = false
			break
		}
	}
	if same {
		t.Fatal("bias unchanged after mutating Table — the cache served a stale value, which is " +
			"the exact failure an inexact invalidation key would produce")
	}

	// And the fresh value must equal an uncached recompute.
	want := func() []float32 {
		b, err := r.computeHeadBias(ctx, 0)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), b.Contiguous().Storage().F32()...)
	}()
	for i := range want {
		if math.Float32bits(after[i]) != math.Float32bits(want[i]) {
			t.Fatalf("post-invalidation element %d = %08x, uncached recompute %08x",
				i, math.Float32bits(after[i]), math.Float32bits(want[i]))
		}
	}
}

// A taped context must never be served a cached tensor: the bias has to be a real graph node
// for the gradient to reach Table, and a tensor from an earlier step carries the wrong edges.
func TestSwinHeadBiasTapedContextBypassesCache(t *testing.T) {
	r := newTestRelBias(t, 3, 2)
	if _, err := r.headBias(backend.NewContext(), 0); err != nil { // populate
		t.Fatal(err)
	}
	cached := r.biasCache[0]
	if cached == nil {
		t.Fatal("cache not populated; test would be vacuous")
	}
	taped, err := r.headBias(autograd.NewTape().Context(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if taped == cached {
		t.Fatal("taped call returned the cached tensor — its gradient edges belong to an earlier step")
	}
}
