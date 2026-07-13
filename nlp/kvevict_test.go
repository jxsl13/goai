package nlp_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// §R73: StreamingLLM keeps the first `sink` + last `recent` tokens and evicts the
// middle; if sink+recent covers the cache nothing is dropped.
func TestStreamingKeep(t *testing.T) {
	if got := nlp.StreamingKeep(10, 4, 3); !reflect.DeepEqual(got, []int{0, 1, 2, 3, 7, 8, 9}) {
		t.Errorf("StreamingKeep(10,4,3) = %v, want [0 1 2 3 7 8 9]", got)
	}
	// sink+recent ≥ n → keep everything
	if got := nlp.StreamingKeep(5, 4, 3); !reflect.DeepEqual(got, []int{0, 1, 2, 3, 4}) {
		t.Errorf("no eviction expected, got %v", got)
	}
	// property: size == sink+recent, first sink + last recent present
	keep := nlp.StreamingKeep(100, 4, 8)
	if len(keep) != 12 {
		t.Fatalf("size %d, want 12", len(keep))
	}
	for i := range 4 {
		if keep[i] != i {
			t.Errorf("sink %d missing", i)
		}
	}
	for i := range 8 {
		if keep[4+i] != 92+i {
			t.Errorf("recent token %d missing", 92+i)
		}
	}
}

// §R73: H2O accumulated-attention score = column sum Σ_i A[i,j].
func TestH2OScores(t *testing.T) {
	attn := [][]float64{
		{0.5, 0.3, 0.2},
		{0.1, 0.1, 0.8},
		{0.0, 0.6, 0.4},
	}
	got := nlp.H2OScores(attn)
	want := []float64{0.6, 1.0, 1.4} // column sums
	for j := range want {
		if d := got[j] - want[j]; d > 1e-12 || d < -1e-12 {
			t.Errorf("score[%d] = %v, want %v", j, got[j], want[j])
		}
	}
}

// §R73: H2O keeps recent ∪ top heavy-hitters within budget; evicted non-recent
// tokens all score ≤ any kept heavy-hitter, and the budget is respected.
func TestH2OKeep(t *testing.T) {
	scores := []float64{0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.0, 0.0}
	keep := nlp.H2OKeep(scores, 2 /*recent*/, 5 /*budget*/)
	if !reflect.DeepEqual(keep, []int{1, 3, 5, 6, 7}) {
		t.Fatalf("H2OKeep = %v, want [1 3 5 6 7]", keep)
	}
	if len(keep) != 5 {
		t.Errorf("budget not respected: %d rows", len(keep))
	}
	// recent (6,7) retained; every evicted non-recent (0,2,4) scores ≤ min kept HH
	kept := map[int]bool{}
	for _, i := range keep {
		kept[i] = true
	}
	if !kept[6] || !kept[7] {
		t.Error("recent window not fully retained")
	}
	minKeptHeavy := 1e18
	for _, i := range []int{1, 3, 5} {
		if scores[i] < minKeptHeavy {
			minKeptHeavy = scores[i]
		}
	}
	for _, i := range []int{0, 2, 4} {
		if kept[i] {
			t.Errorf("low-scorer %d should be evicted", i)
		}
		if scores[i] > minKeptHeavy {
			t.Errorf("evicted %d (%.1f) outscores a kept heavy-hitter (%.1f)", i, scores[i], minKeptHeavy)
		}
	}
	// n ≤ budget → keep all
	if got := nlp.H2OKeep([]float64{1, 2, 3}, 1, 5); !reflect.DeepEqual(got, []int{0, 1, 2}) {
		t.Errorf("under-budget should keep all, got %v", got)
	}
}

// GatherRows selects rows in order, byte-for-byte (round-trip).
func TestGatherRows(t *testing.T) {
	m := tensor.FromFloat64(tensor.Shape{4, 2}, []float64{10, 11, 20, 21, 30, 31, 40, 41})
	got := nlp.GatherRows(m, []int{0, 3})
	if !got.Shape().Equal(tensor.Shape{2, 2}) {
		t.Fatalf("shape %v", got.Shape())
	}
	want := []float64{10, 11, 40, 41}
	for i := range 2 {
		for j := range 2 {
			if got.AtF64(i, j) != want[i*2+j] {
				t.Errorf("gathered[%d,%d] = %v", i, j, got.AtF64(i, j))
			}
		}
	}
}

// EvictStreaming bounds every block's cache to sink+recent and retains the exact
// sink/recent rows across all layers.
func TestKVCacheEvictStreaming(t *testing.T) {
	mk := func(seed float64) *tensor.Tensor {
		d := make([]float64, 8*2)
		for i := range d {
			d[i] = seed + float64(i)
		}
		return tensor.FromFloat64(tensor.Shape{8, 2}, d)
	}
	c := &nlp.KVCache{K: []*tensor.Tensor{mk(0), mk(100)}, V: []*tensor.Tensor{mk(1000), mk(2000)}}
	orig := mk(0)

	c.EvictStreaming(2 /*sink*/, 3 /*recent*/)
	if c.Len() != 5 {
		t.Fatalf("cache len %d, want 5 (sink 2 + recent 3)", c.Len())
	}
	// block 0 K: rows [0,1,5,6,7] of the original
	keptRows := []int{0, 1, 5, 6, 7}
	for r, src := range keptRows {
		for j := range 2 {
			if c.K[0].AtF64(r, j) != orig.AtF64(src, j) {
				t.Errorf("evicted K[0][%d,%d] != original row %d", r, j, src)
			}
		}
	}
}

// StreamingLLM keeps a few "attention sink" tokens at the start plus a rolling
// window of recent tokens, so a fixed-size cache can decode indefinitely. Here a
// 20-token cache is bounded to 4 sinks + 6 recent = 10 entries.
func ExampleStreamingKeep() {
	keep := nlp.StreamingKeep(20, 4 /*sinks*/, 6 /*recent*/)
	fmt.Println(keep)
	// Output: [0 1 2 3 14 15 16 17 18 19]
}
