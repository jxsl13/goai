package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
)

func has(s []int, v int) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

// §V16 tier-1 (arXiv:2404.14469, the pooling step): max-pooling spreads a peak to its
// neighbours (kernel window), so a cluster around an important position survives top-k.
func TestSnapKVPool(t *testing.T) {
	scores := []float64{0, 0, 9, 0, 0}  // single spike at index 2
	pooled := nlp.SnapKVPool(scores, 3) // kernel 3 ⇒ indices 1,2,3 become 9
	want := []float64{0, 9, 9, 9, 0}
	for i := range want {
		if pooled[i] != want[i] {
			t.Errorf("pooled[%d]=%g want %g", i, pooled[i], want[i])
		}
	}
	// kernel ≤ 1 is a no-op.
	if got := nlp.SnapKVPool(scores, 1); got[2] != 9 || got[0] != 0 {
		t.Errorf("kernel 1 should be identity, got %v", got)
	}
}

// §V16 tier-1 (always keep the observation window): the last `win` prompt positions are
// always retained.
func TestSnapKVKeepsObservationWindow(t *testing.T) {
	// 6 prompt keys, observation window = last 2 queries; attention favors early key 0.
	obsAttn := [][]float64{
		{0.8, 0.05, 0.05, 0.05, 0.03, 0.02},
		{0.7, 0.1, 0.05, 0.05, 0.05, 0.05},
	}
	keep := nlp.SnapKVKeep(obsAttn, 4, 1) // budget 4, no pooling
	if !has(keep, 4) || !has(keep, 5) {   // window = positions 4,5
		t.Errorf("observation window (4,5) not kept: %v", keep)
	}
}

// §V16 tier-1 (top-k by importance): a high-attention prefix position is retained and a
// low-attention one is evicted under a tight budget.
func TestSnapKVKeepsImportantPrefix(t *testing.T) {
	// keys 0..5; window = last 2 (positions 4,5). Prefix 0..3: key 1 is the heavy hitter.
	obsAttn := [][]float64{
		{0.05, 0.80, 0.05, 0.02, 0.05, 0.03},
		{0.05, 0.75, 0.05, 0.05, 0.05, 0.05},
	}
	keep := nlp.SnapKVKeep(obsAttn, 3, 1) // budget 3 = window(2) + 1 prefix slot
	if !has(keep, 1) {
		t.Errorf("high-attention prefix key 1 not kept: %v", keep)
	}
	if has(keep, 3) { // key 3 has the lowest attention among the prefix
		t.Errorf("low-attention prefix key 3 should be evicted: %v", keep)
	}
}

// §V2: the retained set is exactly `budget` when the prompt exceeds it.
func TestSnapKVBudgetSize(t *testing.T) {
	obsAttn := make([][]float64, 3) // window 3
	for i := range obsAttn {
		obsAttn[i] = make([]float64, 10) // seq 10
		for j := range obsAttn[i] {
			obsAttn[i][j] = float64(j) // increasing importance
		}
	}
	keep := nlp.SnapKVKeep(obsAttn, 6, 3)
	if len(keep) != 6 {
		t.Errorf("kept %d, want budget 6", len(keep))
	}
	// ascending order preserved.
	for i := 1; i < len(keep); i++ {
		if keep[i] <= keep[i-1] {
			t.Errorf("indices not ascending: %v", keep)
		}
	}
}

// §V2 (edge): a prompt already within budget keeps everything.
func TestSnapKVUnderBudget(t *testing.T) {
	obsAttn := [][]float64{{0.5, 0.5, 0}, {0.3, 0.4, 0.3}} // seq 3
	keep := nlp.SnapKVKeep(obsAttn, 10, 7)
	if len(keep) != 3 || !has(keep, 0) || !has(keep, 1) || !has(keep, 2) {
		t.Errorf("under-budget should keep all 3, got %v", keep)
	}
}

func TestSnapKVEmpty(t *testing.T) {
	if got := nlp.SnapKVKeep(nil, 4, 7); got != nil {
		t.Errorf("empty observation window: want nil, got %v", got)
	}
}

func ExampleSnapKVKeep() {
	// 6-token prompt, observation window = last 2 (positions 4,5). Early key 1 is the heavy
	// hitter; with budget 3 SnapKV keeps it plus the window.
	obsAttn := [][]float64{
		{0.05, 0.80, 0.05, 0.02, 0.05, 0.03},
		{0.05, 0.75, 0.05, 0.05, 0.05, 0.05},
	}
	fmt.Println(nlp.SnapKVKeep(obsAttn, 3, 1))
	// Output:
	// [1 4 5]
}

// BenchmarkSnapKVKeep exercises the pooled-score sort that dominates prompt
// compression at long context. The radix path (sortIdxDescByProb) cut this from
// ~1.07ms to ~0.33ms (3.27×) at seq=8192.
func BenchmarkSnapKVKeep(b *testing.B) {
	const seq, heads = 8192, 8
	obs := make([][]float64, heads)
	for h := range obs {
		obs[h] = make([]float64, seq)
		for i := range obs[h] {
			obs[h][i] = math.Abs(math.Sin(float64(i*h) * 0.001))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = nlp.SnapKVKeep(obs, 1024, 7)
	}
}
