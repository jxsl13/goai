package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 (§T593): masked positions are excluded exactly, the result is
// unit-norm, and a nil mask averages everything.
func TestMeanPoolMaskingAndNorm(t *testing.T) {
	emb := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{
		1, 0, // kept
		0, 2, // kept
		99, 99, // masked padding
	})
	v, err := nlp.MeanPool(emb, []bool{true, true, false})
	if err != nil {
		t.Fatal(err)
	}
	// mean = (0.5, 1) → normalized (1,2)/√5.
	want := []float64{0.5 / math.Sqrt(1.25), 1 / math.Sqrt(1.25)}
	for i := range want {
		if math.Abs(v[i]-want[i]) > 1e-12 {
			t.Errorf("v[%d] = %.12f, want %.12f", i, v[i], want[i])
		}
	}
	var norm float64
	for _, x := range v {
		norm += x * x
	}
	if math.Abs(norm-1) > 1e-12 {
		t.Errorf("not unit norm: %.15f", norm)
	}
	if _, err := nlp.MeanPool(emb, []bool{false, false, false}); err == nil {
		t.Error("all-masked must error")
	}
	if _, err := nlp.MeanPool(emb, []bool{true}); err == nil {
		t.Error("mask length mismatch must error")
	}
}

// §V16 tier-1 (§T593): rerank orders by cosine, deterministic on ties, and
// rejects dimension mismatches.
func TestCosineRerank(t *testing.T) {
	q := []float64{1, 0}
	res, err := nlp.CosineRerank(q, [][]float64{
		{0, 1},  // orthogonal → 0
		{2, 0},  // parallel → 1
		{-1, 0}, // opposite → −1
		{0, -3}, // orthogonal → 0 (tie with index 0)
		{1, 1},  // 45° → ~0.707
	})
	if err != nil {
		t.Fatal(err)
	}
	order := []int{1, 4, 0, 3, 2}
	for i, want := range order {
		if res[i].Index != want {
			t.Fatalf("rank %d = candidate %d, want %d (%v)", i, res[i].Index, want, res)
		}
	}
	if res[0].Score != 1 || res[4].Score != -1 {
		t.Errorf("extreme scores wrong: %v", res)
	}
	if res[2].Score != res[3].Score || res[2].Index > res[3].Index {
		t.Error("tie must keep original index order")
	}
	if _, err := nlp.CosineRerank(q, [][]float64{{1, 2, 3}}); err == nil {
		t.Error("dim mismatch must error")
	}
}
