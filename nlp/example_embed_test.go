package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// Pool a sentence's token embeddings (ignoring the padding position) into one
// unit vector, then rank two candidate passages against it: each RerankResult
// carries the candidate's original index and cosine score.
func ExampleMeanPool() {
	tokens := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{
		4, 0,
		0, 4,
		9, 9, // padding — masked away
	})
	query, err := nlp.MeanPool(tokens, []bool{true, true, false})
	if err != nil {
		panic(err)
	}
	results, err := nlp.CosineRerank(query, [][]float64{
		{1, 0},
		{1, 1}, // same direction as the pooled query
	})
	if err != nil {
		panic(err)
	}
	best := results[0]
	fmt.Printf("best=%d score=%.3f\n", best.Index, best.Score)
	// Output: best=1 score=1.000
}

// CosineRerank alone: candidates come back best-first with their indices.
func ExampleCosineRerank() {
	res, err := nlp.CosineRerank([]float64{1, 0}, [][]float64{{0, 1}, {3, 0}})
	if err != nil {
		panic(err)
	}
	fmt.Println(res[0].Index, res[0].Score, res[1].Index)
	// Output: 1 1 0
}
