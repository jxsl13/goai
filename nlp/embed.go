package nlp

import (
	"fmt"
	"math"
	"sort"

	"github.com/jxsl13/goai/tensor"
)

// MeanPool collapses per-token embeddings [seq, d] into ONE L2-normalized
// sentence vector by masked averaging — the sentence-transformers default
// pooling (§T593/§R236): only positions where mask is true contribute (pass a
// nil mask to use every token; padding positions are what the mask is for),
// the mean is taken per dimension, and the result is normalized to unit
// length so downstream cosine similarity is a plain dot product.
//
// In plain terms: a sentence's meaning vector is just the average of its word
// vectors, ignoring padding — simple, fast, and the standard baseline every
// embedding model builds on.
//
// Further reading: Reimers & Gurevych 2019, "Sentence-BERT: Sentence
// Embeddings using Siamese BERT-Networks" (the pooling's origin); Muennighoff
// et al. 2022, "MTEB: Massive Text Embedding Benchmark", for how such
// embeddings are evaluated.
func MeanPool(emb *tensor.Tensor, mask []bool) ([]float64, error) {
	if emb == nil || emb.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: MeanPool expects [seq, d] embeddings")
	}
	seq, d := emb.Shape()[0], emb.Shape()[1]
	if mask != nil && len(mask) != seq {
		return nil, fmt.Errorf("nlp: MeanPool mask length %d != seq %d", len(mask), seq)
	}
	read := embReader(emb.Contiguous())
	out := make([]float64, d)
	count := 0
	for i := range seq {
		if mask != nil && !mask[i] {
			continue
		}
		count++
		for j := range d {
			out[j] += read(i*d + j)
		}
	}
	if count == 0 {
		return nil, fmt.Errorf("nlp: MeanPool has no unmasked positions")
	}
	var norm float64
	for j := range d {
		out[j] /= float64(count)
		norm += out[j] * out[j]
	}
	if norm == 0 {
		return out, nil // the zero vector cannot be normalized; return it as-is
	}
	norm = math.Sqrt(norm)
	for j := range d {
		out[j] /= norm
	}
	return out, nil
}

// embReader returns a flat reader over a contiguous F32/F64 tensor.
func embReader(t *tensor.Tensor) func(int) float64 {
	if t.Dtype() == tensor.F64 {
		s := t.Storage().F64()
		return func(i int) float64 { return s[i] }
	}
	s := t.Storage().F32()
	return func(i int) float64 { return float64(s[i]) }
}

// RerankResult is one scored candidate from CosineRerank: the candidate's
// original index and its cosine similarity to the query.
type RerankResult struct {
	Index int     // position in the candidates slice passed to CosineRerank
	Score float64 // cosine similarity in [−1, 1]
}

// CosineRerank scores every candidate embedding against the query by cosine
// similarity and returns them best-first (ties broken by original index, so
// the order is deterministic). This is the bi-encoder reranking step used
// after retrieval; for late-interaction reranking over per-TOKEN embeddings
// use nn.MaxSim (ColBERT) instead.
//
// In plain terms: given "what the user asked" and a list of "candidate
// passages" as vectors, sort the passages by how directionally similar they
// are to the question.
//
// Further reading: Karpukhin et al. 2020, "Dense Passage Retrieval" (the
// bi-encoder retrieval pattern); Khattab & Zaharia 2020, "ColBERT", for the
// late-interaction upgrade.
func CosineRerank(query []float64, candidates [][]float64) ([]RerankResult, error) {
	if len(query) == 0 {
		return nil, fmt.Errorf("nlp: CosineRerank needs a non-empty query vector")
	}
	// The query's self-norm is invariant across candidates — hoist it out of the per-candidate
	// loop (the compiler can't: nq was re-initialized to 0 and re-summed each candidate). qNorm
	// accumulates in the identical j=0..dim-1 order over the identical operands, so score =
	// dot/√(qNorm·nc) is bit-identical to the per-candidate nq form → identical ranking.
	var qNorm float64
	for j := range query {
		qNorm += query[j] * query[j]
	}
	out := make([]RerankResult, len(candidates))
	for i, c := range candidates {
		if len(c) != len(query) {
			return nil, fmt.Errorf("nlp: candidate %d has dim %d, query has %d", i, len(c), len(query))
		}
		var dot, nc float64
		for j := range query {
			dot += query[j] * c[j]
			nc += c[j] * c[j]
		}
		score := 0.0
		if qNorm > 0 && nc > 0 {
			score = dot / math.Sqrt(qNorm*nc)
		}
		out[i] = RerankResult{Index: i, Score: score}
	}
	// (Score desc, Index asc) is already a TOTAL order — Index is unique — so stability
	// is redundant; unstable sort.Slice (pdqsort) yields the identical ranking faster.
	sort.Slice(out, func(a, b int) bool {
		if out[a].Score != out[b].Score {
			return out[a].Score > out[b].Score
		}
		return out[a].Index < out[b].Index
	})
	return out, nil
}
