package llamagpu

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

func TestSampleTopKCandidatesIntoSortsPairsWithoutAllocating(t *testing.T) {
	indices := []int32{7, 2, 9, 1}
	values := []float32{0.25, 3, 1, 2}
	scratch := make([]float64, len(indices))
	sampler := nlp.NewSampler(1, nlp.WithTemperature(0), nlp.WithTopK(4))
	if got := sampleTopKCandidatesInto(sampler, indices, values, scratch); got != 2 {
		t.Fatalf("selected token = %d, want 2", got)
	}
	wantIndices := [...]int32{1, 2, 7, 9}
	wantValues := [...]float32{2, 3, 0.25, 1}
	for i := range indices {
		if indices[i] != wantIndices[i] || values[i] != wantValues[i] {
			t.Fatalf("pair[%d] = (%d,%v), want (%d,%v)", i, indices[i], values[i], wantIndices[i], wantValues[i])
		}
	}

	allocs := testing.AllocsPerRun(100, func() {
		copy(indices, []int32{7, 2, 9, 1})
		copy(values, []float32{0.25, 3, 1, 2})
		if got := sampleTopKCandidatesInto(sampler, indices, values, scratch); got != 2 {
			panic("candidate selection changed")
		}
	})
	if allocs != 0 {
		t.Fatalf("sampleTopKCandidatesInto allocations = %v, want 0", allocs)
	}
}
