package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

func BenchmarkDocumentCausalMask(b *testing.B) {
	// a packed block of 2048 tokens split into 8 documents of 256.
	docIDs := make([]int, 2048)
	for i := range docIDs {
		docIDs[i] = i / 256
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = nlp.DocumentCausalMask(docIDs)
	}
}
