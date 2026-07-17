package nlp

import (
	"testing"

	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/tensor"
)

// Benchmarks for the per-token decode hot path at REALISTIC dims (§V22 evidence).
// Two per-call costs the decode paths pay every single token:
//   - embedding one token (was a per-element AtF64/SetF64 loop);
//   - Gemma/Gemma2's tied LM head, which re-transposed the whole [vocab, dim]
//     embedding table per call before the fix cached it at load.

const (
	benchVocab = 32000 // Llama-scale vocabulary
	benchDim   = 2048
)

func benchEmbTable() *tensor.Tensor {
	emb := tensor.New(tensor.F64, tensor.Shape{benchVocab, benchDim})
	s := emb.Storage().F64()
	for i := range s {
		s[i] = float64(i%97) * 0.01
	}
	return emb
}

// BenchmarkEmbedOnePerElement is the OLD per-element embedOne pattern.
func BenchmarkEmbedOnePerElement(b *testing.B) {
	emb := benchEmbTable()
	b.ResetTimer()
	for range b.N {
		x := tensor.New(emb.Dtype(), tensor.Shape{1, benchDim})
		for j := range benchDim {
			x.SetF64(emb.AtF64(1234, j), 0, j)
		}
	}
}

// BenchmarkEmbedOneRowCopy is the typed row-copy replacement (embedRow).
func BenchmarkEmbedOneRowCopy(b *testing.B) {
	emb := benchEmbTable()
	b.ResetTimer()
	for range b.N {
		embedRow(emb, 1234, benchDim)
	}
}

// BenchmarkTiedHeadTransposePerCall measures what Gemma/Gemma2 paid per Forward
// AND per decode token before the cached-transpose fix: a full [vocab, dim]
// transpose of the embedding table.
func BenchmarkTiedHeadTransposePerCall(b *testing.B) {
	emb := benchEmbTable()
	b.ResetTimer()
	for range b.N {
		transpose2D(emb)
	}
}
