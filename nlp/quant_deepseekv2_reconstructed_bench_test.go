package nlp_test

import (
	"bytes"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// QuantDeepSeekV2 has TWO per-head attention loops and only one of them had a benchmark.
// Which one runs is a property of the loaded FILE, not a config flag: a split-form GGUF
// (attn_k_b/attn_v_b) yields the absorbed operator, while the legacy unsplit form
// (attn_kv_b) yields the fused reconstruction operator and takes attnReconstructed. The
// twelve-architecture matrix builds the split form, so the reconstructed path was reached by
// tests and by no benchmark at all — the path excluded from an earlier fusion was also the
// path excluded from measurement.
func benchQuantDeepSeekV2Legacy(b *testing.B) (*nlp.QuantDeepSeekV2, []int) {
	raw := legacyQuantDeepSeekV2GGUFBytes(b, newQuantTestDeepSeekV2())
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		b.Fatal(err)
	}
	q, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		b.Fatal(err)
	}
	if q.Blocks[0].WkvB == nil {
		b.Fatal("legacy fixture did not produce the fused WkvB, so this benchmark would " +
			"measure the absorbed path the matrix already covers")
	}
	return q, []int{1, 3, 2, 5, 4}
}

func BenchmarkQuantDeepSeekV2ReconstructedDecode(b *testing.B) {
	q, toks := benchQuantDeepSeekV2Legacy(b)
	defer q.Close()
	b.ReportAllocs()
	for b.Loop() {
		ctx := backend.NewContext()
		cache := q.NewCache()
		for pos, tok := range toks {
			if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkQuantDeepSeekV2ReconstructedPrefill(b *testing.B) {
	q, toks := benchQuantDeepSeekV2Legacy(b)
	defer q.Close()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := q.Forward(backend.NewContext(), toks); err != nil {
			b.Fatal(err)
		}
	}
}
