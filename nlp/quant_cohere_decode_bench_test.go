package nlp_test

import (
	"bytes"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// BenchmarkQuantCohereDecodeStep covers the QUANTIZED decode layer loop, which no benchmark
// reached. BenchmarkCohereDecode builds the float model, so a change to quant_cohere.go was
// invisible to it — the allocation counts came out bit-identical across both arms of an A/B,
// which reads as "no effect" and actually meant "different code". A per-layer attrs change
// needs a benchmark that runs the layers it touched.
func BenchmarkQuantCohereDecodeStep(b *testing.B) {
	m := newQuantTestCohere()
	raw := quantCohereGGUFBytes(b, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		b.Fatal(err)
	}
	q, err := nlp.QuantCohereFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		b.Fatal(err)
	}
	defer q.Close()
	toks := []int{1, 3, 2, 5, 4}
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

// BenchmarkQuantCohereForward covers the PREFILL path. Forward and DecodeStep are separate
// layer loops with separate call sites, so a change to one is invisible to a benchmark of the
// other — the same coverage trap that made an earlier A/B report identical allocation counts
// on both arms. Both paths need a benchmark before either can be claimed.
func BenchmarkQuantCohereForward(b *testing.B) {
	m := newQuantTestCohere()
	raw := quantCohereGGUFBytes(b, m)
	rf, err := gguf.ReadRaw(bytes.NewReader(raw))
	if err != nil {
		b.Fatal(err)
	}
	q, err := nlp.QuantCohereFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		b.Fatal(err)
	}
	defer q.Close()
	prompt := []int{1, 3, 2, 5, 4}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := q.Forward(backend.NewContext(), prompt); err != nil {
			b.Fatal(err)
		}
	}
}
