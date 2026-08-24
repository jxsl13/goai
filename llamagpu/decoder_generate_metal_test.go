//go:build darwin && cgo

package llamagpu

import (
	"math"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
)

var decoderGenerateTokensSink []int

func TestDecoderGenerateLogitsReusePreservesTokensAndCache(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 32, Dim: 64, Heads: 8, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 17)
	if err != nil {
		t.Fatal(err)
	}
	candidate, err := newDecoder(m, metalDecoderOps())
	if err != nil {
		t.Fatal(err)
	}
	defer candidate.Release()
	control, err := newDecoder(m, metalDecoderOps())
	if err != nil {
		t.Fatal(err)
	}
	defer control.Release()
	control.eagerGenerateResultControl = true

	prompt := []int{5, 12, 3}
	const maxNew = 8
	got, err := candidate.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	want, err := control.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("token count = %d, historical control %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("token[%d] = %d, historical control %d", i, got[i], want[i])
		}
	}

	probe := (got[len(got)-1] + 7) % cfg.Vocab
	gotLogits, err := candidate.Step(probe, len(got))
	if err != nil {
		t.Fatal(err)
	}
	wantLogits, err := control.Step(probe, len(want))
	if err != nil {
		t.Fatal(err)
	}
	for i := range gotLogits {
		if math.Float32bits(gotLogits[i]) != math.Float32bits(wantLogits[i]) {
			t.Fatalf("continuation logit[%d] = %v, historical control %v", i, gotLogits[i], wantLogits[i])
		}
	}
}

func decoderGenerateBenchmark(tb testing.TB) (dec *Decoder, prompt []int) {
	tb.Helper()
	if !metal.Available() {
		tb.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 32000, Ctx: 64, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 688, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 23)
	if err != nil {
		tb.Fatal(err)
	}
	dec, err = newDecoder(m, metalDecoderOps())
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(dec.Release)
	prompt = make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % cfg.Vocab
	}
	return dec, prompt
}

// BenchmarkDecoderGenerateAllocationsMetal isolates the allocation effect of reusing one
// Vocab-sized logits slice across an eight-token generation at TinyLlama's 32k vocabulary.
func BenchmarkDecoderGenerateAllocationsMetal(b *testing.B) {
	dec, prompt := decoderGenerateBenchmark(b)
	const maxNew = 8
	sampler := nlp.Greedy()
	for _, control := range []bool{false, true} {
		dec.eagerGenerateResultControl = control
		if _, err := dec.Generate(prompt, maxNew, sampler); err != nil {
			b.Fatal(err)
		}
	}
	for _, tc := range []struct {
		name    string
		control bool
	}{
		{name: "reuse"},
		{name: "historical", control: true},
	} {
		b.Run(tc.name, func(b *testing.B) {
			dec.eagerGenerateResultControl = tc.control
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				out, err := dec.Generate(prompt, maxNew, sampler)
				if err != nil {
					b.Fatal(err)
				}
				decoderGenerateTokensSink = out
			}
		})
	}
}

// BenchmarkDecoderGeneratePairedMetal measures complete generation in alternating order.
func BenchmarkDecoderGeneratePairedMetal(b *testing.B) {
	dec, prompt := decoderGenerateBenchmark(b)
	const maxNew = 8
	sampler := nlp.Greedy()
	for _, control := range []bool{false, true} {
		dec.eagerGenerateResultControl = control
		if _, err := dec.Generate(prompt, maxNew, sampler); err != nil {
			b.Fatal(err)
		}
	}
	var candidateDuration, controlDuration time.Duration
	run := func(control bool, elapsed *time.Duration) {
		dec.eagerGenerateResultControl = control
		start := time.Now()
		out, err := dec.Generate(prompt, maxNew, sampler)
		*elapsed += time.Since(start)
		if err != nil {
			b.Fatal(err)
		}
		decoderGenerateTokensSink = out
	}
	b.ResetTimer()
	for i := range b.N {
		if i&1 == 0 {
			run(true, &controlDuration)
			run(false, &candidateDuration)
		} else {
			run(false, &candidateDuration)
			run(true, &controlDuration)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(controlDuration)/float64(candidateDuration), "candidate/control")
}
