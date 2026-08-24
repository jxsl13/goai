//go:build darwin && cgo

package llamagpu

import (
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
)

func llamaBoundaryModel(b *testing.B) (*nlp.Llama, nlp.LlamaConfig) {
	b.Helper()
	cfg := nlp.LlamaConfig{
		Vocab: 1024, Ctx: 128, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		b.Fatal(err)
	}
	return m, cfg
}

// BenchmarkLlamaDecodeStepMetal measures the public F32 Decoder.Step boundary at a representative
// six-layer GQA/SwiGLU geometry. Construction and a 16-token prefill stay outside timing.
func BenchmarkLlamaDecodeStepMetal(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no gpu")
	}
	m, cfg := llamaBoundaryModel(b)
	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % cfg.Vocab
	}
	for _, tc := range []struct {
		name  string
		eager bool
	}{
		{name: "lazy"},
		{name: "eager-control", eager: true},
	} {
		b.Run(tc.name, func(b *testing.B) {
			ops := metalDecoderOps()
			ops.eagerFullDecoderScratch = tc.eager
			dec, err := newDecoder(m, ops)
			if err != nil {
				b.Fatal(err)
			}
			b.Cleanup(dec.Release)
			if _, err := dec.StepNLast(prompt, 0); err != nil {
				b.Fatal(err)
			}
			pos := len(prompt)
			if _, err := dec.Step((pos*97+11)%cfg.Vocab, pos); err != nil {
				b.Fatal(err)
			}
			pos++
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if pos >= cfg.Ctx {
					pos = len(prompt) + 1
				}
				if _, err := dec.Step((pos*97+11)%cfg.Vocab, pos); err != nil {
					b.Fatal(err)
				}
				pos++
			}
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}

// BenchmarkLlamaDecodeStepIntoMetal measures the caller-buffer sibling of Decoder.Step at the same
// geometry and warm state. It makes the host-boundary allocation contract visible beside throughput.
func BenchmarkLlamaDecodeStepIntoMetal(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no gpu")
	}
	m, cfg := llamaBoundaryModel(b)
	dec, err := New(m)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(dec.Release)
	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % cfg.Vocab
	}
	if _, err := dec.StepNLast(prompt, 0); err != nil {
		b.Fatal(err)
	}
	pos := len(prompt)
	out := make([]float32, cfg.Vocab)
	if err := dec.StepInto((pos*97+11)%cfg.Vocab, pos, out); err != nil {
		b.Fatal(err)
	}
	pos++
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if pos >= cfg.Ctx {
			pos = len(prompt) + 1
		}
		if err := dec.StepInto((pos*97+11)%cfg.Vocab, pos, out); err != nil {
			b.Fatal(err)
		}
		pos++
	}
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

// BenchmarkLlamaPrefillLastMetal measures generation prefill through StepNLast. Iterations
// overwrite the same 16 cache rows, keeping shape, commands, and cache positions identical.
func BenchmarkLlamaPrefillLastMetal(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no gpu")
	}
	m, cfg := llamaBoundaryModel(b)
	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % cfg.Vocab
	}
	for _, tc := range []struct {
		name  string
		eager bool
	}{
		{name: "lazy"},
		{name: "eager-control", eager: true},
	} {
		b.Run(tc.name, func(b *testing.B) {
			ops := metalDecoderOps()
			ops.eagerFullDecoderScratch = tc.eager
			dec, err := newDecoder(m, ops)
			if err != nil {
				b.Fatal(err)
			}
			b.Cleanup(dec.Release)
			if _, err := dec.StepNLast(prompt, 0); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := dec.StepNLast(prompt, 0); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(len(prompt)*b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}
