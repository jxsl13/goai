//go:build darwin && cgo

package llamagpu

import (
	"testing"
	"time"

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

// BenchmarkLlamaPrefillLastIntoMetal measures the allocation-free caller-buffer boundary at the
// same shape as BenchmarkLlamaPrefillLastMetal.
func BenchmarkLlamaPrefillLastIntoMetal(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no gpu")
	}
	m, cfg := llamaBoundaryModel(b)
	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % cfg.Vocab
	}
	dec, err := newDecoder(m, metalDecoderOps())
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(dec.Release)
	out := make([]float32, cfg.Vocab)
	if err := dec.StepNLastInto(prompt, 0, out); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := dec.StepNLastInto(prompt, 0, out); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(len(prompt)*b.N)/b.Elapsed().Seconds(), "tok/s")
}

// BenchmarkLlamaPrefillHostStagingMetal compares retained high-water embedding staging with the
// historical per-prefill allocation in one binary while keeping StepNLast's public result allocation.
func BenchmarkLlamaPrefillHostStagingMetal(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no gpu")
	}
	m, cfg := llamaBoundaryModel(b)
	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % cfg.Vocab
	}
	for _, tc := range []struct {
		name    string
		control bool
	}{
		{name: "retained"},
		{name: "allocating-control", control: true},
	} {
		b.Run(tc.name, func(b *testing.B) {
			ops := metalDecoderOps()
			ops.eagerPrefillHostStaging = tc.control
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

// BenchmarkLlamaPrefillHostStagingPairedMetal times retained and allocating-control decoders inside
// each iteration and reverses their order. The custom ratio removes process startup and GPU warm-up
// bias from the throughput promotion gate; values at or above 0.97 pass.
func BenchmarkLlamaPrefillHostStagingPairedMetal(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal: no gpu")
	}
	m, cfg := llamaBoundaryModel(b)
	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % cfg.Vocab
	}
	retained, err := newDecoder(m, metalDecoderOps())
	if err != nil {
		b.Fatal(err)
	}
	defer retained.Release()
	controlOps := metalDecoderOps()
	controlOps.eagerPrefillHostStaging = true
	control, err := newDecoder(m, controlOps)
	if err != nil {
		b.Fatal(err)
	}
	defer control.Release()
	retainedOut := make([]float32, cfg.Vocab)
	controlOut := make([]float32, cfg.Vocab)
	if err := retained.StepNLastInto(prompt, 0, retainedOut); err != nil {
		b.Fatal(err)
	}
	if err := control.StepNLastInto(prompt, 0, controlOut); err != nil {
		b.Fatal(err)
	}
	run := func(dec *Decoder, out []float32) time.Duration {
		start := time.Now()
		if err := dec.StepNLastInto(prompt, 0, out); err != nil {
			b.Fatal(err)
		}
		return time.Since(start)
	}
	var retainedDuration, controlDuration time.Duration
	b.ResetTimer()
	for i := range b.N {
		if i&1 == 0 {
			controlDuration += run(control, controlOut)
			retainedDuration += run(retained, retainedOut)
		} else {
			retainedDuration += run(retained, retainedOut)
			controlDuration += run(control, controlOut)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(controlDuration)/float64(retainedDuration), "control/retained")
}
