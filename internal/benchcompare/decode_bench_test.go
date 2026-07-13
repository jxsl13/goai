//go:build darwin && cgo && vulkan

package benchcompare

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// Batched-decode benchmarks (§T425): keep the ADR-0019 wins (§T404 24×, §T418 41× prefill)
// permanently re-measurable next to the per-op rows, so a regression in the recorder path shows up
// in `make bench-compare` like any kernel regression would.

func benchLlama(b *testing.B) *nlp.Llama {
	b.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 1024, Ctx: 128, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}, 1)
	if err != nil {
		b.Fatal(err)
	}
	return m
}

// BenchmarkLlamaDecode times one decode token on the batched decoders (metal + vulkan) and on the
// library's per-op DecodeStep (metal) — the §T404/T409 comparison as a standing benchmark.
func BenchmarkLlamaDecode(b *testing.B) {
	m := benchLlama(b)

	runDecoder := func(b *testing.B, dec *llamagpu.Decoder) {
		b.Helper()
		defer dec.Release()
		if _, err := dec.Step(0, 0); err != nil {
			b.Fatal(err)
		}
		pos := 1
		b.ResetTimer()
		for i := range b.N {
			if pos >= m.Config.Ctx {
				pos = 1 // overwrite old cache rows; timing-only
			}
			if _, err := dec.Step(i%m.Config.Vocab, pos); err != nil {
				b.Fatal(err)
			}
			pos++
		}
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
	}

	b.Run("batched-metal", func(b *testing.B) {
		dec, err := llamagpu.New(m)
		if err != nil {
			b.Skip(err)
		}
		runDecoder(b, dec)
	})
	b.Run("batched-vulkan", func(b *testing.B) {
		dec, err := llamagpu.NewVulkan(m)
		if err != nil {
			b.Skip(err)
		}
		runDecoder(b, dec)
	})
	b.Run("per-op-metal", func(b *testing.B) {
		be, ok := backend.Get(backend.Metal)
		if !ok {
			b.Skip("metal not registered")
		}
		ctx := backend.NewContext().WithBackend(be)
		cache := m.NewCache()
		if _, err := m.DecodeStep(ctx, cache, 0, 0); err != nil {
			b.Fatal(err)
		}
		pos := 1
		b.ResetTimer()
		for i := range b.N {
			if pos >= m.Config.Ctx {
				b.StopTimer()
				cache = m.NewCache()
				pos = 0
				b.StartTimer()
			}
			if _, err := m.DecodeStep(ctx, cache, i%m.Config.Vocab, pos); err != nil {
				b.Fatal(err)
			}
			pos++
		}
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
	})
}

// BenchmarkLlamaPrefill times a 64-token prompt prefill: one batched StepN vs 64 sequential Steps
// (§T418's 41×, as a standing benchmark).
func BenchmarkLlamaPrefill(b *testing.B) {
	m := benchLlama(b)
	prompt := make([]int, 64)
	for i := range prompt {
		prompt[i] = (i * 7) % m.Config.Vocab
	}

	b.Run("stepn-metal", func(b *testing.B) {
		dec, err := llamagpu.New(m)
		if err != nil {
			b.Skip(err)
		}
		defer dec.Release()
		if _, err := dec.StepN(prompt, 0); err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for range b.N {
			if _, err := dec.StepN(prompt, 0); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(b.N)*float64(len(prompt))/b.Elapsed().Seconds(), "tok/s")
	})
	b.Run("sequential-metal", func(b *testing.B) {
		dec, err := llamagpu.New(m)
		if err != nil {
			b.Skip(err)
		}
		defer dec.Release()
		b.ResetTimer()
		for range b.N {
			for pos, tok := range prompt {
				if _, err := dec.Step(tok, pos); err != nil {
					b.Fatal(err)
				}
			}
		}
		b.ReportMetric(float64(b.N)*float64(len(prompt))/b.Elapsed().Seconds(), "tok/s")
	})
}

// BenchmarkLlamaDecodeLongContext times one decode token at a ~1920-token context on both batched
// decoders — the §T428/T429 cooperative decode-attention guard (the serial kernel took 242ms here).
func BenchmarkLlamaDecodeLongContext(b *testing.B) {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 1024, Ctx: 2048, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}, 1)
	if err != nil {
		b.Fatal(err)
	}
	run := func(b *testing.B, dec *llamagpu.Decoder) {
		b.Helper()
		defer dec.Release()
		win := make([]int, 128)
		for i := range win {
			win[i] = (i * 7) % m.Config.Vocab
		}
		for pos := 0; pos+128 <= 1920; pos += 128 { // fill the KV cache
			if _, err := dec.StepN(win, pos); err != nil {
				b.Fatal(err)
			}
		}
		if _, err := dec.Step(0, 1920); err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for i := range b.N {
			if _, err := dec.Step(i%m.Config.Vocab, 1920); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "tok/s")
	}
	b.Run("batched-metal", func(b *testing.B) {
		dec, err := llamagpu.New(m)
		if err != nil {
			b.Skip(err)
		}
		run(b, dec)
	})
	b.Run("batched-vulkan", func(b *testing.B) {
		dec, err := llamagpu.NewVulkan(m)
		if err != nil {
			b.Skip(err)
		}
		run(b, dec)
	})
}
