//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

func tinyQ8(tb testing.TB) *nlp.Llama {
	m, err := nlp.NewLlama(nlp.LlamaConfig{Vocab: 32000, Ctx: 640, Dim: 2048, Heads: 32, KVHeads: 4, Layers: 22, Hidden: 5632, Eps: 1e-5, RopeBase: 10000}, 7)
	if err != nil {
		tb.Fatal(err)
	}
	return m
}

func benchQ8Prefill(b *testing.B, f16 bool) {
	if !cuda.Available() {
		b.Skip("no cuda")
	}
	cuda.Q8PrefillF16 = f16
	defer func() { cuda.Q8PrefillF16 = false }()
	dec, err := llamagpu.NewLlamaQ8CUDA(tinyQ8(b))
	if err != nil {
		b.Fatal(err)
	}
	defer dec.Release()
	prompt := make([]int, 512)
	for i := range prompt {
		prompt[i] = (i*7)%31991 + 1
	}
	dec.StepNLast(prompt, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dec.StepNLast(prompt, 0)
	}
	b.StopTimer()
	b.ReportMetric(512*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkQ8Prefill_mmq(b *testing.B) { benchQ8Prefill(b, false) }
func BenchmarkQ8Prefill_f16(b *testing.B) { benchQ8Prefill(b, true) }
