//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/llamagpu"
)

// A/B: full-k StepN (projects [k,vocab] LM head + downloads k·vocab) vs StepNLast (last row only). The
// transformer body is identical; the delta is the wasted LM head + PCIe download over the [k-1] rows
// that generation / KV-seeding never read.
func BenchmarkLlamaPrefillQ8_seq512_fullk(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no cuda")
	}
	dec, err := llamagpu.NewLlamaQ8CUDA(prefillLlama(b, 512))
	if err != nil {
		b.Fatal(err)
	}
	defer dec.Release()
	prompt := make([]int, 512)
	for i := range prompt {
		prompt[i] = (i*7)%31991 + 1
	}
	dec.StepN(prompt, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dec.StepN(prompt, 0)
	}
	b.StopTimer()
	b.ReportMetric(512*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkLlamaPrefillQ8_seq512_lastrow(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no cuda")
	}
	dec, err := llamagpu.NewLlamaQ8CUDA(prefillLlama(b, 512))
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
