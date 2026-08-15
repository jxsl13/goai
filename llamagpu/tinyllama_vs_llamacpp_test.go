//go:build darwin && cgo

package llamagpu_test

import (
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestTinyLlamaVsLlamaCpp decodes the SAME GGUF file llama-bench measures, so the two
// numbers are directly comparable rather than related by argument.
//
// Reference on this host, llama.cpp build 48d22e295 (10360), Metal + BLAS backends:
//
//	llama-bench -m models/tinyllama-1.1b-q4km.gguf -p 0 -n 64 -r 3
//	  llama 1B Q4_K - Medium, 636.18 MiB, 1.10 B params -> tg64 172.19 +/- 7.57 t/s
//
// Skips when the model is absent so the suite stays hermetic; GOAI_TINYLLAMA_GGUF
// overrides the path.
func TestTinyLlamaVsLlamaCpp(t *testing.T) {
	if testing.Short() {
		t.Skip("1.1B model; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("metal unavailable")
	}
	path := os.Getenv("GOAI_TINYLLAMA_GGUF")
	if path == "" {
		path = "../models/tinyllama-1.1b-q4km.gguf"
	}
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("model not present: %v", err)
	}
	defer f.Close()
	raw, err := gguf.ReadRaw(f)
	if err != nil {
		t.Skipf("ReadRaw: %v", err)
	}
	qm, err := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	if err != nil {
		t.Skipf("QuantLlamaFromGGUF: %v", err)
	}
	defer qm.Close()

	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Skipf("NewQuant: %v", err)
	}
	defer dec.Release()

	prompt := []int{1, 15043, 29892, 590, 1024, 338}
	const genN = 64
	sample := func() float64 {
		start := time.Now()
		if _, err := dec.Generate(prompt, genN, nlp.Greedy()); err != nil {
			t.Fatal(err)
		}
		return float64(genN) / time.Since(start).Seconds()
	}
	sample() // warm
	var tps []float64
	for range 3 {
		tps = append(tps, sample())
	}
	got := coopMedianMetal(tps)
	const llamaCpp = 172.19
	t.Logf("TinyLlama-1.1B Q4_K_M, 64 tokens, Metal: GoAI %.2f tok/s vs llama.cpp %.2f tok/s = %.3fx  (samples %v)",
		got, llamaCpp, got/llamaCpp, tps)
}
