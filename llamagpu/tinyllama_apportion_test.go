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

// TestTinyLlamaApportion splits a TinyLlama decode token into quant-matmul time and
// everything else, which decides where the remaining gap to llama.cpp can be closed.
//
// Toggling the cooperative kernels changes ONLY the quant matmuls, so the delta bounds
// their share. The split is sensitive to the true kernel speedup k at these shapes,
// which this does not measure: with total_off = T_matmul + T_other and total_on =
// T_matmul/k + T_other, k > 2.63 is forced by the measured end-to-end ratio (below
// that T_other goes negative). Across k in 3.4 to 7.9, T_other lands between 13 and 31
// ms — in every case MORE than llama.cpp's entire 5.81 ms token.
func TestTinyLlamaApportion(t *testing.T) {
	path := os.Getenv("GOAI_TINYLLAMA_GGUF")
	f, err := os.Open(path)
	if err != nil {
		t.Skip("no model")
	}
	defer f.Close()
	raw, _ := gguf.ReadRaw(f)
	qm, err := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	if err != nil {
		t.Skip(err)
	}
	defer qm.Close()
	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Skip(err)
	}
	defer dec.Release()
	prompt := []int{1, 15043, 29892, 590, 1024, 338}
	const genN = 32
	sample := func(on bool) float64 {
		p4 := metal.SetQ4KCooperative(on)
		p6 := metal.SetQ6KCooperative(on)
		p8 := metal.SetQ8_0Cooperative(on)
		defer func() { metal.SetQ4KCooperative(p4); metal.SetQ6KCooperative(p6); metal.SetQ8_0Cooperative(p8) }()
		if _, err := dec.Generate(prompt, genN, nlp.Greedy()); err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		if _, err := dec.Generate(prompt, genN, nlp.Greedy()); err != nil {
			t.Fatal(err)
		}
		return float64(genN) / time.Since(start).Seconds()
	}
	var off, on []float64
	for range 3 {
		off = append(off, sample(false))
		on = append(on, sample(true))
	}
	mo, mn := coopMedianMetal(off), coopMedianMetal(on)
	msOff, msOn := 1000/mo, 1000/mn
	t.Logf("TinyLlama: scalar %.2f tok/s (%.2f ms/tok) -> cooperative %.2f tok/s (%.2f ms/tok) = %.3fx",
		mo, msOff, mn, msOn, mn/mo)
	t.Logf("  quant-matmul time removed by the kernels: %.2f ms of %.2f ms per token (%.0f%%)",
		msOff-msOn, msOff, 100*(msOff-msOn)/msOff)
}
