//go:build darwin && cgo

package llamagpu_test

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestLongContextDecodeGap records how decode throughput degrades with context, and corrects a
// like-for-unlike comparison I published that overstated the gap by ~4x.
//
// THE ERROR: I timed dec.Generate(prompt, 32) and called the result decode throughput. Generate
// PREFILLS the whole prompt first, so a 1536-token prompt costs ~808ms of prefill amortised over 32
// generated tokens — ~25ms/token of prefill counted as decode. llama.cpp's figure, derived from
// `llama-bench -pg N,32`, has prefill SUBTRACTED. Prefill-inclusive against prefill-exclusive.
//
// Measured correctly — prefill once, then time only the per-token steps (M2 Pro, TinyLlama-1.1B
// Q4_K_M, live in one session):
//
//	ctx    decode-only   Generate(incl. prefill)   llama.cpp   ratio
//	   8      157.5             149.3                158.4     0.99x
//	 512      146.0              62.2               ~161.6     0.90x
//	1536      117.6              28.2               ~161       0.73x
//
// The middle column is what the earlier version of this test reported as "decode". The arithmetic
// closes exactly: 808ms prefill + 32 x 8.66ms = 1085ms -> 29.5 tok/s, which is what it printed.
//
// WHAT IS ACTUALLY TRUE. Per-step GPU-busy time grows 5.44 -> 6.62 -> 8.24 ms across those contexts
// (96-97% of wall, so this is GPU work, not host overhead). llama.cpp is flat at ~161 t/s. So the
// real problem is that our forward pass degrades ~1.5x with context where theirs does not — a 0.73x
// gap at ctx=1536, not 0.19x.
//
// Sizing what is left: the extra 2.8ms/token at ctx=1536 is attention over the KV cache, which reads
// ~69 MB/token — about 0.38ms at this machine's ~180 GB/s. So attention is ~7x off its bandwidth
// floor. That is a real and correctly-scoped target, and split-K's 1.07x is a first bite at it.
//
// This test asserts DECODE-ONLY throughput, and deliberately also reports the Generate figure beside
// it, so the two can never again be confused for one another.
func TestLongContextDecodeGap(t *testing.T) {
	if !metal.Available() {
		t.Skip("no metal")
	}
	f, _ := os.Open(os.Getenv("GOAI_TINYLLAMA_GGUF"))
	if f == nil {
		t.Skip("no model")
	}
	defer f.Close()
	raw, _ := gguf.ReadRaw(f)
	qm, _ := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	defer qm.Close()
	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Skip(err)
	}
	defer dec.Release()

	for _, c := range []struct {
		plen  int
		floor float64 // recorded decode-only tok/s, with headroom
	}{{8, 110}, {512, 100}, {1536, 80}} {
		p := make([]int, c.plen)
		for i := range p {
			p[i] = 1 + i%2000
		}
		if _, e := dec.StepNLast(p, 0); e != nil {
			t.Fatal(e)
		}
		for i := 0; i < 3; i++ { // prime encode-overlap
			dec.Step(5, c.plen+i)
		}
		const n = 12
		var wall, gpu float64
		for i := 0; i < n; i++ {
			st := time.Now()
			if _, e := dec.Step(5, c.plen+3+i); e != nil {
				t.Fatal(e)
			}
			wall += time.Since(st).Seconds() * 1e3
			gpu += metal.LastGPUSeconds() * 1e3
		}
		wall /= n
		gpu /= n
		tps := 1000 / wall
		st := time.Now()
		if _, e := dec.Generate(p, 32, nlp.Greedy()); e != nil {
			t.Fatal(e)
		}
		gen := 32 / time.Since(st).Seconds()
		fmt.Printf("LONGCTX ctx=%5d decode-only %6.1f tok/s (gpuBusy %.2fms, %.0f%%)  Generate-incl-prefill %6.1f tok/s\n",
			c.plen, tps, gpu, 100*gpu/wall, gen)
		if tps < c.floor {
			t.Errorf("ctx=%d decode-only %.1f tok/s below recorded floor %.0f", c.plen, tps, c.floor)
		}
	}
}
