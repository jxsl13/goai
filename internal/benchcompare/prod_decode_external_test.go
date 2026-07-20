//go:build darwin && cgo && vulkan

package benchcompare

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestProdDecodeGGUF loads a real quantized Llama GGUF (env TINYLLAMA_GGUF) and
// measures GoAI's Metal batched decoder: single-token decode (tg) and prefill (pp)
// throughput, to race llama.cpp's llama-bench on the SAME file — the production-size
// Apple head-to-head that discharges the 17.7M-toy caveat of §T607 (T887). Gated.
func TestProdDecodeGGUF(t *testing.T) {
	path := os.Getenv("TINYLLAMA_GGUF")
	if path == "" {
		t.Skip("set TINYLLAMA_GGUF to a quantized Llama GGUF")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rf, err := gguf.ReadRaw(f)
	if err != nil {
		t.Fatalf("gguf.ReadRaw: %v", err)
	}
	qm, err := nlp.QuantLlamaFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Fatalf("QuantLlamaFromGGUF: %v", err)
	}
	c := qm.Config
	t.Logf("loaded: vocab=%d dim=%d hidden=%d heads=%d kv=%d layers=%d ctx=%d",
		c.Vocab, c.Dim, c.Hidden, c.Heads, c.KVHeads, c.Layers, c.Ctx)

	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Fatalf("llamagpu.NewQuant: %v", err)
	}
	defer dec.Release()

	const nGen, nProm, reps = 64, 64, 3

	// Sanity: the first decode step must return finite logits of the right width.
	logits, err := dec.Step(1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(logits) != c.Vocab {
		t.Fatalf("logits width %d, want vocab %d", len(logits), c.Vocab)
	}

	// tg (decode): best-of-reps over nGen single-token steps at growing position.
	bestTg := time.Hour
	for range reps {
		s := time.Now()
		for i := range nGen {
			if _, err := dec.Step((i*7+1)%c.Vocab, 1+i); err != nil {
				t.Fatal(err)
			}
		}
		if d := time.Since(s); d < bestTg {
			bestTg = d
		}
	}

	// pp (prefill): best-of-reps over one nProm-token StepN.
	prompt := make([]int, nProm)
	for i := range prompt {
		prompt[i] = (i*13 + 1) % c.Vocab
	}
	if _, err := dec.StepN(prompt, 0); err != nil { // warm
		t.Fatal(err)
	}
	bestPp := time.Hour
	for range reps {
		s := time.Now()
		if _, err := dec.StepN(prompt, 0); err != nil {
			t.Fatal(err)
		}
		if d := time.Since(s); d < bestPp {
			bestPp = d
		}
	}

	fmt.Printf("GOAI_PROD metal: decode(tg%d) %.1f tok/s | prefill(pp%d) %.1f tok/s\n",
		nGen, float64(nGen)/bestTg.Seconds(),
		nProm, float64(nProm)/bestPp.Seconds())
}
