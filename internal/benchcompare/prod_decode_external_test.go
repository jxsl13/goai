//go:build darwin && cgo && vulkan

package benchcompare

import (
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
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

	q4kMode := "auto"
	if value := os.Getenv("GOAI_Q4K_COOPERATIVE"); value != "" {
		on, err := strconv.ParseBool(value)
		if err != nil {
			t.Fatalf("GOAI_Q4K_COOPERATIVE: %v", err)
		}
		previous := metal.SetQ4KCooperative(on)
		defer metal.SetQ4KCooperative(previous)
		if on {
			q4kMode = "cooperative"
		} else {
			q4kMode = "scalar"
		}
	}
	q6kMode := "auto"
	if value := os.Getenv("GOAI_Q6K_COOPERATIVE"); value != "" {
		on, err := strconv.ParseBool(value)
		if err != nil {
			t.Fatalf("GOAI_Q6K_COOPERATIVE: %v", err)
		}
		previous := metal.SetQ6KCooperative(on)
		defer metal.SetQ6KCooperative(previous)
		if on {
			q6kMode = "cooperative"
		} else {
			q6kMode = "scalar"
		}
	}

	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Fatalf("llamagpu.NewQuant: %v", err)
	}
	defer dec.Release()

	nGen := benchEnvInt(t, "GOAI_PROD_DECODE_TOKENS", 64)
	nProm := benchEnvInt(t, "GOAI_PROD_PREFILL_TOKENS", 64)
	reps := benchEnvInt(t, "GOAI_PROD_REPS", 3)

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

	fmt.Printf("GOAI_PROD metal q4k=%s q6k=%s: decode(tg%d) %.1f tok/s | prefill(pp%d) %.1f tok/s\n",
		q4kMode, q6kMode,
		nGen, float64(nGen)/bestTg.Seconds(),
		nProm, float64(nProm)/bestPp.Seconds())
}

func benchEnvInt(t *testing.T, name string, fallback int) int {
	t.Helper()
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	n, err := strconv.Atoi(value)
	if err != nil || n <= 0 {
		t.Fatalf("%s must be a positive integer, got %q", name, value)
	}
	return n
}
