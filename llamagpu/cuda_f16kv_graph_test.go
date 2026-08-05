//go:build cuda && cgo && (linux || windows)

package llamagpu

import (
	"github.com/jxsl13/goai/format/gguf"
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/nlp"
)

// TestCUDAF16KVGraphDecode validates NewLlamaQ4KGraphCUDAF16 (f16 KV storage): it builds the f16-KV
// decoder AFTER an f32-KV one in the same process — the exact multi-decoder failure mode that corrupted
// the int8 KV wiring — and a SECOND f16-KV decoder after that, checking the f16 path produces VALID,
// DETERMINISTIC tokens (no pool corruption) that closely track the f32-KV reference (f16 KV is lossy,
// ~2^-11 rel, so near- but not necessarily bit-identical).
func TestCUDAF16KVGraphDecode(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 256, Ctx: 128, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{1, 9, 42, 17}
	const maxNew = 24

	gd32, err := NewLlamaQ4KGraphCUDA(m, cfg.Ctx)
	if err != nil {
		t.Fatal(err)
	}
	out32, err := gd32.Generate(prompt, maxNew, nlp.Greedy())
	gd32.Release()
	if err != nil {
		t.Fatal(err)
	}

	// f16-KV built AFTER the f32 decoder (int8's corruption trigger)
	gd16, err := NewLlamaQ4KGraphCUDAF16(m, cfg.Ctx)
	if err != nil {
		t.Fatal(err)
	}
	out16, err := gd16.Generate(prompt, maxNew, nlp.Greedy())
	gd16.Release()
	if err != nil {
		t.Fatal(err)
	}

	// a SECOND f16-KV decoder after that — determinism / pool-state check
	gd16b, err := NewLlamaQ4KGraphCUDAF16(m, cfg.Ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer gd16b.Release()
	out16b, err := gd16b.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}

	if len(out16) != len(out32) || len(out16b) != len(out32) {
		t.Fatalf("length mismatch f32=%d f16=%d f16b=%d", len(out32), len(out16), len(out16b))
	}
	for i, tok := range out16 {
		if tok < 0 || tok >= cfg.Vocab {
			t.Fatalf("f16-KV token %d = %d out of vocab (corruption)", i, tok)
		}
		if tok != out16b[i] {
			t.Fatalf("f16-KV non-deterministic at %d: %d vs %d (multi-decoder pool corruption)", i, tok, out16b[i])
		}
	}
	agree := 0
	for i := range out16 {
		if out16[i] == out32[i] {
			agree++
		}
	}
	t.Logf("f16-KV graph decode: %d valid tokens, deterministic across 2 decoders, %d/%d agree with f32-KV", len(out16), agree, len(out32))
}

// TestCUDAF16KVGraphRealModel checks f16-KV on a REAL (trained) model, where logits are peaked so f16
// KV rounding should rarely flip the greedy argmax — disambiguating "f16 lossy on random weights" from
// "wiring bug". High f32/f16 token agreement ⇒ the wiring is correct and f16-KV is near-lossless.
func TestCUDAF16KVGraphRealModel(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	if _, err := os.Stat(glModelPath); err != nil {
		t.Skipf("model not present (%s)", glModelPath)
	}
	f, err := gguf.ReadFile(glModelPath)
	if err != nil {
		t.Fatal(err)
	}
	m, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	const maxNew = 24

	gd32, err := NewLlamaQ4KGraphCUDA(m, 64)
	if err != nil {
		t.Fatal(err)
	}
	out32, err := gd32.Generate(prompt, maxNew, nlp.Greedy())
	gd32.Release()
	if err != nil {
		t.Fatal(err)
	}
	gd16, err := NewLlamaQ4KGraphCUDAF16(m, 64)
	if err != nil {
		t.Fatal(err)
	}
	defer gd16.Release()
	out16, err := gd16.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	agree := 0
	for i := range out16 {
		if out16[i] == out32[i] {
			agree++
		}
	}
	t.Logf("REAL TinyLlama f16-KV vs f32-KV greedy: %d/%d tokens agree", agree, len(out32))
	if agree*2 < len(out32) { // <50% agreement on a trained model ⇒ likely a wiring bug, not f16 loss
		t.Fatalf("f16-KV agrees only %d/%d with f32-KV on a trained model — wiring bug suspected", agree, len(out32))
	}
}

// TestCUDAF16KVDecodeSpeed A/Bs long-context decode throughput: at ~1024 cached tokens the flash kernel
// is K/V-read-bound, so f16 storage (half the bytes) should decode faster than f32. Logs the ratio.
func TestCUDAF16KVDecodeSpeed(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test")
	}
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	if _, err := os.Stat(glModelPath); err != nil {
		t.Skipf("model not present (%s)", glModelPath)
	}
	f, err := gguf.ReadFile(glModelPath)
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	const prefill, steps = 1024, 96
	measure := func(useF16 bool) float64 {
		var gd *GraphLlamaDecoder
		if useF16 {
			gd, err = NewLlamaQ4KGraphCUDAF16(model, prefill+steps+8)
		} else {
			gd, err = NewLlamaQ4KGraphCUDA(model, prefill+steps+8)
		}
		if err != nil {
			t.Fatal(err)
		}
		defer gd.Release()
		prompt := make([]int, prefill)
		for i := range prompt {
			prompt[i] = (i*131 + 1) % model.Config.Vocab
		}
		if _, err = gd.prefillForward(prompt); err != nil {
			t.Fatal(err)
		}
		if err = gd.captureGraph(); err != nil {
			t.Fatal(err)
		}
		tk := 1
		must0(t, gd.pos.Set(prefill))
		must0(t, gd.emb.EmbedInto([]int32{int32(tk)}, gd.dx))
		must0(t, gd.graph.Launch())
		must0(t, cuda.GraphSync())
		t0 := time.Now()
		for d := 0; d < steps; d++ {
			tk = gd.logits.Argmax()
			must0(t, gd.pos.Set(prefill+1+d))
			must0(t, gd.emb.EmbedInto([]int32{int32(tk)}, gd.dx))
			must0(t, gd.graph.Launch())
			must0(t, cuda.GraphSync())
		}
		return float64(steps) / time.Since(t0).Seconds()
	}
	f32 := measure(false)
	f16 := measure(true)
	t.Logf("long-ctx(%d) decode: f32-KV %.1f tok/s, f16-KV %.1f tok/s = %.2fx", prefill, f32, f16, f16/f32)
}
