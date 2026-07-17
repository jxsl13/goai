//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// The serving capstone: the COMPLETE user story on the unified path at full speed —
// tokenize → batched f16 tensor-core prefill (seeding the KV caches) → CUDA-graph
// Q4_K decode with on-device argmax → detokenize. This is the configuration a real
// server would run; the test logs the end-to-end latency breakdown and gates on
// coherent output plus the pillars' known floors (prefill ≥250 tok/s at this tiny P, decode ≥100
// tok/s on TinyLlama — both far under the measured 4184/199, so only a real
// regression trips them).
func TestCUDAUnifiedServeDemo(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
	skipNoGPU(t)
	if _, err := os.Stat(tinyLlamaPath); err != nil {
		t.Skipf("model not present (%s)", tinyLlamaPath)
	}
	fraw, err := os.Open(tinyLlamaPath)
	must(t, err)
	rf, err := gguf.ReadRaw(fraw)
	fraw.Close()
	must(t, err)
	f, err := gguf.ReadFile(tinyLlamaPath)
	must(t, err)
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	must(t, err)
	tok, err := nlp.UnigramFromGGUF(rf.Metadata)
	must(t, err)

	prompt := "The capital of France is"
	ids := append([]int{1}, tok.Encode(prompt)...)
	P := len(ids)
	const gen = 24
	maxSeq := P + gen + 8

	// decode side: Q4_K raw decoder (graph-capturable fixed buffers)
	dec := buildRawGraphDecoder(t, rf, "llama", maxSeq, fromF32(quantQ4K))
	defer dec.free()
	for _, l := range dec.layers {
		must(t, l.cache.ZeroCache())
		l.cache.SetLen(0)
	}
	// prefill side: f16 tensor-core stack
	pf := buildResidentLlamaF16(t, model)
	defer pf.free()

	ids32 := make([]int32, P)
	for i, id := range ids {
		ids32[i] = int32(id)
	}

	// ---- serve: prefill (batched, seeds caches); warmup pass first (§V22) ----
	prefill := func() *tensor.Tensor {
		dx, err := pf.emb.Embed(ids32)
		must(t, err)
		for li, l := range pf.layers {
			dx, err = l.seedForward(dx, dec.layers[li].cache)
			must(t, err)
		}
		must(t, dx.RMSNorm(pf.norm, float32(pf.eps)))
		dl, err := pf.out.MatMulDevice(dx)
		must(t, err)
		dx.Free()
		lg, err := dl.ToHost()
		must(t, err)
		dl.Free()
		must(t, cuda.GraphSync())
		return lg
	}
	prefill() // warmup: kernel compiles + cuBLAS heuristics
	for _, l := range dec.layers {
		must(t, l.cache.ZeroCache())
		l.cache.SetLen(0)
	}
	t0 := time.Now()
	lg := prefill()
	prefillDur := time.Since(t0)
	for _, l := range dec.layers {
		l.cache.SetLen(maxSeq) // padded fixed-shape form for the captured graph
	}

	// first token from the prefill logits
	vocab := lg.Shape()[1]
	first, bv := 0, -1e300
	for c := 0; c < vocab; c++ {
		if v := lg.AtF64(P-1, c); v > bv {
			bv, first = v, c
		}
	}

	// ---- serve: graph-captured greedy decode with on-device argmax ----
	// (one eager step warms the fixed buffers at a real position before capture)
	warm := dec.step(t, int32(first), P)
	runtime.LockOSThread()
	must(t, cuda.CaptureBegin())
	dec.forwardBody(t)
	g, err := cuda.CaptureEnd()
	runtime.UnlockOSThread()
	must(t, err)
	dec.graph = g

	toks := []int{first, warm}
	last := warm
	t0 = time.Now()
	for n := len(toks); n < gen; n++ {
		last = dec.stepGraph(t, int32(last), P+n-1)
		toks = append(toks, last)
	}
	decodeDur := time.Since(t0)
	decoded := len(toks) - 2

	text := strings.TrimSpace(renderPieces(tok.Decode(toks)))
	prefillTPS := float64(P) / prefillDur.Seconds()
	decodeTPS := float64(decoded) / decodeDur.Seconds()
	t.Logf("prompt: %q (P=%d)", prompt, P)
	t.Logf("output: %q", text)
	t.Logf("prefill: %.1f ms (%.0f tok/s) | decode: %d tok in %.1f ms (%.0f tok/s) | total %.1f ms",
		float64(prefillDur.Microseconds())/1000, prefillTPS,
		decoded, float64(decodeDur.Microseconds())/1000, decodeTPS,
		float64((prefillDur+decodeDur).Microseconds())/1000)

	if !strings.Contains(strings.ToLower(text), "paris") {
		t.Errorf("serve demo incoherent (no 'Paris'): %q", text)
	}
	if prefillTPS < 250 { // P=11 is launch-bound (the P=6→1.3× / P=94→19.4× finding); the floor catches order-of-magnitude breaks only
		t.Errorf("prefill regression: %.0f tok/s (floor 250)", prefillTPS)
	}
	if decodeTPS < 100 {
		t.Errorf("decode regression: %.0f tok/s (floor 100)", decodeTPS)
	}
}
