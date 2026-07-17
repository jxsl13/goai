//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"runtime"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// argmaxF32 returns the index of the max element (host greedy pick).
func argmaxF32(v []float32) int {
	best, bi := v[0], 0
	for i, x := range v {
		if x > best {
			best, bi = x, i
		}
	}
	return bi
}

// TestCUDAUnifiedVsGraphDecodeThroughput measures, in ONE quiet window on the SAME TinyLlama-1.1B
// Q8 weights, the decode throughput of two CUDA paths:
//
//   - the UNIFIED llamagpu path (NewQuantCUDA): the backend-agnostic batched Decoder driving the
//     cuda.Recorder — eager per-op submit, no graph capture, but Generate/samplers for free;
//   - the BESPOKE graph path (q8GraphDecoder): fixed persistent buffers + a captured CUDA graph
//     replayed per token + on-device argmax (the §PERF 164.7 tok/s engine).
//
// Same box, same window, same weights → the ratio is the honest cost of API uniformity vs the
// hand-tuned graph, and tells us whether wiring graph capture into the unified path is worth it.
func TestCUDAUnifiedVsGraphDecodeThroughput(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
	skipNoGPU(t)
	m := loadTinyLlama(t)
	const prefill, decode = 32, 64

	// --- UNIFIED llamagpu path (NewQuantCUDA) ---
	qm, err := nlp.QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.NewQuantCUDA(qm)
	if err != nil {
		qm.Close()
		t.Fatal(err)
	}
	for p := 0; p < prefill; p++ { // prefill / warm
		if _, err := dec.Step((p*131+1)%m.Config.Vocab, p); err != nil {
			t.Fatal(err)
		}
	}
	logits, err := dec.Step(0, prefill)
	if err != nil {
		t.Fatal(err)
	}
	tok := argmaxF32(logits)
	tUni := time.Now()
	for d := 0; d < decode; d++ {
		logits, err = dec.Step(tok, prefill+1+d)
		if err != nil {
			t.Fatal(err)
		}
		tok = argmaxF32(logits)
	}
	uniTokps := float64(decode) / time.Since(tUni).Seconds()
	dec.Release()
	qm.Close()

	// --- BESPOKE graph path (q8GraphDecoder) ---
	gd := buildQ8GraphDecoder(t, m, prefill+decode+2)
	defer gd.free()
	for p := 0; p < prefill; p++ {
		gd.step(t, int32((p*131+1)%m.Config.Vocab), p)
	}
	runtime.LockOSThread()
	must(t, cuda.CaptureBegin())
	gd.forwardBody(t)
	g, err := cuda.CaptureEnd()
	runtime.UnlockOSThread()
	must(t, err)
	gd.graph = g
	var gtok int32
	gd.pos.Set(prefill)
	gd.emb.EmbedInto([]int32{0}, gd.dx)
	gd.graph.Launch()
	cuda.GraphSync()
	{
		l, _ := gd.logits.ToHost()
		gtok = int32(argmaxRow(l, 0))
	}
	tGraph := time.Now()
	for d := 0; d < decode; d++ {
		gd.pos.Set(prefill + 1 + d)
		gd.emb.EmbedInto([]int32{gtok}, gd.dx)
		gd.graph.Launch()
		cuda.GraphSync()
		l, _ := gd.logits.ToHost()
		gtok = int32(argmaxRow(l, 0))
	}
	graphTokps := float64(decode) / time.Since(tGraph).Seconds()

	t.Logf("TinyLlama-1.1B Q8 decode (%d prefill + %d decode), SAME weights/window:", prefill, decode)
	t.Logf("  UNIFIED llamagpu (recorder, eager submit, no graph): %.1f tok/s", uniTokps)
	t.Logf("  BESPOKE graph (fixed buffers + CUDA graph + dev argmax): %.1f tok/s", graphTokps)
	t.Logf("  → graph is %.2fx the unified path (cost of API uniformity)", graphTokps/uniTokps)
}
