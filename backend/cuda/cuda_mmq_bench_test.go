//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
)

// TestCUDATinyLlamaMMQThroughput measures GoAI's e2e prefill (int8 MMQ, ResidentMMQ) + decode
// (Q4_K graph GEMV) tok/s on TinyLlama, at the SAME shapes llama.cpp's llama-bench reports
// (pp512 / tg128) so the numbers are directly comparable to the refreshed incumbent baseline
// (llama.cpp b10012 Vulkan on the RTX 3060: pp512 9552 t/s, tg128 245.6 t/s). Informational
// (t.Logf), not a hard gate — it establishes standing.
func TestCUDATinyLlamaMMQThroughput(t *testing.T) {
	skipNoGPU(t)
	const path = "../../models/tinyllama-1.1b-chat-q8_0.gguf"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present (%s)", path)
	}
	f, err := os.Open(path)
	must(t, err)
	rf, err := gguf.ReadRaw(f)
	f.Close()
	must(t, err)

	const P = 512
	const gen = 128
	maxSeq := P + gen + 8

	dec := buildRawGraphDecoder(t, rf, "llama", maxSeq, fromF32(quantQ4K))
	defer dec.free()
	pf := buildRawPrefill(t, rf, "llama", true) // int8 MMQ prefill weights
	defer pf.free()
	pf16 := buildRawPrefill(t, rf, "llama", false) // f16 prefill weights (isolate MMQ vs pipeline)
	defer pf16.free()

	ids32 := make([]int32, P)
	for i := range ids32 {
		ids32[i] = int32((i*131 + 7) % 32000)
	}
	reset := func() {
		for _, l := range dec.layers {
			must(t, l.cache.ZeroCache())
			l.cache.SetLen(0)
		}
	}
	seedWith := func(p *rawF16Prefill) {
		reset()
		dx, err := p.emb.Embed(ids32)
		must(t, err)
		for li, l := range p.layers {
			dx, err = l.seedForward(dx, dec.layers[li].cache)
			must(t, err)
		}
		dx.Free()
	}
	timePrefill := func(p *rawF16Prefill) float64 {
		seedWith(p)
		must(t, cuda.GraphSync()) // warmup
		t0 := time.Now()
		seedWith(p)
		must(t, cuda.GraphSync())
		return float64(P) / time.Since(t0).Seconds()
	}
	seed := func() { seedWith(pf) }

	// prefill (pp512): MMQ vs f16 to localize the gap (both run the same seedForward pipeline).
	prefillTPS := timePrefill(pf)
	prefillF16TPS := timePrefill(pf16)

	// decode (tg128): per-op step (understates vs the graph-captured path by the per-launch overhead).
	reset()
	seed()
	must(t, cuda.GraphSync())
	last := 1
	dec.step(t, int32(last), P) // warmup
	t1 := time.Now()
	pos := P
	for n := 0; n < gen; n++ {
		last = dec.step(t, int32(last), pos)
		pos++
	}
	must(t, cuda.GraphSync())
	decodeT := time.Since(t1)
	decodeTPS := float64(gen) / decodeT.Seconds()

	t.Logf("GoAI TinyLlama (CUDA): pp%d MMQ %.0f t/s | pp%d f16 %.0f t/s | tg%d %.1f t/s",
		P, prefillTPS, P, prefillF16TPS, gen, decodeTPS)
	t.Logf("llama.cpp b10012 Vulkan (Q8, same RTX 3060):          pp512 9552 tok/s | tg128 245.6 tok/s")
	t.Logf("→ MMQ prefill is %.2fx the f16 prefill (isolates the scale-epilogue+device-quant cost); the"+
		" rest of the pp gap vs llcpp is the shared seedForward pipeline (per-op launches, attention)",
		prefillTPS/prefillF16TPS)
	t.Logf("→ prefill %.2fx, decode %.2fx vs llama.cpp-Vulkan (CAVEATS: GoAI here = test harness, not"+
		" production; decode is per-op step (graph-captured is faster); llcpp is Vulkan not CUDA; Q4_K vs Q8)",
		prefillTPS/9552, decodeTPS/245.6)
}
