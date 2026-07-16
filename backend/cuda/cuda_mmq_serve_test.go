//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDAUnifiedServeMMQ runs the full unified serve path with int8 MMQ prefill (ResidentMMQ) on a
// real model: MMQ-batched prefill seeds the Q8 decode caches, then the decoder continues. It gates
// on (1) the seeded layer-0 K cache matching the pure decode-path rows within the int8 budget and
// (2) the continuation staying coherent ("Paris") — the end-to-end proof that the int8 prefill
// weight is a valid drop-in through decode, at ~half the prefill weight VRAM.
func TestCUDAUnifiedServeMMQ(t *testing.T) {
	skipNoGPU(t)
	const path = "../../models/qwen2.5-0.5b-instruct-q8_0.gguf" // dim 896 -> Q8 decode
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present (%s)", path)
	}
	f, err := os.Open(path)
	must(t, err)
	rf, err := gguf.ReadRaw(f)
	f.Close()
	must(t, err)
	tok, err := nlp.BPEFromGGUF(rf.Metadata)
	must(t, err)

	prompt := "France is a country in Western Europe, famous for its art, cuisine and history. " +
		"Millions of tourists visit its museums and monuments every year. The capital of France is"
	ids := tok.Encode(prompt)
	P := len(ids)
	const gen = 16
	maxSeq := P + gen + 8

	// Q8 decode reference (0.5B dim 896 is not %256 -> Q4_K-ineligible, resident Q8).
	dec := buildRawGraphDecoder(t, rf, "qwen2", maxSeq, fromF32(func(w *tensor.Tensor) (qProj, error) {
		return cuda.NewResidentBQ8(w)
	}))
	defer dec.free()

	// pure decode-path reference: process the prompt, capture layer-0 K rows + the continuation.
	var last int
	for i, id := range ids {
		last = dec.step(t, int32(id), i)
	}
	var refToks []int
	for n := 0; n < gen; n++ {
		refToks = append(refToks, last)
		last = dec.step(t, int32(last), P+n)
	}
	dec.layers[0].cache.SetLen(P)
	refK, err := dec.layers[0].cache.K().ToHost()
	must(t, err)
	dec.layers[0].cache.SetLen(maxSeq)

	// MMQ batched prefill seeds fresh caches.
	pf := buildRawPrefill(t, rf, "qwen2", true)
	defer pf.free()
	ids32 := make([]int32, P)
	for i, id := range ids {
		ids32[i] = int32(id)
	}
	for _, l := range dec.layers {
		must(t, l.cache.ZeroCache())
		l.cache.SetLen(0)
	}
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

	// gate 1: seeded layer-0 K rows vs decode-path rows (int8 budget).
	dec.layers[0].cache.SetLen(P)
	seedK, err := dec.layers[0].cache.K().ToHost()
	must(t, err)
	dec.layers[0].cache.SetLen(maxSeq)
	var sumAbs, sumDiff float64
	for r := 0; r < P; r++ {
		for c := 0; c < seedK.Shape()[1]; c++ {
			rv := refK.AtF64(r, c)
			sumAbs += mabs(rv)
			sumDiff += mabs(seedK.AtF64(r, c) - rv)
		}
	}
	l1 := sumDiff / sumAbs

	// gate 2: greedy continuation from the MMQ-seeded state.
	vocab := lg.Shape()[1]
	first, bv := 0, -1e300
	for c := 0; c < vocab; c++ {
		if v := lg.AtF64(P-1, c); v > bv {
			bv, first = v, c
		}
	}
	mmqToks := []int{first}
	lastTok := first
	for n := 1; n < gen; n++ {
		lastTok = dec.step(t, int32(lastTok), P+n-1)
		mmqToks = append(mmqToks, lastTok)
	}

	t.Logf("pure-decode text: %q", strings.TrimSpace(tok.Decode(refToks)))
	t.Logf("MMQ-prefill text: %q", strings.TrimSpace(tok.Decode(mmqToks)))
	t.Logf("layer-0 K cache: MMQ-seeded vs decode-path, rel L1 %.4f", l1)

	if l1 > 0.15 {
		t.Errorf("MMQ-seeded cache diverges beyond int8 budget: rel L1 %.4f", l1)
	}
	if !strings.Contains(strings.ToLower(tok.Decode(mmqToks)), "paris") {
		t.Errorf("MMQ unified continuation incoherent (no 'Paris'): %q", tok.Decode(mmqToks))
	}

	// e2e prefill latency: MMQ vs f16 batched prefill (device work only, warmup + sync).
	timeSeed := func(p *rawF16Prefill) time.Duration {
		reset := func() {
			for _, l := range dec.layers {
				must(t, l.cache.ZeroCache())
				l.cache.SetLen(0)
			}
		}
		doSeed := func() {
			d, err := p.emb.Embed(ids32)
			must(t, err)
			for li, l := range p.layers {
				d, err = l.seedForward(d, dec.layers[li].cache)
				must(t, err)
			}
			must(t, d.RMSNorm(p.norm, float32(p.eps)))
			dlx, err := p.out.MatMulDevice(d)
			must(t, err)
			d.Free()
			dlx.Free()
		}
		reset()
		doSeed()
		must(t, cuda.GraphSync()) // warmup
		reset()
		t0 := time.Now()
		doSeed()
		must(t, cuda.GraphSync())
		return time.Since(t0)
	}
	mmqDur := timeSeed(pf)
	pf16 := buildRawPrefill(t, rf, "qwen2", false)
	defer pf16.free()
	f16Dur := timeSeed(pf16)
	t.Logf("prefill P=%d latency: MMQ %.1f ms | f16 %.1f ms (%.2fx) — MMQ at ~half the prefill weight VRAM",
		P, float64(mmqDur.Microseconds())/1000, float64(f16Dur.Microseconds())/1000, float64(mmqDur)/float64(f16Dur))

	t.Log("UNIFIED SERVE WITH int8 MMQ PREFILL: coherent + within int8 K-cache budget (half prefill VRAM)")
}
