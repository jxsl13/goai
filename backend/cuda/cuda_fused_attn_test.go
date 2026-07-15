//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// The flash decode-attention kernel (GQA K/V-shared split-K online-softmax
// partials + merge) must match the 3-kernel dpos chain it replaces across head
// configs, head dims, cache sizes and positions — including the partial-warp
// tail (lim not a multiple of the warp count) and the pos=0 single-key edge.
//
// (A simpler one-block-per-q-head fused kernel — v1 fp64 output 74 tok/s, v2
// f32 warp-partitioned 102 — was BUILT, measured strictly dominated by both
// the chain and flash at every context depth, and removed; see the Tw52 spec
// entry for the record.)
func TestCUDAFlashGQADposParity(t *testing.T) {
	skipNoGPU(t)
	cases := []struct {
		qHeads, kvHeads, hd, seqKV, pos int
	}{
		{8, 2, 8, 10, 9},      // small smoke (mirrors the dpos twin test)
		{8, 2, 8, 10, 0},      // pos=0: exactly one key contributes
		{4, 4, 64, 33, 31},    // MHA ratio 1, odd cache size, partial warp tail
		{32, 8, 64, 128, 127}, // TinyLlama-class GQA shape
		{32, 4, 128, 512, 300},
		{8, 1, 128, 2048, 2047}, // MQA at full fixed-cache depth
	}
	pos, err := cuda.NewDevicePos()
	must(t, err)
	defer pos.Free()
	for _, c := range cases {
		t.Run(fmt.Sprintf("q%d_kv%d_hd%d_ctx%d_p%d", c.qHeads, c.kvHeads, c.hd, c.seqKV, c.pos), func(t *testing.T) {
			wq, wkv := c.qHeads*c.hd, c.kvHeads*c.hd
			q := bench.RandF32(tensor.Shape{1, wq}, 2)
			k := bench.RandF32(tensor.Shape{c.seqKV, wkv}, 3)
			v := bench.RandF32(tensor.Shape{c.seqKV, wkv}, 4)
			dq, _ := cuda.UploadF32(q)
			defer dq.Free()
			dk, _ := cuda.UploadF32(k)
			defer dk.Free()
			dv, _ := cuda.UploadF32(v)
			defer dv.Free()
			must(t, pos.Set(c.pos))

			ref, err := cuda.GroupedQueryAttentionKVDpos(dq, dk, dv, c.qHeads, c.kvHeads, pos)
			must(t, err)
			defer ref.Free()

			hr, _ := ref.ToHost()
			outFl, _ := cuda.UploadF32(bench.RandF32(tensor.Shape{1, wq}, 6)) // junk-filled: the kernel must overwrite every dim
			defer outFl.Free()
			must(t, cuda.GroupedQueryAttentionKVDposFlashInto(dq, dk, dv, c.qHeads, c.kvHeads, pos, outFl))
			hl, _ := outFl.ToHost()
			if m := maxAbsDiff(hr, hl); m > 1e-5 {
				t.Fatalf("flash != 3-kernel chain, maxAbs %.3e", m)
			}
		})
	}
}

// Hostile-magnitude inputs: the flash kernel's online softmax must stay finite
// where naive exp would overflow f32 (logits ~ ±1e4·scale) — the running-max
// subtraction, the m==-INF empty-state correction and the l==0 merge guard are
// the failure spots. Parity vs the chain (whose softmax is double) must hold.
func TestCUDAFlashGQAExtremeMagnitudes(t *testing.T) {
	skipNoGPU(t)
	const qHeads, kvHeads, hd, seqKV, p = 8, 2, 64, 257, 255 // odd sizes: partial tiles + partial warps
	wq, wkv := qHeads*hd, kvHeads*hd
	pos, err := cuda.NewDevicePos()
	must(t, err)
	defer pos.Free()
	must(t, pos.Set(p))
	for _, scale := range []float32{1e4, -1e4, 1e-30} { // huge, huge-negative, denormal-tiny
		qd := make([]float32, wq)
		rq := bench.RandF32(tensor.Shape{1, wq}, 7)
		for i := range qd {
			qd[i] = float32(rq.AtF64(0, i)) * scale
		}
		q := tensor.FromFloat32(tensor.Shape{1, wq}, qd)
		k := bench.RandF32(tensor.Shape{seqKV, wkv}, 8)
		v := bench.RandF32(tensor.Shape{seqKV, wkv}, 9)
		dq, _ := cuda.UploadF32(q)
		dk, _ := cuda.UploadF32(k)
		dv, _ := cuda.UploadF32(v)
		ref, err := cuda.GroupedQueryAttentionKVDpos(dq, dk, dv, qHeads, kvHeads, pos)
		must(t, err)
		out, _ := cuda.UploadF32(bench.RandF32(tensor.Shape{1, wq}, 10))
		must(t, cuda.GroupedQueryAttentionKVDposFlashInto(dq, dk, dv, qHeads, kvHeads, pos, out))
		hr, _ := ref.ToHost()
		hf, _ := out.ToHost()
		for i := 0; i < wq; i++ {
			if x := hf.AtF64(0, i); x != x || x > 1e30 || x < -1e30 {
				t.Fatalf("scale %g: non-finite output at %d: %g", scale, i, x)
			}
		}
		if m := maxAbsDiff(hr, hf); m > 1e-4 { // wider bar: saturated softmax amplifies exp rounding
			t.Fatalf("scale %g: flash != chain, maxAbs %.3e", scale, m)
		}
		dq.Free()
		dk.Free()
		dv.Free()
		ref.Free()
		out.Free()
	}
}

// TestCUDAFusedAttnLongCtx measures graph decode throughput DEEP in the context
// window (TinyLlama Q4_K, 2048-slot fixed cache, ~2000 tokens of state), where
// the 3-kernel chain's per-q-head K/V re-reads and score round-trip scale with
// seqKV but the flash kernel's GQA-shared tiles cut K/V traffic by the group
// factor. A/B via GOAI_CUDA_FUSED_ATTN=0 (chain) vs unset (flash) — the
// short-ctx (160) verdict comes from the sweep A/B; this is the long-ctx leg.
//
// MEASURED (RTX 3060, 2026-07-15, interleaved pairs): chain 168 tok/s — it
// sits at the K/V bandwidth limit (~90µs/layer INCLUDING the 8× GQA
// duplication), so one-block-per-q-head fusion lost (v1 fp64 output 74, v2
// f32 warp-partitioned 102, both removed). FLASH (K/V staged in shared once
// per kv-head for all group q-heads, split-K online softmax + merge) wins:
// 212 tok/s = +26% over the chain here, +3.5% at ctx160.
func TestCUDAFusedAttnLongCtx(t *testing.T) {
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

	const maxSeq = 2048
	const depth = 1980 // eager-prefilled state so the timed window attends ~2000 keys
	gd := buildRawGraphDecoder(t, rf, "llama", maxSeq, fromF32(quantQ4K))
	defer gd.free()

	tok := int32(1)
	for p := 0; p < depth; p++ {
		tok = int32(gd.step(t, tok, p))
	}
	gd.capture(t)
	// warm the graph, then time a 32-step window deep in the context.
	for d := 0; d < 8; d++ {
		tok = int32(gd.stepGraph(t, tok, depth+d))
	}
	const steps = 32
	t0 := time.Now()
	for d := 0; d < steps; d++ {
		tok = int32(gd.stepGraph(t, tok, depth+8+d))
	}
	tps := float64(steps) / time.Since(t0).Seconds()
	mode := "flash"
	if os.Getenv("GOAI_CUDA_FUSED_ATTN") == "0" {
		mode = "3-kernel chain"
	}
	t.Logf("TinyLlama-1.1B Q4_K GRAPH decode @ctx≈%d (%s): %.1f tok/s", depth+8+steps/2, mode, tps)
}
