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

// The fused decode-attention kernel (QKᵀ + scale/causal/softmax + ·V in one
// launch, scores in shared memory) must match the 3-kernel dpos chain it
// replaces across head configs, head dims, cache sizes and positions —
// including the partial-warp tail (lim not a multiple of the warp count) and
// the pos=0 single-key edge.
func TestCUDAFusedGQADposParity(t *testing.T) {
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

			out, _ := cuda.UploadF32(bench.RandF32(tensor.Shape{1, wq}, 5)) // junk-filled: the kernel must overwrite every dim
			defer out.Free()
			must(t, cuda.GroupedQueryAttentionKVDposFusedInto(dq, dk, dv, c.qHeads, c.kvHeads, pos, out))

			hr, _ := ref.ToHost()
			hf, _ := out.ToHost()
			if m := maxAbsDiff(hr, hf); m > 1e-5 {
				t.Fatalf("fused != 3-kernel chain, maxAbs %.3e", m)
			}
		})
	}
}

// TestCUDAFusedAttnLongCtx measures graph decode throughput DEEP in the context
// window (TinyLlama Q4_K, 2048-slot fixed cache, ~2000 tokens of state), where
// the 3-kernel chain's [heads,seqKV] score round-trip through global memory
// scales with seqKV but the fused kernel's shared-memory scores do not. A/B via
// GOAI_CUDA_FUSED_ATTN=1 (fused) vs unset (chain) — the short-ctx (160) verdict
// comes from the sweep A/B; this is the long-ctx leg.
//
// MEASURED (RTX 3060, 2026-07-15): chain 168 tok/s, fused v1 (fp64 output) 74,
// fused v2 (f32 + warp-partitioned output) 102 — the chain sits at the K/V
// bandwidth limit (~90µs/layer incl. the 8× GQA duplication), the one-block-
// per-q-head kernel is latency-bound. The lever that can beat the chain is
// GQA K/V SHARING (stage K/V tiles in shared once per kv-head for all group
// q-heads, online softmax, split-K + merge — flash decoding), which cuts K/V
// traffic 8×; the chain structurally cannot do that.
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
	mode := "3-kernel chain"
	if os.Getenv("GOAI_CUDA_FUSED_ATTN") == "1" {
		mode = "fused"
	}
	t.Logf("TinyLlama-1.1B Q4_K GRAPH decode @ctx≈%d (%s): %.1f tok/s", depth+8+steps/2, mode, tps)
}
