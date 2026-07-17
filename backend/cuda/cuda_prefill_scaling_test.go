//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// TestCUDAPrefillAttnScaling measures how the attention share of prefill grows with
// context length. The GEMMs are O(seq); the materialize-scores attention path
// (cu_gqa_scores → cu_attn_softmax → cu_gqa_out, a full [heads,seq,seq] buffer) is
// O(seq²), so its share must rise with seq. This is the measure-first (§V22) input to
// the fused-attention prefill lever: build flash-prefill only where the curve says it
// pays. Reports attention ms + share + whole-prefill tok/s at each seq.
func TestCUDAPrefillAttnScaling(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
	skipNoGPU(t)
	if _, err := os.Stat(tinyLlamaPath); err != nil {
		t.Skipf("model not present (%s)", tinyLlamaPath)
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	must(t, err)
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	must(t, err)
	rl := buildResidentLlama(t, model)
	defer rl.free()

	seqs := []int{128, 256, 512, 1024}
	for _, seq := range seqs {
		ids := make([]int32, seq)
		for i := range ids {
			ids[i] = int32((i*131 + 1) % model.Config.Vocab)
		}

		var attn, wall time.Duration
		// One full prefill; time only the attention op per layer (sync-bracketed), plus
		// the untimed remainder for the whole-prefill wall (single sync at the end).
		run := func(measure bool) {
			dx, err := rl.emb.Embed(ids)
			must(t, err)
			for _, l := range rl.layers {
				rq := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.heads}
				rk := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.kv}
				dh, err := dx.RMSNormTo(l.gAttn, float32(l.eps))
				must(t, err)
				dq, err := l.wq.MatMulDevice(dh)
				must(t, err)
				dk, err := l.wk.MatMulDevice(dh)
				must(t, err)
				dv, err := l.wv.MatMulDevice(dh)
				must(t, err)
				dh.Free()
				must(t, dq.RoPE(rq))
				must(t, dk.RoPE(rk))
				var da *cuda.DeviceF32
				if measure {
					must(t, cuda.GraphSync())
					s := time.Now()
					da, err = cuda.GroupedQueryAttention(dq, dk, dv, l.heads, l.kv, true)
					must(t, err)
					must(t, cuda.GraphSync())
					attn += time.Since(s)
				} else {
					da, err = cuda.GroupedQueryAttention(dq, dk, dv, l.heads, l.kv, true)
					must(t, err)
				}
				dq.Free()
				dk.Free()
				dv.Free()
				must(t, l.wo.MatMulAccInto(da, dx))
				da.Free()
				dh2, err := dx.RMSNormTo(l.gFFN, float32(l.eps))
				must(t, err)
				dgate, err := l.wg.MatMulDevice(dh2)
				must(t, err)
				dup, err := l.wu.MatMulDevice(dh2)
				must(t, err)
				dh2.Free()
				must(t, dgate.SwiGLU(dup))
				dup.Free()
				must(t, l.wd.MatMulAccInto(dgate, dx))
				dgate.Free()
			}
			must(t, dx.RMSNorm(rl.norm, float32(rl.eps)))
			dl, err := rl.out.MatMulDevice(dx)
			must(t, err)
			dl.Free()
			dx.Free()
		}

		run(false) // warm-up
		const reps = 3
		attn = 0
		w0 := time.Now()
		for i := 0; i < reps; i++ {
			run(true)
		}
		must(t, cuda.GraphSync())
		wall = time.Since(w0)
		attnMs := float64(attn.Microseconds()) / 1000 / reps
		wallMs := float64(wall.Microseconds()) / 1000 / reps
		tokS := float64(seq) / (wallMs / 1000)
		t.Logf("seq=%-4d  prefill %.2f ms (%.0f tok/s)  attention %.2f ms (%.1f%%)",
			seq, wallMs, tokS, attnMs, 100*attnMs/wallMs)
	}
}
