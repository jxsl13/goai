//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// forwardKV is resLayer.forward in KV-cache form: it appends this step's (post-
// RoPE) keys/values to kc and attends the WHOLE accumulated cache, so a decode
// step reuses every prior key/value instead of recomputing them. dx holds the
// nNew new tokens ([nNew, dim]); the new rows sit at absolute positions
// pos..pos+nNew-1 where pos = kc.Len() before the append. Consumes/free dx.
func (l *resLayer) forwardKV(dx *cuda.DeviceF32, kc *cuda.KVCache) (*cuda.DeviceF32, error) {
	pos := kc.Len()
	rq := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.heads, PosOffset: pos}
	rk := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.kv, PosOffset: pos}
	dh, err := dx.RMSNormTo(l.gAttn, float32(l.eps))
	if err != nil {
		return nil, err
	}
	dq, err := l.wq.MatMulDevice(dh)
	if err != nil {
		return nil, err
	}
	dk, _ := l.wk.MatMulDevice(dh)
	dv, _ := l.wv.MatMulDevice(dh)
	dh.Free()
	dq.RoPE(rq)
	dk.RoPE(rk)
	if err := kc.Append(dk, dv); err != nil {
		return nil, err
	}
	da, err := cuda.GroupedQueryAttentionKV(dq, kc.K(), kc.V(), l.heads, l.kv)
	if err != nil {
		return nil, err
	}
	dq.Free()
	dk.Free()
	dv.Free()
	if err := l.wo.MatMulAccInto(da, dx); err != nil { // dx += Wo·attn → dx = x + attn
		return nil, err
	}
	da.Free()
	dh2, _ := dx.RMSNormTo(l.gFFN, float32(l.eps))
	dgate, _ := l.wg.MatMulDevice(dh2)
	dup, _ := l.wu.MatMulDevice(dh2)
	dh2.Free()
	dgate.SwiGLU(dup)
	dup.Free()
	if err := l.wd.MatMulAccInto(dgate, dx); err != nil { // dx += Wd·ffn → dx = x1 + ffn
		return nil, err
	}
	dgate.Free()
	return dx, nil
}

func (rl *residentLlama) newCaches(maxSeq int) ([]*cuda.KVCache, error) {
	wkv := rl.layers[0].kv * (rl.layers[0].dim / rl.layers[0].heads)
	caches := make([]*cuda.KVCache, len(rl.layers))
	for i := range caches {
		c, err := cuda.NewKVCache(maxSeq, wkv)
		if err != nil {
			return nil, err
		}
		caches[i] = c
	}
	return caches, nil
}

// step runs embed → KV layers → final norm → output for the given new tokens,
// growing every layer's cache. Returns host logits [len(ids), vocab].
func (rl *residentLlama) step(tb testing.TB, ids []int32, caches []*cuda.KVCache) *tensor.Tensor {
	dx, err := rl.emb.Embed(ids)
	if err != nil {
		tb.Fatal(err)
	}
	for i, l := range rl.layers {
		if dx, err = l.forwardKV(dx, caches[i]); err != nil {
			tb.Fatal(err)
		}
	}
	if err := dx.RMSNorm(rl.norm, float32(rl.eps)); err != nil {
		tb.Fatal(err)
	}
	dl, err := rl.out.MatMulDevice(dx)
	if err != nil {
		tb.Fatal(err)
	}
	dx.Free()
	out, err := dl.ToHost()
	if err != nil {
		tb.Fatal(err)
	}
	dl.Free()
	return out
}

func argmaxRow(t *tensor.Tensor, r int) int {
	best, bv := 0, math.Inf(-1)
	for c := 0; c < t.Shape()[1]; c++ {
		if v := t.AtF64(r, c); v > bv {
			bv, best = v, c
		}
	}
	return best
}

// The core KV-cache correctness property: greedy autoregressive decode with the
// cache must produce the SAME logits (and thus the same tokens) as re-forwarding
// the whole growing sequence from scratch each step. If the cache, its RoPE
// PosOffset, or the seqQ×seqKV attention were wrong, the two would diverge.
func TestCUDATinyLlamaKVCacheMatchesReforward(t *testing.T) {
	m := loadTinyLlama(t)
	rl := buildResidentLlama(t, m)
	defer rl.free()

	prompt := []int32{1, 15043, 29892, 590, 1024, 338} // "Hello, my name is"
	const steps = 5
	caches, err := rl.newCaches(len(prompt) + steps + 1)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, c := range caches {
			c.Free()
		}
	}()

	seq := append([]int32{}, prompt...)
	kv := rl.step(t, prompt, caches) // prefill; [P, vocab]
	for s := 0; s < steps; s++ {
		ref := rl.forward(t, seq) // full re-forward of the current sequence
		kvRow, refRow := kv.Shape()[0]-1, ref.Shape()[0]-1
		var maxRel float64
		for c := 0; c < ref.Shape()[1]; c++ {
			a, b := kv.AtF64(kvRow, c), ref.AtF64(refRow, c)
			if r := math.Abs(a-b) / (math.Abs(b) + 1); r > maxRel {
				maxRel = r
			}
		}
		gkv, gref := argmaxRow(kv, kvRow), argmaxRow(ref, refRow)
		if gkv != gref {
			t.Fatalf("step %d: KV-cache argmax %d != re-forward %d (maxRel %.2e)", s, gkv, gref, maxRel)
		}
		if maxRel > 5e-4 {
			t.Fatalf("step %d: KV vs re-forward logits diverge, maxRel %.2e", s, maxRel)
		}
		t.Logf("step %d: next token %d, KV==re-forward (maxRel %.2e, cache len %d)", s, gkv, maxRel, caches[0].Len())
		seq = append(seq, int32(gkv))
		kv = rl.step(t, []int32{int32(gkv)}, caches) // decode one token via cache
	}
}

// Decode throughput: prefill P tokens, then time D single-token cache-backed
// decode steps. tok/s is directly comparable to llama.cpp llama-bench tg
// (worker sub-spec §PERF: llama.cpp Vulkan tg128 = 243 tok/s).
func benchDecode(b *testing.B, prefill, decode int) {
	m := loadTinyLlama(b)
	rl := buildResidentLlama(b, m)
	defer rl.free()
	prompt := make([]int32, prefill)
	for i := range prompt {
		prompt[i] = int32((i*131 + 1) % m.Config.Vocab)
	}
	for range b.N {
		b.StopTimer()
		caches, err := rl.newCaches(prefill + decode + 1)
		if err != nil {
			b.Fatal(err)
		}
		kv := rl.step(b, prompt, caches) // prefill (not timed)
		tok := int32(argmaxRow(kv, kv.Shape()[0]-1))
		b.StartTimer()
		for d := 0; d < decode; d++ {
			kv = rl.step(b, []int32{tok}, caches)
			tok = int32(argmaxRow(kv, 0))
		}
		b.StopTimer()
		for _, c := range caches {
			c.Free()
		}
		b.StartTimer()
	}
	b.ReportMetric(float64(decode)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkTinyLlamaDecodeGPU_p32(b *testing.B)  { benchDecode(b, 32, 64) }
func BenchmarkTinyLlamaDecodeGPU_p128(b *testing.B) { benchDecode(b, 128, 64) }
