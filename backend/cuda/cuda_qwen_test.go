//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

const qwenPath = "../../models/qwen2.5-0.5b-instruct-q8_0.gguf"

// transpose2D returns xᵀ as f32 (GGUF stores projection weights [out,in]; the
// resident matmul wants B[in,out]).
func transpose2D(x *tensor.Tensor) *tensor.Tensor {
	xf := x.Cast(tensor.F32).Contiguous().Storage().F32()
	r, c := x.Shape()[0], x.Shape()[1]
	out := tensor.New(tensor.F32, tensor.Shape{c, r})
	of := out.Storage().F32()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			of[j*r+i] = xf[i*c+j]
		}
	}
	return out
}

type qwenLayer struct {
	an, fn         *cuda.ResidentVec
	wq, wk, wv, wo *cuda.ResidentBQ8
	bq, bk, bv     *cuda.ResidentVec
	wg, wu, wd     *cuda.ResidentBQ8
	cache          *cuda.KVCache
}

// Qwen on the GPU: arch=qwen2 (rejected by nlp.LlamaFromGGUF), so the
// weights/biases/config are read straight from the GGUF and made resident here —
// the DISTINCT bit vs Llama is the Q/K/V projection BIASES, added via the new
// DeviceF32.AddBias. Demonstrates additional model families + the AddBias
// primitive, and that the engine generalizes across Qwen scales (0.5B → 3B).
func TestCUDAQwenGenerate(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
	for _, path := range []string{
		qwenPath,
		"../../models/qwen2.5-1.5b-instruct-q8_0.gguf",
		"../../models/qwen2.5-3b-instruct-q8_0.gguf", // config-driven, skips if absent
	} {
		t.Run(path, func(t *testing.T) { runQwen(t, path) })
	}
}

func runQwen(t *testing.T, qwenPath string) []int {
	skipNoGPU(t)
	if _, err := os.Stat(qwenPath); err != nil {
		t.Skipf("model not present (%s)", qwenPath)
	}
	f, err := gguf.ReadFile(qwenPath)
	if err != nil {
		t.Fatal(err)
	}
	mdInt := func(k string) int {
		switch v := f.Metadata[k].(type) {
		case uint32:
			return int(v)
		case int32:
			return int(v)
		case uint64:
			return int(v)
		case int64:
			return int(v)
		case int:
			return v
		default:
			t.Fatalf("meta %s type %T", k, v)
			return 0
		}
	}
	mdF := func(k string) float64 {
		switch v := f.Metadata[k].(type) {
		case float32:
			return float64(v)
		case float64:
			return v
		default:
			t.Fatalf("meta %s type %T", k, v)
			return 0
		}
	}
	dim := mdInt("qwen2.embedding_length")
	nLayers := mdInt("qwen2.block_count")
	heads := mdInt("qwen2.attention.head_count")
	kv := mdInt("qwen2.attention.head_count_kv")
	eps := mdF("qwen2.attention.layer_norm_rms_epsilon")
	ropeBase := mdF("qwen2.rope.freq_base")
	hd := dim / heads
	wkv := kv * hd

	ten := func(name string) *tensor.Tensor {
		x := f.Tensors[name]
		if x == nil {
			t.Fatalf("missing tensor %s", name)
		}
		return x
	}
	q8 := func(name string) *cuda.ResidentBQ8 {
		r, e := cuda.NewResidentBQ8(transpose2D(ten(name)))
		must(t, e)
		return r
	}
	vec := func(name string) *cuda.ResidentVec {
		r, e := cuda.NewResidentVec(ten(name).Cast(tensor.F32))
		must(t, e)
		return r
	}

	emb, _ := cuda.NewResidentB(ten("token_embd.weight").Cast(tensor.F32)) // [vocab,dim] gather table
	onorm := vec("output_norm.weight")
	out := q8("output.weight")
	const maxGen = 28
	prompt := "The capital of France is"
	tokzr, err := nlp.BPEFromGGUF(f.Metadata)
	must(t, err)
	ids := tokzr.Encode(prompt)
	maxSeq := len(ids) + maxGen + 2

	layers := make([]*qwenLayer, nLayers)
	for i := 0; i < nLayers; i++ {
		p := "blk." + itoa(i) + "."
		c, _ := cuda.NewKVCache(maxSeq, wkv)
		layers[i] = &qwenLayer{
			an: vec(p + "attn_norm.weight"), fn: vec(p + "ffn_norm.weight"),
			wq: q8(p + "attn_q.weight"), wk: q8(p + "attn_k.weight"), wv: q8(p + "attn_v.weight"), wo: q8(p + "attn_output.weight"),
			bq: vec(p + "attn_q.bias"), bk: vec(p + "attn_k.bias"), bv: vec(p + "attn_v.bias"),
			wg: q8(p + "ffn_gate.weight"), wu: q8(p + "ffn_up.weight"), wd: q8(p + "ffn_down.weight"),
			cache: c,
		}
	}

	// one decode step at position pos → device logits argmax
	step := func(tok int32, pos int) int {
		dx, e := emb.Embed([]int32{tok})
		must(t, e)
		for _, l := range layers {
			dh, _ := dx.RMSNormTo(l.an, float32(eps))
			dq, _ := l.wq.QMatMulDevice(dh)
			dk, _ := l.wk.QMatMulDevice(dh)
			dv, _ := l.wv.QMatMulDevice(dh)
			dh.Free()
			must(t, dq.AddBias(l.bq))
			must(t, dk.AddBias(l.bk))
			must(t, dv.AddBias(l.bv))
			dq.RoPE(backend.RoPEAttrs{Base: ropeBase, Heads: heads, PosOffset: pos})
			dk.RoPE(backend.RoPEAttrs{Base: ropeBase, Heads: kv, PosOffset: pos})
			must(t, l.cache.Append(dk, dv))
			da, e := cuda.GroupedQueryAttentionKV(dq, l.cache.K(), l.cache.V(), heads, kv)
			must(t, e)
			dq.Free()
			dk.Free()
			dv.Free()
			must(t, l.wo.QMatMulAccInto(da, dx))
			da.Free()
			dh2, _ := dx.RMSNormTo(l.fn, float32(eps))
			dg, _ := l.wg.QMatMulDevice(dh2)
			du, _ := l.wu.QMatMulDevice(dh2)
			dh2.Free()
			dg.SwiGLU(du)
			du.Free()
			must(t, l.wd.QMatMulAccInto(dg, dx))
			dg.Free()
		}
		must(t, dx.RMSNorm(onorm, float32(eps)))
		dl, e := out.QMatMulDevice(dx)
		must(t, e)
		dx.Free()
		tk := dl.Argmax()
		dl.Free()
		return tk
	}

	var last int
	for i, id := range ids {
		last = step(int32(id), i) // prefill (untimed)
	}
	gen := make([]int, 0, maxGen)
	t0 := time.Now()
	for n := 0; n < maxGen; n++ {
		if last == 151643 || last == 151645 { // Qwen EOS / im_end
			break
		}
		gen = append(gen, last)
		last = step(int32(last), len(ids)+n)
	}
	tokPerSec := float64(len(gen)) / time.Since(t0).Seconds()

	t.Logf("MODEL: qwen2 dim=%d layers=%d GQA %d:%d QKV-bias (%s) on RTX 3060",
		dim, nLayers, heads, kv, qwenPath[strings.LastIndex(qwenPath, "/")+1:])
	t.Logf("DECODE: %.1f tok/s (alloc-path, un-graphed)", tokPerSec)
	t.Logf("PROMPT:    %q", prompt)
	t.Logf("GENERATED: %q", strings.TrimSpace(tokzr.Decode(gen)))
	if len(gen) == 0 {
		t.Fatal("generated nothing")
	}
	// Free the resident weights before returning — CUDA device memory is NOT reclaimed
	// by Go's GC, so leaking a full model here would double VRAM residency when a caller
	// (qwenFixedVsAlloc) then builds a second copy, OOMing at 3B scale.
	emb.Free()
	onorm.Free()
	out.Free()
	for _, l := range layers {
		l.an.Free()
		l.fn.Free()
		l.wq.Free()
		l.wk.Free()
		l.wv.Free()
		l.wo.Free()
		l.bq.Free()
		l.bk.Free()
		l.bv.Free()
		l.wg.Free()
		l.wu.Free()
		l.wd.Free()
		l.cache.Free()
	}
	return gen
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [12]byte
	p := len(b)
	for i > 0 {
		p--
		b[p] = byte('0' + i%10)
		i /= 10
	}
	return string(b[p:])
}
