//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// cpuTailLayer holds one CPU-offloaded layer's f32 weights + its growing CPU-side
// K/V cache (the dual-cache half of the hybrid: GPU layers cache on-device, CPU
// layers cache on host).
type cpuTailLayer struct {
	an, fn         *tensor.Tensor
	wq, wk, wv, wo *tensor.Tensor
	wg, wu, wd     *tensor.Tensor
	cacheK, cacheV *tensor.Tensor // [pos, wkv], grown per token
}

func castF32(x *tensor.Tensor) *tensor.Tensor { return x.Cast(tensor.F32) }

// concatRows appends row b [1,c] to a [r,c] (nil → [1,c]).
func concatRows(a, b *tensor.Tensor) *tensor.Tensor {
	c := b.Shape()[1]
	r := 0
	if a != nil {
		r = a.Shape()[0]
	}
	out := tensor.New(tensor.F32, tensor.Shape{r + 1, c})
	of := out.Storage().F32()
	if a != nil {
		copy(of[:r*c], a.Contiguous().Storage().F32())
	}
	copy(of[r*c:], b.Contiguous().Storage().F32())
	return out
}

// T631 hybrid DECODE (autoregressive, dual KV caches): the first `split` layers
// decode on the GPU with device KV-caches; the rest are offloaded to the CPU-SIMD
// backend with host-side KV-caches; the hidden state crosses GPU→host once per
// token at the split. Must produce the SAME greedy tokens as the all-GPU decode —
// proving the split works across tokens with caches on both sides (the actual T631
// serving path for models that exceed VRAM).
func TestCUDAHybridDecode(t *testing.T) {
	skipNoGPU(t)
	if _, err := os.Stat(tinyLlamaPath); err != nil {
		t.Skipf("model not present (%s)", tinyLlamaPath)
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	must(t, err)
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	must(t, err)
	cfg := model.Config
	kvh := cfg.KVHeads
	if kvh == 0 {
		kvh = cfg.Heads
	}
	nL := len(model.Blocks)
	prompt := []int32{1, 15043, 29892, 590, 1024, 338}
	const steps = 5

	// --- all-GPU decode reference ---
	rl := buildResidentLlama(t, model)
	defer rl.free()
	gpuCaches, _ := rl.newCaches(len(prompt) + steps + 1)
	defer func() {
		for _, c := range gpuCaches {
			c.Free()
		}
	}()
	gpuDecode := func(ids []int32) *tensor.Tensor {
		dx, _ := rl.emb.Embed(ids)
		for i, l := range rl.layers {
			dx, _ = l.forwardKV(dx, gpuCaches[i])
		}
		dx.RMSNorm(rl.norm, float32(cfg.Eps))
		dl, _ := rl.out.MatMulDevice(dx)
		dx.Free()
		out, _ := dl.ToHost()
		dl.Free()
		return out
	}
	var refToks []int
	{
		var last *tensor.Tensor
		for i, id := range prompt {
			last = gpuDecode([]int32{id})
			_ = i
		}
		for s := 0; s < steps; s++ {
			tok := argmaxRow(last, 0)
			refToks = append(refToks, tok)
			last = gpuDecode([]int32{int32(tok)})
		}
	}

	// --- hybrid decode: GPU layers [0,split) + CPU-SIMD layers [split,nL) ---
	const split = 11 // half on CPU
	hrl := buildResidentLlama(t, model)
	defer hrl.free()
	hCaches := make([]*cuda.KVCache, split)
	wkv := kvh * (cfg.Dim / cfg.Heads)
	for i := range hCaches {
		hCaches[i], _ = cuda.NewKVCache(len(prompt)+steps+1, wkv)
	}
	defer func() {
		for _, c := range hCaches {
			c.Free()
		}
	}()
	tail := make([]*cpuTailLayer, nL)
	for i := split; i < nL; i++ {
		b := model.Blocks[i]
		tail[i] = &cpuTailLayer{
			an: castF32(b.AttnNorm.Gamma), fn: castF32(b.FFNNorm.Gamma),
			wq: castF32(b.Wq), wk: castF32(b.Wk), wv: castF32(b.Wv), wo: castF32(b.Wo),
			wg: castF32(b.FFN.Wgate), wu: castF32(b.FFN.Wup), wd: castF32(b.FFN.Wdown),
		}
	}
	cpu, _ := backend.Get(backend.CPU)
	cctx := backend.NewContext().WithBackend(cpu)
	ex := func(op backend.Op, a backend.Attrs, ins ...*tensor.Tensor) *tensor.Tensor {
		o, e := backend.Execute(cctx, op, ins, a)
		must(t, e)
		return o[0]
	}
	hybridDecode := func(tok int32, pos int) *tensor.Tensor {
		dx, _ := hrl.emb.Embed([]int32{tok})
		for i := 0; i < split; i++ {
			dx, _ = hrl.layers[i].forwardKV(dx, hCaches[i])
		}
		x, _ := dx.ToHost() // GPU→host transfer at the split
		dx.Free()
		for i := split; i < nL; i++ {
			l := tail[i]
			xb := ex(backend.OpRMSNorm, backend.NormAttrs{Eps: cfg.Eps}, x, l.an)
			q := ex(backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads, PosOffset: pos}, ex(backend.OpMatMul, nil, xb, l.wq))
			k := ex(backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kvh, PosOffset: pos}, ex(backend.OpMatMul, nil, xb, l.wk))
			v := ex(backend.OpMatMul, nil, xb, l.wv)
			l.cacheK = concatRows(l.cacheK, k)
			l.cacheV = concatRows(l.cacheV, v)
			a := ex(backend.OpMHA, backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kvh, Causal: false}, q, l.cacheK, l.cacheV)
			x1 := ex(backend.OpAdd, nil, x, ex(backend.OpMatMul, nil, a, l.wo))
			xf := ex(backend.OpRMSNorm, backend.NormAttrs{Eps: cfg.Eps}, x1, l.fn)
			gate := ex(backend.OpSiLU, nil, ex(backend.OpMatMul, nil, xf, l.wg))
			act := ex(backend.OpMul, nil, gate, ex(backend.OpMatMul, nil, xf, l.wu))
			x = ex(backend.OpAdd, nil, x1, ex(backend.OpMatMul, nil, act, l.wd))
		}
		nx := ex(backend.OpRMSNorm, backend.NormAttrs{Eps: cfg.Eps}, x, castF32(model.Norm.Gamma))
		return ex(backend.OpMatMul, nil, nx, castF32(model.Out))
	}
	var last *tensor.Tensor
	for i, id := range prompt {
		last = hybridDecode(id, i)
	}
	match := 0
	for s := 0; s < steps; s++ {
		tok := argmaxRow(last, 0)
		if tok == refToks[s] {
			match++
		} else {
			t.Errorf("step %d: hybrid %d != all-GPU %d", s, tok, refToks[s])
		}
		last = hybridDecode(int32(tok), len(prompt)+s)
	}
	t.Logf("HYBRID DECODE %d GPU + %d CPU-SIMD layers, dual KV-caches: %d/%d tokens == all-GPU decode",
		split, nL-split, match, steps)
	if match < steps {
		t.Fatal("hybrid decode diverged from all-GPU")
	}
	t.Log("T631 decode mechanism proven: autoregressive generation split across GPU+CPU with caches on both sides.")
}
