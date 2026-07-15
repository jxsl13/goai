//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// cpuBlock runs one TinyLlama decoder block on the CPU backend (prefill self-
// attention), mirroring nlp.Llama.DecodeStep's per-block forward — used for the
// layers offloaded to CPU-SIMD in the hybrid split.
func cpuBlock(t *testing.T, ctx *backend.Context, x *tensor.Tensor, b *nlp.LlamaBlock, cfg nlp.LlamaConfig) *tensor.Tensor {
	f32 := func(tt *tensor.Tensor) *tensor.Tensor { return tt.Cast(tensor.F32) }
	kv := cfg.KVHeads
	if kv == 0 {
		kv = cfg.Heads
	}
	ex := func(op backend.Op, a backend.Attrs, ins ...*tensor.Tensor) *tensor.Tensor {
		o, e := backend.Execute(ctx, op, ins, a)
		must(t, e)
		return o[0]
	}
	xb := ex(backend.OpRMSNorm, backend.NormAttrs{Eps: cfg.Eps}, x, f32(b.AttnNorm.Gamma))
	q := ex(backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads}, ex(backend.OpMatMul, nil, xb, f32(b.Wq)))
	k := ex(backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}, ex(backend.OpMatMul, nil, xb, f32(b.Wk)))
	v := ex(backend.OpMatMul, nil, xb, f32(b.Wv))
	a := ex(backend.OpMHA, backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}, q, k, v)
	x1 := ex(backend.OpAdd, nil, x, ex(backend.OpMatMul, nil, a, f32(b.Wo)))
	xf := ex(backend.OpRMSNorm, backend.NormAttrs{Eps: cfg.Eps}, x1, f32(b.FFNNorm.Gamma))
	gate := ex(backend.OpSiLU, nil, ex(backend.OpMatMul, nil, xf, f32(b.FFN.Wgate)))
	act := ex(backend.OpMul, nil, gate, ex(backend.OpMatMul, nil, xf, f32(b.FFN.Wup)))
	return ex(backend.OpAdd, nil, x1, ex(backend.OpMatMul, nil, act, f32(b.FFN.Wdown)))
}

// T631 hybrid-split PROOF-OF-MECHANISM: run a prompt's forward with the first N
// decoder layers on the GPU (device-resident) and the rest OFFLOADED to the CPU-SIMD
// backend, with a single GPU→host transfer of the hidden state at the split. The
// hybrid must predict the SAME token as the all-GPU forward — proving a model can be
// split across GPU + CPU (the mechanism T631 needs for >VRAM models).
func TestCUDAHybridSplitDecode(t *testing.T) {
	skipNoGPU(t)
	if _, err := os.Stat(tinyLlamaPath); err != nil {
		t.Skipf("model not present (%s)", tinyLlamaPath)
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	must(t, err)
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	must(t, err)
	cfg := model.Config
	nL := len(model.Blocks)

	rl := buildResidentLlama(t, model) // GPU (f32) full model
	defer rl.free()

	tokens := []int{1, 15043, 29892, 590, 1024, 338}
	// all-GPU reference: full forward, last-position argmax
	allGPU := rl.forward(t, i32(tokens))
	wantTok := argmaxRow(allGPU, allGPU.Shape()[0]-1)

	cpu, _ := backend.Get(backend.CPU)
	cctx := backend.NewContext().WithBackend(cpu)

	for _, split := range []int{nL, nL * 3 / 4, nL / 2} { // all-GPU, 25% on CPU, 50% on CPU
		// GPU head: embed + layers [0,split)
		dx, e := rl.emb.Embed(i32(tokens))
		must(t, e)
		for i := 0; i < split; i++ {
			dx, e = rl.layers[i].forward(dx)
			must(t, e)
		}
		hidden, e := dx.ToHost() // the GPU→host transfer at the split boundary
		must(t, e)
		dx.Free()
		// CPU tail: layers [split, nL) + final norm + output projection
		x := hidden
		for i := split; i < nL; i++ {
			x = cpuBlock(t, cctx, x, model.Blocks[i], cfg)
		}
		o, e := backend.Execute(cctx, backend.OpRMSNorm, []*tensor.Tensor{x, model.Norm.Gamma.Cast(tensor.F32)}, backend.NormAttrs{Eps: cfg.Eps})
		must(t, e)
		lg, e := backend.Execute(cctx, backend.OpMatMul, []*tensor.Tensor{o[0], model.Out.Cast(tensor.F32)}, nil)
		must(t, e)
		gotTok := argmaxRow(lg[0], lg[0].Shape()[0]-1)
		onGPU := split
		if gotTok != wantTok {
			t.Fatalf("split=%d (%d layers on CPU): hybrid token %d != all-GPU %d", split, nL-onGPU, gotTok, wantTok)
		}
		t.Logf("split=%2d → %2d GPU layers + %2d CPU-SIMD layers: token %d == all-GPU ✓", split, onGPU, nL-onGPU, gotTok)
	}
	t.Log("T631 mechanism proven: a model runs correctly split across GPU + CPU-SIMD (hidden transferred at the boundary).")
}

func i32(x []int) []int32 {
	o := make([]int32, len(x))
	for i, v := range x {
		o[i] = int32(v)
	}
	return o
}
