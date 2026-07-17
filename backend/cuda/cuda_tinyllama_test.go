//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

const tinyLlamaPath = "../../models/tinyllama-1.1b-chat-q8_0.gguf"

// First REAL-MODEL validation on the GPU: run TinyLlama-1.1B's decoder block 0
// (arch=llama, GQA 32/4, no bias) on the resident CUDA path with the actual GGUF
// weights, and compare against the same block computed op-by-op on the CPU
// backend (the nlp reference: RMSNorm + projections + RoPE + OpMHA + Wo +
// residual, then SwiGLU FFN + residual). Skips if the model file isn't present.
func TestCUDATinyLlamaBlockMatchesCPU(t *testing.T) {
	if testing.Short() {
		t.Skip("model-loading integration test; -short = kernel-level gate")
	}
	skipNoGPU(t)
	if _, err := os.Stat(tinyLlamaPath); err != nil {
		t.Skipf("model not present (%s) — fetch per models/README.md", tinyLlamaPath)
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	if err != nil {
		t.Fatalf("gguf: %v", err)
	}
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatalf("LlamaFromGGUF: %v", err)
	}
	cfg := model.Config
	kv := cfg.KVHeads
	if kv == 0 {
		kv = cfg.Heads
	}
	hd := cfg.Dim / cfg.Heads
	blk := model.Blocks[0]
	t.Logf("TinyLlama: dim=%d heads=%d kv=%d hd=%d hidden=%d ropeBase=%v eps=%v",
		cfg.Dim, cfg.Heads, kv, hd, cfg.Hidden, cfg.RopeBase, cfg.Eps)

	const seq = 4
	x := bench.RandF32(tensor.Shape{seq, cfg.Dim}, 1)
	ropeQ := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads}
	ropeK := backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}

	// LlamaFromGGUF builds the model in f64; the CUDA path is f32, so cast the
	// weights to f32 (and run the CPU reference in f32 too) for a like-for-like
	// comparison of the f32 GPU inference.
	f32 := func(t *tensor.Tensor) *tensor.Tensor { return t.Cast(tensor.F32) }
	gAttn, gFFN := f32(blk.AttnNorm.Gamma), f32(blk.FFNNorm.Gamma)
	wq, wk, wv, wo := f32(blk.Wq), f32(blk.Wk), f32(blk.Wv), f32(blk.Wo)
	wg, wu, wd := f32(blk.FFN.Wgate), f32(blk.FFN.Wup), f32(blk.FFN.Wdown)

	// ---------- CPU reference (matches nlp.Llama block forward) ----------
	cpu, _ := backend.Get(backend.CPU)
	rc := backend.NewContext().WithBackend(cpu)
	ex := func(op backend.Op, a backend.Attrs, ins ...*tensor.Tensor) *tensor.Tensor {
		o, e := backend.Execute(rc, op, ins, a)
		must(t, e)
		return o[0]
	}
	xb := ex(backend.OpRMSNorm, backend.NormAttrs{Eps: cfg.Eps}, x, gAttn)
	q := ex(backend.OpRoPE, ropeQ, ex(backend.OpMatMul, nil, xb, wq))
	k := ex(backend.OpRoPE, ropeK, ex(backend.OpMatMul, nil, xb, wk))
	v := ex(backend.OpMatMul, nil, xb, wv)
	a := ex(backend.OpMHA, backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}, q, k, v)
	x1 := ex(backend.OpAdd, nil, x, ex(backend.OpMatMul, nil, a, wo))
	xf := ex(backend.OpRMSNorm, backend.NormAttrs{Eps: cfg.Eps}, x1, gFFN)
	gate := ex(backend.OpSiLU, nil, ex(backend.OpMatMul, nil, xf, wg))
	act := ex(backend.OpMul, nil, gate, ex(backend.OpMatMul, nil, xf, wu))
	want := ex(backend.OpAdd, nil, x1, ex(backend.OpMatMul, nil, act, wd))

	// ---------- CUDA resident block with the SAME real weights ----------
	rGa, _ := cuda.NewResidentVec(gAttn)
	rGf, _ := cuda.NewResidentVec(gFFN)
	rWq, _ := cuda.NewResidentB(wq)
	rWk, _ := cuda.NewResidentB(wk)
	rWv, _ := cuda.NewResidentB(wv)
	rWo, _ := cuda.NewResidentB(wo)
	rWg, _ := cuda.NewResidentB(wg)
	rWu, _ := cuda.NewResidentB(wu)
	rWd, _ := cuda.NewResidentB(wd)
	defer func() {
		for _, r := range []interface{ Free() }{rGa, rGf, rWq, rWk, rWv, rWo, rWg, rWu, rWd} {
			r.Free()
		}
	}()

	dx, _ := cuda.UploadF32(x)
	dh, _ := dx.Clone()
	must(t, dh.RMSNorm(rGa, float32(cfg.Eps)))
	dq, err := rWq.MatMulDevice(dh)
	must(t, err)
	dk, err := rWk.MatMulDevice(dh)
	must(t, err)
	dv, err := rWv.MatMulDevice(dh)
	must(t, err)
	must(t, dq.RoPE(ropeQ))
	must(t, dk.RoPE(ropeK))
	da, err := cuda.GroupedQueryAttention(dq, dk, dv, cfg.Heads, kv, true)
	must(t, err)
	do, err := rWo.MatMulDevice(da)
	must(t, err)
	must(t, do.Add(dx))
	dh2, _ := do.Clone()
	must(t, dh2.RMSNorm(rGf, float32(cfg.Eps)))
	dgate, err := rWg.MatMulDevice(dh2)
	must(t, err)
	dup, err := rWu.MatMulDevice(dh2)
	must(t, err)
	must(t, dgate.SwiGLU(dup))
	ddown, err := rWd.MatMulDevice(dgate)
	must(t, err)
	must(t, ddown.Add(do))
	got, err := ddown.ToHost()
	must(t, err)
	for _, fr := range []*cuda.DeviceF32{dx, dh, dq, dk, dv, da, do, dh2, dgate, dup, ddown} {
		fr.Free()
	}

	var maxRel float64
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		g, w := got.AtF64(idx...), want.AtF64(idx...)
		if r := math.Abs(g-w) / (math.Abs(w) + 1e-2); r > maxRel {
			maxRel = r
		}
		if math.Abs(g-w) > 3e-2*math.Max(1, math.Abs(w)) {
			t.Fatalf("[%d]: cuda-block %v vs cpu %v", i, g, w)
		}
	}
	t.Logf("TinyLlama block-0 CUDA-vs-CPU max rel err = %.2e", maxRel)
}
