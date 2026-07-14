//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type resLayer struct {
	gAttn, gFFN            *cuda.ResidentVec
	wq, wk, wv, wo         *cuda.ResidentB
	wg, wu, wd             *cuda.ResidentB
	heads, kv, dim, hidden int
	ropeBase, eps          float64
}

func (l *resLayer) free() {
	for _, f := range []interface{ Free() }{l.gAttn, l.gFFN, l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
		f.Free()
	}
}

// forward consumes dx (frees it) and returns the layer output residual.
func (l *resLayer) forward(dx *cuda.DeviceF32) (*cuda.DeviceF32, error) {
	rq := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.heads}
	rk := backend.RoPEAttrs{Base: l.ropeBase, Heads: l.kv}
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
	da, err := cuda.GroupedQueryAttention(dq, dk, dv, l.heads, l.kv, true)
	if err != nil {
		return nil, err
	}
	dq.Free()
	dk.Free()
	dv.Free()
	if err := l.wo.MatMulAccInto(da, dx); err != nil { // dx += Wo·attn  → dx = x + attn
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

// End-to-end: the FULL TinyLlama-1.1B forward on the GPU — embedding → 22 resident
// decoder layers → final RMSNorm → output projection → logits — must predict the
// SAME token (greedy argmax) at every position as nlp.Llama on the CPU. This is
// the real-model inference validation the CUDA arc was built for.
func TestCUDATinyLlamaFullForwardMatchesCPU(t *testing.T) {
	skipNoGPU(t)
	if _, err := os.Stat(tinyLlamaPath); err != nil {
		t.Skipf("model not present (%s)", tinyLlamaPath)
	}
	f, err := gguf.ReadFile(tinyLlamaPath)
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	cfg := model.Config
	kv := cfg.KVHeads
	if kv == 0 {
		kv = cfg.Heads
	}
	cast := func(t *tensor.Tensor) *tensor.Tensor { return t.Cast(tensor.F32) }

	// --- CPU reference (nlp.Llama, f64) ---
	tokens := []int{1, 15043, 29892, 590, 1024, 338}
	cpu, _ := backend.Get(backend.CPU)
	logitsCPU, err := model.Forward(backend.NewContext().WithBackend(cpu), tokens)
	if err != nil {
		t.Fatal(err)
	}

	// --- upload resident model (f32) ---
	rEmb, _ := cuda.NewResidentB(cast(model.TokEmb)) // [vocab, dim]
	rNorm, _ := cuda.NewResidentVec(cast(model.Norm.Gamma))
	rOut, _ := cuda.NewResidentB(cast(model.Out)) // [dim, vocab]
	layers := make([]*resLayer, len(model.Blocks))
	for i, b := range model.Blocks {
		rGa, _ := cuda.NewResidentVec(cast(b.AttnNorm.Gamma))
		rGf, _ := cuda.NewResidentVec(cast(b.FFNNorm.Gamma))
		mk := func(w *tensor.Tensor) *cuda.ResidentB { r, _ := cuda.NewResidentB(cast(w)); return r }
		layers[i] = &resLayer{
			gAttn: rGa, gFFN: rGf,
			wq: mk(b.Wq), wk: mk(b.Wk), wv: mk(b.Wv), wo: mk(b.Wo),
			wg: mk(b.FFN.Wgate), wu: mk(b.FFN.Wup), wd: mk(b.FFN.Wdown),
			heads: cfg.Heads, kv: kv, dim: cfg.Dim, hidden: cfg.Hidden,
			ropeBase: cfg.RopeBase, eps: cfg.Eps,
		}
	}
	defer func() {
		rEmb.Free()
		rNorm.Free()
		rOut.Free()
		for _, l := range layers {
			l.free()
		}
	}()

	// --- GPU forward ---
	ids := make([]int32, len(tokens))
	for i, tk := range tokens {
		ids[i] = int32(tk)
	}
	dx, err := rEmb.Embed(ids)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range layers {
		dx, err = l.forward(dx)
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := dx.RMSNorm(rNorm, float32(cfg.Eps)); err != nil {
		t.Fatal(err)
	}
	dlogits, err := rOut.MatMulDevice(dx) // dx[seq,dim]·Out[dim,vocab] → [seq,vocab]
	if err != nil {
		t.Fatal(err)
	}
	dx.Free()
	logitsGPU, err := dlogits.ToHost()
	if err != nil {
		t.Fatal(err)
	}
	dlogits.Free()

	// --- compare greedy argmax per position + logit error ---
	seq, vocab := logitsCPU.Shape()[0], logitsCPU.Shape()[1]
	argmax := func(lg *tensor.Tensor, r int) (int, float64) {
		best, bv := 0, math.Inf(-1)
		for c := 0; c < vocab; c++ {
			if v := lg.AtF64(r, c); v > bv {
				bv, best = v, c
			}
		}
		return best, bv
	}
	var maxRel float64
	mism := 0
	for r := 0; r < seq; r++ {
		gc, _ := argmax(logitsCPU, r)
		gg, _ := argmax(logitsGPU, r)
		if gc != gg {
			mism++
			t.Errorf("pos %d: GPU argmax token %d != CPU %d", r, gg, gc)
		}
		for c := 0; c < vocab; c++ {
			cv := logitsCPU.AtF64(r, c)
			if rr := math.Abs(logitsGPU.AtF64(r, c)-cv) / (math.Abs(cv) + 1); rr > maxRel {
				maxRel = rr
			}
		}
	}
	t.Logf("TinyLlama full forward (22 layers): seq=%d vocab=%d, %d/%d positions match (greedy), logit max rel err %.2e",
		seq, vocab, seq-mism, seq, maxRel)
}
