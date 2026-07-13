//go:build darwin && cgo

// Package gpudecode's test is the §C3 real-workload wire-in for ADR-0019 Phase 2 (§T392): it drives
// a REAL nlp.Llama's decode through the exported Metal batched-decode Recorder and confirms the
// logits match the model's own per-op DecodeStep on the reference backend. This is the end-to-end
// proof that the batched recorder reproduces the actual library model (catching any convention
// mismatch — RoPE freqs, RMSNorm eps, weight orientation — that a hand-written host reference could
// get wrong). The ~3.45× speedup itself is measured in backend/metal §T389.
package gpudecode_test

import (
	"math"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	_ "github.com/jxsl13/goai/backend/ref" // register the reference backend (§I4) for DecodeStep
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// flat2D flattens a [r,c] tensor to row-major f32.
func flat2D(t *tensor.Tensor) []float32 {
	r, c := t.Shape()[0], t.Shape()[1]
	o := make([]float32, r*c)
	for i := range r {
		for j := range c {
			o[i*c+j] = float32(t.AtF64(i, j))
		}
	}
	return o
}

func flat1D(t *tensor.Tensor) []float32 {
	n := t.Shape()[0]
	o := make([]float32, n)
	for i := range n {
		o[i] = float32(t.AtF64(i))
	}
	return o
}

func TestLlamaRecorderMatchesReference(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 40, Ctx: 64, Dim: 64, Heads: 8, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	D := cfg.Dim
	H := cfg.Heads
	KVH := cfg.KVHeads
	dk := D / H
	kvDim := KVH * dk
	half := dk / 2
	hidden := cfg.Hidden
	V := cfg.Vocab
	eps := float32(cfg.Eps)
	scale := float32(1.0 / math.Sqrt(float64(dk)))

	// RoPE inverse frequencies exactly as the Metal OpRoPE path computes them, so the recorder's
	// rotation matches the library's per-op RoPE bit-for-bit in intent.
	invF64, posDiv64 := backend.RoPEFreqs(dk, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: H})
	inv := make([]float32, half)
	for i := range inv {
		inv[i] = float32(invF64[i])
	}
	posDiv := float32(posDiv64)

	// upload the real model weights (F64→F32) into device buffers.
	mk := func(data []float32) *metal.DeviceBuffer {
		b, e := metal.NewDeviceBufferF32(data)
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	type gpuBlock struct {
		wq, wk, wv, wo, gAttn, gFFN, wG, wU, wD, kC, vC *metal.DeviceBuffer
	}
	var blocks []gpuBlock
	var all []*metal.DeviceBuffer
	const maxLen = 64
	for _, b := range m.Blocks {
		gb := gpuBlock{
			wq: mk(flat2D(b.Wq)), wk: mk(flat2D(b.Wk)), wv: mk(flat2D(b.Wv)), wo: mk(flat2D(b.Wo)),
			gAttn: mk(flat1D(b.AttnNorm.Gamma)), gFFN: mk(flat1D(b.FFNNorm.Gamma)),
			wG: mk(flat2D(b.FFN.Wgate)), wU: mk(flat2D(b.FFN.Wup)), wD: mk(flat2D(b.FFN.Wdown)),
			kC: mk(make([]float32, maxLen*kvDim)), vC: mk(make([]float32, maxLen*kvDim)),
		}
		blocks = append(blocks, gb)
		all = append(all, gb.wq, gb.wk, gb.wv, gb.wo, gb.gAttn, gb.gFFN, gb.wG, gb.wU, gb.wD, gb.kC, gb.vC)
	}
	emb := mk(flat2D(m.TokEmb))
	gFinal := mk(flat1D(m.Norm.Gamma))
	wOut := mk(flat2D(m.Out))
	dinv := mk(inv)
	// per-step scratch
	dx := mk(make([]float32, D))
	xn := mk(make([]float32, D))
	q := mk(make([]float32, D))
	k := mk(make([]float32, kvDim))
	v := mk(make([]float32, kvDim))
	attn := mk(make([]float32, D))
	ao := mk(make([]float32, D))
	xn2 := mk(make([]float32, D))
	gate := mk(make([]float32, hidden))
	up := mk(make([]float32, hidden))
	mo := mk(make([]float32, D))
	logits := mk(make([]float32, V))
	all = append(all, emb, gFinal, wOut, dinv, dx, xn, q, k, v, attn, ao, xn2, gate, up, mo, logits)
	defer func() {
		for _, b := range all {
			b.Release()
		}
	}()

	// reference: the library's own per-op decode on the reference backend (f64 ground truth).
	refCtx := backend.NewContext().WithBackend(backend.Reference())
	cache := m.NewCache()

	// GPU: one Recorder-driven decode step over the device-resident model, mirroring DecodeStep.
	gpuStep := func(token, pos int) []float32 {
		if err := dx.UploadF32(flat2Drow(m.TokEmb, token, D)); err != nil {
			t.Fatal(err)
		}
		r, e := metal.NewRecorder()
		if e != nil {
			t.Fatal(e)
		}
		for _, b := range blocks {
			must(t, r.RMSNorm(dx, b.gAttn, xn, 1, D, eps))
			must(t, r.MatMul(xn, b.wq, q, 1, D, D))
			must(t, r.MatMul(xn, b.wk, k, 1, D, kvDim))
			must(t, r.MatMul(xn, b.wv, v, 1, D, kvDim))
			must(t, r.RoPE(q, dinv, q, 1, D, H, dk, half, pos, posDiv))
			must(t, r.RoPE(k, dinv, k, 1, kvDim, KVH, dk, half, pos, posDiv))
			must(t, r.Blit(k, 0, b.kC, pos*kvDim, kvDim))
			must(t, r.Blit(v, 0, b.vC, pos*kvDim, kvDim))
			must(t, r.MHA(q, b.kC, b.vC, attn, 1, pos+1, D, H, KVH, dk, 1, 0, scale))
			must(t, r.MatMul(attn, b.wo, ao, 1, D, D))
			must(t, r.Binary(dx, ao, dx, 0))
			must(t, r.RMSNorm(dx, b.gFFN, xn2, 1, D, eps))
			must(t, r.MatMul(xn2, b.wG, gate, 1, D, hidden))
			must(t, r.Unary(gate, gate, 6)) // 6 = SiLU (shaders/unary.comp selector)
			must(t, r.MatMul(xn2, b.wU, up, 1, D, hidden))
			must(t, r.Binary(gate, up, gate, 2))
			must(t, r.MatMul(gate, b.wD, mo, 1, hidden, D))
			must(t, r.Binary(dx, mo, dx, 0))
		}
		must(t, r.RMSNorm(dx, gFinal, xn, 1, D, eps))
		must(t, r.MatMul(xn, wOut, logits, 1, D, V))
		must(t, r.Finish())
		r.Free()
		out := make([]float32, V)
		if err := logits.DownloadF32(out); err != nil {
			t.Fatal(err)
		}
		return out
	}

	toks := []int{7, 3, 11, 0, 5}
	for pos, tok := range toks {
		gpu := gpuStep(tok, pos)
		refT, err := m.DecodeStep(refCtx, cache, tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		for j := range V {
			want := refT.AtF64(0, j)
			if math.Abs(float64(gpu[j])-want) > 3e-2*math.Max(1, math.Abs(want)) {
				t.Fatalf("pos %d tok %d logit[%d]: recorder %v vs reference %v", pos, tok, j, gpu[j], want)
			}
		}
	}
	t.Logf("recorder decode matches the reference nlp.Llama DecodeStep across %d tokens (GQA %d:%d, SwiGLU, %d layers)", len(toks), H, KVH, cfg.Layers)
}

func flat2Drow(t *tensor.Tensor, row, cols int) []float32 {
	o := make([]float32, cols)
	for j := range cols {
		o[j] = float32(t.AtF64(row, j))
	}
	return o
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// §T404 (§C3 decode proof): the metal DECODE audit (§T402) found metal per-op decode is dispatch-
// bound — SLOWER than cpu on a small model. ADR-0019's batched recorder (one command buffer/step)
// is the fix. This measures the REAL win: a batched-recorder decode of a REAL nlp.Llama vs the
// library's own per-op DecodeStep on the SAME metal backend, at a realistic size, in tokens/s.
func TestLlamaDecodeBatchedVsPerOpThroughput(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	metalBE, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("metal backend not registered")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 1024, Ctx: 128, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	D, H, KVH := cfg.Dim, cfg.Heads, cfg.KVHeads
	dk := D / H
	kvDim := KVH * dk
	half := dk / 2
	hidden, V := cfg.Hidden, cfg.Vocab
	eps := float32(cfg.Eps)
	scale := float32(1.0 / math.Sqrt(float64(dk)))
	const maxLen = 128

	invF64, posDiv64 := backend.RoPEFreqs(dk, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: H})
	inv := make([]float32, half)
	for i := range inv {
		inv[i] = float32(invF64[i])
	}
	posDiv := float32(posDiv64)

	var all []*metal.DeviceBuffer
	mk := func(data []float32) *metal.DeviceBuffer {
		b, e := metal.NewDeviceBufferF32(data)
		if e != nil {
			t.Fatal(e)
		}
		all = append(all, b)
		return b
	}
	type gpuBlock struct {
		wq, wk, wv, wo, gAttn, gFFN, wG, wU, wD, kC, vC *metal.DeviceBuffer
	}
	var blocks []gpuBlock
	for _, b := range m.Blocks {
		blocks = append(blocks, gpuBlock{
			wq: mk(flat2D(b.Wq)), wk: mk(flat2D(b.Wk)), wv: mk(flat2D(b.Wv)), wo: mk(flat2D(b.Wo)),
			gAttn: mk(flat1D(b.AttnNorm.Gamma)), gFFN: mk(flat1D(b.FFNNorm.Gamma)),
			wG: mk(flat2D(b.FFN.Wgate)), wU: mk(flat2D(b.FFN.Wup)), wD: mk(flat2D(b.FFN.Wdown)),
			kC: mk(make([]float32, maxLen*kvDim)), vC: mk(make([]float32, maxLen*kvDim)),
		})
	}
	gFinal, wOut, dinv := mk(flat1D(m.Norm.Gamma)), mk(flat2D(m.Out)), mk(inv)
	dx, xn, xn2 := mk(make([]float32, D)), mk(make([]float32, D)), mk(make([]float32, D))
	q, k, v := mk(make([]float32, D)), mk(make([]float32, kvDim)), mk(make([]float32, kvDim))
	attn, ao := mk(make([]float32, D)), mk(make([]float32, D))
	gate, up, mo := mk(make([]float32, hidden)), mk(make([]float32, hidden)), mk(make([]float32, D))
	logits := mk(make([]float32, V))
	defer func() {
		for _, b := range all {
			b.Release()
		}
	}()

	recStep := func(token, pos int) {
		must(t, dx.UploadF32(flat2Drow(m.TokEmb, token, D)))
		r, e := metal.NewRecorder()
		if e != nil {
			t.Fatal(e)
		}
		for _, b := range blocks {
			must(t, r.RMSNorm(dx, b.gAttn, xn, 1, D, eps))
			must(t, r.MatMul(xn, b.wq, q, 1, D, D))
			must(t, r.MatMul(xn, b.wk, k, 1, D, kvDim))
			must(t, r.MatMul(xn, b.wv, v, 1, D, kvDim))
			must(t, r.RoPE(q, dinv, q, 1, D, H, dk, half, pos, posDiv))
			must(t, r.RoPE(k, dinv, k, 1, kvDim, KVH, dk, half, pos, posDiv))
			must(t, r.Blit(k, 0, b.kC, pos*kvDim, kvDim))
			must(t, r.Blit(v, 0, b.vC, pos*kvDim, kvDim))
			must(t, r.MHA(q, b.kC, b.vC, attn, 1, pos+1, D, H, KVH, dk, 1, 0, scale))
			must(t, r.MatMul(attn, b.wo, ao, 1, D, D))
			must(t, r.Binary(dx, ao, dx, 0))
			must(t, r.RMSNorm(dx, b.gFFN, xn2, 1, D, eps))
			must(t, r.MatMul(xn2, b.wG, gate, 1, D, hidden))
			must(t, r.Unary(gate, gate, 6))
			must(t, r.MatMul(xn2, b.wU, up, 1, D, hidden))
			must(t, r.Binary(gate, up, gate, 2))
			must(t, r.MatMul(gate, b.wD, mo, 1, hidden, D))
			must(t, r.Binary(dx, mo, dx, 0))
		}
		must(t, r.RMSNorm(dx, gFinal, xn, 1, D, eps))
		must(t, r.MatMul(xn, wOut, logits, 1, D, V))
		must(t, r.Finish())
		r.Free()
		out := make([]float32, V)
		must(t, logits.DownloadF32(out))
	}

	const steps = 32
	// batched recorder decode
	recStep(0, 0) // warmup
	sRec := time.Now()
	for pos := range steps {
		recStep(pos%V, pos)
	}
	recTokps := float64(steps) / time.Since(sRec).Seconds()

	// library per-op decode on the SAME metal backend
	mctx := backend.NewContext().WithBackend(metalBE)
	cache := m.NewCache()
	if _, err := m.DecodeStep(mctx, cache, 0, 0); err != nil { // warmup
		t.Fatal(err)
	}
	cache = m.NewCache()
	sPer := time.Now()
	for pos := range steps {
		if _, err := m.DecodeStep(mctx, cache, pos%V, pos); err != nil {
			t.Fatal(err)
		}
	}
	perTokps := float64(steps) / time.Since(sPer).Seconds()

	t.Logf("Llama decode D=%d GQA%d:%d %d-layer V=%d: batched recorder %.1f tok/s | nlp per-op DecodeStep(metal) %.1f tok/s | speedup %.2fx",
		D, H, KVH, cfg.Layers, V, recTokps, perTokps, recTokps/perTokps)
}
